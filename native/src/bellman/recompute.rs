use super::fingerprint::{packet_fingerprint, theta_fingerprint};
use super::{BellmanDiagnostics, BellmanError, BellmanLoss, BellmanSelection};
use crate::model::FusionDistanceField;
use tch::nn::VarStore;
use tch::{Reduction, Tensor};

/// Recompute the frozen greedy targets with autograd and return mean smooth-L1 loss.
pub fn recompute_backup(
    model: &FusionDistanceField,
    variables: &VarStore,
    selection: &BellmanSelection,
) -> Result<BellmanLoss, BellmanError> {
    if theta_fingerprint(variables)? != selection.theta_digest {
        return Err(BellmanError::Invalid(
            "model parameters changed after Bellman selection".into(),
        ));
    }
    if variables.device() != model.device() {
        return Err(BellmanError::Invalid(
            "model and variable store are on different devices".into(),
        ));
    }

    let mut residuals = Vec::with_capacity(selection.branches.len());
    let mut terminal_count = 0usize;
    let mut candidate_count = 0usize;
    let mut macro_margins = Vec::new();
    for branch in &selection.branches {
        if packet_fingerprint(&branch.parent_packet, branch.successor_packet.as_ref())
            != branch.packet_digest
        {
            return Err(BellmanError::Invalid(
                "feature packet changed after Bellman selection".into(),
            ));
        }
        let parent_value = model.forward(&branch.parent_packet)?.value;
        if parent_value.numel() != 1 || parent_value.isfinite().all().int64_value(&[]) == 0 {
            return Err(BellmanError::Invalid(
                "Bellman parent value is not a finite scalar".into(),
            ));
        }
        let residual = if let (Some(_successor), Some(successor_packet)) =
            (&branch.successor, &branch.successor_packet)
        {
            let successor_value = model.forward(successor_packet)?.value;
            if successor_value.numel() != 1
                || successor_value.isfinite().all().int64_value(&[]) == 0
            {
                return Err(BellmanError::Invalid(
                    "Bellman successor value is not a finite scalar".into(),
                ));
            }
            parent_value.select(0, 0) - branch.cost as f64 - successor_value.select(0, 0)
        } else {
            terminal_count += 1;
            parent_value.select(0, 0)
        };
        residuals.push(residual);
        candidate_count = candidate_count
            .checked_add(branch.audit.candidate_count)
            .ok_or_else(|| BellmanError::Invalid("candidate count overflow".into()))?;
        if let Some(margin) = branch.audit.macro_margin {
            macro_margins.push(margin);
        }
    }
    if residuals.is_empty() {
        return Err(BellmanError::Invalid("empty Bellman selection".into()));
    }
    let residuals = Tensor::stack(&residuals, 0);
    let loss = residuals.smooth_l1_loss(&Tensor::zeros_like(&residuals), Reduction::Mean, 1.0);
    if loss.numel() != 1 || loss.isfinite().all().int64_value(&[]) == 0 {
        return Err(BellmanError::Invalid("nonfinite Bellman loss".into()));
    }
    Ok(BellmanLoss {
        loss,
        diagnostics: BellmanDiagnostics {
            examples: selection.branches.len(),
            terminal_count,
            candidate_count,
            macro_margins,
            approximate: false,
        },
    })
}
