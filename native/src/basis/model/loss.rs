use super::super::{
    objective, partial_shear, BasisBinding, BasisLossAudit, BasisLossBackend, BasisTemplate,
};
use super::BasisField;
use crate::model::ModelError;
use crate::packet::FeaturePacket;
use crate::state::MatrixState;
use tch::Tensor;

impl BasisField {
    /// Train the basis proposal against the paid cost of a bound shear.
    ///
    /// The packet must contain all 103 resolved features for the same state;
    /// the selected source binding is re-derived before the objective runs.
    #[allow(clippy::too_many_arguments)]
    pub fn basis_loss(
        &self,
        state: &MatrixState,
        packet: &FeaturePacket,
        template: &BasisTemplate,
        block: &[Vec<u8>],
        batch_index: usize,
        binding: BasisBinding,
        backend: BasisLossBackend,
        samples: usize,
        seed: u64,
        max_evaluations: usize,
        max_bytes: usize,
    ) -> Result<(Tensor, BasisLossAudit), ModelError> {
        template.validate()?;
        if state.dimension() as usize != template.n {
            return Err(ModelError("state/template width mismatch".into()));
        }
        let packet_state = packet
            .state_keys
            .get(batch_index)
            .ok_or_else(|| ModelError("basis state batch index is outside packet".into()))?;
        if packet_state != &crate::packet::StateIdentity::from(state) {
            return Err(ModelError(
                "feature packet is not bound to the basis objective state".into(),
            ));
        }
        if block.len() != template.target_wires.len()
            || block.iter().any(|row| {
                row.len() != template.source_wires.len() || row.iter().any(|bit| *bit > 1)
            })
        {
            return Err(ModelError(
                "basis block must be a binary target-by-source matrix".into(),
            ));
        }
        match binding {
            BasisBinding::FullShear => bind_full_shear(state, template, block)?,
            BasisBinding::PartialElimination => {
                let parsed = partial_shear(state)?.ok_or_else(|| {
                    ModelError("matrix has no nonzero partial-elimination shear".into())
                })?;
                if parsed.source_wires != template.source_wires
                    || parsed.target_wires != template.target_wires
                    || parsed.block != block
                {
                    return Err(ModelError(
                        "basis block is not bound to the partial elimination".into(),
                    ));
                }
            }
        }

        let encoded = self.field.encode(packet)?;
        let probabilities = self.proposer.forward(&encoded, template, batch_index)?;
        let result = objective::paid_cost_loss(
            template,
            block,
            &probabilities,
            binding,
            backend,
            samples,
            seed,
            max_evaluations,
            max_bytes,
        )?;
        Ok((result.loss, result.audit))
    }
}

fn bind_full_shear(
    state: &MatrixState,
    template: &BasisTemplate,
    block: &[Vec<u8>],
) -> Result<(), ModelError> {
    let identity = MatrixState::identity(state.dimension(), state.alphabet_id())
        .map_err(|error| ModelError(error.to_string()))?;
    let words_per_row = identity.words_per_row();
    let mut expected_rows = identity.packed_rows().to_vec();
    for (target_index, &target_wire) in template.target_wires.iter().enumerate() {
        for (source_index, &source_wire) in template.source_wires.iter().enumerate() {
            if block[target_index][source_index] == 1 {
                let offset = target_wire * words_per_row + source_wire / 64;
                expected_rows[offset] ^= 1u64 << (source_wire % 64);
            }
        }
    }
    if expected_rows != state.packed_rows() {
        return Err(ModelError(
            "basis block is not bound to the full matrix shear".into(),
        ));
    }
    Ok(())
}
