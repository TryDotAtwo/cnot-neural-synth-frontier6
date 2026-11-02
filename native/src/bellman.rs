//! Exact full-neighbor Bellman selection and frozen-provenance loss.
//!
//! Selection runs without autograd, across every primitive action and optional
//! verified macro. The chosen parent/successor packets are retained and checked
//! before the differentiable pass so the objective cannot silently change its
//! target between selection and backward.

use crate::alphabet::{CertificateError, CnotAlphabet, WordCertificate};
use crate::features::{FeatureBuildError, FeaturePolicy, FeatureRegistry};
use crate::model::{FusionDistanceField, ModelError};
use crate::packet::{FeaturePacket, FeaturePacketError};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt::{Display, Formatter};
use tch::nn::VarStore;
use tch::{Device, Kind, Reduction, Tensor};

#[derive(Debug)]
pub enum BellmanError {
    Invalid(String),
    FeatureBuild(FeatureBuildError),
    FeaturePacket(FeaturePacketError),
    Model(ModelError),
    Certificate(CertificateError),
    Torch(tch::TchError),
}

impl Display for BellmanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(reason) => formatter.write_str(reason),
            Self::FeatureBuild(error) => Display::fmt(error, formatter),
            Self::FeaturePacket(error) => Display::fmt(error, formatter),
            Self::Model(error) => Display::fmt(error, formatter),
            Self::Certificate(error) => Display::fmt(error, formatter),
            Self::Torch(error) => write!(formatter, "Bellman tensor operation failed: {error}"),
        }
    }
}

impl Error for BellmanError {}

impl From<FeatureBuildError> for BellmanError {
    fn from(value: FeatureBuildError) -> Self {
        Self::FeatureBuild(value)
    }
}

impl From<FeaturePacketError> for BellmanError {
    fn from(value: FeaturePacketError) -> Self {
        Self::FeaturePacket(value)
    }
}

impl From<ModelError> for BellmanError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl From<CertificateError> for BellmanError {
    fn from(value: CertificateError) -> Self {
        Self::Certificate(value)
    }
}

impl From<tch::TchError> for BellmanError {
    fn from(value: tch::TchError) -> Self {
        Self::Torch(value)
    }
}

/// Supplies optional verified paid macros in deterministic caller order.
pub trait MacroProvider {
    fn candidates(&self, state: &MatrixState) -> Result<Vec<WordCertificate>, String>;
}

impl<F> MacroProvider for F
where
    F: Fn(&MatrixState) -> Result<Vec<WordCertificate>, String>,
{
    fn candidates(&self, state: &MatrixState) -> Result<Vec<WordCertificate>, String> {
        self(state)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BellmanBranchAudit {
    pub parent: crate::packet::StateIdentity,
    pub selected_index: Option<usize>,
    pub selected_cost: Option<u64>,
    pub candidate_count: usize,
    pub primitive_count: usize,
    pub packet_sha256: String,
    pub macro_margin: Option<f64>,
    pub primitive_totals: Vec<f64>,
}

#[derive(Debug)]
struct PreparedBranch {
    parent: MatrixState,
    parent_packet: FeaturePacket,
    successor: Option<MatrixState>,
    successor_packet: Option<FeaturePacket>,
    cost: u64,
    packet_digest: [u8; 32],
    audit: BellmanBranchAudit,
}

/// A detached greedy selection tied to one exact set of model weights.
#[derive(Debug)]
pub struct BellmanSelection {
    branches: Vec<PreparedBranch>,
    theta_digest: [u8; 32],
}

impl BellmanSelection {
    pub fn branches(&self) -> impl Iterator<Item = &BellmanBranchAudit> {
        self.branches.iter().map(|branch| &branch.audit)
    }

    /// States whose values participate in the differentiable Bellman residual.
    pub fn gradient_exposure_states(&self) -> Vec<MatrixState> {
        self.branches
            .iter()
            .flat_map(|branch| {
                std::iter::once(branch.parent.clone()).chain(branch.successor.iter().cloned())
            })
            .collect()
    }

    pub fn theta_sha256(&self) -> String {
        hex_digest(&self.theta_digest)
    }
}

#[derive(Debug)]
pub struct BellmanLoss {
    pub loss: Tensor,
    pub diagnostics: BellmanDiagnostics,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BellmanDiagnostics {
    pub examples: usize,
    pub terminal_count: usize,
    pub candidate_count: usize,
    pub macro_margins: Vec<f64>,
    pub approximate: bool,
}

/// Select the minimum paid-cost primitive successor for every state.
pub fn select_backup(
    model: &FusionDistanceField,
    variables: &VarStore,
    states: &[MatrixState],
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    chunk_size: Option<usize>,
) -> Result<BellmanSelection, BellmanError> {
    select_backup_with_macros(
        model, variables, states, registry, policy, alphabet, None, chunk_size,
    )
}

/// Select the minimum paid-cost successor across all primitive and verified macro actions.
pub fn select_backup_with_macros(
    model: &FusionDistanceField,
    variables: &VarStore,
    states: &[MatrixState],
    registry: &FeatureRegistry,
    policy: &FeaturePolicy,
    alphabet: &CnotAlphabet,
    macro_provider: Option<&dyn MacroProvider>,
    chunk_size: Option<usize>,
) -> Result<BellmanSelection, BellmanError> {
    if states.is_empty() {
        return Err(BellmanError::Invalid("empty Bellman batch".into()));
    }
    if chunk_size.is_some_and(|size| size == 0) {
        return Err(BellmanError::Invalid(
            "positive neighbor chunk size required".into(),
        ));
    }
    if policy.evaluations.is_some() {
        return Err(BellmanError::Invalid(
            "finite feature-evaluation budgets can change the selected packets; use frozen unlimited deterministic producers".into(),
        ));
    }
    if variables.device() != model.device() {
        return Err(BellmanError::Invalid(
            "model and variable store are on different devices".into(),
        ));
    }

    let theta_digest = theta_fingerprint(variables)?;
    let mut branches = Vec::with_capacity(states.len());
    for state in states {
        if state.alphabet_id() != alphabet.id() || state.dimension() != alphabet.dimension() {
            return Err(BellmanError::Invalid(
                "state/alphabet mismatch in Bellman batch".into(),
            ));
        }
        let parent_packet = registry.build(std::slice::from_ref(state), policy)?;
        if state.is_identity() {
            let packet_digest = packet_fingerprint(&parent_packet, None);
            branches.push(PreparedBranch {
                parent: state.clone(),
                audit: BellmanBranchAudit {
                    parent: parent_packet.state_keys[0].clone(),
                    selected_index: None,
                    selected_cost: None,
                    candidate_count: 0,
                    primitive_count: 0,
                    packet_sha256: hex_digest(&packet_digest),
                    macro_margin: None,
                    primitive_totals: Vec::new(),
                },
                parent_packet,
                successor: None,
                successor_packet: None,
                cost: 0,
                packet_digest,
            });
            continue;
        }

        let mut successors = Vec::with_capacity(alphabet.gates().len());
        let mut costs = Vec::with_capacity(alphabet.gates().len());
        for (index, gate) in alphabet.gates().iter().copied().enumerate() {
            let mut successor = state.clone();
            successor
                .apply_cnot(gate)
                .map_err(|error| BellmanError::Invalid(error.to_string()))?;
            successors.push(successor);
            costs.push(u64::from(alphabet.cost_at(index).ok_or_else(|| {
                BellmanError::Invalid("alphabet gate cost is missing".into())
            })?));
        }
        let primitive_count = successors.len();
        if let Some(provider) = macro_provider {
            for certificate in provider.candidates(state).map_err(BellmanError::Invalid)? {
                certificate.verify(alphabet)?;
                if certificate.start != *state || certificate.physical_cost == 0 {
                    return Err(BellmanError::Invalid(
                        "macro certificate is not bound to its parent or has nonpositive cost"
                            .into(),
                    ));
                }
                successors.push(certificate.end.clone());
                costs.push(certificate.physical_cost);
            }
        }
        if successors.is_empty() {
            return Err(BellmanError::Invalid(
                "nonterminal state has an empty move alphabet".into(),
            ));
        }

        let width = chunk_size.unwrap_or(successors.len());
        let mut best = f64::INFINITY;
        let mut selected_index = None;
        let mut selected_cost = None;
        let mut selected_packet = None;
        let mut totals = Vec::with_capacity(successors.len());
        for start in (0..successors.len()).step_by(width) {
            let end = (start + width).min(successors.len());
            let chunk = &successors[start..end];
            let packet = registry.build(chunk, policy)?;
            let chunk_totals = tch::no_grad(|| -> Result<Vec<f64>, BellmanError> {
                let values = model.forward(&packet)?.value;
                if values.isfinite().all().int64_value(&[]) == 0 {
                    return Err(BellmanError::Invalid(
                        "nonfinite Bellman candidate value".into(),
                    ));
                }
                let flat = values
                    .to_device(Device::Cpu)
                    .to_kind(Kind::Double)
                    .contiguous()
                    .view([-1]);
                let values = Vec::<f64>::try_from(&flat)?;
                if values.len() != end - start {
                    return Err(BellmanError::Invalid(
                        "model candidate count differs from successor chunk".into(),
                    ));
                }
                values
                    .into_iter()
                    .zip(costs[start..end].iter().copied())
                    .map(|(value, cost)| candidate_total(value, cost))
                    .collect()
            })?;
            if chunk_totals.len() != end - start {
                return Err(BellmanError::Invalid(
                    "model candidate count differs from successor chunk".into(),
                ));
            }
            for (local_index, total) in chunk_totals.iter().copied().enumerate() {
                totals.push(total);
                if total < best {
                    best = total;
                    let global_index = start + local_index;
                    selected_index = Some(global_index);
                    selected_cost = Some(costs[global_index]);
                    selected_packet = Some(packet.single_row(local_index)?);
                }
            }
        }
        let selected_index = selected_index
            .ok_or_else(|| BellmanError::Invalid("no finite Bellman candidate selected".into()))?;
        let min_primitive = totals[..primitive_count].iter().copied().reduce(f64::min);
        let min_macro = totals[primitive_count..].iter().copied().reduce(f64::min);
        let macro_margin = min_macro
            .zip(min_primitive)
            .map(|(macro_cost, primitive_cost)| macro_cost - primitive_cost);
        let successor = successors[selected_index].clone();
        let selected_packet = selected_packet
            .ok_or_else(|| BellmanError::Invalid("selected successor packet is missing".into()))?;
        let packet_digest = packet_fingerprint(&parent_packet, Some(&selected_packet));
        branches.push(PreparedBranch {
            parent: state.clone(),
            audit: BellmanBranchAudit {
                parent: parent_packet.state_keys[0].clone(),
                selected_index: Some(selected_index),
                selected_cost,
                candidate_count: successors.len(),
                primitive_count,
                packet_sha256: hex_digest(&packet_digest),
                macro_margin,
                primitive_totals: totals[..primitive_count].to_vec(),
            },
            parent_packet,
            successor: Some(successor),
            successor_packet: Some(selected_packet),
            cost: selected_cost.unwrap_or_default(),
            packet_digest,
        });
    }

    if theta_fingerprint(variables)? != theta_digest {
        return Err(BellmanError::Invalid(
            "model parameters changed during Bellman selection".into(),
        ));
    }
    Ok(BellmanSelection {
        branches,
        theta_digest,
    })
}

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

fn candidate_total(value: f64, cost: u64) -> Result<f64, BellmanError> {
    let total = value + cost as f64;
    if total.is_finite() {
        Ok(total)
    } else {
        Err(BellmanError::Invalid(
            "nonfinite Bellman candidate cost".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn candidate_cost_order_is_not_collapsed_to_float32_precision() {
        let lower = super::candidate_total(0.0, 16_777_216).unwrap();
        let higher = super::candidate_total(0.0, 16_777_217).unwrap();
        assert!(lower < higher);
        assert_eq!(lower as f32, higher as f32);
    }
}

fn theta_fingerprint(variables: &VarStore) -> Result<[u8; 32], BellmanError> {
    let mut named: Vec<_> = variables.variables().into_iter().collect();
    named.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = Sha256::new();
    for (name, value) in named {
        hash_bytes(&mut digest, name.as_bytes());
        hash_bytes(&mut digest, format!("{:?}", value.kind()).as_bytes());
        hash_u64(&mut digest, value.size().len() as u64);
        for extent in value.size() {
            digest.update(extent.to_le_bytes());
        }
        let value = value.detach().to_device(Device::Cpu).contiguous();
        let byte_count = value
            .numel()
            .checked_mul(value.kind().elt_size_in_bytes())
            .ok_or_else(|| BellmanError::Invalid("parameter byte size overflow".into()))?;
        let mut bytes = vec![0u8; byte_count];
        value.f_copy_data_u8(&mut bytes, value.numel())?;
        hash_bytes(&mut digest, &bytes);
    }
    Ok(digest.finalize().into())
}

fn packet_fingerprint(parent: &FeaturePacket, successor: Option<&FeaturePacket>) -> [u8; 32] {
    let mut digest = Sha256::new();
    hash_packet(&mut digest, parent);
    if let Some(successor) = successor {
        digest.update([1]);
        hash_packet(&mut digest, successor);
    } else {
        digest.update([0]);
    }
    digest.finalize().into()
}

fn hash_packet(digest: &mut Sha256, packet: &FeaturePacket) {
    hash_bytes(digest, packet.feature_id_manifest_id.as_bytes());
    hash_text(digest, &packet.schema_id);
    hash_text(digest, &packet.policy_id);
    hash_u64(digest, packet.batch as u64);
    hash_u64(digest, packet.nodes as u64);
    for value in &packet.valid_nodes {
        digest.update([*value]);
    }
    for (identity, alphabet) in packet.state_keys.iter().zip(&packet.alphabet_ids) {
        hash_u64(digest, u64::from(identity.dimension));
        hash_text(digest, &identity.alphabet_id);
        hash_text(digest, alphabet);
        hash_u64(digest, identity.packed_rows.len() as u64);
        for word in &identity.packed_rows {
            digest.update(word.to_le_bytes());
        }
    }
    hash_u64(digest, packet.entries.len() as u64);
    for entry in &packet.entries {
        hash_text(digest, &entry.spec.id.to_string());
        hash_text(digest, &format!("{:?}", entry.spec.role));
        hash_u64(digest, entry.spec.channels as u64);
        hash_text(digest, &entry.spec.scope);
        hash_text(digest, &entry.spec.producer_version);
        hash_u64(digest, entry.spec.dependencies.len() as u64);
        for dependency in &entry.spec.dependencies {
            hash_text(digest, dependency);
        }
        hash_text(digest, &entry.spec.exactness);
        hash_text(digest, &entry.spec.cost_tier);
        hash_u64(digest, entry.spec.wire_roles.len() as u64);
        for role in &entry.spec.wire_roles {
            hash_text(digest, role);
        }
        digest.update([u8::from(entry.spec.requires_solver_assisted)]);
        hash_u64(digest, entry.shape.len() as u64);
        for extent in &entry.shape {
            hash_u64(digest, *extent as u64);
        }
        for value in &entry.values {
            digest.update(value.to_bits().to_le_bytes());
        }
        for value in &entry.available {
            digest.update([*value]);
        }
        for status in &entry.status {
            hash_text(digest, status.as_str());
        }
        for reason in &entry.reason {
            hash_text(digest, reason);
        }
        for role in &entry.wire_roles {
            hash_text(digest, role);
        }
        hash_optional_values(digest, entry.token_valid.as_deref());
        hash_optional_f32(digest, entry.incidence.as_deref());
    }
}

fn hash_optional_values(digest: &mut Sha256, values: Option<&[u8]>) {
    match values {
        Some(values) => {
            digest.update([1]);
            hash_u64(digest, values.len() as u64);
            digest.update(values);
        }
        None => digest.update([0]),
    }
}

fn hash_optional_f32(digest: &mut Sha256, values: Option<&[f32]>) {
    match values {
        Some(values) => {
            digest.update([1]);
            hash_u64(digest, values.len() as u64);
            for value in values {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
        None => digest.update([0]),
    }
}

fn hash_text(digest: &mut Sha256, value: &str) {
    hash_bytes(digest, value.as_bytes());
}

fn hash_bytes(digest: &mut Sha256, value: &[u8]) {
    hash_u64(digest, value.len() as u64);
    digest.update(value);
}

fn hash_u64(digest: &mut Sha256, value: u64) {
    digest.update(value.to_le_bytes());
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
