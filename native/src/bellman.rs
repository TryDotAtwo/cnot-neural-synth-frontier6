//! Exact full-neighbor Bellman selection and frozen-provenance loss.
//!
//! Selection runs without autograd, across every primitive action and optional
//! verified macro. The chosen parent/successor packets are retained and checked
//! before the differentiable pass so the objective cannot silently change its
//! target between selection and backward.

mod fingerprint;
mod recompute;
mod selection;

pub use recompute::recompute_backup;
pub(crate) use selection::select_backup_with_model_input_coverage;
pub use selection::{select_backup, select_backup_with_macros, BellmanSelectionConfig};

use crate::alphabet::{CertificateError, WordCertificate};
use crate::features::FeatureBuildError;
use crate::model::ModelError;
use crate::packet::{FeaturePacket, FeaturePacketError};
use crate::state::MatrixState;
use std::error::Error;
use std::fmt::{Display, Formatter};
use tch::Tensor;

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

    pub(crate) fn parent_packet(&self, index: usize) -> Option<&FeaturePacket> {
        self.branches.get(index).map(|branch| &branch.parent_packet)
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
        fingerprint::hex_digest(&self.theta_digest)
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
