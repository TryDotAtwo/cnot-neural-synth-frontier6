//! Paid-basis proposal head used by the production full-width model.
//!
//! The facade keeps the stable public API while separating the serialized
//! template contract from its neural proposer and exact loss implementation.

mod compiler;
mod decomposition;
mod model;
mod objective;
mod template;

pub use crate::paid_basis::BasisSide;
pub use compiler::{compile_basis, BasisCandidate};
pub use decomposition::{partial_shear, recognize_shear, BasisBinding, ShearDecomposition};
pub use model::{BasisField, BasisFieldOutput};
pub use objective::{BasisLossAudit, BasisLossBackend};
pub use template::{BasisChoice, BasisChoiceKind, BasisTemplate};
