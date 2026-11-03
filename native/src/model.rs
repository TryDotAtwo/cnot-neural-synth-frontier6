//! Neural distance model contract for the canonical CNOT feature manifest.
//!
//! The model consumes the executable model-input features; H07 remains a
//! post-forward diagnostic. This module re-exports the stable API and composes
//! the model contract, tensor conversion, and network components.

mod auxiliary;
mod contracts;
mod family_heads;
mod field;
mod fusion;
mod layers;
mod packet;
mod stem;

pub(super) const AUXILIARY_HIDDEN_DIM: i64 = 128;

pub use contracts::{FieldOutput, ModelConfig, ModelError};
pub use field::FusionDistanceField;
pub(super) use layers::{pytorch_linear_config, Mlp, MultiHeadAttention, StructuralBlock};
use packet::{matrix_batch_values, shape_i64};
pub(crate) use stem::EncodedState;
