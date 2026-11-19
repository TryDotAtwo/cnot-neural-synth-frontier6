//! Supervised losses for Bellman targets and verified teacher words.

mod qhat;
mod teacher;

pub(super) use qhat::qhat_loss;
#[cfg(feature = "cuda")]
pub(super) use teacher::first_gate_labels;
pub(super) use teacher::teacher_loss;
