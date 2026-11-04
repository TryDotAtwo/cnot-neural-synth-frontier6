//! Supervised losses for Bellman targets and verified teacher words.

mod qhat;
mod teacher;

pub(super) use qhat::qhat_loss;
pub(super) use teacher::teacher_loss;
