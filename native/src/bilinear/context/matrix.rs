//! Packed GF(2) matrix type and focused operation modules.

mod algebra;
mod construction;
mod packed;

pub(in crate::bilinear) use packed::{insert_row_basis, words_for};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::bilinear) struct BitMatrix {
    pub(in crate::bilinear) rows: Vec<Vec<u64>>,
    pub(in crate::bilinear) ncols: usize,
}
