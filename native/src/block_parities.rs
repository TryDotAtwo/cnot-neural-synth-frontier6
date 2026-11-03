//! Exact parity histograms for explicit physical source/target wire blocks.
//!
//! These values describe the supplied physical block. They do not infer a
//! free change of basis, a candidate program, or a distance bound.

use crate::features::FeatureBuildError;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlockContext {
    n: u32,
    sources: Vec<u32>,
    targets: Vec<u32>,
}

impl BlockContext {
    pub fn new(n: u32, sources: Vec<u32>, targets: Vec<u32>) -> Result<Self, FeatureBuildError> {
        if n < 2
            || !valid_indices(&sources, n)
            || !valid_indices(&targets, n)
            || sources
                .iter()
                .any(|source| targets.binary_search(source).is_ok())
        {
            return Err(FeatureBuildError(
                "block context needs nonempty disjoint sorted source and target wires".into(),
            ));
        }
        Ok(Self {
            n,
            sources,
            targets,
        })
    }
}

mod producer;
pub use producer::register_block_parities;
use producer::valid_indices;
