//! Cached exhaustive breadth-first atlases for local GL(width, 2) bases.

mod build;
mod optimizer;

pub use optimizer::optimize_exact_basis;

use super::BasisSide;
use std::sync::OnceLock;

#[derive(Debug)]
struct ExactAtlas {
    distance: Vec<u8>,
    parent: Vec<Option<usize>>,
    parent_gate: Vec<Option<(usize, usize)>>,
    population: Vec<u8>,
}

struct AtlasCache {
    source: [OnceLock<ExactAtlas>; 5],
    target: [OnceLock<ExactAtlas>; 5],
}

static ATLAS_CACHE: OnceLock<AtlasCache> = OnceLock::new();

fn atlas(width: usize, side: BasisSide) -> &'static ExactAtlas {
    let cache = ATLAS_CACHE.get_or_init(|| AtlasCache {
        source: std::array::from_fn(|_| OnceLock::new()),
        target: std::array::from_fn(|_| OnceLock::new()),
    });
    let slots = match side {
        BasisSide::Source => &cache.source,
        BasisSide::Target => &cache.target,
    };
    slots[width].get_or_init(|| ExactAtlas::build(width, side))
}
#[cfg(test)]
mod tests {
    use super::{atlas, BasisSide};

    #[test]
    fn atlas_enumerates_every_local_linear_basis() {
        let expected = [0usize, 1, 6, 168, 20_160];
        for width in 1..=4 {
            assert_eq!(
                atlas(width, BasisSide::Source).population.len() / (1usize << width),
                expected[width]
            );
            assert_eq!(
                atlas(width, BasisSide::Target).population.len() / (1usize << width),
                expected[width]
            );
        }
    }
}
