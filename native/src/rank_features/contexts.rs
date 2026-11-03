//! Validated user-supplied contexts for exact rank features.

use std::fmt;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LabeledCut {
    pub(super) n: u32,
    pub(super) a: Vec<u32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RestrictedIntersection {
    pub(super) n: u32,
    pub(super) left: Vec<u32>,
    pub(super) right: Vec<u32>,
    pub(super) coordinates: Vec<u32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RankIncrementProbe {
    pub(super) n: u32,
    pub(super) span_rows: Vec<u32>,
    pub(super) coordinates: Vec<u32>,
    pub(super) candidate_support: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LabeledCutError(pub String);

impl fmt::Display for LabeledCutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for LabeledCutError {}

impl LabeledCut {
    pub fn new(n: u32, a: Vec<u32>) -> Result<Self, LabeledCutError> {
        if n < 2 {
            return Err(LabeledCutError("cut width must be at least two".into()));
        }
        if a.is_empty() || a.len() >= n as usize {
            return Err(LabeledCutError(
                "cut A must be a nonempty proper subset of the wires".into(),
            ));
        }
        if a.iter().any(|wire| *wire >= n) {
            return Err(LabeledCutError("cut wire is outside the matrix".into()));
        }
        if a.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(LabeledCutError(
                "cut A must be sorted and contain no repeated wire".into(),
            ));
        }
        Ok(Self { n, a })
    }

    pub fn dimension(&self) -> u32 {
        self.n
    }

    pub fn side_a(&self) -> &[u32] {
        &self.a
    }

    pub fn side_b(&self) -> Vec<u32> {
        let mut b = Vec::with_capacity(self.n as usize - self.a.len());
        let mut next_a = 0usize;
        for wire in 0..self.n {
            if self.a.get(next_a).copied() == Some(wire) {
                next_a += 1;
            } else {
                b.push(wire);
            }
        }
        b
    }

    /// Relabels each selected wire by its source-to-destination image.
    pub fn relabel(&self, permutation: &[u32]) -> Result<Self, LabeledCutError> {
        let width = self.n as usize;
        if permutation.len() != width {
            return Err(LabeledCutError(
                "a full wire permutation is required".into(),
            ));
        }
        let mut seen = Vec::new();
        seen.try_reserve_exact(width)
            .map_err(|_| LabeledCutError("wire permutation validation allocation failed".into()))?;
        seen.resize(width, false);
        for &destination in permutation {
            let index = destination as usize;
            if index >= width || std::mem::replace(&mut seen[index], true) {
                return Err(LabeledCutError(
                    "wire relabeling must be a permutation of all wires".into(),
                ));
            }
        }
        let mut a = self
            .a
            .iter()
            .map(|source| permutation[*source as usize])
            .collect::<Vec<_>>();
        a.sort_unstable();
        Self::new(self.n, a)
    }
}

impl RestrictedIntersection {
    pub fn new(
        n: u32,
        left: Vec<u32>,
        right: Vec<u32>,
        coordinates: Vec<u32>,
    ) -> Result<Self, LabeledCutError> {
        if n == 0
            || !valid_sorted_indices(&left, n, false)
            || !valid_sorted_indices(&right, n, false)
            || !valid_sorted_indices(&coordinates, n, false)
        {
            return Err(LabeledCutError(
                "intersection probes need nonempty sorted unique in-range indices".into(),
            ));
        }
        Ok(Self {
            n,
            left,
            right,
            coordinates,
        })
    }
}

impl RankIncrementProbe {
    pub fn new(
        n: u32,
        span_rows: Vec<u32>,
        coordinates: Vec<u32>,
        candidate_support: Vec<u32>,
    ) -> Result<Self, LabeledCutError> {
        if n == 0
            || !valid_sorted_indices(&span_rows, n, false)
            || !valid_sorted_indices(&coordinates, n, false)
            || !valid_sorted_indices(&candidate_support, n, true)
        {
            return Err(LabeledCutError(
                "rank-increment probes need sorted unique in-range indices".into(),
            ));
        }
        Ok(Self {
            n,
            span_rows,
            coordinates,
            candidate_support,
        })
    }
}

fn valid_sorted_indices(values: &[u32], n: u32, allow_empty: bool) -> bool {
    (allow_empty || !values.is_empty())
        && values.iter().all(|value| *value < n)
        && !values.windows(2).any(|pair| pair[0] >= pair[1])
}
