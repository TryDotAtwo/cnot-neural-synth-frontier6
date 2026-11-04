use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CnotGate {
    pub control: u32,
    pub target: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MatrixState {
    n: u32,
    words_per_row: usize,
    alphabet_id: String,
    rows: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MatrixError {
    InvalidDimension,
    InvalidAlphabet,
    WrongPackedLength { expected: usize, actual: usize },
    NonzeroPadding,
    Singular,
    InvalidGate { control: u32, target: u32 },
    SizeOverflow,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension => write!(f, "matrix dimension must be positive"),
            Self::InvalidAlphabet => write!(f, "alphabet identity must be nonempty"),
            Self::WrongPackedLength { expected, actual } => {
                write!(f, "packed matrix has {actual} words; expected {expected}")
            }
            Self::NonzeroPadding => write!(f, "matrix has nonzero bits outside its dimension"),
            Self::Singular => write!(f, "CNOT state must be invertible over GF(2)"),
            Self::InvalidGate { control, target } => {
                write!(f, "invalid CNOT gate ({control}, {target})")
            }
            Self::SizeOverflow => write!(f, "matrix storage size overflow"),
        }
    }
}

impl std::error::Error for MatrixError {}

fn packed_matrix_layout(n: u32) -> Result<(usize, usize), MatrixError> {
    let words_per_row = usize::try_from(n.checked_add(63).ok_or(MatrixError::SizeOverflow)? / 64)
        .map_err(|_| MatrixError::SizeOverflow)?;
    let word_count = usize::try_from(n)
        .ok()
        .and_then(|dimension| dimension.checked_mul(words_per_row))
        .ok_or(MatrixError::SizeOverflow)?;
    Ok((words_per_row, word_count))
}

impl MatrixState {
    /// Bits are packed row-major: bit `j` of row `i` is matrix entry `(i,j)`.
    /// Word zero contains columns 0 through 63, with column 0 in bit 0.
    pub fn from_packed_rows(
        n: u32,
        alphabet_id: impl Into<String>,
        rows: Vec<u64>,
    ) -> Result<Self, MatrixError> {
        if n == 0 {
            return Err(MatrixError::InvalidDimension);
        }
        let alphabet_id = alphabet_id.into();
        if alphabet_id.is_empty() {
            return Err(MatrixError::InvalidAlphabet);
        }
        let (words_per_row, expected) = packed_matrix_layout(n)?;
        if rows.len() != expected {
            return Err(MatrixError::WrongPackedLength {
                expected,
                actual: rows.len(),
            });
        }
        if !n.is_multiple_of(64) {
            let allowed = (1u64 << (n % 64)) - 1;
            if rows
                .chunks_exact(words_per_row)
                .any(|row| row[words_per_row - 1] & !allowed != 0)
            {
                return Err(MatrixError::NonzeroPadding);
            }
        }
        let state = Self {
            n,
            words_per_row,
            alphabet_id,
            rows,
        };
        if !state.is_invertible() {
            return Err(MatrixError::Singular);
        }
        Ok(state)
    }

    pub fn identity(n: u32, alphabet_id: impl Into<String>) -> Result<Self, MatrixError> {
        if n == 0 {
            return Err(MatrixError::InvalidDimension);
        }
        let (words_per_row, word_count) = packed_matrix_layout(n)?;
        let mut rows = vec![0u64; word_count];
        for row in 0..n as usize {
            rows[row * words_per_row + row / 64] |= 1u64 << (row % 64);
        }
        let alphabet_id = alphabet_id.into();
        if alphabet_id.is_empty() {
            return Err(MatrixError::InvalidAlphabet);
        }

        // The constructed identity is invertible by definition; do not repeat
        // the cubic elimination performed by the general packed-row validator.
        Ok(Self {
            n,
            words_per_row,
            alphabet_id,
            rows,
        })
    }

    pub fn dimension(&self) -> u32 {
        self.n
    }

    pub fn words_per_row(&self) -> usize {
        self.words_per_row
    }

    pub fn alphabet_id(&self) -> &str {
        &self.alphabet_id
    }

    pub fn packed_rows(&self) -> &[u64] {
        &self.rows
    }

    pub fn row_words(&self, row: u32) -> Option<&[u64]> {
        if row >= self.n {
            return None;
        }
        let start = row as usize * self.words_per_row;
        Some(&self.rows[start..start + self.words_per_row])
    }

    pub fn is_identity(&self) -> bool {
        for row in 0..self.n as usize {
            for word in 0..self.words_per_row {
                let expected = if word == row / 64 {
                    1u64 << (row % 64)
                } else {
                    0
                };
                if self.rows[row * self.words_per_row + word] != expected {
                    return false;
                }
            }
        }
        true
    }

    /// Applies the physical convention `row_target ^= row_control`.
    pub fn apply_cnot(&mut self, gate: CnotGate) -> Result<(), MatrixError> {
        if gate.control >= self.n || gate.target >= self.n || gate.control == gate.target {
            return Err(MatrixError::InvalidGate {
                control: gate.control,
                target: gate.target,
            });
        }
        let source = gate.control as usize * self.words_per_row;
        let target = gate.target as usize * self.words_per_row;
        for word in 0..self.words_per_row {
            self.rows[target + word] ^= self.rows[source + word];
        }
        Ok(())
    }

    pub fn is_invertible(&self) -> bool {
        let n = self.n as usize;
        let mut rows = self.rows.clone();
        for (pivot_row, column) in (0..n).enumerate() {
            let word = column / 64;
            let bit = 1u64 << (column % 64);
            let Some(found) =
                (pivot_row..n).find(|row| rows[*row * self.words_per_row + word] & bit != 0)
            else {
                return false;
            };
            if found != pivot_row {
                for index in 0..self.words_per_row {
                    rows.swap(
                        found * self.words_per_row + index,
                        pivot_row * self.words_per_row + index,
                    );
                }
            }
            for row in 0..n {
                if row != pivot_row && rows[row * self.words_per_row + word] & bit != 0 {
                    for index in 0..self.words_per_row {
                        let pivot = rows[pivot_row * self.words_per_row + index];
                        rows[row * self.words_per_row + index] ^= pivot;
                    }
                }
            }
        }
        true
    }
}
