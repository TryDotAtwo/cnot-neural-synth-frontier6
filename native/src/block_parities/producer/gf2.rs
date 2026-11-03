use super::patterns::ParityPattern;

pub(super) fn rank_patterns(
    patterns: &[&ParityPattern],
    source_count: usize,
) -> Result<usize, &'static str> {
    let words = source_count
        .checked_add(63)
        .ok_or("parity matrix width overflow")?
        / 64;
    let word_count = patterns
        .len()
        .checked_mul(words)
        .ok_or("parity matrix storage size overflow")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(word_count)
        .map_err(|_| "parity matrix allocation failed")?;
    for pattern in patterns {
        matrix.extend_from_slice(&pattern.0);
    }
    Ok(eliminate_packed_rows(
        &mut matrix,
        patterns.len(),
        source_count,
        words,
    ))
}

fn eliminate_packed_rows(
    matrix: &mut [u64],
    row_count: usize,
    columns: usize,
    words: usize,
) -> usize {
    let mut rank = 0usize;
    for column in 0..columns {
        let word = column / 64;
        let bit = 1u64 << (column % 64);
        let Some(pivot) = (rank..row_count).find(|row| matrix[*row * words + word] & bit != 0)
        else {
            continue;
        };
        if pivot != rank {
            for index in 0..words {
                matrix.swap(pivot * words + index, rank * words + index);
            }
        }
        for row in (rank + 1)..row_count {
            if matrix[row * words + word] & bit != 0 {
                for index in 0..words {
                    let pivot_value = matrix[rank * words + index];
                    matrix[row * words + index] ^= pivot_value;
                }
            }
        }
        rank += 1;
        if rank == row_count {
            break;
        }
    }
    rank
}
