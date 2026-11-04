//! Packed GF(2) rank, inversion, and dense expansion helpers.

use super::zeroed;

pub(in crate::structural) fn gf2_rank(rows: &[u64], n: usize) -> usize {
    let words = n.div_ceil(64);
    let mut matrix = rows.to_vec();
    let mut rank = 0usize;
    for col in 0..n {
        let pivot =
            (rank..n).find(|row| matrix[row * words + col / 64] & (1u64 << (col % 64)) != 0);
        let Some(pivot) = pivot else { continue };
        matrix.swap(rank * words, pivot * words);
        for word in 1..words {
            matrix.swap(rank * words + word, pivot * words + word);
        }
        for row in 0..n {
            if row != rank && matrix[row * words + col / 64] & (1u64 << (col % 64)) != 0 {
                for word in 0..words {
                    matrix[row * words + word] ^= matrix[rank * words + word];
                }
            }
        }
        rank += 1;
        if rank == n {
            break;
        }
    }
    rank
}

pub(crate) fn gf2_inverse(rows: &[u64], n: usize) -> Result<Vec<u64>, String> {
    let words = n.div_ceil(64);
    let mut matrix = rows.to_vec();
    let mut inverse = zeroed(n.checked_mul(words).ok_or("inverse size overflow")?)?;
    for i in 0..n {
        inverse[i * words + i / 64] |= 1u64 << (i % 64);
    }
    for col in 0..n {
        let pivot = (col..n)
            .find(|row| matrix[row * words + col / 64] & (1u64 << (col % 64)) != 0)
            .ok_or("input matrix is singular")?;
        for word in 0..words {
            matrix.swap(col * words + word, pivot * words + word);
            inverse.swap(col * words + word, pivot * words + word);
        }
        for row in 0..n {
            if row != col && matrix[row * words + col / 64] & (1u64 << (col % 64)) != 0 {
                for word in 0..words {
                    matrix[row * words + word] ^= matrix[col * words + word];
                    inverse[row * words + word] ^= inverse[col * words + word];
                }
            }
        }
    }
    Ok(inverse)
}

pub(in crate::structural) fn packed_to_dense(rows: &[u64], n: usize) -> Vec<i64> {
    let words = n.div_ceil(64);
    (0..n * n)
        .map(|index| {
            ((rows[(index / n) * words + (index % n) / 64] >> ((index % n) % 64)) & 1) as i64
        })
        .collect()
}
