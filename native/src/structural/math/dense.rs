//! Dense integer-matrix operations used by structural feature families.

use super::zeroed;

pub(in crate::structural) fn transpose_dense(matrix: &[i64], n: usize) -> Vec<i64> {
    let mut result = vec![0; matrix.len()];
    for i in 0..n {
        for j in 0..n {
            result[j * n + i] = matrix[i * n + j];
        }
    }
    result
}

pub(in crate::structural) fn row_sums(matrix: &[i64], n: usize) -> Vec<i64> {
    (0..n)
        .map(|i| matrix[i * n..(i + 1) * n].iter().sum())
        .collect()
}

pub(in crate::structural) fn col_sums(matrix: &[i64], n: usize) -> Vec<i64> {
    (0..n)
        .map(|j| (0..n).map(|i| matrix[i * n + j]).sum())
        .collect()
}

pub(in crate::structural) fn matmul(
    left: &[i64],
    right: &[i64],
    n: usize,
) -> Result<Vec<i64>, String> {
    let cells = n.checked_mul(n).ok_or("matrix product size overflow")?;
    let mut result = zeroed(cells)?;
    for i in 0..n {
        for k in 0..n {
            let value = left[i * n + k];
            if value == 0 {
                continue;
            }
            for j in 0..n {
                result[i * n + j] += value * right[k * n + j];
            }
        }
    }
    Ok(result)
}

pub(in crate::structural) fn hadamard(left: &[i64], right: &[i64], n: usize) -> Vec<i64> {
    (0..n * n).map(|i| left[i] * right[i]).collect()
}
