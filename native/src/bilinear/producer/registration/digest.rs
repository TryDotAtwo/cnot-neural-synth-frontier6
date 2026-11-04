use crate::bilinear::context::BitMatrix;
use crate::bilinear::{BilinearContext, STAGE55_SOURCE_SHA256};
use sha2::{Digest, Sha256};

pub(super) fn producer_version(
    contexts: &[BilinearContext],
    max_dimension: usize,
    max_matrix_bits: usize,
) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"stage55_i05_port_reduction_v1");
    hash_field(&mut digest, STAGE55_SOURCE_SHA256.as_bytes());
    hash_field(&mut digest, &max_dimension.to_le_bytes());
    hash_field(&mut digest, &max_matrix_bits.to_le_bytes());
    hash_field(&mut digest, &(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hash_field(&mut digest, &context.n.to_le_bytes());
        hash_field(&mut digest, &(context.hidden_width as u64).to_le_bytes());
        hash_field(&mut digest, &(context.q as u64).to_le_bytes());
        hash_indices(&mut digest, &context.rows);
        hash_indices(&mut digest, &context.columns);
        hash_matrix(&mut digest, &context.u);
        hash_matrix(&mut digest, &context.v);
        hash_matrix(&mut digest, &context.a0);
        hash_matrix(&mut digest, &context.d0);
    }
    format!(
        "stage55_i05_port_reduction_v1:{}",
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn hash_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hash_indices(digest: &mut Sha256, indices: &[u32]) {
    hash_field(digest, &(indices.len() as u64).to_le_bytes());
    for index in indices {
        hash_field(digest, &index.to_le_bytes());
    }
}

fn hash_matrix(digest: &mut Sha256, matrix: &BitMatrix) {
    hash_field(digest, &(matrix.nrows() as u64).to_le_bytes());
    hash_field(digest, &(matrix.ncols as u64).to_le_bytes());
    for row in &matrix.rows {
        hash_field(digest, &(row.len() as u64).to_le_bytes());
        for word in row {
            hash_field(digest, &word.to_le_bytes());
        }
    }
}
