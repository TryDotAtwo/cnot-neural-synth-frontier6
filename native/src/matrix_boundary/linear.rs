use super::MatrixBoundarySpec;

pub(super) fn words_for_bits(bits: usize) -> usize {
    bits.saturating_add(63) / 64
}

pub(super) fn bit_is_set(words: &[u64], bit: usize) -> bool {
    words
        .get(bit / 64)
        .is_some_and(|word| (word & (1u64 << (bit % 64))) != 0)
}

pub(super) fn set_bit(words: &mut [u64], bit: usize) {
    words[bit / 64] |= 1u64 << (bit % 64);
}

pub(super) fn matrix_bit(words: &[u64], bit: usize) -> bool {
    bit_is_set(words, bit)
}

pub(super) fn xor_rows(left: &mut [u64], right: &[u64]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left ^= *right;
    }
}

pub(super) fn canonicalize_rows(mut rows: Vec<Vec<u64>>, bits: usize) -> Vec<Vec<u64>> {
    let mut pivot = 0usize;
    for bit in (0..bits).rev() {
        let Some(selected) = (pivot..rows.len()).find(|index| bit_is_set(&rows[*index], bit))
        else {
            continue;
        };
        rows.swap(pivot, selected);
        let pivot_row = rows[pivot].clone();
        for (index, row) in rows.iter_mut().enumerate() {
            if index != pivot && bit_is_set(row, bit) {
                xor_rows(row, &pivot_row);
            }
        }
        pivot += 1;
    }
    rows.into_iter()
        .filter(|row| row.iter().any(|word| *word != 0))
        .collect()
}

pub(super) fn rank_binary_rows(rows: Vec<Vec<u64>>, bits: usize) -> usize {
    canonicalize_rows(rows, bits).len()
}

pub(super) fn protected_projection_rank(
    spec: &MatrixBoundarySpec,
    relation_rows: &[Vec<u64>],
    ports: usize,
) -> usize {
    let protected_ports = spec
        .protected_inputs
        .iter()
        .filter_map(|wire| spec.inputs.binary_search(wire).ok())
        .collect::<Vec<_>>();
    let protected_port_set = protected_ports
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();

    let mut rows = relation_rows.to_vec();
    let mut pivot_count = 0usize;
    for bit in 0..ports {
        if bit < spec.inputs.len() && protected_port_set.contains(&bit) {
            continue;
        }
        let Some(selected) = (pivot_count..rows.len()).find(|index| bit_is_set(&rows[*index], bit))
        else {
            continue;
        };
        rows.swap(pivot_count, selected);
        let pivot_row = rows[pivot_count].clone();
        for (index, row) in rows.iter_mut().enumerate() {
            if index != pivot_count && bit_is_set(row, bit) {
                xor_rows(row, &pivot_row);
            }
        }
        pivot_count += 1;
    }

    let mut projected = Vec::new();
    for row in rows.iter().skip(pivot_count) {
        let mut output = vec![0u64; words_for_bits(protected_ports.len())];
        for (new_index, old_index) in protected_ports.iter().enumerate() {
            if bit_is_set(row, *old_index) {
                set_bit(&mut output, new_index);
            }
        }
        projected.push(output);
    }
    canonicalize_rows(projected, protected_ports.len()).len()
}
