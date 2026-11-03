use super::super::BasisSide;

pub(super) fn side_vectors(block: &[Vec<u8>], side: BasisSide) -> Vec<Vec<u8>> {
    match side {
        BasisSide::Source => (0..block[0].len())
            .map(|column| block.iter().map(|row| row[column]).collect())
            .collect(),
        BasisSide::Target => block.to_vec(),
    }
}

pub(super) fn local_vectors(
    block: &[Vec<u8>],
    group_indices: &[usize],
    side: BasisSide,
) -> Vec<Vec<u8>> {
    match side {
        BasisSide::Source => block
            .iter()
            .map(|row| group_indices.iter().map(|&index| row[index]).collect())
            .collect(),
        BasisSide::Target => (0..block[0].len())
            .map(|column| {
                group_indices
                    .iter()
                    .map(|&row| block[row][column])
                    .collect()
            })
            .collect(),
    }
}

pub(super) fn select_groups(vectors: &[Vec<u8>], max_width: usize) -> Vec<Vec<usize>> {
    if vectors.len() < 2 {
        return Vec::new();
    }
    let weights = vectors
        .iter()
        .map(|row| row.iter().map(|bit| usize::from(*bit)).sum::<usize>())
        .collect::<Vec<_>>();
    let active = weights
        .iter()
        .enumerate()
        .filter_map(|(index, weight)| (*weight > 0).then_some(index))
        .collect::<Vec<_>>();
    if active.len() < 2 {
        return Vec::new();
    }
    let width = max_width.min(active.len());
    let mut leaders = active.clone();
    leaders.sort_by_key(|index| (std::cmp::Reverse(weights[*index]), *index));
    let mut seen = std::collections::HashSet::<Vec<usize>>::new();
    let mut groups = Vec::new();

    for leader in leaders {
        let mut others = active
            .iter()
            .copied()
            .filter(|index| *index != leader)
            .collect::<Vec<_>>();
        others.sort_by_key(|other| {
            let distance = vectors[leader]
                .iter()
                .zip(&vectors[*other])
                .filter(|(left, right)| left != right)
                .count() as isize;
            (
                distance - weights[leader].max(weights[*other]) as isize,
                *other,
            )
        });
        let mut group = Vec::with_capacity(width);
        group.push(leader);
        group.extend(others.into_iter().take(width - 1));
        group.sort_unstable();
        if seen.insert(group.clone()) {
            groups.push(group);
        }
    }

    let step = width.saturating_sub(1).max(1);
    for start in (0..active.len()).step_by(step) {
        let end = (start + width).min(active.len());
        let group = active[start..end].to_vec();
        if group.len() > 1 && seen.insert(group.clone()) {
            groups.push(group);
        }
    }
    groups
}
