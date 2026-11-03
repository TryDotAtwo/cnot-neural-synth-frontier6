use crate::schema::FeatureId;

pub fn is_builtin(id: FeatureId) -> bool {
    match id.family() {
        'A' | 'B' => true,
        'F' => id.ordinal() == 1,
        'G' => id.ordinal() == 5,
        'C' => id.ordinal() == 1,
        'D' => id.ordinal() == 3,
        'H' => (1..=5).contains(&id.ordinal()),
        _ => false,
    }
}

#[derive(Clone, Copy)]
pub struct BuiltinMetadata {
    pub scope: &'static str,
    pub producer_version: &'static str,
    pub dependencies: &'static [&'static str],
    pub exactness: &'static str,
    pub cost_tier: &'static str,
}

pub fn metadata(id: FeatureId) -> Option<BuiltinMetadata> {
    let (version, cost, dependencies) = match (id.family(), id.ordinal()) {
        ('A', 1..=20) | ('B', 1..=12) | ('F', 1) => ("structural_v1", "T1", &["M"][..]),
        ('G', 5) => (
            "exact_column_pair_histogram_v1",
            "T1",
            &["M", "column_weights", "column_intersections"][..],
        ),
        ('C', 1) => ("gf2_rank_residual_v1", "T1", &["M"][..]),
        ('D', 3) => (
            "width2_parity_multiplicity_v1",
            "T1",
            &["M", "width2_source_pair"][..],
        ),
        ('H', 1) => ("matrix_view_v1", "T0", &["M"][..]),
        ('H', 2) => ("matrix_transpose_view_v1", "T0", &["M"][..]),
        ('H', 3) => ("matrix_inverse_view_v1", "T1", &["M"][..]),
        ('H', 4) => ("matrix_inverse_transpose_view_v1", "T1", &["M"][..]),
        ('H', 5) => (
            "aligned_four_matrix_views_pair_tokens_v1",
            "T2",
            &["M", "transpose", "inverse", "inverse_transpose"][..],
        ),
        _ => return None,
    };
    Some(BuiltinMetadata {
        scope: "full_labeled_matrix",
        producer_version: version,
        dependencies,
        exactness: "exact",
        cost_tier: cost,
    })
}
