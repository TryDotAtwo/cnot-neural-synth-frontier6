use crate::features::FeatureValue;
use crate::schema::{feature_spec, FeatureId, FeatureRole};
use std::collections::HashMap;

pub(super) fn id(family: char, ordinal: u8) -> FeatureId {
    FeatureId::new(family as u8, ordinal).expect("built-in feature IDs are in the manifest")
}

pub(super) fn put_node(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    n: usize,
    channels: usize,
    values: Vec<f32>,
) {
    put(
        output,
        family,
        ordinal,
        FeatureRole::Node,
        vec![n, channels],
        values,
    );
}

pub(super) fn put_pair(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    n: usize,
    channels: usize,
    values: Vec<f32>,
) {
    put(
        output,
        family,
        ordinal,
        FeatureRole::Pair,
        vec![n, n, channels],
        values,
    );
}

pub(super) fn put_global(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    value: i64,
    channels: usize,
) {
    put_global_values(output, family, ordinal, channels, vec![value as f32]);
}

pub(super) fn put_global_values(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    channels: usize,
    values: Vec<f32>,
) {
    put(
        output,
        family,
        ordinal,
        FeatureRole::Global,
        vec![channels],
        values,
    );
}

pub(super) fn put(
    output: &mut HashMap<FeatureId, FeatureValue>,
    family: char,
    ordinal: u8,
    role: FeatureRole,
    shape: Vec<usize>,
    values: Vec<f32>,
) {
    let feature_id = id(family, ordinal);
    debug_assert_eq!(feature_spec(feature_id).role, role);
    output.insert(
        feature_id,
        FeatureValue {
            shape,
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        },
    );
}
