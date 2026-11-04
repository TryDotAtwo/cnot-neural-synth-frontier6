mod id;
mod spec;

pub const FEATURE_COUNT: usize = 103;
pub const MODEL_INPUT_COUNT: usize = FEATURE_COUNT - 1;
pub const H07_POST_FORWARD_DIAGNOSTIC_ID: FeatureId = FeatureId {
    family: b'H',
    ordinal: 7,
};
pub const FEATURE_ID_MANIFEST_ID: &str = "cnot_feature_ids_a01_i11_v1";
pub const FEATURE_LAYOUT_ID: &str = "runtime_packet_layout_v24";

pub const FAMILY_COUNTS: [(u8, u8); 9] = [
    (b'A', 20),
    (b'B', 12),
    (b'C', 10),
    (b'D', 14),
    (b'E', 9),
    (b'F', 10),
    (b'G', 10),
    (b'H', 7),
    (b'I', 11),
];

pub use id::FeatureId;
pub(crate) use spec::is_reserved_native_layout;
pub use spec::{
    all_feature_specs, all_model_input_specs, feature_spec, FeatureRole, FeatureSpec,
    ProviderMetadata,
};
