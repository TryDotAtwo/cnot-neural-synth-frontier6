use super::{FeatureId, FEATURE_COUNT, H07_POST_FORWARD_DIAGNOSTIC_ID, MODEL_INPUT_COUNT};

mod roles;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FeatureRole {
    Node,
    Pair,
    Global,
    Group,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureSpec {
    pub id: FeatureId,
    pub role: FeatureRole,
    pub channels: usize,
    pub scope: String,
    pub producer_version: String,
    pub dependencies: Vec<String>,
    pub exactness: String,
    pub cost_tier: String,
    pub wire_roles: Vec<String>,
    pub requires_solver_assisted: bool,
}

impl FeatureSpec {
    pub fn with_provider_metadata(
        mut self,
        scope: impl Into<String>,
        producer_version: impl Into<String>,
        dependencies: impl IntoIterator<Item = impl Into<String>>,
        exactness: impl Into<String>,
        cost_tier: impl Into<String>,
        wire_roles: impl IntoIterator<Item = impl Into<String>>,
        requires_solver_assisted: bool,
    ) -> Self {
        self.scope = scope.into();
        self.producer_version = producer_version.into();
        self.dependencies = dependencies.into_iter().map(Into::into).collect();
        self.exactness = exactness.into();
        self.cost_tier = cost_tier.into();
        self.wire_roles = wire_roles.into_iter().map(Into::into).collect();
        self.requires_solver_assisted = requires_solver_assisted;
        self
    }
}

pub(crate) fn is_reserved_native_layout(id: FeatureId) -> bool {
    matches!(
        (id.family, id.ordinal),
        (b'C', 8)
            | (b'D', 9..=14)
            | (b'E', 3)
            | (b'E', 6..=9)
            | (b'F', 3..=10)
            | (b'H', 6..=7)
            | (b'I', 1..=11)
    )
}

fn default_wire_roles(id: FeatureId, role: FeatureRole) -> Vec<String> {
    if role == FeatureRole::Group && !is_reserved_native_layout(id) {
        vec!["source".into(), "target".into()]
    } else {
        Vec::new()
    }
}

/// Canonical model inputs from the stable 103-ID manifest. H07 is a
/// post-forward diagnostic and therefore is not an input to the model.
pub fn all_model_input_specs() -> Vec<FeatureSpec> {
    let specs: Vec<_> = all_feature_specs()
        .into_iter()
        .filter(|spec| spec.id != H07_POST_FORWARD_DIAGNOSTIC_ID)
        .collect();
    debug_assert_eq!(specs.len(), MODEL_INPUT_COUNT);
    specs
}

pub fn all_feature_specs() -> Vec<FeatureSpec> {
    (0..FEATURE_COUNT)
        .map(|index| {
            let id = FeatureId::from_index(index).expect("feature index is within manifest");
            let (role, channels) = roles::role_channels(id);
            FeatureSpec {
                id,
                role,
                channels,
                scope: "full_labeled_matrix".into(),
                producer_version: "unconfigured".into(),
                dependencies: vec!["M".into()],
                exactness: "unknown".into(),
                cost_tier: "T1".into(),
                wire_roles: default_wire_roles(id, role),
                requires_solver_assisted: false,
            }
        })
        .collect()
}

pub fn feature_spec(id: FeatureId) -> FeatureSpec {
    let (role, channels) = roles::role_channels(id);
    FeatureSpec {
        id,
        role,
        channels,
        scope: "full_labeled_matrix".into(),
        producer_version: "unconfigured".into(),
        dependencies: vec!["M".into()],
        exactness: "unknown".into(),
        cost_tier: "T1".into(),
        wire_roles: default_wire_roles(id, role),
        requires_solver_assisted: false,
    }
}
