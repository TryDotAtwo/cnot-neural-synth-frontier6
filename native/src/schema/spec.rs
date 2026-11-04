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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderMetadata {
    scope: String,
    producer_version: String,
    dependencies: Vec<String>,
    exactness: String,
    cost_tier: String,
    wire_roles: Vec<String>,
    requires_solver_assisted: bool,
}

impl ProviderMetadata {
    pub fn new(
        scope: impl Into<String>,
        producer_version: impl Into<String>,
        exactness: impl Into<String>,
        cost_tier: impl Into<String>,
    ) -> Self {
        Self {
            scope: scope.into(),
            producer_version: producer_version.into(),
            dependencies: Vec::new(),
            exactness: exactness.into(),
            cost_tier: cost_tier.into(),
            wire_roles: Vec::new(),
            requires_solver_assisted: false,
        }
    }

    pub fn dependencies(mut self, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.dependencies = values.into_iter().map(Into::into).collect();
        self
    }

    pub fn wire_roles(mut self, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.wire_roles = values.into_iter().map(Into::into).collect();
        self
    }

    pub fn requires_solver_assisted(mut self, required: bool) -> Self {
        self.requires_solver_assisted = required;
        self
    }
}

impl FeatureSpec {
    pub fn with_provider_metadata(mut self, metadata: ProviderMetadata) -> Self {
        self.scope = metadata.scope;
        self.producer_version = metadata.producer_version;
        self.dependencies = metadata.dependencies;
        self.exactness = metadata.exactness;
        self.cost_tier = metadata.cost_tier;
        self.wire_roles = metadata.wire_roles;
        self.requires_solver_assisted = metadata.requires_solver_assisted;
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
