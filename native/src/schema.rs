use std::fmt;

pub const FEATURE_COUNT: usize = 103;
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FeatureId {
    family: u8,
    ordinal: u8,
}

impl FeatureId {
    pub fn new(family: u8, ordinal: u8) -> Option<Self> {
        FAMILY_COUNTS
            .iter()
            .any(|(known, count)| *known == family && (1..=*count).contains(&ordinal))
            .then_some(Self { family, ordinal })
    }

    pub fn from_index(index: usize) -> Option<Self> {
        if index >= FEATURE_COUNT {
            return None;
        }
        let mut offset = 0usize;
        for (family, count) in FAMILY_COUNTS {
            let end = offset + usize::from(count);
            if index < end {
                return Some(Self {
                    family,
                    ordinal: (index - offset + 1) as u8,
                });
            }
            offset = end;
        }
        None
    }

    pub fn family(self) -> char {
        char::from(self.family)
    }

    pub fn ordinal(self) -> u8 {
        self.ordinal
    }

    pub fn index(self) -> usize {
        let mut offset = 0usize;
        for (family, count) in FAMILY_COUNTS {
            if family == self.family {
                return offset + usize::from(self.ordinal - 1);
            }
            offset += usize::from(count);
        }
        unreachable!("FeatureId is validated on construction")
    }
}

impl fmt::Display for FeatureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:02}", char::from(self.family), self.ordinal)
    }
}

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

fn in_set(value: u8, values: &[u8]) -> bool {
    values.contains(&value)
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

fn role_channels(id: FeatureId) -> (FeatureRole, usize) {
    if is_reserved_native_layout(id) {
        // Static fallback only: a concrete provider must declare its shape and
        // identity before the feature can enter a model.
        return (FeatureRole::Group, 1);
    }
    let family = id.family;
    let ordinal = id.ordinal;
    match family {
        b'A' => {
            let role = if in_set(ordinal, &[1, 2, 3, 4]) || (13..=20).contains(&ordinal) {
                FeatureRole::Pair
            } else {
                FeatureRole::Node
            };
            let channels = if in_set(ordinal, &[3, 5, 12, 17, 18, 19, 20]) {
                2
            } else {
                1
            };
            (role, channels)
        }
        b'B' => match ordinal {
            1 => (FeatureRole::Global, 1),
            2 | 3 | 5 | 9 => (FeatureRole::Node, 1),
            4 | 11 => (FeatureRole::Node, 3),
            6 | 7 => (FeatureRole::Pair, 1),
            8 => (FeatureRole::Pair, 2),
            10 => (FeatureRole::Node, 2),
            12 => (FeatureRole::Global, 16),
            _ => unreachable!("validated B feature ordinal"),
        },
        b'C' => {
            if in_set(ordinal, &[1, 5, 8]) {
                (FeatureRole::Global, 1)
            } else {
                (FeatureRole::Group, 1)
            }
        }
        b'D' => match ordinal {
            3 => (FeatureRole::Pair, 3),
            5 => (FeatureRole::Node, 2),
            7 => (FeatureRole::Group, 5),
            9 | 10 => (FeatureRole::Group, 1),
            11 | 12 => (FeatureRole::Group, 2),
            13 => (FeatureRole::Group, 12),
            14 => (FeatureRole::Pair, 4),
            _ => (FeatureRole::Group, 1),
        },
        b'E' => match ordinal {
            1..=3 | 7 => (FeatureRole::Global, 1),
            4 => (FeatureRole::Pair, 2),
            5 => (FeatureRole::Global, 4),
            6 => (FeatureRole::Group, 1),
            8 => (FeatureRole::Global, 2),
            9 => (FeatureRole::Global, 5),
            _ => unreachable!("validated E feature ordinal"),
        },
        b'F' => match ordinal {
            1 | 5 => (FeatureRole::Global, 2),
            2 => (FeatureRole::Global, 1),
            3 | 10 => (FeatureRole::Global, 5),
            4 => (FeatureRole::Pair, 3),
            6 => (FeatureRole::Group, 6),
            7 | 8 => (FeatureRole::Global, 3),
            9 => (FeatureRole::Global, 4),
            _ => unreachable!("validated F feature ordinal"),
        },
        b'G' => match ordinal {
            1 | 2 => (FeatureRole::Node, 28),
            3 => (FeatureRole::Node, 16),
            4 => (FeatureRole::Node, 3),
            5 => (FeatureRole::Pair, 4),
            6 => (FeatureRole::Pair, 28),
            7 => (FeatureRole::Node, 1),
            8 => (FeatureRole::Pair, 5),
            9 => (FeatureRole::Pair, 2),
            10 => (FeatureRole::Pair, 7),
            _ => unreachable!("validated G feature ordinal"),
        },
        b'H' => match ordinal {
            1..=4 => (FeatureRole::Pair, 1),
            5 => (FeatureRole::Pair, 4),
            6 => (FeatureRole::Pair, 15),
            7 => (FeatureRole::Global, 6),
            _ => unreachable!("validated H feature ordinal"),
        },
        b'I' => match ordinal {
            1 | 3 => (FeatureRole::Group, 2),
            2 => (FeatureRole::Group, 3),
            4 | 11 => (FeatureRole::Global, 4),
            5 | 7 => (FeatureRole::Global, 2),
            6 => (FeatureRole::Group, 5),
            8 => (FeatureRole::Group, 4),
            9 => (FeatureRole::Group, 132),
            10 => (FeatureRole::Global, 8),
            _ => unreachable!("validated I feature ordinal"),
        },
        _ => unreachable!("FeatureId family is validated on construction"),
    }
}

fn default_wire_roles(id: FeatureId, role: FeatureRole) -> Vec<String> {
    if role == FeatureRole::Group && !is_reserved_native_layout(id) {
        vec!["source".into(), "target".into()]
    } else {
        Vec::new()
    }
}

pub fn all_feature_specs() -> Vec<FeatureSpec> {
    (0..FEATURE_COUNT)
        .map(|index| {
            let id = FeatureId::from_index(index).expect("feature index is within manifest");
            let (role, channels) = role_channels(id);
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
    let (role, channels) = role_channels(id);
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
