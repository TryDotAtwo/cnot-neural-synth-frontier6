//! Canonical role and channel layouts for feature families D through F.

use super::super::{FeatureId, FeatureRole};

pub(super) fn role_channels(id: FeatureId) -> (FeatureRole, usize) {
    let family = id.family;
    let ordinal = id.ordinal;
    match family {
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
        _ => unreachable!("family dispatched to D-F layout group"),
    }
}
