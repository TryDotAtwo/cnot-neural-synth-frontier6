//! Canonical role and channel layouts for feature families A through C.

use super::super::{FeatureId, FeatureRole};

fn in_set(value: u8, values: &[u8]) -> bool {
    values.contains(&value)
}

pub(super) fn role_channels(id: FeatureId) -> (FeatureRole, usize) {
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
        _ => unreachable!("family dispatched to A-C layout group"),
    }
}
