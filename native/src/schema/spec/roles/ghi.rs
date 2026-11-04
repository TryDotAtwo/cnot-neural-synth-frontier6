//! Canonical role and channel layouts for feature families G through I.

use super::super::{FeatureId, FeatureRole};

pub(super) fn role_channels(id: FeatureId) -> (FeatureRole, usize) {
    let family = id.family;
    let ordinal = id.ordinal;
    match family {
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
            // H07 is reserved as a post-forward diagnostic, not a model input.
            7 => (FeatureRole::Group, 1),
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
        _ => unreachable!("family dispatched to G-I layout group"),
    }
}
