//! Stable 103-feature tensor role and channel layout.

mod abc;
mod def;
mod ghi;

use super::{FeatureId, FeatureRole};

pub(super) fn role_channels(id: FeatureId) -> (FeatureRole, usize) {
    if super::is_reserved_native_layout(id) {
        // Static fallback only: a concrete provider must declare its shape and
        // identity before the feature can enter a model.
        return (FeatureRole::Group, 1);
    }
    match id.family {
        b'A' | b'B' | b'C' => abc::role_channels(id),
        b'D' | b'E' | b'F' => def::role_channels(id),
        b'G' | b'H' | b'I' => ghi::role_channels(id),
        _ => unreachable!("FeatureId family is validated on construction"),
    }
}
