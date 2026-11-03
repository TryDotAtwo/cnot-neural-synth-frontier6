//! Selected-feature structural distance field and forward contract.

use super::auxiliary::AuxiliaryHeads;
use super::family_heads::FamilyHeads;
use super::fusion::FusionTransformer;
use super::stem::StructuralStem;
use crate::schema::FeatureSpec;
use tch::Device;

mod construct;
mod packet;
mod predict;

/// The selected-feature structural distance field used by research_v3.
///
/// `specs` is the selected subset of the canonical 103-ID manifest. Built-in
/// shapes are fixed; a reserved ID receives its role and channel count from
/// its registered provider. The packet still carries all 103 manifest rows.
/// The model rejects packets whose selected IDs, producer metadata, or policy
/// differ from the schema selected at construction. Row availability is an
/// input mask: unavailable rows stay zero-masked and do not block other states
/// in a batch. This v24 contract is distinct from the legacy fixed-shape v22
/// packet adapter.
#[derive(Debug)]
pub struct FusionDistanceField {
    specs: Vec<FeatureSpec>,
    expected_schema_id: String,
    expected_policy_id: String,
    device: Device,
    stem: StructuralStem,
    family_heads: FamilyHeads,
    fusion: FusionTransformer,
    auxiliary: AuxiliaryHeads,
}
