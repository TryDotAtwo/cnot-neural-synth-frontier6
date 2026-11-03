use super::fingerprint::{identity_set_digest, sorted_identities};
use crate::packet::StateIdentity;
use crate::state::MatrixState;
use std::collections::{HashMap, HashSet};

pub(super) const EXPOSURE_SCHEMA: &str = "native_full_graph_exposures_v1";

#[derive(Clone, Debug, Default)]
pub(super) struct ExposureLedger {
    pub(super) seen_as_label: HashSet<StateIdentity>,
    pub(super) seen_by_forward: HashSet<StateIdentity>,
    pub(super) seen_by_gradient: HashSet<StateIdentity>,
    pub(super) compact_counts: [u64; 3],
}

impl ExposureLedger {
    pub(super) fn observe(&mut self, compact: bool, kind: usize, state: &MatrixState) {
        if compact {
            self.compact_counts[kind] = self.compact_counts[kind].saturating_add(1);
        } else {
            let target = match kind {
                0 => &mut self.seen_as_label,
                1 => &mut self.seen_by_forward,
                _ => &mut self.seen_by_gradient,
            };
            target.insert(StateIdentity::from(state));
        }
    }

    pub(super) fn audit(&self, forbidden: &HashSet<StateIdentity>, compact: bool) -> ExposureAudit {
        let label_overlap = self.seen_as_label.intersection(forbidden).count();
        let forward_overlap = self.seen_by_forward.intersection(forbidden).count();
        let gradient_overlap = self.seen_by_gradient.intersection(forbidden).count();
        let strict_unseen = label_overlap == 0 && forward_overlap == 0 && gradient_overlap == 0;
        let (label_count, forward_count, gradient_count) = if compact {
            (
                self.compact_counts[0] as usize,
                self.compact_counts[1] as usize,
                self.compact_counts[2] as usize,
            )
        } else {
            (
                self.seen_as_label.len(),
                self.seen_by_forward.len(),
                self.seen_by_gradient.len(),
            )
        };
        let set_sha256 = if compact {
            HashMap::new()
        } else {
            HashMap::from([
                ("labels".into(), identity_set_digest(&self.seen_as_label)),
                ("forward".into(), identity_set_digest(&self.seen_by_forward)),
                (
                    "gradient".into(),
                    identity_set_digest(&self.seen_by_gradient),
                ),
                ("forbidden".into(), identity_set_digest(forbidden)),
            ])
        };
        ExposureAudit {
            strict_unseen,
            audit_mode: if compact {
                "external_admission_before_forward"
            } else {
                "stored_full_state_identities"
            },
            label_count,
            forward_count,
            gradient_count,
            forbidden_count: forbidden.len(),
            label_overlap,
            forward_overlap,
            gradient_overlap,
            set_sha256,
        }
    }

    pub(super) fn state_dict(&self, compact: bool) -> serde_json::Value {
        if compact {
            serde_json::json!({
                "schema": EXPOSURE_SCHEMA,
                "mode": "compact",
                "counts": {
                    "label": self.compact_counts[0],
                    "forward": self.compact_counts[1],
                    "gradient": self.compact_counts[2]
                }
            })
        } else {
            serde_json::json!({
                "schema": EXPOSURE_SCHEMA,
                "mode": "full",
                "seen_as_label": sorted_identities(&self.seen_as_label),
                "seen_by_forward": sorted_identities(&self.seen_by_forward),
                "seen_by_gradient": sorted_identities(&self.seen_by_gradient)
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExposureAudit {
    pub strict_unseen: bool,
    pub audit_mode: &'static str,
    pub label_count: usize,
    pub forward_count: usize,
    pub gradient_count: usize,
    pub forbidden_count: usize,
    pub label_overlap: usize,
    pub forward_overlap: usize,
    pub gradient_overlap: usize,
    pub set_sha256: HashMap<String, String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObjectiveDiagnostics {
    pub examples: usize,
    pub primitive_candidates: usize,
    pub exact_examples: usize,
    pub teacher_examples: usize,
}

/// Checks a state against an external sealed-data/admission policy before any
/// feature forward. Implementations must fail closed for unknown identities.
pub trait StateAdmission {
    /// Stable identity of the admission implementation and the frozen split
    /// manifest it enforces. It is stored in the objective/checkpoint contract.
    fn contract_sha256(&self) -> &str;
    fn admit(&self, state: &MatrixState) -> Result<(), String>;
}
