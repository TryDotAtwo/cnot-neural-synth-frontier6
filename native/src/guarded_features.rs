//! Source-compatible guarded recognizer feature views.
//!
//! F07 exposes the bounded stage-37 zero-excess recognizer as three status
//! channels. A negative result excludes only the changed-row lower-bound
//! length; it does not claim that longer CNOT words do not exist.

use crate::alphabet::CnotAlphabet;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::stage35::{recognize_stage35, verify_stage35_result, Stage35Error, Stage35Result};
use crate::stage37::{recognize_stage37, verify_stage37_result, Stage37Error, Stage37Result};
use crate::stage38::{recognize_stage38, verify_stage38_result, Stage38Error, Stage38Result};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::sync::Mutex;

const DEFAULT_MAX_ENTRIES: usize = 100_000;
const CACHE_CAPACITY: usize = 32;
const STAGE35_SOURCE_SHA256: &str =
    "1b8ef924af6c68af51dd67e0be98148ad7f8e8c272d379ad44943b7baf3fade5";
const STAGE38_SOURCE_NOTES_SHA256: &str =
    "0815dc96c12052e04d26a04295b26f883c8936fb62d5b4c484b7d3841ee8292d";
const STAGE38_SOURCE_IMPLEMENTATION_SHA256: &str =
    "90f4aac082958111293665010a34332f28d5145e957ca2ae49551d199c4cd82e";

struct CachedPair {
    state: MatrixState,
    result: Stage35Result,
}

struct GuardedPairProducer {
    spec: FeatureSpec,
    max_n: u32,
    cache: Mutex<VecDeque<CachedPair>>,
}

struct CachedRecognition {
    state: MatrixState,
    result: Stage37Result,
}

struct GuardedZeroExcessProducer {
    spec: FeatureSpec,
    max_n: u32,
    cache: Mutex<VecDeque<CachedRecognition>>,
}

struct CachedOneExcess {
    state: MatrixState,
    result: Stage38Result,
}

struct GuardedOneExcessProducer {
    spec: FeatureSpec,
    max_n: u32,
    max_entries: usize,
    cache: Mutex<VecDeque<CachedOneExcess>>,
}

/// Registers F07 with the source's default one-excess contraction budget.
pub fn register_guarded_zero_excess(
    registry: &mut FeatureRegistry,
    max_n: u32,
) -> Result<(), FeatureBuildError> {
    register_guarded_zero_excess_with_budget(registry, max_n, DEFAULT_MAX_ENTRIES)
}

/// Registers F07 while preserving the source bundle's producer identity.
///
/// `max_entries` does not change F07's computation; the source uses a shared
/// producer version for F02/F06/F07/F08, so it is still part of the version
/// identity for checkpoint and packet compatibility.
pub fn register_guarded_zero_excess_with_budget(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
) -> Result<(), FeatureBuildError> {
    if max_n < 2 {
        return Err(FeatureBuildError(
            "F07 requires an explicit maximum width of at least two".into(),
        ));
    }
    if max_entries == 0 {
        return Err(FeatureBuildError(
            "guarded recognizers require a positive contraction-entry budget".into(),
        ));
    }

    let id = FeatureId::new(b'F', 7).expect("F07 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 3;
    spec.scope = "solver_assisted;verified_zero_excess_scope".into();
    spec.producer_version = guarded_producer_version(max_n, max_entries);
    spec.dependencies = vec![
        "M".into(),
        "full_unit_alphabet".into(),
        "theorem_sources".into(),
        "explicit_oracle_budget".into(),
    ];
    spec.exactness = "exact".into();
    spec.cost_tier = "T3".into();
    spec.wire_roles.clear();
    spec.requires_solver_assisted = true;

    registry.register(Box::new(GuardedZeroExcessProducer {
        spec,
        max_n,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}

/// Registers source F08 with explicit width and exact-contraction budgets.
pub fn register_guarded_one_excess(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
) -> Result<(), FeatureBuildError> {
    if max_n < 2 {
        return Err(FeatureBuildError(
            "F08 requires an explicit maximum width of at least two".into(),
        ));
    }
    if max_entries == 0 {
        return Err(FeatureBuildError(
            "guarded recognizers require a positive contraction-entry budget".into(),
        ));
    }

    let id = FeatureId::new(b'F', 8).expect("F08 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 3;
    spec.scope = "solver_assisted;verified_one_excess_scope;resource_limit_unknown".into();
    spec.producer_version = guarded_producer_version(max_n, max_entries);
    spec.dependencies = vec![
        "M".into(),
        "full_unit_alphabet".into(),
        "theorem_sources".into(),
        "explicit_oracle_budget".into(),
    ];
    spec.exactness = "exact".into();
    spec.cost_tier = "T3".into();
    spec.wire_roles.clear();
    spec.requires_solver_assisted = true;

    registry.register(Box::new(GuardedOneExcessProducer {
        spec,
        max_n,
        max_entries,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}

/// Registers F02's guarded stage-35 connectivity/diagonal descriptor.
///
/// This is a model input, not a distance lower bound. It is available only
/// after the stage-35 recognizer admits the matrix under the complete unit
/// CNOT alphabet and the resulting certificate passes independent replay.
pub fn register_guarded_pair_f02_view(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
) -> Result<(), FeatureBuildError> {
    if max_n < 2 {
        return Err(FeatureBuildError(
            "F02 requires an explicit maximum width of at least two".into(),
        ));
    }
    if max_entries == 0 {
        return Err(FeatureBuildError(
            "guarded recognizers require a positive contraction-entry budget".into(),
        ));
    }

    let id = FeatureId::new(b'F', 2).expect("F02 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Global;
    spec.channels = 1;
    spec.scope = "solver_assisted;stage35_verified_pair_scope_only".into();
    spec.producer_version = guarded_producer_version(max_n, max_entries);
    spec.dependencies = vec![
        "M".into(),
        "full_unit_alphabet".into(),
        "theorem_sources".into(),
        "explicit_oracle_budget".into(),
    ];
    spec.exactness = "exact".into();
    spec.cost_tier = "T3".into();
    // The Python reference inherits unused default roles for global features;
    // the native packet schema keeps incidence roles group-only.
    spec.wire_roles.clear();
    spec.requires_solver_assisted = true;

    registry.register(Box::new(GuardedPairProducer {
        spec,
        max_n,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}

/// Registers source F06: a verified stage-35 pair certificate with physical
/// active-wire and external-source incidence.
pub fn register_guarded_pair(
    registry: &mut FeatureRegistry,
    max_n: u32,
    max_entries: usize,
) -> Result<(), FeatureBuildError> {
    if max_n < 2 {
        return Err(FeatureBuildError(
            "F06 requires an explicit maximum width of at least two".into(),
        ));
    }
    if max_entries == 0 {
        return Err(FeatureBuildError(
            "guarded recognizers require a positive contraction-entry budget".into(),
        ));
    }

    let id = FeatureId::new(b'F', 6).expect("F06 belongs to the canonical feature manifest");
    let mut spec = feature_spec(id);
    spec.role = FeatureRole::Group;
    spec.channels = 6;
    spec.scope = "solver_assisted;stage35_verified_pair_scope_only".into();
    spec.producer_version = guarded_producer_version(max_n, max_entries);
    spec.dependencies = vec![
        "M".into(),
        "full_unit_alphabet".into(),
        "theorem_sources".into(),
        "explicit_oracle_budget".into(),
    ];
    spec.exactness = "exact".into();
    spec.cost_tier = "T3".into();
    spec.wire_roles = vec![
        "active_first".into(),
        "active_second".into(),
        "external_sources".into(),
    ];
    spec.requires_solver_assisted = true;

    registry.register(Box::new(GuardedPairProducer {
        spec,
        max_n,
        cache: Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)),
    }))
}

impl FeatureProducer for GuardedZeroExcessProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if !policy.solver_assisted {
            return unavailable(
                FeatureStatus::Unsupported,
                "guarded_recognizer_requires_solver_assisted_policy",
            );
        }
        if state.dimension() > self.max_n {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "f07_configured_width_limit_exceeded",
            );
        }

        let alphabet = match CnotAlphabet::all_to_all_unit(state.dimension()) {
            Ok(alphabet) => alphabet,
            Err(_) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "f07_complete_alphabet_construction_failed",
                );
            }
        };
        if state.alphabet_id() != alphabet.id() {
            return unavailable(
                FeatureStatus::Unsupported,
                "f07_requires_complete_unit_all_to_all_cnot_alphabet",
            );
        }

        let result = match self.recognize(state, &alphabet) {
            Ok(result) => result,
            Err(Stage37Error::AlphabetMismatch) => {
                return unavailable(FeatureStatus::Unsupported, "f07_state_alphabet_mismatch");
            }
            Err(Stage37Error::IndexOverflow | Stage37Error::SizeOverflow) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "f07_recognizer_size_limit_exceeded",
                );
            }
            Err(error) => panic!("F07 exact recognizer integrity failure: {error}"),
        };

        let values = match &result {
            Stage37Result::Exact(_) => match verify_stage37_result(&result, &alphabet) {
                Ok(true) => vec![1.0, 0.0, 0.0],
                Ok(false) => panic!("F07 exact result failed independent admission"),
                Err(error) => panic!("F07 exact result verification failed: {error}"),
            },
            Stage37Result::NoZeroExcess(_) => vec![0.0, 1.0, 0.0],
            Stage37Result::Unsupported(_) => vec![0.0, 0.0, 1.0],
        };
        FeatureComputation::Computed(FeatureValue {
            shape: vec![3],
            values,
            token_valid: Vec::new(),
            incidence: Vec::new(),
        })
    }
}

impl FeatureProducer for GuardedPairProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        if self.spec.id.family() == 'F' && self.spec.id.ordinal() == 2 {
            "exact_relabel"
        } else {
            "augmentation_only"
        }
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        let is_f02 = self.spec.id.family() == 'F' && self.spec.id.ordinal() == 2;
        let prefix = if is_f02 { "f02" } else { "f06" };
        if !policy.solver_assisted {
            return unavailable(
                FeatureStatus::Unsupported,
                "guarded_recognizer_requires_solver_assisted_policy",
            );
        }
        if state.dimension() > self.max_n {
            return unavailable(
                FeatureStatus::ResourceLimit,
                &format!("{prefix}_configured_width_limit_exceeded"),
            );
        }

        let alphabet = match CnotAlphabet::all_to_all_unit(state.dimension()) {
            Ok(alphabet) => alphabet,
            Err(_) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    &format!("{prefix}_complete_alphabet_construction_failed"),
                );
            }
        };
        if state.alphabet_id() != alphabet.id() {
            return unavailable(
                FeatureStatus::Unsupported,
                &format!("{prefix}_requires_complete_unit_all_to_all_cnot_alphabet"),
            );
        }

        let result = match self.recognize(state, &alphabet) {
            Ok(result) => result,
            Err(Stage35Error::AlphabetMismatch) => {
                return unavailable(
                    FeatureStatus::Unsupported,
                    &format!("{prefix}_state_alphabet_mismatch"),
                );
            }
            Err(Stage35Error::IndexOverflow | Stage35Error::CostOverflow) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    &format!("{prefix}_recognizer_size_limit_exceeded"),
                );
            }
            Err(error) => panic!("{prefix} exact recognizer integrity failure: {error}"),
        };

        match &result {
            Stage35Result::Unsupported(_) => unavailable(
                FeatureStatus::Unsupported,
                &format!("{prefix}_outside_verified_pair_scope"),
            ),
            Stage35Result::Exact(certificate) => {
                match verify_stage35_result(&result, &alphabet) {
                    Ok(true) => {}
                    Ok(false) => panic!("{prefix} exact result failed independent admission"),
                    Err(error) => panic!("{prefix} exact result verification failed: {error}"),
                }

                if is_f02 {
                    let Some(value) = certificate
                        .connectivity
                        .checked_add(certificate.zero_diagonal)
                    else {
                        return unavailable(
                            FeatureStatus::ResourceLimit,
                            "f02_connectivity_diagonal_value_overflow",
                        );
                    };
                    return FeatureComputation::Computed(FeatureValue {
                        shape: vec![1],
                        values: vec![value as f32],
                        token_valid: Vec::new(),
                        incidence: Vec::new(),
                    });
                }

                let dimension = state.dimension() as usize;
                let first = certificate.active_rows[0] as usize;
                let second = certificate.active_rows[1] as usize;
                if first >= dimension || second >= dimension || first == second {
                    panic!("F06 exact result contains invalid active rows");
                }
                let [first_frame, second_frame] = certificate.frame_rows;
                let values = vec![
                    f32::from(first_frame & 1),
                    f32::from((first_frame >> 1) & 1),
                    f32::from(second_frame & 1),
                    f32::from((second_frame >> 1) & 1),
                    certificate.nonzero_external_columns as f32,
                    certificate.shared_external_columns as f32,
                ];
                let mut incidence = vec![0.0; dimension * 3];
                incidence[first * 3] = 1.0;
                incidence[second * 3 + 1] = 1.0;
                for row in 0..dimension {
                    if row != first && row != second {
                        incidence[row * 3 + 2] = 1.0;
                    }
                }
                FeatureComputation::Computed(FeatureValue {
                    shape: vec![1, 6],
                    values,
                    token_valid: vec![1],
                    incidence,
                })
            }
        }
    }
}

impl GuardedZeroExcessProducer {
    fn recognize(
        &self,
        state: &MatrixState,
        alphabet: &CnotAlphabet,
    ) -> Result<Stage37Result, Stage37Error> {
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache.iter().position(|entry| entry.state == *state) {
                let entry = cache.remove(index).expect("cache index was found");
                let result = entry.result.clone();
                cache.push_front(entry);
                return Ok(result);
            }
        }

        let result = recognize_stage37(state, alphabet)?;
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(CachedRecognition {
                state: state.clone(),
                result: result.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        Ok(result)
    }
}

impl FeatureProducer for GuardedOneExcessProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "exact_relabel"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if !policy.solver_assisted {
            return unavailable(
                FeatureStatus::Unsupported,
                "guarded_recognizer_requires_solver_assisted_policy",
            );
        }
        if state.dimension() > self.max_n {
            return unavailable(
                FeatureStatus::ResourceLimit,
                "f08_configured_width_limit_exceeded",
            );
        }
        let alphabet = match CnotAlphabet::all_to_all_unit(state.dimension()) {
            Ok(alphabet) => alphabet,
            Err(_) => {
                return unavailable(
                    FeatureStatus::ResourceLimit,
                    "f08_complete_alphabet_construction_failed",
                );
            }
        };
        if state.alphabet_id() != alphabet.id() {
            return unavailable(
                FeatureStatus::Unsupported,
                "f08_requires_complete_unit_all_to_all_cnot_alphabet",
            );
        }

        let result = match self.recognize(state, &alphabet) {
            Ok(result) => result,
            Err(Stage38Error::AlphabetMismatch) => {
                return unavailable(FeatureStatus::Unsupported, "f08_state_alphabet_mismatch");
            }
            Err(Stage38Error::InvalidBudget | Stage38Error::SizeOverflow) => {
                return unavailable(FeatureStatus::ResourceLimit, "f08_resource_limit");
            }
            Err(error) => panic!("F08 exact recognizer integrity failure: {error}"),
        };

        match &result {
            Stage38Result::Exact(_) => match verify_stage38_result(&result, &alphabet) {
                Ok(true) => computed_status(vec![1.0, 0.0, 0.0]),
                Ok(false) => panic!("F08 exact result failed independent admission"),
                Err(error) => panic!("F08 exact result verification failed: {error}"),
            },
            Stage38Result::CertifiedNoSolutionWithinScope(_) => {
                computed_status(vec![0.0, 1.0, 0.0])
            }
            Stage38Result::Unsupported(_) => computed_status(vec![0.0, 0.0, 1.0]),
            Stage38Result::ResourceLimit(limit) => {
                unavailable(FeatureStatus::ResourceLimit, &limit.reason)
            }
        }
    }
}

impl GuardedPairProducer {
    fn recognize(
        &self,
        state: &MatrixState,
        alphabet: &CnotAlphabet,
    ) -> Result<Stage35Result, Stage35Error> {
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache.iter().position(|entry| entry.state == *state) {
                let entry = cache.remove(index).expect("cache index was found");
                let result = entry.result.clone();
                cache.push_front(entry);
                return Ok(result);
            }
        }

        let result = recognize_stage35(state, alphabet)?;
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(CachedPair {
                state: state.clone(),
                result: result.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        Ok(result)
    }
}

impl GuardedOneExcessProducer {
    fn recognize(
        &self,
        state: &MatrixState,
        alphabet: &CnotAlphabet,
    ) -> Result<Stage38Result, Stage38Error> {
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(index) = cache.iter().position(|entry| entry.state == *state) {
                let entry = cache.remove(index).expect("cache index was found");
                let result = entry.result.clone();
                cache.push_front(entry);
                return Ok(result);
            }
        }

        let result = recognize_stage38(state, alphabet, self.max_entries)?;
        if let Ok(mut cache) = self.cache.lock() {
            cache.push_front(CachedOneExcess {
                state: state.clone(),
                result: result.clone(),
            });
            cache.truncate(CACHE_CAPACITY);
        }
        Ok(result)
    }
}

fn computed_status(values: Vec<f32>) -> FeatureComputation {
    FeatureComputation::Computed(FeatureValue {
        shape: vec![3],
        values,
        token_valid: Vec::new(),
        incidence: Vec::new(),
    })
}

fn guarded_producer_version(max_n: u32, max_entries: usize) -> String {
    // Python's json.dumps tuple encoding is a JSON array with the default
    // separators containing one space after commas. All string fields here
    // are lowercase SHA-256 hex, so this spelling is byte-for-byte stable.
    let source_identity = format!(
        "[{max_n}, {max_entries}, \"{STAGE35_SOURCE_SHA256}\", \"{STAGE38_SOURCE_NOTES_SHA256}\", \"{STAGE38_SOURCE_IMPLEMENTATION_SHA256}\"]"
    );
    let digest = Sha256::digest(source_identity.as_bytes());
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("guarded_certificate_features_v1:{hex}")
}

fn unavailable(status: FeatureStatus, reason: &str) -> FeatureComputation {
    FeatureComputation::Unavailable {
        status,
        reason: reason.into(),
    }
}
