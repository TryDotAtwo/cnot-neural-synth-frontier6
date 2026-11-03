use super::{guarded_producer_version, unavailable, CACHE_CAPACITY};
use crate::alphabet::CnotAlphabet;
use crate::features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
use crate::packet::FeatureStatus;
use crate::schema::{feature_spec, FeatureId, FeatureRole, FeatureSpec};
use crate::stage35::{recognize_stage35, verify_stage35_result, Stage35Error, Stage35Result};
use crate::state::MatrixState;
use std::collections::VecDeque;
use std::sync::Mutex;

struct CachedPair {
    state: MatrixState,
    result: Stage35Result,
}

struct GuardedPairProducer {
    spec: FeatureSpec,
    max_n: u32,
    cache: Mutex<VecDeque<CachedPair>>,
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
