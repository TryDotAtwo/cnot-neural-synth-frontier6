//! Policy, recognition admission, and packet assembly for guarded pair views.

use super::super::unavailable;
use super::GuardedPairProducer;
use crate::alphabet::CnotAlphabet;
use crate::features::{FeatureComputation, FeaturePolicy, FeatureValue};
use crate::packet::FeatureStatus;
use crate::stage35::{verify_stage35_result, Stage35Error, Stage35Result};
use crate::state::MatrixState;

pub(super) fn compute(
    producer: &GuardedPairProducer,
    state: &MatrixState,
    policy: &FeaturePolicy,
) -> FeatureComputation {
    let is_f02 = producer.spec.id.family() == 'F' && producer.spec.id.ordinal() == 2;
    let prefix = if is_f02 { "f02" } else { "f06" };
    if !policy.solver_assisted {
        return unavailable(
            FeatureStatus::Unsupported,
            "guarded_recognizer_requires_solver_assisted_policy",
        );
    }
    if state.dimension() > producer.max_n {
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

    let result = match producer.recognize(state, &alphabet) {
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
