use super::{BoundEvidence, CachedBoundEvidence};
use crate::alphabet::CnotAlphabet;
use crate::features::FeatureValue;
use crate::general_compiler::{
    compile_general_cnot_word, GeneralCompilerError, GeneralCompilerLimits,
};
use crate::packet::FeatureStatus;
use crate::state::MatrixState;
use crate::training_data::elementary_lower_bound;

const MAX_EXACT_F32_INTEGER: u64 = 1 << 24;

pub(super) fn compute_bound_evidence(
    state: &MatrixState,
    limits: GeneralCompilerLimits,
) -> CachedBoundEvidence {
    let alphabet = match CnotAlphabet::all_to_all_unit(state.dimension()) {
        Ok(alphabet) => alphabet,
        Err(error) => {
            return unavailable_result(FeatureStatus::Unsupported, error.to_string());
        }
    };
    if state.alphabet_id() != alphabet.id() {
        return unavailable_result(
            FeatureStatus::Unsupported,
            "general compiler only supports all-to-all unit-cost CNOT states",
        );
    }

    let certificate = match compile_general_cnot_word(state, limits) {
        Ok(certificate) => certificate,
        Err(GeneralCompilerError::Unsupported(reason)) => {
            return unavailable_result(FeatureStatus::Unsupported, reason);
        }
        Err(GeneralCompilerError::ResourceLimit(reason)) => {
            return unavailable_result(FeatureStatus::ResourceLimit, reason);
        }
        Err(GeneralCompilerError::Invariant(reason)) => {
            return unavailable_result(FeatureStatus::Unsupported, reason);
        }
    };
    let lower_bound = match elementary_lower_bound(state) {
        Ok(value) => value,
        Err(error) => {
            return unavailable_result(FeatureStatus::Unsupported, error.to_string());
        }
    };
    let upper_bound = certificate.physical_cost;
    if lower_bound > upper_bound {
        return unavailable_result(
            FeatureStatus::Unsupported,
            "certified lower bound exceeds the verified compiler upper bound",
        );
    }
    if upper_bound > MAX_EXACT_F32_INTEGER {
        return unavailable_result(
            FeatureStatus::ResourceLimit,
            "bound values exceed exact integer representation in float32 packets",
        );
    }

    let gap = upper_bound - lower_bound;
    let tight = u8::from(gap == 0) as f32;
    let f09 = FeatureValue {
        shape: vec![4],
        values: vec![upper_bound as f32, lower_bound as f32, gap as f32, tight],
        token_valid: Vec::new(),
        incidence: Vec::new(),
    };
    // Flags: verified UB, certified LB, bounds meet, exhaustive optimality
    // search completed, bounded compiler completed. The fourth flag stays zero:
    // Gauss-Jordan produces a valid candidate but does not search for an optimum.
    let f10 = FeatureValue {
        shape: vec![5],
        values: vec![1.0, 1.0, tight, 0.0, 1.0],
        token_valid: Vec::new(),
        incidence: Vec::new(),
    };
    let proof = if gap == 0 {
        "bounds_meet_optimality_proved"
    } else {
        "certified_interval_only"
    };
    CachedBoundEvidence::Computed(BoundEvidence {
        f09,
        f10,
        provenance: vec![
            format!("proof={proof}"),
            format!("compiler={}", certificate.producer),
            format!("compiler_word_gates={upper_bound}"),
            format!("lower_bound=elementary_changed_rows_columns_rank:{lower_bound}"),
            "optimality_search=not_performed".into(),
        ],
    })
}

fn unavailable_result(status: FeatureStatus, reason: impl Into<String>) -> CachedBoundEvidence {
    CachedBoundEvidence::Unavailable(status, reason.into())
}
