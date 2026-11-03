use super::certificate::{
    cheap_lower_bound, high_remainder, independently_verify, make_certificate, EvaluationError,
};
use crate::state::MatrixState;
use num_bigint::BigUint;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Stage26Values {
    pub(super) f03: Vec<f32>,
    pub(super) f04: Vec<f32>,
    pub(super) f05: Vec<f32>,
}

pub(super) fn evaluate(
    state: &MatrixState,
    max_n: u32,
    max_checker_pairs: usize,
) -> Result<Stage26Values, EvaluationError> {
    let n = state.dimension();
    let expected_alphabet = format!("all_to_all_unit_cnot:{n}");
    if state.alphabet_id() != expected_alphabet.as_str() {
        return Err(EvaluationError::UnsupportedAlphabet);
    }
    let pool_size = usize::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(2))
        .ok_or(EvaluationError::ResourceLimit(
            "stage26 checker pool size overflow",
        ))?;
    let checker_pairs = pool_size
        .checked_mul(pool_size.saturating_sub(1))
        .map(|pairs| pairs / 2)
        .ok_or(EvaluationError::ResourceLimit(
            "stage26 checker pair count overflow",
        ))?;
    if n > max_n || checker_pairs > max_checker_pairs {
        return Err(EvaluationError::ResourceLimit(
            "stage26 source/checker budget exceeded",
        ));
    }

    let certificate = make_certificate(state);
    if !independently_verify(state, &certificate) {
        return Err(EvaluationError::VerificationFailed(
            "independent stage26 certificate checker rejected the generated bound",
        ));
    }
    let n = n as usize;
    let mut cosets = Vec::with_capacity(n);
    let mut multiplicities = HashMap::<BigUint, usize>::new();
    for value in &certificate.projected_rows {
        let coset = high_remainder(value.clone(), &certificate.h_basis);
        cosets.push(coset.clone());
        *multiplicities.entry(value.clone()).or_default() += 1;
    }
    let mut f04 = Vec::with_capacity(n * n * 3);
    for row in 0..n {
        for column in 0..n {
            let coset = &cosets[row];
            f04.push(
                if coset != &BigUint::from(0u8) && coset == &cosets[column] {
                    1.0
                } else {
                    0.0
                },
            );
            f04.push(
                if certificate.projected_rows[row] == certificate.projected_rows[column] {
                    1.0
                } else {
                    0.0
                },
            );
            f04.push(multiplicities[&certificate.projected_rows[row]] as f32);
        }
    }
    let cheap = cheap_lower_bound(&certificate.rows);
    let f03 = vec![
        certificate.deletions as f32,
        certificate.target_count as f32,
        certificate.obstruction_rank as f32,
        certificate.h_dimension as f32,
        certificate.w0_dimension as f32,
    ];
    let f05 = vec![
        certificate.lower_bound as f32,
        certificate.lower_bound as f32 - cheap as f32,
    ];
    Ok(Stage26Values { f03, f04, f05 })
}
