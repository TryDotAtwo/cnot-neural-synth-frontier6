use super::compiler::{
    compile_equations, equation_bound, variable_count, working_bytes_upper_bound,
};
use super::summaries::{
    compute_alpha, compute_beta, equation_support, rank_beta, rank_mixed_terms, variable_names,
};
use super::types::{I09Context, I09Error, I09ErrorKind, I09Limits, I09Response};
use super::MAX_EXACT_F32_INTEGER;
use crate::features::FeatureValue;
use crate::state::MatrixState;

impl I09Response {
    pub(super) fn to_feature_value(
        &self,
        limits: I09Limits,
        n: usize,
    ) -> Result<FeatureValue, I09Error> {
        let groups = self.equations.len();
        let channels = limits
            .max_variables
            .checked_add(4)
            .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_channel_overflow"))?;
        let value_count = groups.checked_mul(channels).ok_or_else(|| {
            I09Error::new(I09ErrorKind::ResourceLimit, "i09_output_size_overflow")
        })?;
        if groups > MAX_EXACT_F32_INTEGER || channels > MAX_EXACT_F32_INTEGER {
            return Err(I09Error::new(
                I09ErrorKind::ResourceLimit,
                "i09_float_labels_exceed_exact_integer_range",
            ));
        }
        let mut values = Vec::new();
        values.try_reserve_exact(value_count).map_err(|_| {
            I09Error::new(I09ErrorKind::ResourceLimit, "i09_output_allocation_failed")
        })?;
        values.resize(value_count, 0.0f32);
        let mut incidence = Vec::new();
        incidence
            .try_reserve_exact(groups.checked_mul(n).ok_or_else(|| {
                I09Error::new(I09ErrorKind::ResourceLimit, "i09_incidence_size_overflow")
            })?)
            .map_err(|_| {
                I09Error::new(
                    I09ErrorKind::ResourceLimit,
                    "i09_incidence_allocation_failed",
                )
            })?;
        for equation_index in 0..groups {
            let base = equation_index * channels;
            values[base] = equation_index as f32;
            values[base + 1] = f32::from(self.alpha[equation_index]);
            values[base + 2] = self.rank_b as f32;
            values[base + 3] = self.mixed_rank as f32;
            for (future, bit) in self.beta[equation_index].iter().copied().enumerate() {
                if future < limits.max_variables {
                    values[base + 4 + future] = f32::from(bit);
                }
            }
            let support_start = equation_index * n;
            incidence.extend(
                self.equation_support[support_start..support_start + n]
                    .iter()
                    .map(|value| f32::from(*value)),
            );
        }
        Ok(FeatureValue {
            shape: vec![groups, channels],
            values,
            token_valid: vec![1; groups],
            incidence,
        })
    }
}

/// Compile the bounded optional-CNOT equations and exact Stage-60 summaries.
pub fn compute_i09_response(
    state: &MatrixState,
    context: &I09Context,
    limits: I09Limits,
) -> Result<I09Response, I09Error> {
    if state.dimension() as usize != context.n {
        return Err(I09Error::new(
            I09ErrorKind::Unsupported,
            "i09_state_width_does_not_match_context",
        ));
    }
    let expected_alphabet = format!("all_to_all_unit_cnot:{}", context.n);
    if state.alphabet_id() != expected_alphabet {
        return Err(I09Error::new(
            I09ErrorKind::Unsupported,
            "i09_requires_matching_all_to_all_unit_cnot_alphabet",
        ));
    }
    let variables = variable_count(context.n, context.horizon)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_variable_count_overflow"))?;
    let equation_bound = equation_bound(context.n, context.horizon)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_equation_bound_overflow"))?;
    if limits.max_variables == 0
        || limits.max_equations == 0
        || limits.max_bytes == 0
        || variables > limits.max_variables
        || equation_bound > limits.max_equations
        || variables > MAX_EXACT_F32_INTEGER
        || equation_bound > MAX_EXACT_F32_INTEGER
    {
        return Err(I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_context_exceeds_declared_variable_equation_or_label_budget",
        ));
    }
    let estimate = working_bytes_upper_bound(context.n, context.horizon, variables, equation_bound)
        .ok_or_else(|| I09Error::new(I09ErrorKind::ResourceLimit, "i09_working_size_overflow"))?;
    if estimate > limits.max_bytes {
        return Err(I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_working_memory_budget_exceeded_before_allocation",
        ));
    }

    let equations = compile_equations(state, context, variables)?;
    if equations.len() > limits.max_equations {
        return Err(I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_compiled_equation_count_exceeded_declared_budget",
        ));
    }
    let alpha = compute_alpha(&equations, context);
    let beta = compute_beta(&equations, context, variables)?;
    let rank_b = rank_beta(&beta, equations.len())?;
    let mixed_rank = rank_mixed_terms(&equations, context, variables)?;
    let equation_support = equation_support(&equations, context.n, context.horizon, variables)?;
    let variable_names = variable_names(context.n, context.horizon, variables)?;
    Ok(I09Response {
        variable_names,
        equations,
        alpha,
        beta,
        rank_b,
        mixed_rank,
        equation_support,
    })
}
