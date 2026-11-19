use super::super::{Histories, Stage38Error};
use super::factorization::{history_factors, Factor};
use super::ordering::{components, elimination_order, has_mandatory_cycle};
use super::tables::{checked_power, lattice_sum};
use num_bigint::{BigInt, Sign};
use std::collections::HashSet;

pub(in crate::stage38) struct CountAudit {
    pub(in crate::stage38) count: BigInt,
    pub(in crate::stage38) max_bag: usize,
}

/// One deterministic allowance shared by enumeration, recount and recovery.
pub(in crate::stage38) struct HistoryCountBudget {
    max: Option<u32>,
    used: u32,
}
impl HistoryCountBudget {
    pub(in crate::stage38) fn new(max: Option<u32>) -> Self { Self { max, used: 0 } }
}

pub(in crate::stage38) fn count_histories_with_budget(
    histories: &Histories,
    max_entries: usize,
    protected_rows: &HashSet<usize>,
    max_component_events: Option<usize>,
    budget: &mut HistoryCountBudget,
) -> Result<CountAudit, Stage38Error> {
    if budget.max.is_some_and(|limit| budget.used >= limit) {
        return Err(Stage38Error::ContractionBudgetExceeded("history count limit exceeded".into()));
    }
    budget.used = budget.used.saturating_add(1);
    count_histories_with_component_limit(histories, max_entries, protected_rows, max_component_events)
}

pub(in crate::stage38) fn count_histories(
    histories: &Histories,
    max_entries: usize,
    protected_rows: &HashSet<usize>,
) -> Result<CountAudit, Stage38Error> {
    count_histories_with_component_limit(histories, max_entries, protected_rows, None)
}

pub(in crate::stage38) fn count_histories_with_component_limit(
    histories: &Histories,
    max_entries: usize,
    protected_rows: &HashSet<usize>,
    max_component_events: Option<usize>,
) -> Result<CountAudit, Stage38Error> {
    if max_entries == 0 {
        return Err(Stage38Error::InvalidBudget);
    }
    let (events, factors) = history_factors(histories)?;
    let k = events.len();
    if factors.iter().any(|factor| factor.terms.is_empty()) {
        return Ok(CountAudit {
            count: BigInt::from(0u8),
            max_bag: 0,
        });
    }
    if has_mandatory_cycle(k, &factors) {
        return Ok(CountAudit {
            count: BigInt::from(0u8),
            max_bag: 0,
        });
    }

    let mut constant = BigInt::from(1u8);
    let mut active = Vec::new();
    for factor in factors {
        if factor.scope.is_empty() {
            constant *= BigInt::from(factor.terms.len() as u64);
        } else {
            active.push(factor);
        }
    }
    if k == 0 {
        return Ok(CountAudit {
            count: constant,
            max_bag: 0,
        });
    }

    let components = components(k, &active);
    let retained_rows: HashSet<usize> = protected_rows
        .iter()
        .copied()
        .chain(
            histories
                .iter()
                .enumerate()
                .filter_map(|(row, history)| (history.len() > 2).then_some(row)),
        )
        .collect();
    let protected_events: HashSet<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(event, (row, _))| retained_rows.contains(row).then_some(event))
        .collect();
    let mut component_counts = Vec::with_capacity(components.len());
    let mut max_bag = 0usize;

    for component in &components {
        if max_component_events.is_some_and(|limit| component.len() > limit) {
            return Err(Stage38Error::ContractionBudgetExceeded(
                "component event limit exceeded".into(),
            ));
        }
        let component_set: HashSet<usize> = component.iter().copied().collect();
        let component_factors: Vec<Factor> = active
            .iter()
            .filter(|factor| {
                factor
                    .scope
                    .iter()
                    .all(|variable| component_set.contains(variable))
            })
            .cloned()
            .collect();
        let (order, bag) = elimination_order(component, &component_factors, &protected_events)?;
        max_bag = max_bag.max(bag);
        checked_power(component.len(), bag, max_entries)?;

        let mut lattice_values = vec![BigInt::from(0u8); component.len() + 1];
        for (q, value) in lattice_values.iter_mut().enumerate().skip(1) {
            *value = lattice_sum(q, &component_factors, &order, max_entries)?;
        }
        let m = component.len();
        let mut leading = BigInt::from(0u8);
        for (q, lattice_value) in lattice_values.iter().enumerate() {
            let mut term = binomial(m, q) * lattice_value;
            if (m - q) % 2 == 1 {
                term = -term;
            }
            leading += term;
        }
        if leading.sign() == Sign::Minus {
            return Err(Stage38Error::ContractionInvariant(
                "negative exact leading coefficient",
            ));
        }
        if leading.sign() == Sign::NoSign {
            return Ok(CountAudit {
                count: BigInt::from(0u8),
                max_bag,
            });
        }
        component_counts.push(leading);
    }

    let mut numerator = constant * factorial(k);
    let mut denominator = BigInt::from(1u8);
    for (component, count) in components.iter().zip(component_counts) {
        numerator *= count;
        denominator *= factorial(component.len());
    }
    if (&numerator % &denominator).sign() != Sign::NoSign {
        return Err(Stage38Error::ContractionInvariant(
            "component interleaving count is not integral",
        ));
    }
    let count = numerator / denominator;
    if count.sign() == Sign::Minus {
        return Err(Stage38Error::ContractionInvariant(
            "negative temporal history count",
        ));
    }
    Ok(CountAudit { count, max_bag })
}

fn binomial(n: usize, k: usize) -> BigInt {
    let k = k.min(n - k);
    let mut result = BigInt::from(1u8);
    for index in 0..k {
        result *= BigInt::from((n - index) as u64);
        result /= BigInt::from((index + 1) as u64);
    }
    result
}

fn factorial(n: usize) -> BigInt {
    (1..=n).fold(BigInt::from(1u8), |value, factor| {
        value * BigInt::from(factor as u64)
    })
}
