//! Exact factor contraction and temporal-history recovery for stage 38.

use super::{Constraint, Histories, Stage38Error, xor_rows};
use crate::state::{CnotGate, MatrixState};
use num_bigint::{BigInt, Sign};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Factor {
    terms: Vec<Vec<Constraint>>,
    scope: Vec<usize>,
}

impl Factor {
    fn new(terms: Vec<Vec<Constraint>>) -> Self {
        let scope = terms
            .iter()
            .flat_map(|term| term.iter().flat_map(|(left, right)| [*left, *right]))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Self { terms, scope }
    }
}

#[derive(Clone)]
struct Table {
    scope: Vec<usize>,
    values: Vec<BigInt>,
}

pub(super) struct CountAudit {
    pub(super) count: BigInt,
    pub(super) max_bag: usize,
}

pub(super) fn count_histories(
    histories: &Histories,
    max_entries: usize,
    protected_rows: &HashSet<usize>,
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
        for q in 1..=component.len() {
            lattice_values[q] = lattice_sum(q, &component_factors, &order, max_entries)?;
        }
        let m = component.len();
        let mut leading = BigInt::from(0u8);
        for q in 0..=m {
            let mut term = binomial(m, q) * &lattice_values[q];
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

fn history_factors(
    histories: &Histories,
) -> Result<(Vec<(usize, usize)>, Vec<Factor>), Stage38Error> {
    validate_histories(histories)?;
    let n = histories.len();
    let mut events = Vec::new();
    let mut labels = HashMap::new();
    for (row, history) in histories.iter().enumerate() {
        for transition in 0..history.len() - 1 {
            let label = events.len();
            labels.insert((row, transition), label);
            events.push((row, transition));
        }
    }

    let mut factors = Vec::new();
    for (row, history) in histories.iter().enumerate() {
        for transition in 0..history.len().saturating_sub(2) {
            factors.push(Factor::new(vec![vec![(
                label(&labels, row, transition)?,
                label(&labels, row, transition + 1)?,
            )]]));
        }
    }

    for (event, (row, transition)) in events.iter().copied().enumerate() {
        let difference = xor_rows(&histories[row][transition], &histories[row][transition + 1]);
        let mut terms = Vec::new();
        for (control_row, history) in histories.iter().enumerate() {
            if control_row == row {
                continue;
            }
            for (position, value) in history.iter().enumerate() {
                if *value != difference {
                    continue;
                }
                let mut term = Vec::new();
                if position > 0 {
                    term.push((label(&labels, control_row, position - 1)?, event));
                }
                if position < history.len() - 1 {
                    term.push((event, label(&labels, control_row, position)?));
                }
                term.sort_unstable();
                terms.push(term);
            }
        }
        factors.push(Factor::new(terms));
    }

    let mut seen = HashSet::new();
    factors.retain(|factor| factor.terms.len() != 1 || seen.insert(factor.clone()));
    if events.len() != labels.len() || n != histories.len() {
        return Err(Stage38Error::ContractionInvariant("event labeling changed"));
    }
    Ok((events, factors))
}

fn validate_histories(histories: &Histories) -> Result<(), Stage38Error> {
    let n = histories.len();
    if n == 0 || histories.iter().any(Vec::is_empty) {
        return Err(Stage38Error::InvalidHistory("empty matrix or row"));
    }
    let dimension = u32::try_from(n).map_err(|_| Stage38Error::SizeOverflow)?;
    let words = n.checked_add(63).ok_or(Stage38Error::SizeOverflow)? / 64;
    for history in histories {
        for value in history {
            if value.len() != words || value.iter().all(|word| *word == 0) {
                return Err(Stage38Error::InvalidHistory("row width or nonzero value"));
            }
            if n % 64 != 0 && value[words - 1] >> (n % 64) != 0 {
                return Err(Stage38Error::InvalidHistory("row padding bits"));
            }
        }
        if history.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Stage38Error::InvalidHistory(
                "event leaves target unchanged",
            ));
        }
    }
    for endpoint in [0, 1] {
        let rows: Vec<u64> = histories
            .iter()
            .flat_map(|history| {
                if endpoint == 0 {
                    history[0].clone()
                } else {
                    history[history.len() - 1].clone()
                }
            })
            .collect();
        MatrixState::from_packed_rows(dimension, "stage38-history-validation", rows)
            .map_err(|_| Stage38Error::InvalidHistory("singular endpoint matrix"))?;
    }
    Ok(())
}

fn has_mandatory_cycle(event_count: usize, factors: &[Factor]) -> bool {
    let mut edges = vec![BTreeSet::new(); event_count];
    for factor in factors {
        let Some(first) = factor.terms.first() else {
            continue;
        };
        let mut mandatory: BTreeSet<Constraint> = first.iter().copied().collect();
        for term in factor.terms.iter().skip(1) {
            let constraints: HashSet<Constraint> = term.iter().copied().collect();
            mandatory.retain(|constraint| constraints.contains(constraint));
        }
        for (before, after) in mandatory {
            edges[before].insert(after);
        }
    }
    let mut indegree = vec![0usize; event_count];
    for targets in &edges {
        for target in targets {
            indegree[*target] += 1;
        }
    }
    let mut ready: Vec<usize> = (0..event_count)
        .filter(|event| indegree[*event] == 0)
        .collect();
    let mut visited = 0usize;
    while let Some(event) = ready.pop() {
        visited += 1;
        for target in edges[event].iter().copied() {
            indegree[target] -= 1;
            if indegree[target] == 0 {
                ready.push(target);
            }
        }
    }
    visited != event_count
}

fn components(event_count: usize, factors: &[Factor]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![BTreeSet::new(); event_count];
    for factor in factors {
        for variable in factor.scope.iter().copied() {
            adjacency[variable].extend(
                factor
                    .scope
                    .iter()
                    .copied()
                    .filter(|other| *other != variable),
            );
        }
    }
    let mut seen = vec![false; event_count];
    let mut output = Vec::new();
    for start in 0..event_count {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();
        while let Some(variable) = stack.pop() {
            component.push(variable);
            for neighbor in adjacency[variable].iter().copied() {
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        component.sort_unstable();
        output.push(component);
    }
    output
}

fn elimination_order(
    vertices: &[usize],
    factors: &[Factor],
    protected: &HashSet<usize>,
) -> Result<(Vec<usize>, usize), Stage38Error> {
    let mut adjacency: BTreeMap<usize, BTreeSet<usize>> = vertices
        .iter()
        .copied()
        .map(|v| (v, BTreeSet::new()))
        .collect();
    for factor in factors {
        for variable in factor.scope.iter().copied() {
            for other in factor
                .scope
                .iter()
                .copied()
                .filter(|other| *other != variable)
            {
                adjacency
                    .get_mut(&variable)
                    .ok_or(Stage38Error::ContractionInvariant(
                        "factor scope is outside component",
                    ))?
                    .insert(other);
            }
        }
    }

    let remaining: BTreeSet<usize> = vertices
        .iter()
        .copied()
        .filter(|vertex| !protected.contains(vertex))
        .collect();
    let mut seen = HashSet::new();
    let mut pseudoforest = true;
    for start in remaining.iter().copied() {
        if seen.contains(&start) {
            continue;
        }
        seen.insert(start);
        let mut stack = vec![start];
        let mut vertex_count = 0usize;
        let mut degree_sum = 0usize;
        while let Some(vertex) = stack.pop() {
            vertex_count += 1;
            let neighbors: Vec<usize> = adjacency[&vertex]
                .intersection(&remaining)
                .copied()
                .collect();
            degree_sum = degree_sum
                .checked_add(neighbors.len())
                .ok_or(Stage38Error::SizeOverflow)?;
            for neighbor in neighbors {
                if seen.insert(neighbor) {
                    stack.push(neighbor);
                }
            }
        }
        if degree_sum / 2 > vertex_count {
            pseudoforest = false;
        }
    }
    let use_protected = !protected.is_empty() && pseudoforest;
    let mut order = Vec::with_capacity(vertices.len());
    let mut max_bag = 0usize;

    while !adjacency.is_empty() {
        let pool: Vec<usize> = if use_protected {
            adjacency
                .keys()
                .copied()
                .filter(|vertex| !protected.contains(vertex))
                .collect()
        } else {
            Vec::new()
        };
        let selected = if !pool.is_empty() {
            let vertex = pool
                .into_iter()
                .min_by_key(|vertex| {
                    (
                        adjacency[vertex]
                            .iter()
                            .filter(|neighbor| !protected.contains(neighbor))
                            .count(),
                        elimination_key(*vertex, &adjacency),
                    )
                })
                .ok_or(Stage38Error::ContractionInvariant("empty elimination pool"))?;
            if adjacency[&vertex]
                .iter()
                .filter(|neighbor| !protected.contains(neighbor))
                .count()
                > 2
            {
                return Err(Stage38Error::ContractionInvariant(
                    "pseudoforest elimination degree exceeded two",
                ));
            }
            vertex
        } else {
            adjacency
                .keys()
                .copied()
                .min_by_key(|vertex| elimination_key(*vertex, &adjacency))
                .ok_or(Stage38Error::ContractionInvariant(
                    "no elimination candidate",
                ))?
        };

        let neighbors: Vec<usize> = adjacency[&selected].iter().copied().collect();
        max_bag = max_bag.max(
            neighbors
                .len()
                .checked_add(1)
                .ok_or(Stage38Error::SizeOverflow)?,
        );
        for (position, left) in neighbors.iter().copied().enumerate() {
            for right in neighbors.iter().copied().skip(position + 1) {
                adjacency.get_mut(&left).unwrap().insert(right);
                adjacency.get_mut(&right).unwrap().insert(left);
            }
        }
        for neighbor in &neighbors {
            adjacency.get_mut(neighbor).unwrap().remove(&selected);
        }
        adjacency.remove(&selected);
        order.push(selected);
    }
    Ok((order, max_bag))
}

fn elimination_key(
    vertex: usize,
    adjacency: &BTreeMap<usize, BTreeSet<usize>>,
) -> (usize, usize, usize) {
    let neighbors = &adjacency[&vertex];
    let mut fill = 0usize;
    for (position, left) in neighbors.iter().copied().enumerate() {
        for right in neighbors.iter().copied().skip(position + 1) {
            if !adjacency[&left].contains(&right) {
                fill += 1;
            }
        }
    }
    (fill, neighbors.len(), vertex)
}

fn lattice_sum(
    q: usize,
    factors: &[Factor],
    order: &[usize],
    max_entries: usize,
) -> Result<BigInt, Stage38Error> {
    let mut tables = factors
        .iter()
        .map(|factor| factor_table(factor, q, max_entries))
        .collect::<Result<Vec<_>, _>>()?;

    for variable in order.iter().copied() {
        let mut selected = Vec::new();
        let mut remaining = Vec::new();
        for table in tables.drain(..) {
            if table.scope.binary_search(&variable).is_ok() {
                selected.push(table);
            } else {
                remaining.push(table);
            }
        }
        tables = remaining;
        if selected.is_empty() {
            tables.push(Table {
                scope: Vec::new(),
                values: vec![BigInt::from(q as u64)],
            });
            continue;
        }

        let joint: Vec<usize> = selected
            .iter()
            .flat_map(|table| table.scope.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let joint_size = checked_power(q, joint.len(), max_entries)?;
        let axis = joint
            .binary_search(&variable)
            .map_err(|_| Stage38Error::ContractionInvariant("elimination variable is absent"))?;
        let mut reduced_scope = joint.clone();
        reduced_scope.remove(axis);
        let reduced_size = checked_power(q, reduced_scope.len(), max_entries)?;
        let mut reduced = zero_vector(reduced_size)?;
        let projections: Vec<Vec<usize>> = selected
            .iter()
            .map(|table| {
                table
                    .scope
                    .iter()
                    .map(|v| {
                        joint.binary_search(v).map_err(|_| {
                            Stage38Error::ContractionInvariant("table scope is outside joint scope")
                        })
                    })
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        let mut digits = vec![0usize; joint.len()];
        for flat in 0..joint_size {
            decode_assignment(flat, q, &mut digits);
            let mut product = BigInt::from(1u8);
            for (table, projection) in selected.iter().zip(&projections) {
                let table_index = projection
                    .iter()
                    .fold(0usize, |index, position| index * q + digits[*position]);
                product *= &table.values[table_index];
            }
            let reduced_index = digits
                .iter()
                .enumerate()
                .filter(|(position, _)| *position != axis)
                .fold(0usize, |index, (_, digit)| index * q + digit);
            reduced[reduced_index] += product;
        }
        tables.push(Table {
            scope: reduced_scope,
            values: reduced,
        });
    }

    let mut product = BigInt::from(1u8);
    for table in tables {
        if !table.scope.is_empty() || table.values.len() != 1 {
            return Err(Stage38Error::ContractionInvariant(
                "uncontracted variable remains after elimination",
            ));
        }
        product *= table.values[0].clone();
    }
    Ok(product)
}

fn factor_table(factor: &Factor, q: usize, max_entries: usize) -> Result<Table, Stage38Error> {
    let size = checked_power(q, factor.scope.len(), max_entries)?;
    let mut values = zero_vector(size)?;
    let positions: HashMap<usize, usize> = factor
        .scope
        .iter()
        .copied()
        .enumerate()
        .map(|(position, variable)| (variable, position))
        .collect();
    let mut digits = vec![0usize; factor.scope.len()];
    for flat in 0..size {
        decode_assignment(flat, q, &mut digits);
        for term in &factor.terms {
            if term
                .iter()
                .all(|(left, right)| digits[positions[left]] < digits[positions[right]])
            {
                values[flat] += BigInt::from(1u8);
            }
        }
    }
    Ok(Table {
        scope: factor.scope.clone(),
        values,
    })
}

fn checked_power(base: usize, exponent: usize, limit: usize) -> Result<usize, Stage38Error> {
    let mut value = 1usize;
    for _ in 0..exponent {
        value = value
            .checked_mul(base)
            .filter(|value| *value <= limit)
            .ok_or_else(|| {
                Stage38Error::ContractionBudgetExceeded(format!(
                    "table requires more than {limit} entries"
                ))
            })?;
    }
    Ok(value)
}

fn zero_vector(size: usize) -> Result<Vec<BigInt>, Stage38Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(size)
        .map_err(|_| Stage38Error::ContractionBudgetExceeded("table allocation failed".into()))?;
    values.resize_with(size, || BigInt::from(0u8));
    Ok(values)
}

fn decode_assignment(mut flat: usize, radix: usize, digits: &mut [usize]) {
    for digit in digits.iter_mut().rev() {
        *digit = flat % radix;
        flat /= radix;
    }
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

pub(super) fn recover_histories(
    initial: &Histories,
    max_entries: usize,
) -> Result<Vec<CnotGate>, Stage38Error> {
    if count_histories(initial, max_entries, &HashSet::new())?
        .count
        .sign()
        == Sign::NoSign
    {
        return Err(Stage38Error::ContractionInvariant(
            "cannot recover a word from zero temporal amplitude",
        ));
    }
    let protected: HashSet<usize> = initial
        .iter()
        .enumerate()
        .filter_map(|(row, history)| (history.len() > 2).then_some(row))
        .collect();
    let mut histories = initial.clone();
    let mut gates = Vec::new();

    while histories.iter().any(|history| history.len() > 1) {
        let mut lookup = HashMap::new();
        for (row, history) in histories.iter().enumerate() {
            lookup.insert(history[0].clone(), row);
        }
        let mut admitted = false;
        for target in 0..histories.len() {
            if histories[target].len() == 1 {
                continue;
            }
            let difference = xor_rows(&histories[target][0], &histories[target][1]);
            let Some(control) = lookup.get(&difference).copied() else {
                continue;
            };
            if control == target {
                continue;
            }
            let mut updated = histories.clone();
            updated[target].remove(0);
            if count_histories(&updated, max_entries, &protected)?
                .count
                .sign()
                != Sign::Plus
            {
                continue;
            }
            gates.push(CnotGate {
                control: u32::try_from(control).map_err(|_| Stage38Error::SizeOverflow)?,
                target: u32::try_from(target).map_err(|_| Stage38Error::SizeOverflow)?,
            });
            histories = updated;
            admitted = true;
            break;
        }
        if !admitted {
            return Err(Stage38Error::ContractionInvariant(
                "positive amplitude has no recoverable next gate",
            ));
        }
    }
    Ok(gates)
}

fn label(
    labels: &HashMap<(usize, usize), usize>,
    row: usize,
    transition: usize,
) -> Result<usize, Stage38Error> {
    labels
        .get(&(row, transition))
        .copied()
        .ok_or(Stage38Error::ContractionInvariant("row event has no label"))
}
