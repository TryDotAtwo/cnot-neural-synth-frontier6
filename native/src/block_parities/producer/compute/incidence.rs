use super::super::patterns::ParityPattern;
use crate::block_parities::BlockContext;

pub(super) enum IncidenceColumn<'a> {
    SourcePattern(&'a ParityPattern),
    MatchingTargets(&'a ParityPattern),
    AllTargets,
}

pub(super) fn append_incidence(
    context: &BlockContext,
    codes: &[ParityPattern],
    role_one_pattern: Option<&ParityPattern>,
    role_two: IncidenceColumn<'_>,
    output: &mut Vec<f32>,
) {
    for wire in 0..context.n {
        let source_position = context.sources.binary_search(&wire).ok();
        output.push(f32::from(source_position.is_some()));
        let role_one = role_one_pattern
            .and_then(|pattern| source_position.map(|index| pattern.bit(index)))
            .unwrap_or_else(|| role_one_pattern.is_none() && source_position.is_some());
        output.push(f32::from(role_one));
        let role_two = match &role_two {
            IncidenceColumn::SourcePattern(pattern) => source_position
                .map(|index| pattern.bit(index))
                .unwrap_or(false),
            IncidenceColumn::MatchingTargets(pattern) => context
                .targets
                .binary_search(&wire)
                .ok()
                .and_then(|index| codes.get(index))
                .is_some_and(|code| code == *pattern),
            IncidenceColumn::AllTargets => context.targets.binary_search(&wire).is_ok(),
        };
        output.push(f32::from(role_two));
    }
}
