//! Exhaustive atlas construction and compact linear-map encoding.

use super::super::BasisSide;
use super::ExactAtlas;
use std::collections::{HashMap, VecDeque};

impl ExactAtlas {
    pub(super) fn build(width: usize, side: BasisSide) -> Self {
        debug_assert!((1..=4).contains(&width));
        let state_count = 1usize << width;
        let identity = (0..state_count).collect::<Vec<_>>();
        let identity_key = encode_mapping(&identity, width);
        let mut states = vec![identity_key];
        let mut distance = vec![0u8];
        let mut parent = vec![None];
        let mut parent_gate = vec![None];
        let mut seen = HashMap::new();
        seen.insert(identity_key, 0usize);
        let mut queue = VecDeque::from([0usize]);

        while let Some(index) = queue.pop_front() {
            let state = states[index];
            for control in 0..width {
                for target in 0..width {
                    if control == target {
                        continue;
                    }
                    let next = apply_to_mapping(state, width, side, control, target);
                    if seen.contains_key(&next) {
                        continue;
                    }
                    let next_index = states.len();
                    seen.insert(next, next_index);
                    states.push(next);
                    distance.push(distance[index] + 1);
                    parent.push(Some(index));
                    parent_gate.push(Some((control, target)));
                    queue.push_back(next_index);
                }
            }
        }

        let mut population = Vec::with_capacity(states.len() * state_count);
        for state in &states {
            for code in 0..state_count {
                population.push(state_image(*state, width, code).count_ones() as u8);
            }
        }

        Self {
            distance,
            parent,
            parent_gate,
            population,
        }
    }
}

fn encode_mapping(mapping: &[usize], width: usize) -> u64 {
    mapping
        .iter()
        .enumerate()
        .fold(0u64, |packed, (input, output)| {
            packed | ((*output as u64) << (input * width))
        })
}

fn state_image(state: u64, width: usize, code: usize) -> usize {
    let mask = (1u64 << width) - 1;
    ((state >> (code * width)) & mask) as usize
}

fn apply_to_mapping(
    state: u64,
    width: usize,
    side: BasisSide,
    control: usize,
    target: usize,
) -> u64 {
    let mut packed = 0u64;
    for input in 0..(1usize << width) {
        let mut output = state_image(state, width, input);
        match side {
            BasisSide::Source => output ^= ((output >> target) & 1) << control,
            BasisSide::Target => output ^= ((output >> control) & 1) << target,
        }
        packed |= (output as u64) << (input * width);
    }
    packed
}
