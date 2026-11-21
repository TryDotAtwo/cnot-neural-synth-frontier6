use super::frontier::{first_unseen, Frontier};
use super::scoring::ScoringBatch;
use super::BeamConfig;
use tch::{Device, Kind, Tensor};

pub(super) struct Schedule {
    pub(super) layers: i64,
    pub(super) value_slots: i64,
    pub(super) action_slots: i64,
    shortlist: i64,
    actions: i64,
    max_packet_rows: i64,
}

impl Schedule {
    pub(super) fn new(
        cfg: BeamConfig,
        actions: i64,
        packed_words: i64,
        max_packet_rows: i64,
    ) -> Result<Self, String> {
        let layers = cfg.max_depth.min(cfg.max_expanded.saturating_add(1));
        let shortlist = cfg.shortlist.min(actions);
        let overflow = "beam workspace or scoring extent overflows i64";
        let mut width = 1i64;
        let mut remaining = layers;
        let mut parent_slots = 0i64;
        let mut value_slots = 0i64;
        let mut action_slots = 0i64;
        let slots = |rows: i64| {
            (rows / cfg.inference_batch) * cfg.inference_batch.min(max_packet_rows)
                + (rows % cfg.inference_batch).min(max_packet_rows)
        };
        while remaining > 0 {
            let children = width.checked_mul(shortlist).ok_or(overflow)?;
            let next = cfg.beam_width.min(children);
            let repeat = if next == width { remaining } else { 1 };
            parent_slots = width
                .checked_mul(repeat)
                .and_then(|p| p.checked_add(parent_slots))
                .ok_or(overflow)?;
            if shortlist < actions {
                action_slots = slots(width)
                    .checked_mul(repeat)
                    .and_then(|n| n.checked_add(action_slots))
                    .ok_or(overflow)?;
            }
            value_slots = slots(children)
                .checked_mul(repeat)
                .and_then(|p| p.checked_add(value_slots))
                .ok_or(overflow)?;
            // Full packed rows, parent word histories, lexicographic rank codes
            // and action-head rows must all fit the tensor indexing ABI.
            parent_slots
                .checked_add(children)
                .and_then(|x| x.checked_add(1))
                .and_then(|x| x.checked_mul(packed_words))
                .and_then(|x| x.checked_mul(8))
                .ok_or(overflow)?;
            next.checked_mul(layers)
                .and_then(|x| x.checked_mul(8))
                .ok_or(overflow)?;
            width
                .checked_mul(actions)
                .and_then(|x| x.checked_mul(8))
                .ok_or(overflow)?;
            width = next;
            remaining -= repeat;
        }
        layers
            .checked_mul(7)
            .and_then(|n| n.checked_add(7))
            .ok_or(overflow)?;
        Ok(Self {
            layers,
            shortlist,
            actions,
            value_slots,
            action_slots,
            max_packet_rows,
        })
    }
}

struct Ledger {
    status: Tensor,
    expanded: Tensor,
    generated: Tensor,
    values: Tensor,
    actions: Tensor,
    word: Tensor,
    rejection: Tensor,
    failed_depth: Tensor,
    rejected_state: Tensor,
}

impl Ledger {
    fn check_scores(
        &mut self,
        rows: &Tensor,
        active: &Tensor,
        finite: &Tensor,
        coverage: &Tensor,
        stage: i64,
        depth: i64,
    ) {
        let failures = active.logical_and(&finite.logical_not().logical_or(&coverage.ne(0)));
        let stop = self.status.eq(0).logical_and(&failures.any());
        let first = failures.to_kind(Kind::Int64).argmax(0, false).view([1]);
        self.rejection = coverage
            .index_select(0, &first)
            .view([])
            .where_self(&stop, &self.rejection);
        self.failed_depth = self
            .failed_depth
            .full_like(depth)
            .where_self(&stop, &self.failed_depth);
        self.rejected_state = rows
            .index_select(0, &first)
            .where_self(&stop, &self.rejected_state);
        self.stop_when(&stop, stage);
    }
    fn stop_when(&mut self, condition: &Tensor, code: i64) {
        let stop = self.status.eq(0).logical_and(condition);
        self.status = self.status.full_like(code).where_self(&stop, &self.status);
    }
}

pub(super) fn run(
    root: &Tensor,
    identity: &Tensor,
    cfg: BeamConfig,
    s: &Schedule,
    score_actions: &mut impl FnMut(&Tensor) -> Result<(Tensor, Tensor), String>,
    score_values: &mut impl FnMut(&Tensor) -> Result<(Tensor, Tensor), String>,
) -> Result<(Vec<i64>, Tensor), String> {
    let device = root.device();
    let scoring = ScoringBatch {
        width: cfg.inference_batch,
        capacity: s.max_packet_rows,
        identity,
    };
    let long = (Kind::Int64, device);
    let zeros = || Tensor::zeros([], long);
    let mut log = Ledger {
        status: root.eq_tensor(identity).all().to_kind(Kind::Int64),
        expanded: zeros(),
        generated: zeros(),
        values: zeros(),
        actions: zeros(),
        word: Tensor::full([s.layers], -1, long),
        rejection: zeros(),
        failed_depth: zeros(),
        rejected_state: root.zeros_like(),
    };
    let mut frontier = Frontier {
        rows: root.shallow_clone(),
        words: Tensor::empty([1, 0], long),
        lex_rank: Tensor::zeros([1], long),
        live: Tensor::ones([1], (Kind::Bool, device)),
    };
    let mut closed = root.shallow_clone();
    let mut trace = Vec::new();
    let ids = Tensor::arange(s.actions, long);
    for step in 0..s.layers {
        let capacity = frontier.rows.size()[0];
        let frontier_count = frontier.live.sum(Kind::Int64) * log.status.eq(0);
        log.stop_when(&(&log.expanded + &frontier_count).gt(cfg.max_expanded), 3);
        let expanded = &frontier_count * log.status.eq(0);
        log.expanded += &expanded;
        let (has_goal, prefix) = frontier.nearest(identity);
        let solved = log.status.eq(0).logical_and(&has_goal);
        let word = Tensor::cat(&[prefix, Tensor::full([s.layers - step - 1], -1, long)], 0);
        log.word = word.where_self(&solved, &log.word);
        log.stop_when(&has_goal, 1);

        let live = frontier.live.logical_and(&log.status.eq(0));
        let ranked = if s.shortlist < s.actions {
            log.actions += live.sum(Kind::Int64);
            let parent_rows = frontier.rows.where_self(&live.view([-1, 1, 1]), root);
            let (q, coverage) =
                scoring.score(&parent_rows, &live, Some(s.actions), score_actions)?;
            log.check_scores(
                &parent_rows,
                &live,
                &q.isfinite().all_dim(1, false),
                &coverage,
                5,
                step + 1,
            );
            q.argsort_stable(true, 1, false).narrow(1, 0, s.shortlist)
        } else {
            ids.unsqueeze(0).expand([capacity, s.actions], true)
        };
        let (children, parents, actions) = frontier.expand(&ranked, s.shortlist);
        let eligible = live
            .index_select(0, &parents)
            .logical_and(&log.status.eq(0));
        let generated = eligible.sum(Kind::Int64);
        log.generated += &generated;
        // Invalid/padded rows use the already-closed root, ensuring they cannot
        // take the representative of a real child during stable deduplication.
        let children = children.where_self(&eligible.view([-1, 1, 1]), root);
        let keep = first_unseen(&closed, &children).logical_and(&eligible);
        let evaluations = keep.sum(Kind::Int64);
        log.stop_when(&evaluations.eq(0), 2);
        log.values += &evaluations;

        // Source value batches consume distinct children in first-generation
        // order. Fixed extents avoid a host dependency on the CUDA survivor count.
        let order = keep.to_kind(Kind::Int64).argsort_stable(true, 0, true);
        let sorted_keep = keep.index_select(0, &order);
        let sorted_rows = children
            .index_select(0, &order)
            .where_self(&sorted_keep.view([-1, 1, 1]), root);
        let (raw, coverage) = scoring.score(&sorted_rows, &sorted_keep, None, score_values)?;
        log.check_scores(
            &sorted_rows,
            &sorted_keep,
            &raw.isfinite(),
            &coverage,
            6,
            step + 1,
        );
        let scores = raw.zeros_like().scatter(0, &order, &raw) + (step + 1) as f64;
        let retained = keep.logical_and(&log.status.eq(0));
        frontier = frontier.retain(
            &children,
            &parents,
            &actions,
            &scores,
            &retained,
            cfg.beam_width.min(children.size()[0]),
            s.actions,
            root,
        );
        trace.push(Tensor::stack(
            &[
                &frontier_count,
                &expanded,
                &generated,
                &evaluations,
                &frontier.live.sum(Kind::Int64),
                &log.status,
            ],
            0,
        ));
        if step + 1 < s.layers {
            closed = Tensor::cat(&[closed, frontier.rows.shallow_clone()], 0);
        }
    }
    log.status = log
        .status
        .full_like(4)
        .where_self(&log.status.eq(0), &log.status);
    let mut fields = vec![
        Tensor::stack(
            &[
                log.status,
                log.expanded,
                log.generated,
                log.values,
                log.actions,
                log.rejection,
                log.failed_depth,
            ],
            0,
        ),
        log.word,
    ];
    fields.extend(trace);
    let receipt = Tensor::cat(&fields, 0).to_device(Device::Cpu);
    Ok((
        Vec::<i64>::try_from(receipt).map_err(|e| e.to_string())?,
        log.rejected_state,
    ))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_search::{search_hamming, BeamStatus};
    use crate::{CnotAlphabet, MatrixState};
    use serde::Deserialize;
    use std::collections::{HashMap, HashSet, VecDeque};

    fn identity(n: usize) -> Vec<u64> {
        let words = n.div_ceil(64);
        let mut rows = vec![0; n * words];
        for row in 0..n {
            rows[row * words + row / 64] |= 1u64 << (row % 64);
        }
        rows
    }

    /// Independent scalar replay for the row_target ^= row_control convention.
    fn apply_action(rows: &mut [u64], n: usize, action: usize) {
        let words = n.div_ceil(64);
        let control = action / (n - 1);
        let slot = action % (n - 1);
        let target = slot + usize::from(slot >= control);
        for word in 0..words {
            rows[target * words + word] ^= rows[control * words + word];
        }
    }

    fn replay(mut rows: Vec<u64>, n: usize, actions: &[usize]) -> Vec<u64> {
        for &action in actions {
            apply_action(&mut rows, n, action);
        }
        rows
    }

    fn state_at_distance(n: usize, wanted: usize) -> Vec<u64> {
        let start = identity(n);
        let mut distances = HashMap::from([(start.clone(), 0usize)]);
        let mut queue = VecDeque::from([start]);
        while let Some(rows) = queue.pop_front() {
            let distance = distances[&rows];
            if distance == wanted {
                return rows;
            }
            for action in 0..n * (n - 1) {
                let mut next = rows.clone();
                apply_action(&mut next, n, action);
                if !distances.contains_key(&next) {
                    distances.insert(next.clone(), distance + 1);
                    queue.push_back(next);
                }
            }
        }
        panic!("no CNOT state exists at distance {wanted}");
    }

    fn breadth_first_group(n: usize) -> Vec<(Vec<u64>, usize)> {
        let start = identity(n);
        let mut distances = HashMap::from([(start.clone(), 0usize)]);
        let mut queue = VecDeque::from([start]);
        while let Some(rows) = queue.pop_front() {
            let distance = distances[&rows];
            for action in 0..n * (n - 1) {
                let mut next = rows.clone();
                apply_action(&mut next, n, action);
                if !distances.contains_key(&next) {
                    distances.insert(next.clone(), distance + 1);
                    queue.push_back(next);
                }
            }
        }
        let mut states: Vec<_> = distances.into_iter().collect();
        states.sort_by(|left, right| left.1.cmp(&right.1).then(left.0.cmp(&right.0)));
        states
    }

    fn shortest_distance(start: Vec<u64>, goal: &[u64], n: usize) -> Option<usize> {
        let mut distances = HashMap::from([(start.clone(), 0usize)]);
        let mut queue = VecDeque::from([start]);
        while let Some(rows) = queue.pop_front() {
            let distance = distances[&rows];
            if rows == goal {
                return Some(distance);
            }
            for action in 0..n * (n - 1) {
                let mut next = rows.clone();
                apply_action(&mut next, n, action);
                if !distances.contains_key(&next) {
                    distances.insert(next.clone(), distance + 1);
                    queue.push_back(next);
                }
            }
        }
        None
    }

    fn packed(rows: &[u64], n: usize) -> Tensor {
        let words = n.div_ceil(64);
        let signed: Vec<i64> = rows.iter().map(|&row| row as i64).collect();
        Tensor::from_slice(&signed).view([1, n as i64, words as i64])
    }

    fn run_runtime(
        root_rows: &[u64],
        n: usize,
        cfg: BeamConfig,
        ties: bool,
        action_override: Option<Vec<f64>>,
    ) -> (Vec<i64>, Schedule) {
        let actions = (n * (n - 1)) as i64;
        let schedule = Schedule::new(cfg, actions, root_rows.len() as i64, i64::MAX).unwrap();
        let root = packed(root_rows, n);
        let goal = packed(&identity(n), n);
        let mut action_scores = |rows: &Tensor| {
            let values = if let Some(scores) = &action_override {
                Tensor::from_slice(scores)
                    .view([1, actions])
                    .expand([rows.size()[0], actions], true)
            } else if ties {
                Tensor::zeros([rows.size()[0], actions], (Kind::Float, Device::Cpu))
            } else {
                Tensor::arange(actions, (Kind::Float, Device::Cpu))
                    .remainder(3)
                    .unsqueeze(0)
                    .expand([rows.size()[0], actions], true)
            };
            Ok((
                values,
                Tensor::zeros([rows.size()[0]], (Kind::Int64, Device::Cpu)),
            ))
        };
        let mut value_scores = |rows: &Tensor| {
            let values = if ties {
                Tensor::zeros([rows.size()[0]], (Kind::Float, Device::Cpu))
            } else {
                rows.sum_dim_intlist([1i64, 2].as_slice(), false, Kind::Int64)
                    .remainder(5)
                    .to_kind(Kind::Float)
            };
            Ok((
                values,
                Tensor::zeros([rows.size()[0]], (Kind::Int64, Device::Cpu)),
            ))
        };
        let (receipt, _) = run(
            &root,
            &goal,
            cfg,
            &schedule,
            &mut action_scores,
            &mut value_scores,
        )
        .unwrap();
        (receipt, schedule)
    }

    fn action_gate(n: usize, action: usize) -> [u32; 2] {
        let control = action / (n - 1);
        let slot = action % (n - 1);
        let target = slot + usize::from(slot >= control);
        [control as u32, target as u32]
    }

    fn receipt_word(receipt: &[i64], layers: i64, n: usize) -> Option<Vec<[u32; 2]>> {
        if receipt[0] != 1 {
            return None;
        }
        Some(
            receipt[7..7 + layers as usize]
                .iter()
                .copied()
                .take_while(|&action| action >= 0)
                .map(|action| action_gate(n, action as usize))
                .collect(),
        )
    }

    fn trace_row(receipt: &[i64], layers: i64, row: usize) -> &[i64] {
        let start = 7 + layers as usize + row * 6;
        &receipt[start..start + 6]
    }

    fn status_code(status: BeamStatus) -> i64 {
        match status {
            BeamStatus::Solved => 1,
            BeamStatus::Exhausted => 2,
            BeamStatus::ResourceLimit => 3,
            BeamStatus::DepthLimit => 4,
        }
    }

    #[derive(Debug, PartialEq)]
    struct Oracle {
        status: BeamStatus,
        word: Option<Vec<usize>>,
        expanded: i64,
        generated: i64,
        values: i64,
        action_rows: i64,
        traces: Vec<[i64; 6]>,
    }

    /// Scalar v13 oracle: charge a complete live frontier before the full-alphabet
    /// one-step goal scan, then shortlist, deduplicate and rank complete words.
    fn scalar_v13(start: &[u64], n: usize, cfg: BeamConfig, ties: bool) -> Oracle {
        let goal = identity(n);
        let actions = n * (n - 1);
        let layers = cfg.max_depth.min(cfg.max_expanded.saturating_add(1)) as usize;
        let mut out = Oracle {
            status: BeamStatus::DepthLimit,
            word: None,
            expanded: 0,
            generated: 0,
            values: 0,
            action_rows: 0,
            traces: Vec::with_capacity(layers),
        };
        if start == goal {
            out.status = BeamStatus::Solved;
            out.word = Some(Vec::new());
            out.traces.resize(layers, [0, 0, 0, 0, 0, status_code(out.status)]);
            return out;
        }

        let mut terminal = false;
        let mut closed = HashSet::from([start.to_vec()]);
        let mut frontier = vec![(start.to_vec(), Vec::<usize>::new())];
        for depth in 0..layers {
            if terminal {
                out.traces.push([0, 0, 0, 0, 0, status_code(out.status)]);
                continue;
            }
            let frontier_count = frontier.len() as i64;
            if out.expanded + frontier_count > cfg.max_expanded {
                out.status = BeamStatus::ResourceLimit;
                out.traces.push([frontier_count, 0, 0, 0, 0, status_code(out.status)]);
                frontier.clear();
                terminal = true;
                continue;
            }
            out.expanded += frontier_count;

            let mut goals = Vec::new();
            for (rows, prefix) in &frontier {
                for action in 0..actions {
                    let mut child = rows.clone();
                    apply_action(&mut child, n, action);
                    if child == goal {
                        let mut word = prefix.clone();
                        word.push(action);
                        goals.push(word);
                    }
                }
            }
            if let Some(word) = goals.into_iter().min() {
                out.status = BeamStatus::Solved;
                out.word = Some(word);
                out.traces.push([frontier_count, frontier_count, 0, 0, 0, status_code(out.status)]);
                frontier.clear();
                terminal = true;
                continue;
            }

            let mut generated_here = 0i64;
            let mut values_here = 0i64;
            let mut generated_states = HashSet::new();
            let mut ranked = Vec::<(i64, Vec<usize>, Vec<u64>)>::new();
            for (rows, prefix) in &frontier {
                let mut moves: Vec<usize> = (0..actions).collect();
                if cfg.shortlist < actions as i64 {
                    out.action_rows += 1;
                    if !ties {
                        moves.sort_by_key(|action| action % 3);
                    }
                    moves.truncate(cfg.shortlist as usize);
                }
                for action in moves {
                    out.generated += 1;
                    generated_here += 1;
                    let mut child = rows.clone();
                    apply_action(&mut child, n, action);
                    if closed.contains(&child) || !generated_states.insert(child.clone()) {
                        continue;
                    }
                    let mut word = prefix.clone();
                    word.push(action);
                    let value = if ties {
                        0
                    } else {
                        child
                            .iter()
                            .map(|row| *row as i64)
                            .fold(0i64, i64::wrapping_add)
                            .rem_euclid(5)
                    };
                    ranked.push((value + depth as i64 + 1, word, child));
                    out.values += 1;
                    values_here += 1;
                }
            }
            if ranked.is_empty() {
                out.status = BeamStatus::Exhausted;
                out.traces.push([
                    frontier_count,
                    frontier_count,
                    generated_here,
                    values_here,
                    0,
                    status_code(out.status),
                ]);
                frontier.clear();
                terminal = true;
                continue;
            }
            ranked.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
            frontier = ranked
                .into_iter()
                .take(cfg.beam_width as usize)
                .map(|(_, word, rows)| {
                    closed.insert(rows.clone());
                    (rows, word)
                })
                .collect();
            out.traces.push([
                frontier_count,
                frontier_count,
                generated_here,
                values_here,
                frontier.len() as i64,
                0,
            ]);
        }
        out
    }

    #[derive(Deserialize)]
    struct V13Fixture {
        n: u32,
        packed: Vec<u64>,
        ties: bool,
        budget: BeamConfig,
        status: BeamStatus,
        word: Option<Vec<[u32; 2]>>,
        expanded: i64,
        generated: i64,
        values: i64,
        actions: i64,
    }

    #[test]
    fn v13_budget_is_charged_before_nearest_and_identity_depth_rules_hold() {
        let root = vec![0b10, 0b11];
        let goal = identity(2);
        assert_eq!(shortest_distance(root.clone(), &goal, 2), Some(2));
        for budget in [1, 2] {
            let cfg = BeamConfig {
                beam_width: 2,
                shortlist: 2,
                max_depth: 2,
                max_expanded: budget,
                inference_batch: 4,
            };
            let (receipt, schedule) = run_runtime(&root, 2, cfg, true, None);
            assert_eq!(receipt[0], 3, "budget {budget} must stop before the next nearest scan");
            assert_eq!(receipt[1], 1, "budget {budget} charges only the root");
            assert!(receipt_word(&receipt, schedule.layers, 2).is_none());
            assert_eq!(trace_row(&receipt, schedule.layers, 0)[1], 1);
            assert_eq!(trace_row(&receipt, schedule.layers, 1), &[2, 0, 0, 0, 0, 3]);
        }

        let cfg = BeamConfig {
            beam_width: 2,
            shortlist: 2,
            max_depth: 2,
            max_expanded: 3,
            inference_batch: 4,
        };
        let expected = scalar_v13(&root, 2, cfg, true);
        let (receipt, schedule) = run_runtime(&root, 2, cfg, true, None);
        assert_eq!(receipt[0], status_code(expected.status));
        assert_eq!(receipt[1], expected.expanded);
        let word = receipt_word(&receipt, schedule.layers, 2).unwrap();
        assert_eq!(word.len(), 2);
        let expected_actions = expected.word.unwrap();
        assert_eq!(word, expected_actions.iter().map(|&a| action_gate(2, a)).collect::<Vec<_>>());
        assert_eq!(replay(root.clone(), 2, &expected_actions), goal);

        let depth_one = BeamConfig {
            max_depth: 1,
            max_expanded: 3,
            ..cfg
        };
        let (receipt, schedule) = run_runtime(&root, 2, depth_one, true, None);
        assert_eq!(receipt[0], 4, "D=1 cannot return a depth-2 word");
        assert_eq!(receipt[1], 1);
        assert!(receipt_word(&receipt, schedule.layers, 2).is_none());

        let zero_depth = BeamConfig {
            max_depth: 0,
            max_expanded: 1,
            ..cfg
        };
        let (receipt, schedule) = run_runtime(&goal, 2, zero_depth, true, None);
        assert_eq!(receipt[0], 1);
        assert_eq!(receipt[1], 0);
        assert_eq!(receipt_word(&receipt, schedule.layers, 2), Some(Vec::new()));
    }

    #[test]
    fn v13_terminal_scans_all_gates_even_when_shortlist_omits_the_solution() {
        let root = vec![0b01, 0b11];
        let cfg = BeamConfig {
            beam_width: 2,
            shortlist: 1,
            max_depth: 1,
            max_expanded: 1,
            inference_batch: 4,
        };
        // In GL(2), gate 0 closes this root. Q deliberately ranks gate 1 first.
        let (receipt, schedule) =
            run_runtime(&root, 2, cfg, true, Some(vec![1.0, 0.0]));
        assert_eq!(receipt[0], 1);
        assert_eq!(receipt[1], 1);
        assert_eq!(receipt[3], 0, "terminal success performs no logical V evaluation");
        assert_eq!(receipt[4], 0, "terminal success performs no logical Q evaluation");
        assert_eq!(receipt_word(&receipt, schedule.layers, 2), Some(vec![[0, 1]]));
        assert_eq!(replay(root, 2, &[0]), identity(2));
    }

    #[test]
    fn v13_depth_three_budget_matches_scalar_oracle_at_seven_and_forty_three() {
        let root = state_at_distance(3, 3);
        let cfg = BeamConfig {
            beam_width: 36,
            shortlist: 6,
            max_depth: 3,
            max_expanded: 7,
            inference_batch: 8,
        };
        assert_eq!(shortest_distance(root.clone(), &identity(3), 3), Some(3));
        let expected = scalar_v13(&root, 3, cfg, true);
        let (receipt, schedule) = run_runtime(&root, 3, cfg, true, None);
        assert_eq!(expected.status, BeamStatus::ResourceLimit);
        assert_eq!(expected.expanded, 7);
        assert_eq!(receipt[0], status_code(expected.status));
        assert_eq!(receipt[1], expected.expanded);
        assert_eq!(receipt[2], expected.generated);
        assert_eq!(receipt[3], expected.values);
        assert_eq!(receipt[4], expected.action_rows);
        for (index, expected_trace) in expected.traces.iter().enumerate() {
            assert_eq!(trace_row(&receipt, schedule.layers, index), expected_trace);
        }

        let enough = BeamConfig {
            max_expanded: 43,
            ..cfg
        };
        let expected = scalar_v13(&root, 3, enough, true);
        let (receipt, schedule) = run_runtime(&root, 3, enough, true, None);
        assert_eq!(expected.status, BeamStatus::Solved);
        assert_eq!(expected.word.as_ref().unwrap().len(), 3);
        assert!(expected.expanded < 43, "oracle must count actual parents, not the cap");
        assert_eq!(receipt[0], status_code(expected.status));
        assert_eq!(receipt[1], expected.expanded);
        assert_eq!(receipt[2], expected.generated);
        assert_eq!(receipt[3], expected.values);
        assert_eq!(receipt[4], expected.action_rows);
        let word = receipt_word(&receipt, schedule.layers, 3).unwrap();
        assert_eq!(word.len(), 3);
        let action_ids: Vec<_> = expected.word.unwrap();
        assert_eq!(word, action_ids.iter().map(|&a| action_gate(3, a)).collect::<Vec<_>>());
        assert_eq!(replay(root, 3, &action_ids), identity(3));
    }

    #[test]
    fn v13_saturated_frontier_is_all_or_none_at_one_plus_b_minus_one_and_one_plus_b() {
        let root = state_at_distance(3, 4);
        let goal = identity(3);
        assert_eq!(shortest_distance(root.clone(), &goal, 3), Some(4));
        for budget in [6, 7] {
            let cfg = BeamConfig {
                beam_width: 6,
                shortlist: 6,
                max_depth: 3,
                max_expanded: budget,
                inference_batch: 8,
            };
            let expected = scalar_v13(&root, 3, cfg, true);
            let (receipt, schedule) = run_runtime(&root, 3, cfg, true, None);
            assert_eq!(expected.status, BeamStatus::ResourceLimit);
            assert_eq!(receipt[0], status_code(expected.status));
            assert_eq!(receipt[1], expected.expanded);
            assert_eq!(receipt[2], expected.generated);
            assert_eq!(receipt[3], expected.values);
            assert_eq!(receipt[4], expected.action_rows);
            for (index, expected_trace) in expected.traces.iter().enumerate() {
                assert_eq!(
                    trace_row(&receipt, schedule.layers, index),
                    expected_trace,
                    "budget {budget}, layer {index}"
                );
            }
            assert!(receipt_word(&receipt, schedule.layers, 3).is_none());
        }
    }

    #[test]
    fn compiled_cpu_runtime_matches_small_gl2_bfs_distances_and_replays() {
        let cfg = BeamConfig {
            beam_width: 6,
            shortlist: 2,
            max_depth: 4,
            max_expanded: 24,
            inference_batch: 4,
        };
        for (root, distance) in breadth_first_group(2) {
            assert_eq!(shortest_distance(root.clone(), &identity(2), 2), Some(distance));
            let expected = scalar_v13(&root, 2, cfg, true);
            let (receipt, schedule) = run_runtime(&root, 2, cfg, true, None);
            assert_eq!(receipt[0], 1);
            assert_eq!(receipt[1], expected.expanded);
            assert_eq!(receipt[2], expected.generated);
            assert_eq!(receipt[3], expected.values);
            assert_eq!(receipt[4], expected.action_rows);
            let word = receipt_word(&receipt, schedule.layers, 2).unwrap();
            assert_eq!(word.len(), distance);
            let action_ids = expected.word.unwrap();
            assert_eq!(word, action_ids.iter().map(|&a| action_gate(2, a)).collect::<Vec<_>>());
            assert_eq!(replay(root, 2, &action_ids), identity(2));
        }
    }

    #[test]
    fn compiled_cpu_runtime_matches_209_pinned_v13_cases_and_scalar_oracle() {
        let mut count = 0;
        let mut statuses = HashSet::new();
        for (line_index, line) in include_str!("../../tests/fixtures/beam_v13.jsonl")
            .lines()
            .enumerate()
        {
            let case: V13Fixture = serde_json::from_str(line).unwrap();
            let n = case.n as usize;
            let expected = scalar_v13(&case.packed, n, case.budget, case.ties);
            let (receipt, schedule) =
                run_runtime(&case.packed, n, case.budget, case.ties, None);
            assert_eq!(receipt[0], status_code(case.status), "fixture {line_index}");
            assert_eq!(expected.status, case.status, "scalar v13 fixture {line_index}");
            let expected_word = case.word.clone();
            let scalar_word = expected
                .word
                .as_ref()
                .map(|word| word.iter().map(|&a| action_gate(n, a)).collect::<Vec<_>>());
            assert_eq!(scalar_word, expected_word, "scalar word fixture {line_index}");
            assert_eq!(
                receipt_word(&receipt, schedule.layers, n),
                expected_word,
                "compiled runtime word fixture {line_index}"
            );
            assert_eq!(receipt[1], case.expanded, "expanded fixture {line_index}");
            assert_eq!(receipt[2], case.generated, "generated fixture {line_index}");
            assert_eq!(receipt[3], case.values, "value rows fixture {line_index}");
            assert_eq!(receipt[4], case.actions, "action rows fixture {line_index}");
            assert_eq!(expected.expanded, case.expanded, "oracle expanded fixture {line_index}");
            assert_eq!(expected.generated, case.generated, "oracle generated fixture {line_index}");
            assert_eq!(expected.values, case.values, "oracle values fixture {line_index}");
            assert_eq!(expected.action_rows, case.actions, "oracle action rows fixture {line_index}");
            for (depth, expected_trace) in expected.traces.iter().enumerate() {
                assert_eq!(
                    trace_row(&receipt, schedule.layers, depth),
                    expected_trace,
                    "compiled trace fixture {line_index}, depth {depth}"
                );
            }
            statuses.insert(case.status);
            count += 1;
        }
        assert_eq!(count, 209);
        assert_eq!(statuses.len(), 4);
    }

    #[test]
    fn cpu_hamming_baseline_runs_through_the_same_search_runtime_and_replays() {
        let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
        let mut root = MatrixState::identity(3, alphabet.id()).unwrap();
        root.apply_cnot(alphabet.gates()[0]).unwrap();
        root.apply_cnot(alphabet.gates()[4]).unwrap();
        let config = BeamConfig {
            beam_width: 36,
            shortlist: 4,
            max_depth: 8,
            max_expanded: 201,
            inference_batch: 32,
        };
        let result = search_hamming(&root, config).unwrap();
        assert_eq!(result.status, BeamStatus::Solved);
        let mut replayed = root.clone();
        for gate in result.word.unwrap() {
            replayed.apply_cnot(gate).unwrap();
        }
        assert_eq!(replayed, MatrixState::identity(3, alphabet.id()).unwrap());
    }
}
