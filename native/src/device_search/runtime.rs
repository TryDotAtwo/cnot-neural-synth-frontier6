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
    use std::collections::{HashMap, VecDeque};

    fn identity(n: usize) -> Vec<u64> {
        (0..n).map(|row| 1u64 << row).collect()
    }

    /// Independent scalar replay for the row_target ^= row_control convention.
    fn apply_action(rows: &mut [u64], n: usize, action: usize) {
        let control = action / (n - 1);
        let slot = action % (n - 1);
        let target = slot + usize::from(slot >= control);
        rows[target] ^= rows[control];
    }

    fn replay(mut rows: Vec<u64>, n: usize, actions: &[i64]) -> Vec<u64> {
        for &action in actions {
            assert!(action >= 0);
            apply_action(&mut rows, n, action as usize);
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

    fn packed(rows: &[u64]) -> Tensor {
        let signed: Vec<i64> = rows.iter().map(|&row| row as i64).collect();
        Tensor::from_slice(&signed).view([1, rows.len() as i64, 1])
    }

    fn run_case(
        root_rows: &[u64],
        max_depth: i64,
        max_expanded: i64,
        beam_width: i64,
        shortlist: i64,
    ) -> (Vec<i64>, Schedule) {
        let n = root_rows.len() as i64;
        let actions = n * (n - 1);
        let cfg = BeamConfig {
            beam_width,
            shortlist,
            max_depth,
            max_expanded,
            inference_batch: 64,
        };
        let schedule = Schedule::new(cfg, actions, n, 64).unwrap();
        let root = packed(root_rows);
        let goal = packed(&identity(n as usize));
        let mut action_scores = |rows: &Tensor| {
            Ok((
                Tensor::zeros([rows.size()[0], actions], (Kind::Double, Device::Cpu)),
                Tensor::zeros([rows.size()[0]], (Kind::Int64, Device::Cpu)),
            ))
        };
        let mut value_scores = |rows: &Tensor| {
            Ok((
                Tensor::zeros([rows.size()[0]], (Kind::Double, Device::Cpu)),
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

    fn solved_word(receipt: &[i64], layers: i64) -> Vec<i64> {
        receipt[7..7 + layers as usize]
            .iter()
            .copied()
            .take_while(|&action| action >= 0)
            .collect()
    }

    fn trace_row(receipt: &[i64], layers: i64, row: usize) -> &[i64] {
        let start = 7 + layers as usize + row * 6;
        &receipt[start..start + 6]
    }

    #[test]
    fn one_step_goal_wins_before_the_next_full_frontier_exceeds_exact_budget() {
        let root = vec![0b10, 0b11];
        let goal = identity(2);
        assert_eq!(shortest_distance(root.clone(), &goal, 2), Some(2));

        let (receipt, schedule) = run_case(&root, 2, 1, 2, 2);
        assert_eq!(receipt[0], 1, "reachable depth-2 goal must be solved");
        assert_eq!(receipt[1], 1, "only the root was expanded");
        let word = solved_word(&receipt, schedule.layers);
        assert_eq!(word.len(), 2);
        assert_eq!(replay(root, 2, &word), goal);
        assert_eq!(trace_row(&receipt, schedule.layers, 0)[1], 1);
    }

    #[test]
    fn chain_at_last_allowed_depth_is_checked_without_spending_another_expansion() {
        let root = state_at_distance(3, 3);
        let goal = identity(3);
        assert_eq!(shortest_distance(root.clone(), &goal, 3), Some(3));

        // One root expansion and one complete B=6 frontier consume the cap.
        // The exact one-step goal check at depth 2 must run before the next gate.
        let (receipt, schedule) = run_case(&root, 3, 7, 36, 6);
        assert_eq!(receipt[0], 1, "depth-3 solution must beat resource_limit");
        assert_eq!(receipt[1], 7, "goal recognition consumes no extra expansions");
        let word = solved_word(&receipt, schedule.layers);
        assert_eq!(word.len(), 3);
        assert_eq!(replay(root, 3, &word), goal);
        assert_eq!(trace_row(&receipt, schedule.layers, 0)[1], 1);
        assert_eq!(trace_row(&receipt, schedule.layers, 1)[1], 6);
    }

    #[test]
    fn saturated_frontier_is_all_or_none_at_one_plus_b_minus_one_and_one_plus_b() {
        let root = state_at_distance(3, 4);
        let goal = identity(3);
        assert_eq!(shortest_distance(root.clone(), &goal, 3), Some(4));

        let (short_receipt, short_schedule) = run_case(&root, 3, 6, 6, 6);
        assert_eq!(short_receipt[0], 3, "1+B-1 cannot expand a B=6 frontier");
        assert_eq!(short_receipt[1], 1, "no partial frontier is charged");
        assert!(solved_word(&short_receipt, short_schedule.layers).is_empty());
        assert_eq!(trace_row(&short_receipt, short_schedule.layers, 0)[1], 1);
        assert_eq!(trace_row(&short_receipt, short_schedule.layers, 1)[0], 6);
        assert_eq!(trace_row(&short_receipt, short_schedule.layers, 1)[1], 0);
        assert_eq!(trace_row(&short_receipt, short_schedule.layers, 1)[5], 3);

        let (exact_receipt, exact_schedule) = run_case(&root, 3, 7, 6, 6);
        assert_eq!(exact_receipt[0], 3, "the next whole frontier still cannot fit");
        assert_eq!(exact_receipt[1], 7, "1+B admits exactly one complete B=6 layer");
        assert!(solved_word(&exact_receipt, exact_schedule.layers).is_empty());
        assert_eq!(trace_row(&exact_receipt, exact_schedule.layers, 0)[1], 1);
        assert_eq!(trace_row(&exact_receipt, exact_schedule.layers, 1)[0], 6);
        assert_eq!(trace_row(&exact_receipt, exact_schedule.layers, 1)[1], 6);
        assert_eq!(trace_row(&exact_receipt, exact_schedule.layers, 2)[0], 6);
        assert_eq!(trace_row(&exact_receipt, exact_schedule.layers, 2)[1], 0);
        assert_eq!(trace_row(&exact_receipt, exact_schedule.layers, 2)[5], 3);
    }
}
