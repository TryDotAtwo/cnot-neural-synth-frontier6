use cnot103_native::device_search::{search_with_scorers, BeamConfig, BeamStatus};
use cnot103_native::{CnotAlphabet, CnotGate, MatrixState};
use std::collections::HashSet;
use tch::{Device, Kind, Tensor};

fn config(beam_width: i64, shortlist: i64, max_depth: i64, max_expanded: i64) -> BeamConfig {
    BeamConfig {
        beam_width,
        shortlist,
        max_depth,
        max_expanded,
        inference_batch: 5,
    }
}

fn root(n: u32, moves: &[usize]) -> MatrixState {
    let a = CnotAlphabet::all_to_all_unit(n).unwrap();
    let mut s = MatrixState::identity(n, a.id()).unwrap();
    for &m in moves {
        s.apply_cnot(a.gates()[m]).unwrap();
    }
    s
}

#[derive(Debug, PartialEq)]
struct Expected {
    status: BeamStatus,
    path: Option<Vec<usize>>,
    expanded: i64,
    generated: i64,
    evaluations: i64,
    action_rows: i64,
}

fn oracle(s: &MatrixState, cfg: BeamConfig, ties: bool) -> Expected {
    let a = CnotAlphabet::all_to_all_unit(s.dimension()).unwrap();
    let mut out = Expected {
        status: BeamStatus::DepthLimit,
        path: None,
        expanded: 0,
        generated: 0,
        evaluations: 0,
        action_rows: 0,
    };
    if s.is_identity() {
        out.status = BeamStatus::Solved;
        out.path = Some(vec![]);
        return out;
    }
    let mut closed = HashSet::from([s.clone()]);
    let mut frontier = vec![(s.clone(), Vec::<usize>::new())];
    for depth in 0..cfg.max_depth {
        if out.expanded + frontier.len() as i64 > cfg.max_expanded {
            out.status = BeamStatus::ResourceLimit;
            return out;
        }
        out.expanded += frontier.len() as i64;
        let next = |s: &MatrixState, g: CnotGate| {
            let mut x = s.clone();
            x.apply_cnot(g).unwrap();
            x
        };
        let mut goals = vec![];
        for (s, path) in &frontier {
            for (i, g) in a.gates().iter().enumerate() {
                if next(s, *g).is_identity() {
                    let mut p = path.clone();
                    p.push(i);
                    goals.push(p);
                }
            }
        }
        if let Some(p) = goals.into_iter().min() {
            out.status = BeamStatus::Solved;
            out.path = Some(p);
            return out;
        }
        let mut generated = HashSet::new();
        let mut ranked = vec![];
        for (s, path) in &frontier {
            let mut moves: Vec<usize> = (0..a.gates().len()).collect();
            if (cfg.shortlist as usize) < moves.len() {
                out.action_rows += 1;
                if !ties {
                    moves.sort_by_key(|i| i % 3);
                }
                moves.truncate(cfg.shortlist as usize);
            }
            for i in moves {
                out.generated += 1;
                let child = next(s, a.gates()[i]);
                if closed.contains(&child) || !generated.insert(child.clone()) {
                    continue;
                }
                let mut p = path.clone();
                p.push(i);
                let value = if ties {
                    0
                } else {
                    child
                        .packed_rows()
                        .iter()
                        .map(|v| *v as i64)
                        .fold(0i64, i64::wrapping_add)
                        .rem_euclid(5)
                };
                ranked.push((value + depth + 1, p, child));
                out.evaluations += 1;
            }
        }
        if ranked.is_empty() {
            out.status = BeamStatus::Exhausted;
            return out;
        }
        ranked.sort_by(|l, r| l.0.cmp(&r.0).then(l.1.cmp(&r.1)));
        frontier = ranked
            .into_iter()
            .take(cfg.beam_width as usize)
            .map(|(_, w, s)| {
                closed.insert(s.clone());
                (s, w)
            })
            .collect();
    }
    out
}

fn canned(
    s: &MatrixState,
    budget: BeamConfig,
    ties: bool,
    device: usize,
) -> cnot103_native::device_search::BeamResult {
    let n = i64::from(s.dimension());
    let actions = n * (n - 1);
    search_with_scorers(
        s,
        device,
        budget,
        |rows| {
            assert!(
                budget.shortlist < actions,
                "complete alphabet bypasses the action head"
            );
            assert_eq!(rows.device(), Device::Cuda(device));
            let q = Tensor::arange(actions, (Kind::Float, rows.device())).remainder(3);
            Ok((if ties { q.zeros_like() } else { q })
                .unsqueeze(0)
                .expand([rows.size()[0], actions], true))
        },
        |rows| {
            let v = rows
                .sum_dim_intlist([1i64, 2].as_slice(), false, Kind::Int64)
                .remainder(5)
                .to_kind(Kind::Float);
            Ok(if ties { v.zeros_like() } else { v })
        },
    )
    .unwrap()
}

#[test]
fn gpu_beam_matches_independent_complete_word_frontiers_and_work_counts() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let mut outcomes = HashSet::new();
    let mut cases = 0;
    for (s, budget, ties) in [
        (root(4, &[]), config(3, 4, 3, 10), true),
        (root(4, &[11]), config(2, 1, 2, 3), true),
        (root(4, &[3, 11]), config(8, 4, 0, 3), false),
        (root(4, &[0, 11]), config(12, 12, 2, 20), false),
        (root(4, &[1, 4, 8, 10]), config(7, 5, 5, 3), false),
        (root(4, &[1, 4, 8, 10]), config(7, 1, 9, 40), true),
        (root(4, &[1, 4, 8, 10]), config(8, 7, 5, 100), false),
        (root(3, &[2, 1, 5, 0]), config(9, 6, 4, 100), true),
        (root(3, &[2, 1, 5, 0]), config(9, 4, 4, 100), false),
        (root(3, &[2, 1, 5, 0]), config(3, 9, 4, 5), false),
    ] {
        let expected = oracle(&s, budget, ties);
        let alphabet = CnotAlphabet::all_to_all_unit(s.dimension()).unwrap();
        let out = canned(&s, budget, ties, 0);
        let actual = Expected {
            status: out.status,
            path: out.word.as_ref().map(|w| {
                w.iter()
                    .map(|g| {
                        alphabet
                            .gates()
                            .iter()
                            .position(|candidate| candidate == g)
                            .unwrap()
                    })
                    .collect()
            }),
            expanded: out.expanded,
            generated: out.generated,
            evaluations: out.value_evaluations,
            action_rows: out.action_evaluations,
        };
        assert_eq!(
            actual,
            expected,
            "root {:?}, budget {budget:?}",
            s.packed_rows()
        );
        outcomes.insert(out.status);
        cases += 1;
    }
    for status in [
        BeamStatus::Solved,
        BeamStatus::DepthLimit,
        BeamStatus::ResourceLimit,
        BeamStatus::Exhausted,
    ] {
        assert!(outcomes.contains(&status), "{status:?} missing");
    }
    println!("resident beam CPU frontier / word / ledger parity {cases} cases, four outcomes");
}

#[derive(serde::Deserialize)]
struct SourceCase {
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
fn matches_original_python_beam_on_gl2_gl3_gl4_and_packed_word_boundaries() {
    let mut count = 0;
    let mut statuses = HashSet::new();
    for (i, line) in include_str!("fixtures/beam_v13.jsonl").lines().enumerate() {
        let c: SourceCase = serde_json::from_str(line).unwrap();
        let a = CnotAlphabet::all_to_all_unit(c.n).unwrap();
        let s = MatrixState::from_packed_rows(c.n, a.id(), c.packed).unwrap();
        let index = usize::from(c.n == 65 && tch::Cuda::device_count() > 1);
        let out = canned(&s, c.budget, c.ties, index);
        let path = out
            .word
            .as_ref()
            .map(|g| g.iter().map(|x| [x.control, x.target]).collect::<Vec<_>>());
        assert_eq!(
            (
                out.status,
                path,
                out.expanded,
                out.generated,
                out.value_evaluations,
                out.action_evaluations
            ),
            (
                c.status,
                c.word,
                c.expanded,
                c.generated,
                c.values,
                c.actions
            ),
            "Python case {i} n={} {:?}",
            c.n,
            c.budget
        );
        assert!(
            out.value_slots >= out.value_evaluations && out.action_slots >= out.action_evaluations
        );
        assert_eq!(
            out.expanded,
            out.depths.iter().map(|d| d.expanded).sum::<i64>()
        );
        assert_eq!(
            out.generated,
            out.depths.iter().map(|d| d.generated).sum::<i64>()
        );
        assert_eq!(
            out.value_evaluations,
            out.depths.iter().map(|d| d.value_evaluations).sum::<i64>()
        );
        statuses.insert(out.status);
        count += 1;
    }
    assert_eq!(count, 209);
    assert_eq!(statuses.len(), 4);
    println!("pinned Python beam: {count} exact words/statuses/ledgers, all GL2 and GL3, sampled GL4, width65");
}

#[test]
fn invalid_active_scores_fail_closed_and_padding_has_no_vote() {
    let q = |rows: &Tensor| {
        Ok(Tensor::full(
            [rows.size()[0], 12],
            f64::NAN,
            (Kind::Float, rows.device()),
        ))
    };
    let v = |rows: &Tensor| {
        Ok(Tensor::full(
            [rows.size()[0]],
            f64::NAN,
            (Kind::Float, rows.device()),
        ))
    };
    let budget = config(2, 3, 2, 5);
    let terminal = search_with_scorers(&root(4, &[8]), 0, budget, q, v).unwrap();
    assert_eq!(terminal.status, BeamStatus::Solved);
    assert_eq!(
        (terminal.action_evaluations, terminal.value_evaluations),
        (0, 0)
    );
    let active = root(4, &[1, 4]);
    assert!(search_with_scorers(&active, 0, budget, q, v)
        .err()
        .unwrap()
        .contains("depth 1 action scoring"));
    assert!(search_with_scorers(&active, 0, config(2, 12, 2, 5), q, v)
        .err()
        .unwrap()
        .contains("depth 1 value scoring"));
    assert!(
        search_with_scorers(&active, 0, config(i64::MAX, 12, i64::MAX, i64::MAX), q, v)
            .err()
            .unwrap()
            .contains("overflows")
    );
}
