//! Optional deterministic dev/test search for a discard-only benchmark.
use super::artifacts;
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::device_search::{search_neural, ActionHead, BeamConfig};
use crate::model::FusionDistanceField;
use crate::{FrozenDataset, TrainingSplit};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::time::Instant;
use tch::{nn::VarStore, Tensor};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QualitySpec {
    schema: String,
    dataset_sha256: String,
    dev_indices: Vec<usize>,
    #[serde(default)]
    test_indices: Vec<usize>,
    search: BeamConfig,
    head: ActionHead,
}

pub(super) struct PinnedQualitySpec {
    spec: QualitySpec,
    sha256: String,
}

fn parse_spec(raw: &[u8]) -> Result<PinnedQualitySpec, String> {
    let spec: QualitySpec = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    let unique = |indices: &[usize]| indices.iter().collect::<BTreeSet<_>>().len() == indices.len();
    if spec.schema != "cnot103_holdout_search_v2"
        || (spec.dev_indices.is_empty() && spec.test_indices.is_empty())
        || !unique(&spec.dev_indices)
        || !unique(&spec.test_indices)
    {
        return Err("holdout search requires its schema and distinct explicit indices".into());
    }
    Ok(PinnedQualitySpec { spec, sha256: artifacts::sha256(raw) })
}

pub(super) fn optional_spec() -> Result<Option<PinnedQualitySpec>, String> {
    std::env::var_os("CNOT103_BENCHMARK_EVAL_SPEC")
        .map(|path| fs::read(path).map_err(|e| e.to_string()).and_then(|raw| parse_spec(&raw)))
        .transpose()
}

pub(super) fn evaluate(
    pinned: &PinnedQualitySpec,
    dataset: &FrozenDataset,
    model: &FusionDistanceField,
    plans: &HashMap<u32, DeviceFeaturePlan>,
    variables: &VarStore,
    selected_count: usize,
) -> Result<Value, String> {
    let spec = &pinned.spec;
    if spec.dataset_sha256 != dataset.sha256 || selected_count != crate::schema::MODEL_INPUT_COUNT {
        return Err("holdout search dataset or full-input contract differs".into());
    }
    let finite: Vec<_> = variables.variables().values().map(|t| t.isfinite().all()).collect();
    if Tensor::stack(&finite, 0).all().int64_value(&[]) == 0 {
        return Err("discard benchmark model has nonfinite weights".into());
    }

    let started = Instant::now();
    let mut rows = Vec::new();
    let mut summaries = Vec::new();
    for (split, indices, records, expected_split) in [
        ("dev", spec.dev_indices.as_slice(), dataset.dev.records(), TrainingSplit::Dev),
        ("test", spec.test_indices.as_slice(), dataset.test.records(), TrainingSplit::Test),
    ] {
        if indices.is_empty() {
            continue;
        }
        let mut solved = 0u64;
        let mut solved_cost = 0u64;
        let mut total_expanded = 0u64;
        let mut total_generated = 0u64;
        let mut total_value_evaluations = 0u64;
        let mut total_action_evaluations = 0u64;
        let mut total_value_slots = 0u64;
        let mut total_action_slots = 0u64;
        for &index in indices {
            let record = records.get(index).ok_or("holdout search index out of range")?;
            let root = &record.state;
            if record.split != expected_split
                || !dataset.forbidden_keys.contains(root)
                || !plans.contains_key(&root.dimension())
            {
                return Err("requested holdout root lacks its frozen identity or resident plan".into());
            }
            let out = search_neural(root, model, &plans[&root.dimension()], spec.search, spec.head)?;
            let physical_cost = out.word.as_ref().map(|gates| gates.len());
            if let Some(cost) = physical_cost {
                solved += 1;
                solved_cost = solved_cost.checked_add(cost as u64).ok_or("quality cost overflow")?;
            }
            total_expanded = total_expanded.checked_add(out.expanded as u64).ok_or("quality expansion overflow")?;
            total_generated = total_generated.checked_add(out.generated as u64).ok_or("quality generation overflow")?;
            total_value_evaluations = total_value_evaluations
                .checked_add(out.value_evaluations as u64)
                .ok_or("quality value-evaluation overflow")?;
            total_action_evaluations = total_action_evaluations
                .checked_add(out.action_evaluations as u64)
                .ok_or("quality action-evaluation overflow")?;
            total_value_slots = total_value_slots
                .checked_add(out.value_slots as u64)
                .ok_or("quality value-slot overflow")?;
            total_action_slots = total_action_slots
                .checked_add(out.action_slots as u64)
                .ok_or("quality action-slot overflow")?;
            let word = out.word.map(|gates| json!({
                "control_target": gates.iter().map(|g| [g.control, g.target]).collect::<Vec<_>>(),
                "physical_cost": gates.len(), "replayed": true,
            }));
            rows.push(json!({
                "split": split, "index": index, "lineage": record.lineage_id,
                "dimension": root.dimension(), "alphabet_id": root.alphabet_id(),
                "start_packed_rows": root.packed_rows(), "status": out.status,
                "solved": physical_cost.is_some(), "physical_cost": physical_cost,
                "word": word, "depths": out.depths, "expanded": out.expanded,
                "generated": out.generated, "value_evaluations": out.value_evaluations,
                "action_evaluations": out.action_evaluations, "value_slots": out.value_slots,
                "action_slots": out.action_slots,
            }));
        }
        summaries.push(json!({
            "split": split,
            "cases": indices.len(),
            "solved": solved,
            "solve_rate": solved as f64 / indices.len() as f64,
            "mean_physical_cost_solved": (solved > 0).then_some(solved_cost as f64 / solved as f64),
            "total_expanded": total_expanded,
            "total_generated": total_generated,
            "total_value_evaluations": total_value_evaluations,
            "total_action_evaluations": total_action_evaluations,
            "total_value_slots": total_value_slots,
            "total_action_slots": total_action_slots,
        }));
    }
    Ok(json!({
        "schema": "cnot103_holdout_search_result_v2",
        "spec_sha256": pinned.sha256,
        "dataset_sha256": dataset.sha256,
        "search": spec.search,
        "head": spec.head,
        "search_seconds": started.elapsed().as_secs_f64(),
        "splits": summaries,
        "rows": rows,
    }))
}

#[derive(Clone, Copy)]
struct RowMetric {
    solved: bool,
    physical_cost: Option<u64>,
}

fn row_metrics(report: &Value) -> Result<BTreeMap<(String, u64), RowMetric>, String> {
    let rows = report["rows"].as_array().ok_or("holdout search report has no rows")?;
    let mut output = BTreeMap::new();
    for row in rows {
        let split = row["split"].as_str().ok_or("holdout search row has no split")?;
        if split != "dev" && split != "test" {
            return Err("holdout search row has an invalid split".into());
        }
        let index = row["index"].as_u64().ok_or("holdout search row has no index")?;
        let solved = row["solved"].as_bool().ok_or("holdout search row has no solved flag")?;
        let physical_cost = row["physical_cost"].as_u64();
        if solved != physical_cost.is_some() {
            return Err("holdout search solution and physical cost disagree".into());
        }
        if output.insert((split.to_owned(), index), RowMetric { solved, physical_cost }).is_some() {
            return Err("holdout search report repeats an index".into());
        }
    }
    Ok(output)
}

pub(super) fn compare(before: &Value, after: &Value) -> Result<Value, String> {
    for field in ["schema", "spec_sha256", "dataset_sha256", "search", "head"] {
        if before.get(field) != after.get(field) {
            return Err("before/after holdout search contracts differ".into());
        }
    }
    let before_rows = row_metrics(before)?;
    let after_rows = row_metrics(after)?;
    if before_rows.len() != after_rows.len() || before_rows.keys().ne(after_rows.keys()) {
        return Err("before/after holdout search indices differ".into());
    }

    let mut splits = Vec::new();
    for split in ["dev", "test"] {
        let mut cases = 0u64;
        let mut solved_before = 0u64;
        let mut solved_after = 0u64;
        let mut newly_solved = 0u64;
        let mut regressed = 0u64;
        let mut before_cost_sum = 0u64;
        let mut after_cost_sum = 0u64;
        let mut paired_before_cost = 0u64;
        let mut paired_after_cost = 0u64;
        let mut paired_solved = 0u64;
        for (key, baseline) in &before_rows {
            if key.0.as_str() != split {
                continue;
            }
            cases += 1;
            let trained = after_rows.get(key).ok_or("after-search row is missing")?;
            if baseline.solved {
                solved_before += 1;
                before_cost_sum += baseline.physical_cost.ok_or("baseline solution has no cost")?;
            }
            if trained.solved {
                solved_after += 1;
                after_cost_sum += trained.physical_cost.ok_or("trained solution has no cost")?;
            }
            match (baseline.solved, trained.solved) {
                (false, true) => newly_solved += 1,
                (true, false) => regressed += 1,
                (true, true) => {
                    paired_solved += 1;
                    paired_before_cost += baseline.physical_cost.ok_or("baseline solution has no cost")?;
                    paired_after_cost += trained.physical_cost.ok_or("trained solution has no cost")?;
                }
                (false, false) => {}
            }
        }
        if cases == 0 {
            continue;
        }
        let paired_cost_delta = (paired_solved > 0).then_some(
            paired_after_cost as f64 / paired_solved as f64
                - paired_before_cost as f64 / paired_solved as f64,
        );
        splits.push(json!({
            "split": split,
            "cases": cases,
            "solved_before": solved_before,
            "solved_after": solved_after,
            "newly_solved": newly_solved,
            "regressed": regressed,
            "solve_rate_before": solved_before as f64 / cases as f64,
            "solve_rate_after": solved_after as f64 / cases as f64,
            "solve_rate_delta": (solved_after as f64 - solved_before as f64) / cases as f64,
            "mean_physical_cost_before_solved": (solved_before > 0)
                .then_some(before_cost_sum as f64 / solved_before as f64),
            "mean_physical_cost_after_solved": (solved_after > 0)
                .then_some(after_cost_sum as f64 / solved_after as f64),
            "mean_paired_physical_cost_delta": paired_cost_delta,
        }));
    }
    Ok(json!({
        "schema": "cnot103_quality_delta_v1",
        "spec_sha256": before["spec_sha256"],
        "dataset_sha256": before["dataset_sha256"],
        "splits": splits,
    }))
}

#[cfg(test)]
mod tests {
    use super::{compare, parse_spec};
    use serde_json::json;

    #[test]
    fn holdout_spec_pins_explicit_unique_dev_and_test_indices() {
        let base = r#"{"schema":"cnot103_holdout_search_v2","dataset_sha256":"abc",
            "dev_indices":[1,2],"test_indices":[3],"search":{"beam_width":1,"shortlist":1,
            "max_depth":1,"max_expanded":1,"inference_batch":1},"head":"qhat"}"#;
        assert!(parse_spec(base.as_bytes()).is_ok());
        assert!(parse_spec(base.replace("[3]", "[3,3]").as_bytes()).is_err());
        assert!(parse_spec(base.replace("[1,2]", "[]").replace("[3]", "[]").as_bytes()).is_err());
    }

    #[test]
    fn quality_delta_reports_solved_transitions_and_paired_cost_change() {
        let before = json!({
            "schema":"cnot103_holdout_search_result_v2","spec_sha256":"spec","dataset_sha256":"data",
            "search":{},"head":"qhat",
            "rows":[
                {"split":"dev","index":1,"solved":true,"physical_cost":5},
                {"split":"dev","index":2,"solved":false,"physical_cost":null},
                {"split":"test","index":1,"solved":false,"physical_cost":null}
            ]
        });
        let after = json!({
            "schema":"cnot103_holdout_search_result_v2","spec_sha256":"spec","dataset_sha256":"data",
            "search":{},"head":"qhat",
            "rows":[
                {"split":"dev","index":1,"solved":true,"physical_cost":4},
                {"split":"dev","index":2,"solved":true,"physical_cost":6},
                {"split":"test","index":1,"solved":false,"physical_cost":null}
            ]
        });
        let delta = compare(&before, &after).unwrap();
        let dev = &delta["splits"][0];
        assert_eq!(dev["newly_solved"], 1);
        assert_eq!(dev["regressed"], 0);
        assert_eq!(dev["solve_rate_delta"].as_f64(), Some(0.5));
        assert_eq!(dev["mean_paired_physical_cost_delta"].as_f64(), Some(-1.0));
    }
}
