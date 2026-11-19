//! Optional, explicitly specified dev search on a discard-only benchmark model.
use super::artifacts;
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::device_search::{search_neural, ActionHead, BeamConfig};
use crate::model::FusionDistanceField;
use crate::{FrozenDataset, TrainingSplit};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::time::Instant;
use tch::{nn::VarStore, Tensor};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QualitySpec {
    schema: String,
    dataset_sha256: String,
    dev_indices: Vec<usize>,
    search: BeamConfig,
    head: ActionHead,
}

pub(super) struct PinnedQualitySpec {
    spec: QualitySpec,
    sha256: String,
}

fn parse_spec(raw: &[u8]) -> Result<PinnedQualitySpec, String> {
    let spec: QualitySpec = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    if spec.schema != "cnot103_discard_dev_search_v1"
        || spec.dev_indices.is_empty()
        || spec.dev_indices.iter().collect::<BTreeSet<_>>().len() != spec.dev_indices.len()
    {
        return Err("discard dev search requires its schema and distinct explicit dev indices".into());
    }
    Ok(PinnedQualitySpec { spec, sha256: artifacts::sha256(raw) })
}

pub(super) fn optional_spec() -> Result<Option<PinnedQualitySpec>, String> {
    std::env::var_os("CNOT103_BENCHMARK_EVAL_SPEC")
        .map(|path| fs::read(path).map_err(|e| e.to_string()).and_then(|raw| parse_spec(&raw)))
        .transpose()
}

pub(super) fn evaluate(
    pinned: PinnedQualitySpec,
    dataset: &FrozenDataset,
    model: &FusionDistanceField,
    plans: &HashMap<u32, DeviceFeaturePlan>,
    variables: &VarStore,
    selected_count: usize,
) -> Result<Value, String> {
    let spec = pinned.spec;
    if spec.dataset_sha256 != dataset.sha256 || selected_count != 102 {
        return Err("discard dev search dataset or full-input contract differs".into());
    }
    for &i in &spec.dev_indices {
        let row = dataset.dev.records().get(i).ok_or("dev search index out of range")?;
        if row.split != TrainingSplit::Dev || !dataset.forbidden_keys.contains(&row.state)
            || !plans.contains_key(&row.state.dimension())
        {
            return Err("requested dev root lacks a frozen holdout identity or resident plan".into());
        }
    }
    let finite: Vec<_> = variables.variables().values().map(|t| t.isfinite().all()).collect();
    if Tensor::stack(&finite, 0).all().int64_value(&[]) == 0 {
        return Err("discard benchmark model has nonfinite weights".into());
    }
    let started = Instant::now();
    let mut rows = Vec::with_capacity(spec.dev_indices.len());
    for &i in &spec.dev_indices {
        let record = &dataset.dev.records()[i];
        let root = &record.state;
        let out = search_neural(root, model, &plans[&root.dimension()], spec.search, spec.head)?;
        let word = out.word.map(|gates| json!({
            "control_target": gates.iter().map(|g| [g.control, g.target]).collect::<Vec<_>>(),
            "physical_cost": gates.len(), "replayed": true,
        }));
        rows.push(json!({
            "dev_index": i, "lineage": record.lineage_id, "dimension": root.dimension(),
            "alphabet_id": root.alphabet_id(), "start_packed_rows": root.packed_rows(),
            "status": out.status, "word": word, "depths": out.depths,
            "expanded": out.expanded, "generated": out.generated,
            "value_evaluations": out.value_evaluations, "action_evaluations": out.action_evaluations,
            "value_slots": out.value_slots, "action_slots": out.action_slots,
        }));
    }
    Ok(json!({"schema": "cnot103_discard_dev_search_result_v1", "spec_sha256": pinned.sha256,
        "dataset_sha256": dataset.sha256, "search": spec.search, "head": spec.head,
        "search_seconds": started.elapsed().as_secs_f64(), "rows": rows}))
}

#[cfg(test)]
mod tests {
    use super::parse_spec;

    #[test]
    fn discard_search_requires_explicit_distinct_dev_indices() {
        let base = r#"{"schema":"cnot103_discard_dev_search_v1","dataset_sha256":"abc",
            "dev_indices":[1,2],"search":{"beam_width":1,"shortlist":1,"max_depth":1,
            "max_expanded":1,"inference_batch":1},"head":"qhat"}"#;
        assert!(parse_spec(base.as_bytes()).is_ok());
        assert!(parse_spec(base.replace("[1,2]", "[]").as_bytes()).is_err());
        assert!(parse_spec(base.replace("[1,2]", "[1,1]").as_bytes()).is_err());
        assert!(parse_spec(base.replace("discard_dev", "other").as_bytes()).is_err());
    }
}
