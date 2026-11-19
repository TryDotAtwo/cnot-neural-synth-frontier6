//! Audit actual producer statuses without loading model weights or opening a checkpoint.
use super::{artifacts, feature_runtime};
use crate::device_state::DeviceStateBatch;
use crate::packet::FeatureStatus;
use crate::run_config::parse_run_manifest;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
use tch::{Device, Kind, Tensor};

pub fn audit_feature_coverage(path: &Path, split: &str) -> Result<Value, String> {
    let profile = crate::stage_profile::Session::start();
    let raw = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let manifest = parse_run_manifest(&raw)?;
    let paths = artifacts::input_paths(path, &raw, &manifest);
    let data = artifacts::load(&manifest, &paths)?;
    let rows = match split {
        "train" => data.dataset.train.records(),
        "dev" => data.dataset.dev.records(),
        _ => return Err("choose train or dev for feature coverage".into()),
    };
    let index = manifest.training.cuda_device;
    crate::stage_profile::measure("cuda_init", || {
        crate::torch_device::require_cuda_device(index)
    })?;
    profile.device(index);
    let runtime = feature_runtime::build(
        manifest.feature_contexts,
        manifest.feature_policy,
        &data.alphabets,
        manifest.training.seed,
        rows.iter().map(|r| r.state.dimension()).collect(),
        index,
    )?;
    let mut specs = runtime
        .registry
        .selected_specs(&runtime.policy)
        .map_err(|e| e.to_string())?;
    specs.sort_by_key(|s| s.id.index());
    let mut by_width = BTreeMap::new();
    for record in rows {
        let state = &record.state;
        let n = state.dimension();
        let alphabet = data
            .alphabets
            .iter()
            .find(|a| a.id() == state.alphabet_id())
            .ok_or("missing bound alphabet")?;
        let packed: Vec<_> = state.packed_rows().iter().map(|&w| w as i64).collect();
        let batch = DeviceStateBatch::from_packed_tensor_for_alphabet(
            Tensor::from_slice(&packed)
                .view([1, i64::from(n), state.words_per_row() as i64])
                .to_device(Device::Cuda(index)),
            alphabet,
            index,
        )?;
        let packet = runtime.plans[&n].build(&batch)?;
        let codes = packet
            .selected_status_matrix()
            .map_err(|e| e.to_string())?
            .view([-1])
            .to_device(Device::Cpu)
            .to_kind(Kind::Uint8);
        let codes = Vec::<u8>::try_from(codes).map_err(|e| e.to_string())?;
        let (samples, rejected, counts) = by_width
            .entry(n)
            .or_insert_with(|| (0usize, 0usize, vec![[0usize; 5]; specs.len()]));
        let mut admitted = true;
        for ((spec, count), code) in specs.iter().zip(counts).zip(codes) {
            let status = match code {
                0 => FeatureStatus::NotRequested,
                1 => FeatureStatus::Computed,
                2 => FeatureStatus::Inapplicable,
                3 => FeatureStatus::Unsupported,
                4 => FeatureStatus::ResourceLimit,
                _ => return Err(format!("{}: invalid provider status {code}", spec.id)),
            };
            count[usize::from(code)] += 1;
            admitted &= status.admitted(spec.id, u8::from(code == 1), packet.schema_id());
        }
        *samples += 1;
        *rejected += usize::from(!admitted);
    }
    let widths: Vec<_> = by_width.into_iter().map(|(width,(samples,rejected,counts))| {
        let features: Vec<_> = specs.iter().zip(counts).map(|(s,c)| json!({
            "id":s.id.to_string(), "role":format!("{:?}",s.role), "scope":s.scope, "provider":s.producer_version,
            "not_requested":c[0], "computed":c[1], "inapplicable":c[2], "unsupported":c[3], "resource_limit":c[4]
        })).collect();
        json!({"width":width, "rows":samples, "rejected_rows":rejected, "features":features})
    }).collect();
    Ok(
        json!({"schema":"cnot103_feature_coverage_v1", "input_schema":runtime.policy.schema_id,
        "policy_id":runtime.policy.id().map_err(|e| e.to_string())?, "manifest_sha256":paths.run_sha256,
        "dataset_sha256":manifest.dataset.sha256, "panel":split, "batch_size":1, "widths":widths}),
    )
}
