//! Uses the same pinned corpus and trained checkpoint as the dev-search regression.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn pinned(path: &Path) -> Value {
    json!({"path":path,"sha256":format!("{:x}",Sha256::digest(fs::read(path).unwrap()))})
}

#[test]
#[ignore = "requires the frozen trained GL4 dev panel"]
fn separate_process_reports_real_view_forwards_from_the_bound_checkpoint() {
    let trained = PathBuf::from(std::env::var_os("CNOT103_H07_TRAINING_DIR").unwrap())
        .canonicalize()
        .unwrap();
    let dir = PathBuf::from(std::env::var_os("CNOT103_H07_TEST_ARTIFACTS").unwrap());
    fs::create_dir(&dir).unwrap();
    let path = dir.join("h07.json");
    let mut spec = json!({"schema":"cnot103_dev_h07_v1","training_manifest":pinned(&trained.join("run.json")),
        "training_result":pinned(&trained.join("training-result.json")),"dev_indices":[0,1,2,3,4]});
    fs::write(&path, serde_json::to_vec_pretty(&spec).unwrap()).unwrap();
    let before = fs::read(trained.join("checkpoint/last.json")).unwrap();
    let binary = env!("CARGO_BIN_EXE_cnot103-h07");
    let out = Command::new(binary).arg(&path).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    fs::write(dir.join("result.json"), &out.stdout).unwrap();
    let training: Value =
        serde_json::from_slice(&fs::read(trained.join("training-result.json")).unwrap()).unwrap();
    assert_eq!(v["checkpoint"], training["checkpoint"]);
    assert_eq!(v["training_exposure_audit"], training["exposure_audit"]);
    assert_eq!(v["feature_model_inputs"], 102);
    assert_eq!(
        v["rows"].as_array().unwrap().len(),
        training["dev_rows"].as_u64().unwrap() as usize
    );
    let mut positive_variances = 0;
    for (idx, row) in v["rows"].as_array().unwrap().iter().enumerate() {
        assert_eq!(row["dev_index"], idx);
        let d = &row["diagnostic"];
        assert_eq!(d["id"], "H07");
        assert_eq!(d["sample_count"], 5);
        assert_eq!(d["view_sample_count"], 4);
        assert_eq!(d["relabel_sample_count"], 1);
        let probes = d["probes"].as_array().unwrap();
        assert_eq!(probes.len(), 5);
        let values: Vec<_> = probes
            .iter()
            .map(|p| p["prediction"].as_f64().unwrap())
            .collect();
        let mean = values.iter().sum::<f64>() / 5.;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 5.;
        positive_variances += usize::from(variance > 0.);
        assert!((d["population_variance"].as_f64().unwrap() - variance).abs() < 1e-14);
        assert_eq!(d["channel"].as_f64().unwrap() as f32, variance as f32);
        for p in probes {
            let c = &p["input_coverage"];
            assert_eq!(c["complete"], true);
            let entries = c["entries"].as_array().unwrap();
            assert_eq!(entries.len(), 102);
            assert!(entries.iter().all(|e| e["id"] != "H07"
                && (e["status"] == "computed" && e["available"] == 1
                    || e["status"] == "inapplicable"
                        && e["available"] == 0
                        && !e["reason"].as_str().unwrap().is_empty())));
        }
        println!(
            "full H07 dev[{idx}] values={values:?} variance={variance:e} probe_sha={}",
            d["probe_set_sha256"]
        );
    }
    assert!(
        positive_variances > 0,
        "the trained field resolves distinct physical views"
    );
    spec["training_result"]["sha256"] = json!("0".repeat(64));
    let wrong = dir.join("wrong-digest.json");
    fs::write(&wrong, serde_json::to_vec(&spec).unwrap()).unwrap();
    let reject = Command::new(binary).arg(wrong).output().unwrap();
    assert!(
        !reject.status.success()
            && String::from_utf8_lossy(&reject.stderr).contains("pinned SHA-256")
    );
    assert_eq!(
        before,
        fs::read(trained.join("checkpoint/last.json")).unwrap()
    );
}
