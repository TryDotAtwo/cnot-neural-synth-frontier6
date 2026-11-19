use super::{gl4_manifest, run_cli};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};

fn pinned(dir: &Path, name: &str) -> Value {
    let hash = format!("{:x}", Sha256::digest(fs::read(dir.join(name)).unwrap()));
    json!({"path":name,"sha256":hash})
}

#[test]
#[ignore = "runs own GL4 training and separate immutable-checkpoint CUDA dev search processes"]
fn trained_checkpoint_solves_two_prespecified_nontrivial_dev_roots() {
    let (dir, retained, manifest) = gl4_manifest(true, "CNOT103_SEARCH_TEST_ARTIFACTS");
    let run = dir.join("run.json");
    fs::write(&run, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let result = run_cli(&run);
    fs::write(
        dir.join("training-result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    assert_eq!(result["end_progress"]["accepted"], 1);
    assert_eq!(result["dev_rows"], 5);
    let before = fs::read(dir.join("checkpoint/last.json")).unwrap();
    let make = |head: &str, ids: &[usize]| {
        json!({"schema":"cnot103_dev_search_v1","training_manifest":pinned(&dir,"run.json"),
        "training_result":pinned(&dir,"training-result.json"),"dev_indices":ids,"head":head,
        "search":{"beam_width":11,"shortlist":11,"max_depth":2,"max_expanded":12,"inference_batch":4}})
    };
    let binary = env!("CARGO_BIN_EXE_cnot103-search");
    for (head, ids) in [("qhat", vec![3, 4]), ("proposal", vec![4])] {
        let spec = make(head, &ids);
        let path = dir.join(format!("search-{head}.json"));
        fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
        let output = Command::new(binary).arg(&path).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        fs::write(dir.join(format!("result-{head}.json")), &output.stdout).unwrap();
        let eval: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(eval["checkpoint"], result["checkpoint"]);
        assert_eq!(eval["training_exposure_audit"], result["exposure_audit"]);
        assert_eq!(eval["training_exposure_audit"]["strict_unseen"], true);
        assert_eq!(eval["panel"], "dev");
        assert_eq!(eval["feature_model_inputs"], 102);
        let rows = eval["rows"].as_array().unwrap();
        assert_eq!(rows.len(), ids.len());
        for (row, index) in rows.iter().zip(ids) {
            assert_eq!(row["dev_index"], index);
            assert_eq!(row["status"], "solved");
            assert_eq!(row["word"]["physical_cost"], 2);
            assert!(
                row["action_evaluations"].as_i64().unwrap() > 0
                    && row["value_evaluations"].as_i64().unwrap() > 0
            );
            let mut rows: Vec<u64> =
                serde_json::from_value(row["start_packed_rows"].clone()).unwrap();
            let gates: Vec<[usize; 2]> =
                serde_json::from_value(row["word"]["control_target"].clone()).unwrap();
            assert_eq!(gates.len(), 2);
            for &[c, t] in &gates {
                assert!(c < 4 && t < 4 && c != t);
                rows[t] ^= rows[c];
            }
            assert_eq!(rows, [1, 2, 4, 8]);
            println!("102-input checkpoint {head} dev[{index}] {:?} -> {:?}, expanded={} generated={} values={} actual slots={}/{}",row["start_packed_rows"],gates,row["expanded"],row["generated"],row["value_evaluations"],row["action_slots"],row["value_slots"]);
        }
    }
    for i in [1, 2] {
        let path = dir.join(format!("outside-scope-{i}.json"));
        fs::write(&path, serde_json::to_vec(&make("qhat", &[i])).unwrap()).unwrap();
        let out = Command::new(binary).arg(&path).output().unwrap();
        let err = String::from_utf8(out.stderr).unwrap();
        assert!(
            !out.status.success() && err.contains("F02 unsupported"),
            "{err}"
        );
        println!("strict coverage rejection dev[{i}]: {}", err.trim());
        fs::write(dir.join(format!("outside-scope-{i}.log")), err).unwrap();
    }
    let mut wrong = make("qhat", &[3]);
    wrong["training_result"]["sha256"] = json!("0".repeat(64));
    let bad = dir.join("mismatched-report.json");
    fs::write(&bad, serde_json::to_vec(&wrong).unwrap()).unwrap();
    let reject = Command::new(binary).arg(&bad).output().unwrap();
    assert!(!reject.status.success());
    assert!(String::from_utf8_lossy(&reject.stderr).contains("pinned SHA-256"));
    assert_eq!(before, fs::read(dir.join("checkpoint/last.json")).unwrap());
    if !retained {
        fs::remove_dir_all(dir).unwrap();
    }
}
