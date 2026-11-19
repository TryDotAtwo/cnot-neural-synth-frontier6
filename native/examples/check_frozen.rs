//! Validate a sealed dataset through the production loader without a GPU.
use cnot103_native::FrozenDataset;
use serde_json::json;
use std::{collections::BTreeMap, env, process::ExitCode};

fn main() -> ExitCode {
    let args: Vec<_> = env::args().skip(1).collect();
    let [path, sha256] = args.as_slice() else {
        eprintln!("Usage: check_frozen <dataset.json> <pinned SHA-256>");
        return ExitCode::from(2);
    };
    let data = match FrozenDataset::load_pinned(path, sha256) {
        Ok(data) => data,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let mut exact_by_source = BTreeMap::new();
    for (record, metadata) in data.train.records().iter().zip(data.train.metadata()) {
        if record.exact_distance.is_some() {
            *exact_by_source.entry(&metadata.source).or_insert(0usize) += 1;
        }
    }
    println!(
        "{}",
        json!({
            "dataset_sha256": data.sha256, "train": data.train.len(), "dev": data.dev.len(),
            "test": data.test.len(), "forbidden": data.forbidden_keys.len(),
            "exact_train_by_source": exact_by_source,
        })
    );
    ExitCode::SUCCESS
}
