use cnot103_native::{nn_training::NeuralTrainer, CheckpointStore};
use std::{collections::BTreeMap, io::Cursor};
use tch::{nn::VarStore, Tensor};

pub fn saved(
    vars: &VarStore,
    trainer: &NeuralTrainer,
    name: &str,
) -> (serde_json::Value, BTreeMap<String, Tensor>) {
    let path = std::env::temp_dir().join(format!(
        "device-adamw-{}-{}-{name}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let store = CheckpointStore::new(&path);
    let contract = serde_json::json!({"id":"device-adamw-parity"});
    trainer
        .save_checkpoint(vars, &store, contract.clone(), false)
        .unwrap();
    let bytes = store.load_latest(&contract).unwrap().optimizer_state;
    let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let meta = serde_json::from_slice(&bytes[16..16 + len]).unwrap();
    let moments = Tensor::load_multi_from_stream(Cursor::new(&bytes[16 + len..]))
        .unwrap()
        .into_iter()
        .collect();
    std::fs::remove_dir_all(path).unwrap();
    (meta, moments)
}
