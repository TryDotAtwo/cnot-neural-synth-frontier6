use super::TchTrainingError;
use std::collections::HashMap;
use std::io::Cursor;
use tch::nn::VarStore;
use tch::Tensor;

pub(super) fn encode_model_state(variables: &VarStore) -> Result<Vec<u8>, TchTrainingError> {
    // VarStore::save_to_stream iterates its internal HashMap. Sort names first
    // so logically identical checkpoints have identical model payload bytes and
    // therefore stable content hashes across process reconstructions.
    let mut named_tensors: Vec<_> = variables.variables().into_iter().collect();
    named_tensors.sort_by(|left, right| left.0.cmp(&right.0));
    let mut bytes = Vec::new();
    Tensor::save_multi_to_stream(&named_tensors, &mut bytes)?;
    Ok(bytes)
}

pub(super) fn validate_model_state(
    variables: &VarStore,
    bytes: &[u8],
) -> Result<(), TchTrainingError> {
    let loaded =
        Tensor::load_multi_from_stream_with_device(Cursor::new(bytes), variables.device())?;
    let loaded_map: HashMap<_, _> = loaded.into_iter().collect();
    let current = variables.variables();
    if loaded_map.len() != current.len() {
        return Err(TchTrainingError::Invalid(
            "model checkpoint variable count differs",
        ));
    }
    for (name, variable) in current {
        let saved = loaded_map.get(&name).ok_or(TchTrainingError::Invalid(
            "model checkpoint variable is missing",
        ))?;
        if saved.size() != variable.size() || saved.kind() != variable.kind() {
            return Err(TchTrainingError::Invalid(
                "model checkpoint variable shape or type differs",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod model_state_serialization_tests {
    use super::{encode_model_state, validate_model_state};
    use tch::nn::{Init, VarStore};
    use tch::Device;

    fn variables(insert_reverse: bool) -> VarStore {
        let store = VarStore::new(Device::Cpu);
        let root = store.root();
        let add_alpha = || root.var("alpha", &[2, 3], Init::Const(0.25));
        let add_omega = || root.var("omega", &[3], Init::Const(-0.5));
        if insert_reverse {
            let _ = add_omega();
            let _ = add_alpha();
        } else {
            let _ = add_alpha();
            let _ = add_omega();
        }
        store
    }

    #[test]
    fn canonical_encoder_is_order_independent_and_reads_legacy_var_store_streams() {
        let alpha_first = variables(false);
        let omega_first = variables(true);
        let canonical_alpha_first = encode_model_state(&alpha_first).unwrap();
        let canonical_omega_first = encode_model_state(&omega_first).unwrap();
        assert_eq!(canonical_alpha_first, canonical_omega_first);

        let mut legacy_stream = Vec::new();
        alpha_first.save_to_stream(&mut legacy_stream).unwrap();
        validate_model_state(&omega_first, &legacy_stream).unwrap();
        validate_model_state(&alpha_first, &canonical_alpha_first).unwrap();
    }
}
