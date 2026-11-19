use cnot103_native::device_context::DeviceSelectedResidualContext;
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::{
    register_selected_residual_ranks, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy,
    FeatureRegistry, MatrixState,
};
use tch::{Device, Kind, Tensor};

fn state_batch(dimension: u32, alphabet_id: &str) -> Vec<MatrixState> {
    (0..3u32)
        .map(|seed| {
            let mut state = MatrixState::identity(dimension, alphabet_id).unwrap();
            for step in 0..dimension + 2 + seed {
                let control = (step * 17 + seed + 1) % dimension;
                let mut target = (step * 29 + seed + 2) % dimension;
                if target == control {
                    target = (target + 1) % dimension;
                }
                state.apply_cnot(CnotGate { control, target }).unwrap();
            }
            state
        })
        .collect()
}

#[test]
fn cuda_selected_residual_ranks_match_cpu_reference_and_alphabet_contract() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let id = FeatureId::new(b'C', 10).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![id],
        seed: 0,
        schema_id: "cuda-selected-residual-v1".into(),
        solver_assisted: false,
        symmetry_contract: "augmentation_only".into(),
        evaluations: None,
        max_bytes: None,
    };

    for dimension in [2u32, 3, 7, 65, 144] {
        let alphabet = CnotAlphabet::all_to_all_unit(dimension).unwrap();
        let gates = alphabet.gates().iter().copied().take(3).collect::<Vec<_>>();
        let states = state_batch(dimension, alphabet.id());
        let mut registry = FeatureRegistry::new();
        register_selected_residual_ranks(&mut registry, &alphabet, gates.clone()).unwrap();
        let expected = registry.build(&states, &policy).unwrap();
        let words_per_row = dimension.div_ceil(64);
        let packed_words = states
            .iter()
            .flat_map(|state| state.packed_rows().iter().map(|word| *word as i64))
            .collect::<Vec<_>>();
        let packed = Tensor::from_slice(&packed_words)
            .view([
                states.len() as i64,
                i64::from(dimension),
                i64::from(words_per_row),
            ])
            .to_device(Device::Cuda(0))
            .contiguous();
        let device_states =
            DeviceStateBatch::from_packed_tensor_for_alphabet(packed, &alphabet, 0).unwrap();
        let context = DeviceSelectedResidualContext::new(&alphabet, &gates, 0).unwrap();
        let actual = device_states.selected_residual_features(&context).unwrap();
        let values = actual.feature(id).unwrap();
        assert_eq!(values.device(), Device::Cuda(0));
        let values =
            Vec::<f32>::try_from(&values.to_device(Device::Cpu).contiguous().view([-1])).unwrap();
        assert_eq!(values, expected.feature(id).values, "C10, n={dimension}");

        let tokens = Vec::<i64>::try_from(
            &actual
                .token_valid()
                .to_kind(Kind::Int64)
                .to_device(Device::Cpu)
                .contiguous()
                .view([-1]),
        )
        .unwrap();
        let expected_tokens = expected
            .feature(id)
            .token_valid
            .as_ref()
            .unwrap()
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();
        assert_eq!(tokens, expected_tokens, "C10 group tokens, n={dimension}");

        let incidence = Vec::<f32>::try_from(
            &actual
                .incidence()
                .to_device(Device::Cpu)
                .contiguous()
                .view([-1]),
        )
        .unwrap();
        assert_eq!(
            incidence.as_slice(),
            expected.feature(id).incidence.as_ref().unwrap().as_slice(),
            "C10 incidence, n={dimension}"
        );

        let mismatched = CnotAlphabet::new(
            dimension,
            format!("wrong-alphabet:{dimension}"),
            alphabet.gates().to_vec(),
            vec![1; alphabet.gates().len()],
        )
        .unwrap();
        let mismatched_context =
            DeviceSelectedResidualContext::new(&mismatched, &gates, 0).unwrap();
        assert!(device_states
            .selected_residual_features(&mismatched_context)
            .is_err());
    }
}

#[test]
fn cuda_multi_width_selected_residual_matches_cpu_and_marks_missing_width() {
    use cnot103_native::device_feature_plan::{DeviceFeatureContextInputs, DeviceFeaturePlan, SelectedResidualPlanInputs};
    use cnot103_native::{register_selected_residual_ranks_multi, SelectedResidualContext};
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let a3 = CnotAlphabet::all_to_all_unit(3).unwrap();
    let a4 = CnotAlphabet::all_to_all_unit(4).unwrap();
    let a5 = CnotAlphabet::all_to_all_unit(5).unwrap();
    let gates3 = vec![a3.gates()[0], a3.gates()[1]];
    let extras = vec![SelectedResidualContext { alphabet: a4.clone(), gates: vec![a4.gates()[2]] }];
    let mut registry = FeatureRegistry::new();
    register_selected_residual_ranks_multi(&mut registry, vec![
        SelectedResidualContext { alphabet: a3.clone(), gates: gates3.clone() }, extras[0].clone()
    ]).unwrap();
    let id = FeatureId::new(b'C', 10).unwrap();
    let policy = FeaturePolicy {
        enabled_ids: vec![id], seed: 0,
        schema_id: cnot103_native::schema::MISSING_TOKEN_LAYOUT_ID.into(),
        solver_assisted: false, symmetry_contract: "augmentation_only".into(),
        evaluations: None, max_bytes: None,
    };
    for alphabet in [&a3, &a4, &a5] {
        let dimension = alphabet.dimension();
        let states = state_batch(dimension, alphabet.id());
        let packed: Vec<i64> = states.iter().flat_map(|s| s.packed_rows().iter().map(|&w| w as i64)).collect();
        let tensor = Tensor::from_slice(&packed).view([states.len() as i64, i64::from(dimension), i64::from(dimension.div_ceil(64))]).to_device(Device::Cuda(0));
        let device_states = DeviceStateBatch::from_packed_tensor_for_alphabet(tensor, alphabet, 0).unwrap();
        let plan = DeviceFeaturePlan::with_contexts(&registry, &policy, dimension, 0,
            DeviceFeatureContextInputs {
                selected_residual: Some(SelectedResidualPlanInputs { alphabet: &a3, gates: &gates3 }),
                selected_residual_additional_contexts: &extras,
                ..Default::default()
            }).unwrap();
        let gpu = plan.build(&device_states).unwrap();
        let cpu = registry.build(&states, &policy).unwrap();
        let actual = gpu.feature(id).unwrap();
        let status = Vec::<i64>::try_from(&actual.status().to_kind(Kind::Int64).to_device(Device::Cpu).view([-1])).unwrap();
        let code = if dimension == 5 { 3 } else { 1 };
        assert_eq!(status, vec![code; states.len()]);
        if dimension != 5 {
            let values = Vec::<f32>::try_from(&actual.values().to_device(Device::Cpu).view([-1])).unwrap();
            assert_eq!(values, cpu.feature(id).values, "n={dimension}");
            let tokens = Vec::<i64>::try_from(&actual.token_valid().unwrap().to_kind(Kind::Int64).to_device(Device::Cpu).view([-1])).unwrap();
            assert_eq!(tokens, cpu.feature(id).token_valid.as_ref().unwrap().iter().map(|&x| i64::from(x)).collect::<Vec<_>>());
        }
    }
}
