//! Readbacks in this file are comparisons against the independent host reference.
use cnot103_native::bellman::device::{DevicePrimitiveBellman, RootBranch};
use cnot103_native::device_feature_plan::DeviceFeaturePlan;
use cnot103_native::device_state::DeviceStateBatch;
use cnot103_native::model::{FusionDistanceField, ModelConfig};
use cnot103_native::{
    recompute_backup, select_backup, CnotAlphabet, CnotGate, FeatureId, FeaturePolicy,
    FeatureRegistry, MatrixState,
};
use tch::{nn::VarStore, Device, Kind, Tensor};

fn config() -> ModelConfig {
    ModelConfig {
        wire_dim: 8,
        pair_dim: 4,
        structural_blocks: 1,
        attention_heads: 2,
        feedforward_dim: 16,
        fusion_blocks: 1,
        basis_proposal_hidden_dim: 8,
    }
}

fn policy() -> FeaturePolicy {
    let mut p = FeaturePolicy::all_model_inputs("device-bellman-test");
    p.enabled_ids = (1..=20).map(|i| FeatureId::new(b'A', i).unwrap()).collect();
    p
}

fn resident(state: &MatrixState, alphabet: &CnotAlphabet, index: usize) -> DeviceStateBatch {
    let n = i64::from(state.dimension());
    let words: Vec<_> = state.packed_rows().iter().map(|x| *x as i64).collect();
    DeviceStateBatch::from_packed_tensor_for_alphabet(
        Tensor::from_slice(&words)
            .view([1, n, (n + 63) / 64])
            .to_device(Device::Cuda(index)),
        alphabet,
        index,
    )
    .unwrap()
}

fn compare_gradients(vars: &VarStore, actual: &Tensor, expected: &Tensor) {
    let mut params: Vec<_> = vars
        .variables()
        .into_iter()
        .filter(|(_, t)| t.requires_grad())
        .collect();
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let p: Vec<_> = params.iter().map(|(_, v)| v).collect();
    let a = Tensor::run_backward(&[actual], &p, false, false);
    let b = Tensor::run_backward(&[expected], &p, false, false);
    assert_eq!(a.len(), params.len());
    let mut nonzero = 0;
    for (((name, _), a), b) in params.iter().zip(a).zip(b) {
        assert_eq!(a.defined(), b.defined(), "{name}");
        if a.defined() {
            assert!(a.allclose(&b, 5e-5, 3e-5, false), "{name}");
            nonzero += usize::from(b.abs().max().double_value(&[]) > 0.0);
        }
    }
    assert!(nonzero > 0);
}

#[test]
fn primitive_selection_retains_all_costs_and_differentiates_the_same_branch() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    tch::manual_seed(4121);
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let registry = FeatureRegistry::new();
    let p = policy();
    let plan = DeviceFeaturePlan::context_free(&registry, &p, 0).unwrap();
    let vars = VarStore::new(Device::Cuda(0));
    let model =
        FusionDistanceField::new_for_policy(&vars.root(), &registry, &p, vars.device(), config())
            .unwrap();
    let kernel = DevicePrimitiveBellman::new(&alphabet, 0).unwrap();
    let mut state = MatrixState::identity(4, alphabet.id()).unwrap();
    for i in [1, 5, 8] {
        state.apply_cnot(alphabet.gates()[i]).unwrap();
    }
    let gpu = resident(&state, &alphabet, 0);
    for chunk in [1, 5, 15] {
        let got = kernel
            .backup(&model, &plan, &gpu, RootBranch::Nonterminal, chunk, false)
            .unwrap();
        assert_eq!(got.valid.int64_value(&[]), 1);
        assert_eq!(got.primitive_totals.device(), vars.device());
        assert_eq!(got.primitive_totals.kind(), Kind::Double);
        assert!(!got.primitive_totals.requires_grad());
        let selection = select_backup(
            &model,
            &vars,
            std::slice::from_ref(&state),
            &registry,
            &p,
            &alphabet,
            Some(chunk),
        )
        .unwrap();
        let branch = selection.branches().next().unwrap();
        assert!(got.primitive_totals.allclose(
            &Tensor::from_slice(&branch.primitive_totals).to_device(vars.device()),
            2e-5,
            2e-5,
            false
        ));
        assert_eq!(
            got.selected_index.as_ref().unwrap().int64_value(&[]),
            branch.selected_index.unwrap() as i64
        );
        let source = recompute_backup(&model, &vars, &selection).unwrap();
        assert!(got.loss.allclose(&source.loss, 2e-5, 2e-5, false));
        compare_gradients(&vars, &got.loss, &source.loss);
    }
    assert!(kernel
        .backup(&model, &plan, &gpu, RootBranch::Nonterminal, 0, false)
        .is_err());
    assert!(kernel
        .backup(&model, &plan, &gpu, RootBranch::Nonterminal, 5, true)
        .is_err());
}

#[test]
fn terminal_qhat_workspace_failure_explains_the_unsplit_packet() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let alphabet = CnotAlphabet::all_to_all_unit(4).unwrap();
    let registry = FeatureRegistry::new();
    let mut p = policy();
    p.max_bytes = Some((4 * 4 * 1024 + 4 * 256) * 2);
    let plan = DeviceFeaturePlan::context_free(&registry, &p, 0).unwrap();
    let vars = VarStore::new(Device::Cuda(0));
    let model =
        FusionDistanceField::new_for_policy(&vars.root(), &registry, &p, vars.device(), config())
            .unwrap();
    let state = MatrixState::identity(4, alphabet.id()).unwrap();
    let gpu = resident(&state, &alphabet, 0);
    let kernel = DevicePrimitiveBellman::new(&alphabet, 0).unwrap();
    let error = kernel
        .backup(
            &model,
            &plan,
            &gpu,
            RootBranch::Terminal { qhat: true },
            2,
            true,
        )
        .err()
        .expect("the terminal packet exceeds the feature budget");
    assert!(error.contains("terminal Q-hat"), "{error}");
    assert!(!error.contains("reduce neighbor_chunk_size"), "{error}");
}

#[test]
fn custom_costs_keep_fp64_order_and_first_tie_across_chunk_boundaries() {
    cnot103_native::torch_device::require_cuda_device(0).unwrap();
    let n = 65;
    let gates = vec![
        CnotGate {
            control: 64,
            target: 0,
        },
        CnotGate {
            control: 0,
            target: 64,
        },
        CnotGate {
            control: 1,
            target: 63,
        },
    ];
    let alphabet = CnotAlphabet::new(
        n,
        "weighted-sparse",
        gates,
        vec![16_777_217, 16_777_216, 16_777_216],
    )
    .unwrap();
    let mut state = MatrixState::identity(n, alphabet.id()).unwrap();
    for gate in alphabet.gates() {
        state.apply_cnot(*gate).unwrap();
    }
    let registry = FeatureRegistry::new();
    let p = policy();
    let plan = DeviceFeaturePlan::context_free(&registry, &p, 0).unwrap();
    let vars = VarStore::new(Device::Cuda(0));
    let model =
        FusionDistanceField::new_for_policy(&vars.root(), &registry, &p, vars.device(), config())
            .unwrap();
    tch::no_grad(|| {
        for (_, mut t) in vars.variables() {
            let _ = t.zero_();
        }
    });
    let kernel = DevicePrimitiveBellman::new(&alphabet, 0).unwrap();
    let gpu = resident(&state, &alphabet, 0);
    for chunk in [1, 2, 3] {
        let got = kernel
            .backup(&model, &plan, &gpu, RootBranch::Nonterminal, chunk, false)
            .unwrap();
        assert_eq!(got.valid.int64_value(&[]), 1);
        assert_eq!(got.selected_index.unwrap().int64_value(&[]), 1);
        let totals = Vec::<f64>::try_from(got.primitive_totals.to_device(Device::Cpu)).unwrap();
        assert_eq!(totals[0] - totals[1], 1.0);
        assert_eq!(totals[1], totals[2]);
    }
    let identity = MatrixState::identity(n, alphabet.id()).unwrap();
    let gpu = resident(&identity, &alphabet, 0);
    for qhat in [false, true] {
        let got = kernel
            .backup(&model, &plan, &gpu, RootBranch::Terminal { qhat }, 2, false)
            .unwrap();
        assert_eq!(got.valid.int64_value(&[]), 1);
        assert!(got.selected_index.is_none());
        assert_eq!(got.primitive_totals.numel(), if qhat { 3 } else { 0 });
        assert_eq!(got.loss.double_value(&[]), 0.0);
    }
    let wrong = kernel
        .backup(&model, &plan, &gpu, RootBranch::Nonterminal, 2, false)
        .unwrap();
    assert_eq!(wrong.valid.int64_value(&[]), 0);
}
