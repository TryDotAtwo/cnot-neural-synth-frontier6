use super::*;
use crate::{compute_h07_prediction_spread, CnotGate, H07ProbeKind};
use std::collections::HashSet;

/// Derive the same five matrices independently by rewriting elementary CNOT
/// words. The GPU implementation never receives these gates.
pub(crate) fn word_views(
    n: u32,
    construction: &[CnotGate],
) -> Vec<(&'static str, H07ProbeKind, MatrixState)> {
    let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
    let run = |gates: Vec<CnotGate>| {
        let mut s = MatrixState::identity(n, alphabet.id()).unwrap();
        for g in gates {
            s.apply_cnot(g).unwrap();
        }
        s
    };
    let swapped = |g: &CnotGate| CnotGate {
        control: g.target,
        target: g.control,
    };
    let reversed = |g: &CnotGate| CnotGate {
        control: n - 1 - g.control,
        target: n - 1 - g.target,
    };
    vec![
        (
            "canonical",
            H07ProbeKind::Canonical,
            run(construction.to_vec()),
        ),
        (
            "inverse",
            H07ProbeKind::Inverse,
            run(construction.iter().rev().copied().collect()),
        ),
        (
            "inverse_transpose",
            H07ProbeKind::InverseTranspose,
            run(construction.iter().map(swapped).collect()),
        ),
        (
            "transpose",
            H07ProbeKind::Transpose,
            run(construction.iter().rev().map(swapped).collect()),
        ),
        (
            "wire_reversal",
            H07ProbeKind::Relabel,
            run(construction.iter().map(reversed).collect()),
        ),
    ]
}

#[test]
fn packed_cuda_transforms_match_elementary_words_across_bit_63() {
    crate::torch_device::require_cuda_device(0).unwrap();
    let index = usize::from(tch::Cuda::device_count() > 1);
    crate::torch_device::require_cuda_device(index).unwrap();
    let device = Device::Cuda(index);
    for n in [4, 63, 64, 65, 67] {
        let word = [[n - 1, 0], [0, 1], [1, n - 1], [n - 2, n - 1], [n - 1, 1]]
            .map(|[control, target]| CnotGate { control, target });
        let expected = word_views(n, &word);
        let root = &expected[0].2;
        let a = CnotAlphabet::all_to_all_unit(n).unwrap();
        let scope = problem_scope(root, &a).unwrap();
        assert_eq!(scope["wire_reversal_old_to_new"][0], n - 1);
        let views = views::packed_views(root, device);
        assert_eq!(
            expected.iter().map(|p| &p.2).collect::<HashSet<_>>().len(),
            5
        );
        for ((packed, (id, kind, state)), meta) in views.iter().zip(&expected).zip(views::PROBES) {
            assert_eq!((*id, *kind), meta);
            assert_eq!(packed.device(), device);
            let wanted: Vec<_> = state.packed_rows().iter().map(|&w| w as i64).collect();
            assert_eq!(
                Vec::<i64>::try_from(packed.to_device(Device::Cpu).view([-1])).unwrap(),
                wanted,
                "width {n} {id}"
            );
        }
        let mut costs = vec![1; a.gates().len()];
        costs[0] = 2;
        let different = CnotAlphabet::new(n, a.id(), a.gates().to_vec(), costs).unwrap();
        assert!(problem_scope(root, &different)
            .unwrap_err()
            .contains("exact ordered"));
    }
    println!("five CUDA matrix transforms matched independent CNOT words at 4,63,64,65,67 on cuda:{index}");
}

#[test]
fn device_moments_match_sorted_source_and_reject_nonfinite_and_float_overflow() {
    crate::torch_device::require_cuda_device(0).unwrap();
    let digest = "89".repeat(32);
    for values in [
        [0., 0., 0., 0., 0.],
        [4., -2., 6., 3., 8.],
        [1.0000001, 1., 1.00000003, 0.99999997, 1.00000011],
        [0., 1.0e24, 0., 0., 0.],
        [0., 1., f64::NAN, 2., 3.],
    ] {
        let tensor = Tensor::from_slice(&values).to_device(Device::Cuda(0));
        let actual = Vec::<f64>::try_from(moments(&tensor).to_device(Device::Cpu)).unwrap();
        let probes: Vec<_> = views::PROBES
            .iter()
            .zip(values)
            .map(|(&(id, kind), prediction)| H07ProbePrediction {
                probe_id: id.into(),
                kind,
                prediction,
                input_coverage: None,
            })
            .collect();
        let source = compute_h07_prediction_spread(&digest, &digest, &probes);
        let actual = validate_probe_metadata(&probes).and_then(|counts| {
            finish_spread(
                digest.clone(),
                digest.clone(),
                probes,
                counts,
                Spread::from_export(actual.try_into().unwrap()),
            )
        });
        assert_eq!(actual, source);
    }
}

#[test]
fn strict_probe_rejects_unavailable_or_unknown_reasons() {
    let mut codes = vec![1.; MODEL_INPUT_COUNT];
    let input_ids: Vec<_> = all_model_input_specs()
        .into_iter()
        .map(|s| s.id.to_string())
        .collect();
    let slot = |id: &str| input_ids.iter().position(|s| s == id).unwrap();
    for id in ["C05", "E01", "E09", "G01", "G06", "I07"] {
        codes[slot(id)] = 2.;
    }
    let complete = coverage("canonical", "s", "p", &codes).unwrap();
    assert_eq!(complete.inapplicable_model_input_count(), 6);
    for (c, label) in [
        (0., "not_requested"),
        (3., "unsupported"),
        (4., "resource_limit"),
        (5., "invalid_status"),
        (2., "no source inapplicability reason"),
    ] {
        codes[slot("F02")] = c;
        let err = coverage("transpose", "s", "p", &codes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("transpose") && err.contains("F02") && err.contains(label));
    }
}
