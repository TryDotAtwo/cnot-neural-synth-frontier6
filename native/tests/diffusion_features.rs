use cnot103_native::diffusion::{register_diffusion, DiffusionConfig};
use cnot103_native::{
    CnotAlphabet, CnotGate, FeatureId, FeaturePolicy, FeatureRegistry, FeatureRole, FeatureStatus,
    MatrixState,
};
use std::collections::{HashMap, HashSet};

fn diffusion_packet(
    n: u32,
    config: DiffusionConfig,
) -> (FeatureRegistry, cnot103_native::FeaturePacket) {
    let alphabet = CnotAlphabet::all_to_all_unit(n).unwrap();
    let state = MatrixState::identity(n, alphabet.id()).unwrap();
    let mut registry = FeatureRegistry::new();
    register_diffusion(&mut registry, config).unwrap();
    let policy = FeaturePolicy::complete_103_augmentation_only("diffusion-test-v1");
    let packet = registry.build(&[state], &policy).unwrap();
    (registry, packet)
}

fn config(max_n: u32) -> DiffusionConfig {
    DiffusionConfig {
        taus: [0.0, 0.25, 1.0, 4.0],
        floor: 1e-30,
        max_n,
        max_pair_states: 10_000,
        mc_samples: 256,
        mc_tau: 1.0,
    }
}

fn mc_policy(seed: u64) -> FeaturePolicy {
    let mut policy = FeaturePolicy::complete_103_augmentation_only("diffusion-mc-test-v1");
    policy.enabled_ids = (7..=10)
        .map(|ordinal| FeatureId::new(b'G', ordinal).unwrap())
        .collect();
    policy.seed = seed;
    policy.symmetry_contract = "augmentation_only".into();
    policy
}

fn assert_values_close(actual: &[f32], expected: &[f64], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected as f32).abs() <= tolerance,
            "value {index}: got {actual}, expected {expected}"
        );
    }
}

fn assert_interleaved_channel(
    actual: &[f32],
    channels: usize,
    channel: usize,
    expected: &[f64],
    tolerance: f32,
) {
    assert_eq!(actual.len(), expected.len() * channels);
    let selected = (0..expected.len())
        .map(|index| actual[index * channels + channel])
        .collect::<Vec<_>>();
    assert_values_close(&selected, expected, tolerance);
}

#[test]
fn two_wire_diffusion_profiles_match_the_closed_form_column_chain() {
    let (_registry, packet) = diffusion_packet(2, config(16));
    let g01 = packet.feature(FeatureId::new(b'G', 1).unwrap());
    let g02 = packet.feature(FeatureId::new(b'G', 2).unwrap());
    let g03 = packet.feature(FeatureId::new(b'G', 3).unwrap());
    let g04 = packet.feature(FeatureId::new(b'G', 4).unwrap());
    assert_eq!(g01.available, vec![1]);
    assert_eq!(g02.available, vec![1]);

    let raw = |tau: f64| {
        let t = 2.0 * tau;
        1.0 / 3.0 + 0.5 * (-0.5 * t).exp() + (1.0 / 6.0) * (-1.5 * t).exp()
    };
    assert_eq!(g01.values[0], 1.0);
    assert_eq!(g01.values[1], 0.0);
    assert!((g01.values[2] - 1.0 / 3.0).abs() < 1e-6);
    assert!((g01.values[3] - 3.0_f32.ln()).abs() < 1e-6);
    assert_eq!(g01.values[4], 0.0); // selected class is the initial class
    assert_eq!(g03.values[0], 0.0); // log mass at t=0
    assert_eq!(g03.values[1], 0.0); // exact-zero mask for the selected class

    for (time_index, tau) in config(16).taus.iter().copied().enumerate().skip(1) {
        assert!((g01.values[time_index * 7] - raw(tau) as f32).abs() < 2e-6);
    }
    for index in 0..3 {
        let expected = raw(config(16).taus[index + 1]) - raw(config(16).taus[index]);
        assert!((g04.values[index] - expected as f32).abs() < 2e-6);
    }
}

#[test]
fn transposed_profiles_use_row_weights_and_width_limits_are_visible() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut state = MatrixState::identity(3, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let mut registry = FeatureRegistry::new();
    register_diffusion(&mut registry, config(3)).unwrap();
    let policy = FeaturePolicy::complete_103_augmentation_only("diffusion-test-v1");
    let packet = registry.build(&[state], &policy).unwrap();
    let g01 = packet.feature(FeatureId::new(b'G', 1).unwrap());
    let g02 = packet.feature(FeatureId::new(b'G', 2).unwrap());
    assert!((g01.values[7] - 0.17683674).abs() < 2e-6);
    assert!((g02.values[7] - 0.79016411).abs() < 2e-6);
    assert_eq!(g01.values[4], 1.0); // noninitial selected class at t=0

    let (_registry, limited) = diffusion_packet(3, config(2));
    let row = limited.feature(FeatureId::new(b'G', 1).unwrap());
    assert_eq!(row.available, vec![0]);
    assert_eq!(row.status, vec![FeatureStatus::ResourceLimit]);
    assert!(row.reason[0].contains("max_n"));
}

#[test]
fn pair_diffusion_matches_the_source_profile_and_zeroes_diagonal() {
    let (_registry, packet) = diffusion_packet(3, config(16));
    let g06 = packet.feature(FeatureId::new(b'G', 6).unwrap());
    assert_eq!(g06.available, vec![1]);
    assert_eq!(g06.status, vec![FeatureStatus::Computed]);
    assert_eq!(g06.shape, vec![1, 3, 3, 28]);
    assert_eq!(g06.spec.role, FeatureRole::Pair);
    assert_eq!(g06.spec.channels, 28);
    assert_eq!(g06.spec.exactness, "floating_semigroup");
    assert_eq!(g06.spec.cost_tier, "T2");
    assert!(g06
        .spec
        .dependencies
        .contains(&"source_relative_rank2_pair_chain".to_owned()));

    let offset =
        |control: usize, target: usize, channel: usize| ((control * 3 + target) * 28) + channel;
    for wire in 0..3 {
        assert!(
            g06.values[offset(wire, wire, 0)..offset(wire, wire, 0) + 28]
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    // Independent SciPy expm_multiply values from research_v3/diffusion/pair.py
    // for n=3, initial histogram (1,1,1,0), and t=n*0.25=0.75.
    let profile = &g06.values[offset(0, 1, 0)..offset(0, 1, 0) + 28];
    assert_eq!(profile[0], 1.0); // raw class mass at t=0
    assert!((profile[1] - 6.0f32.ln()).abs() < 1e-6);
    assert!((profile[2] - (1.0 / 7.0)).abs() < 1e-6);
    assert!((profile[3] - 7.0f32.ln()).abs() < 1e-6);
    assert_eq!(profile[4], 0.0); // selected pair class is initial
    assert_eq!(profile[5], 0.0);
    assert_eq!(profile[6], 0.0);
    assert!((profile[7] - 0.6344311247711041).abs() < 2e-6);
    assert!((profile[10] - 1.4908836009354465).abs() < 2e-6);
    assert_eq!(
        &g06.values[offset(0, 1, 0)..offset(0, 1, 0) + 28],
        &g06.values[offset(1, 0, 0)..offset(1, 0, 0) + 28]
    );
}

#[test]
fn pair_diffusion_reports_unsupported_and_resource_limited_inputs() {
    let (_registry, singleton) = diffusion_packet(1, config(16));
    let g06 = singleton.feature(FeatureId::new(b'G', 6).unwrap());
    assert_eq!(g06.available, vec![0]);
    assert_eq!(g06.status, vec![FeatureStatus::Unsupported]);

    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let state = MatrixState::identity(3, alphabet.id()).unwrap();
    let mut limited_config = config(16);
    limited_config.max_pair_states = 9;
    let mut registry = FeatureRegistry::new();
    register_diffusion(&mut registry, limited_config).unwrap();
    let policy = FeaturePolicy::complete_103_augmentation_only("diffusion-test-v1");
    let packet = registry.build(&[state], &policy).unwrap();
    let limited = packet.feature(FeatureId::new(b'G', 6).unwrap());
    assert_eq!(limited.available, vec![0]);
    assert_eq!(limited.status, vec![FeatureStatus::ResourceLimit]);
    assert!(limited.reason[0].contains("10 states exceeds 9"));
}

#[test]
fn pair_diffusion_uses_each_matrix_column_pair_histogram() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut state = MatrixState::identity(3, alphabet.id()).unwrap();
    state
        .apply_cnot(CnotGate {
            control: 0,
            target: 1,
        })
        .unwrap();
    let mut registry = FeatureRegistry::new();
    register_diffusion(&mut registry, config(16)).unwrap();
    let policy = FeaturePolicy::complete_103_augmentation_only("diffusion-test-v1");
    let packet = registry.build(&[state], &policy).unwrap();
    let g06 = packet.feature(FeatureId::new(b'G', 6).unwrap());
    assert_eq!(g06.available, vec![1]);

    let offset =
        |control: usize, target: usize, channel: usize| ((control * 3 + target) * 28) + channel;
    // The CNOT gives columns 0 and 2 histogram (0,2,1,0). The source SciPy
    // reference for that class at t=0.75 is 0.07000506679912016.
    let changed_pair = &g06.values[offset(0, 2, 0)..offset(0, 2, 0) + 28];
    assert_eq!(changed_pair[0], 0.0);
    assert_eq!(changed_pair[4], 1.0); // exact zero at t=0
    assert_eq!(changed_pair[6], 1.0); // floor applied at t=0
    assert!((changed_pair[7] - 0.07000506679912016).abs() < 2e-6);
    assert!((changed_pair[8] - 3.0f32.ln()).abs() < 1e-6);
    assert!((changed_pair[9] - (1.0 / 14.0)).abs() < 1e-6);
    assert!((changed_pair[10] - (-0.0201303270924571)).abs() < 2e-6);
    assert_ne!(
        &g06.values[offset(0, 1, 0)..offset(0, 1, 0) + 28],
        changed_pair
    );
}

#[test]
fn rank_two_transition_counts_close_on_every_gl3_matrix_and_column_pair() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let mut invertible = Vec::new();
    for packed in 0u16..(1 << 9) {
        let rows = (0..3)
            .map(|row| u64::from((packed >> (row * 3)) & 0b111))
            .collect();
        if let Ok(state) = MatrixState::from_packed_rows(3, alphabet.id(), rows) {
            invertible.push(state);
        }
    }
    assert_eq!(invertible.len(), 168);

    let mut observed_classes = HashSet::new();
    for state in &invertible {
        for first in 0..3 {
            for second in 0..3 {
                if first == second {
                    continue;
                }
                let source = pair_histogram(state, first, second);
                observed_classes.insert(source);

                let mut expected = HashMap::<[usize; 4], usize>::new();
                for control_pattern in 0..4 {
                    for target_pattern in 0..4 {
                        if source[control_pattern] == 0
                            || (control_pattern == target_pattern && source[target_pattern] < 2)
                        {
                            continue;
                        }
                        let ways = source[control_pattern]
                            * (source[target_pattern]
                                - usize::from(control_pattern == target_pattern));
                        if ways == 0 {
                            continue;
                        }
                        let mut destination = source;
                        destination[target_pattern] -= 1;
                        destination[control_pattern ^ target_pattern] += 1;
                        *expected.entry(destination).or_default() += ways;
                    }
                }

                let mut observed = HashMap::<[usize; 4], usize>::new();
                for control in 0..3 {
                    for target in 0..3 {
                        if control == target {
                            continue;
                        }
                        let mut next = state.clone();
                        next.apply_cnot(CnotGate { control, target }).unwrap();
                        *observed
                            .entry(pair_histogram(&next, first, second))
                            .or_default() += 1;
                    }
                }
                assert_eq!(observed, expected, "pair histogram {source:?}");
                assert_eq!(expected.values().sum::<usize>(), 3 * 2);
            }
        }
    }
    assert_eq!(observed_classes.len(), 10);
}

#[test]
fn shared_word_monte_carlo_features_match_numpy_2_2_6_source_golden() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let state = MatrixState::from_packed_rows(3, alphabet.id(), vec![1, 3, 7]).unwrap();
    let mut monte_carlo_config = config(16);
    monte_carlo_config.mc_samples = 128;
    monte_carlo_config.mc_tau = 1.0;
    let mut registry = FeatureRegistry::new();
    register_diffusion(&mut registry, monte_carlo_config).unwrap();
    let policy = mc_policy(91);
    let packet = registry.build(&[state.clone()], &policy).unwrap();

    let g07 = packet.feature(FeatureId::new(b'G', 7).unwrap());
    assert_eq!(g07.shape, vec![1, 3, 1]);
    assert_eq!(g07.available, vec![1]);
    assert_eq!(g07.spec.exactness, "monte_carlo");
    assert_eq!(g07.spec.cost_tier, "T2");
    assert_values_close(&g07.values, &[1.8984375, 1.890625, 1.515625], 1e-6);

    let g08 = packet.feature(FeatureId::new(b'G', 8).unwrap());
    assert_eq!(g08.shape, vec![1, 3, 3, 5]);
    let covariance_weight = [
        0.4699187992,
        -0.0584399606,
        -0.0653297244,
        -0.0584399606,
        0.4761318898,
        -0.0376476378,
        -0.0653297244,
        -0.0376476378,
        0.3934547244,
    ];
    let mean_intersection = [
        1.8984375, 1.1015625, 0.84375, 1.1015625, 1.890625, 0.8828125, 0.84375, 0.8828125, 1.515625,
    ];
    let mean_xor = [
        0.0, 1.5859375, 1.7265625, 1.5859375, 0.0, 1.640625, 1.7265625, 1.640625, 0.0,
    ];
    let covariance_weight_intersection = [
        0.4699187992,
        0.2544906496,
        0.2281003937,
        0.2789124016,
        0.4761318898,
        0.2469242126,
        0.1835629921,
        0.2183809055,
        0.3934547244,
    ];
    let covariance_weight_xor = [
        0.0,
        -0.0975024606,
        -0.0516117126,
        -0.140132874,
        0.0,
        -0.0553641732,
        -0.0390009843,
        -0.0809547244,
        0.0,
    ];
    assert_interleaved_channel(&g08.values, 5, 0, &covariance_weight, 2e-6);
    assert_interleaved_channel(&g08.values, 5, 1, &mean_intersection, 2e-6);
    assert_interleaved_channel(&g08.values, 5, 2, &mean_xor, 2e-6);
    assert_interleaved_channel(&g08.values, 5, 3, &covariance_weight_intersection, 2e-6);
    assert_interleaved_channel(&g08.values, 5, 4, &covariance_weight_xor, 2e-6);

    let g09 = packet.feature(FeatureId::new(b'G', 9).unwrap());
    assert_eq!(g09.shape, vec![1, 3, 3, 2]);
    let character_pair_mean = [
        1.0, -0.359375, -0.015625, -0.359375, 1.0, -0.21875, -0.015625, -0.21875, 1.0,
    ];
    let character_pair_covariance = [
        0.997802734375,
        -0.36083984375,
        -0.00390625,
        -0.36083984375,
        0.9990234375,
        -0.2109375,
        -0.00390625,
        -0.2109375,
        0.9375,
    ];
    assert_interleaved_channel(&g09.values, 2, 0, &character_pair_mean, 1e-6);
    assert_interleaved_channel(&g09.values, 2, 1, &character_pair_covariance, 1e-6);

    let g10 = packet.feature(FeatureId::new(b'G', 10).unwrap());
    assert_eq!(g10.shape, vec![1, 3, 3, 7]);
    let se_intersection = [
        0.0605907635,
        0.0595668458,
        0.0513564767,
        0.0595668458,
        0.0609900024,
        0.0550378960,
        0.0513564767,
        0.0550378960,
        0.0554424479,
    ];
    let se_xor = [
        0.0,
        0.0632604755,
        0.0583934816,
        0.0632604755,
        0.0,
        0.0614922445,
        0.0583934816,
        0.0614922445,
        0.0,
    ];
    let se_character_pair = [
        0.0,
        0.0828075089,
        0.0887248183,
        0.0828075089,
        0.0,
        0.0865865566,
        0.0887248183,
        0.0865865566,
        0.0,
    ];
    assert_interleaved_channel(&g10.values, 7, 0, &se_intersection, 2e-6);
    assert_interleaved_channel(&g10.values, 7, 1, &se_xor, 2e-6);
    assert_interleaved_channel(&g10.values, 7, 2, &se_character_pair, 2e-6);
    for pair in 0..9 {
        let base = pair * 7;
        let expected_stationary = if pair / 3 == pair % 3 {
            1.0
        } else {
            -1.0 / 7.0
        };
        assert!(
            (g10.values[base + 3] - (character_pair_mean[pair] - expected_stationary).abs() as f32)
                .abs()
                < 1e-6
        );
        assert!((g10.values[base + 4] - 128.0).abs() < 1e-6);
        assert!((g10.values[base + 5] - (359.0 / 128.0)).abs() < 1e-6);
        assert!((g10.values[base + 6] - 1.0).abs() < 1e-6);
    }

    let mut exact_policy = policy;
    exact_policy.symmetry_contract = "exact_relabel".into();
    assert!(registry.build(&[state], &exact_policy).is_err());
}

#[test]
fn monte_carlo_diffusion_exposes_configured_resource_limit() {
    let alphabet = CnotAlphabet::all_to_all_unit(3).unwrap();
    let state = MatrixState::identity(3, alphabet.id()).unwrap();
    let mut limited_config = config(2);
    limited_config.mc_samples = 8;
    let mut registry = FeatureRegistry::new();
    register_diffusion(&mut registry, limited_config).unwrap();
    let packet = registry.build(&[state], &mc_policy(91)).unwrap();
    let g07 = packet.feature(FeatureId::new(b'G', 7).unwrap());
    assert_eq!(g07.available, vec![0]);
    assert_eq!(g07.status, vec![FeatureStatus::ResourceLimit]);
    assert!(g07.reason[0].contains("max_n"));
}

fn pair_histogram(state: &MatrixState, first: usize, second: usize) -> [usize; 4] {
    let mut counts = [0usize; 4];
    for row in 0..state.dimension() as usize {
        let bits = state.packed_rows()[row];
        let pattern = (((bits >> first) & 1) | (((bits >> second) & 1) << 1)) as usize;
        counts[pattern] += 1;
    }
    counts
}
