use super::*;

pub(super) fn selection(ordinals: &[u8], seed: u64) -> FeaturePolicy {
    let mut p = policy(ordinals);
    p.symmetry_contract = "augmentation_only".into();
    p.seed = seed;
    p
}

#[test]
fn seeded_words_and_all_moments_match_rust_across_generators_and_packed_widths() {
    for (n, samples, tau, seed) in [
        (2, 17, 0., 0),
        (2, 13, 4.9, 91),
        (3, 33, 10. / 3., u64::MAX),
        (4, 129, 15. / 4., 0x1234567800000000),
        (65, 13, 1., 91),
        (144, 7, 0.75, u64::MAX),
    ] {
        let config = DiffusionConfig {
            mc_samples: samples,
            mc_tau: tau,
            ..Default::default()
        };
        let registry = registry(config);
        let selected = selection(&[7, 8, 9, 10], seed);
        let plan = bind(&registry, &selected, n, config).unwrap();
        let states = class_cover(n);
        compare(&registry, &plan, &selected, &states);
        let packet = plan.build(&device_states(&states[1..2])).unwrap();
        let output = packet.feature(FeatureId::new(b'G', 10).unwrap()).unwrap();
        let _ = output.values().shallow_clone().fill_(-300.);
        let _ = output.status().shallow_clone().fill_(4);
        compare(&registry, &plan, &selected, &states[1..2]);
    }
}

#[test]
fn monte_carlo_limits_match_at_the_exact_256_mib_boundary() {
    let n = 64;
    let states = [MatrixState::identity(n, "column-diffusion-test").unwrap()];
    let selected = selection(&[7, 8, 9, 10], 0);
    for (samples, width, tau, code) in [
        (1024, 64, 0., 1u8),
        (1025, 64, 0., 4),
        (2, 63, 1., 4),
        (2, 64, f64::MAX, 4),
        (2, 64, 1e25, 4),
        (usize::MAX, 64, 1., 4),
    ] {
        let config = DiffusionConfig {
            max_n: width,
            mc_samples: samples,
            mc_tau: tau,
            ..Default::default()
        };
        let registry = registry(config);
        let plan = bind(&registry, &selected, n, config).unwrap();
        compare(&registry, &plan, &selected, &states);
        let result = plan.build(&device_states(&states)).unwrap();
        for &id in &selected.enabled_ids {
            assert_eq!(
                Vec::<u8>::try_from(result.feature(id).unwrap().status().to_device(Device::Cpu))
                    .unwrap(),
                [code]
            );
        }
    }
}

#[test]
fn every_selected_subset_and_its_configuration_are_bound_separately() {
    let states = class_cover(3);
    let config = DiffusionConfig {
        mc_samples: 11,
        mc_tau: 5.,
        ..Default::default()
    };
    let registry = registry(config);
    for mask in 1..16 {
        let ids: Vec<_> = (7..=10).filter(|i| mask & (1 << (i - 7)) != 0).collect();
        let p = selection(&ids, 91);
        let plan = bind(&registry, &p, 3, config).unwrap();
        compare(&registry, &plan, &p, &states);
    }
    let p = selection(&[7, 8, 9, 10], 91);
    assert!(bind(&registry, &policy(&[7]), 3, config).is_err());
    for wrong in [
        DiffusionConfig { max_n: 1, ..config },
        DiffusionConfig {
            mc_tau: 0.,
            ..config
        },
        DiffusionConfig {
            mc_samples: 5,
            ..config
        },
    ] {
        assert!(bind(&registry, &p, 3, wrong)
            .err()
            .unwrap()
            .contains("metadata"));
    }
    let unrelated = DiffusionConfig {
        taus: [0.; 4],
        floor: 0.5,
        max_pair_states: 1,
        ..config
    };
    let plan = bind(&registry, &p, 3, unrelated).unwrap();
    compare(&registry, &plan, &p, &states);
}
