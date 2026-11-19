use super::*;

#[test]
fn exact_pair_state_budget_is_independent_of_the_column_width_limit() {
    let selected = policy(&[1, 6]);
    let states = class_cover(7);
    let count = 10 * 9 * 8 / 6 - 1 - 3 * 7;
    for budget in [count - 1, count] {
        let config = DiffusionConfig {
            taus: [0., 0.001, 0.5, 3.],
            floor: 0.125,
            max_n: 1,
            max_pair_states: budget,
            ..Default::default()
        };
        let registry = registry(config);
        let plan = bind(&registry, &selected, 7, config).unwrap();
        compare(&registry, &plan, &selected, &states);
        let result = plan.build(&device_states(&states)).unwrap();
        for (ordinal, code) in [(1, 4u8), (6, if budget == count { 1 } else { 4 })] {
            let entry = result
                .feature(FeatureId::new(b'G', ordinal).unwrap())
                .unwrap();
            assert_eq!(
                Vec::<u8>::try_from(entry.status().to_device(Device::Cpu)).unwrap(),
                vec![code; states.len()]
            );
        }
    }
}

#[test]
fn computed_g06_crosses_a_packed_word_and_preserves_zero_diagonals() {
    let n = 65;
    let count = 68 * 67 * 66 / 6 - 1 - 3 * 65;
    let config = DiffusionConfig {
        taus: [0.; 4],
        floor: 1e-25,
        max_n: 2,
        max_pair_states: count,
        ..Default::default()
    };
    let registry = registry(config);
    let selected = policy(&[6]);
    let plan = bind(&registry, &selected, n, config).unwrap();
    let states = class_cover(n);
    compare(&registry, &plan, &selected, &states);
    let result = plan.build(&device_states(&states[1..2])).unwrap();
    let entry = result.feature(FeatureId::new(b'G', 6).unwrap()).unwrap();
    assert_eq!(
        Vec::<u8>::try_from(entry.status().to_device(Device::Cpu)).unwrap(),
        [1]
    );
    let payload = values(entry.values());
    for wire in 0..n as usize {
        let start = (wire * n as usize + wire) * 28;
        assert_eq!(payload[start..start + 28], [0.; 28]);
    }
    let _ = entry.values().shallow_clone().fill_(100.);
    compare(&registry, &plan, &selected, &states[1..2]);
}

#[test]
fn pair_numeric_failure_and_selected_zero_floor_follow_source() {
    let states = class_cover(4);
    let selected = policy(&[6]);
    let config = DiffusionConfig {
        taus: [0., 0., 1., 1e100],
        ..Default::default()
    };
    let limited = registry(config);
    let plan = bind(&limited, &selected, 4, config).unwrap();
    compare(&limited, &plan, &selected, &states);
    let zero = DiffusionConfig {
        taus: [0., 0.25, 1., 4.],
        floor: 0.,
        ..Default::default()
    };
    let registry = registry(zero);
    let plan = bind(&registry, &selected, 4, zero).unwrap();
    let identity = MatrixState::identity(4, "column-diffusion-test").unwrap();
    compare(&registry, &plan, &selected, &[identity]);
    assert!(registry
        .build(&states, &selected)
        .unwrap_err()
        .0
        .contains("non-finite"));
    let id = FeatureId::new(b'G', 6).unwrap();
    let mut augmented = selected;
    augmented.symmetry_contract = "augmentation_only".into();
    let legacy = bind(&registry, &augmented, 4, zero).unwrap();
    let same = legacy
        .build(&device_states(&[MatrixState::identity(
            4,
            "column-diffusion-test",
        )
        .unwrap()]))
        .unwrap();
    assert_eq!(
        same.feature(id).unwrap().spec().exactness,
        "floating_semigroup"
    );
}
