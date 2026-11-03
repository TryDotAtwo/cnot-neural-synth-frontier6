use super::NumpyRng;

#[test]
fn pcg64_raw_stream_matches_numpy_2_2_6() {
    let mut rng = NumpyRng::new(91);
    assert_eq!(
        (0..5).map(|_| rng.next_u64()).collect::<Vec<_>>(),
        vec![
            5_221_960_716_401_725_447,
            16_791_166_746_794_900_714,
            15_113_692_621_735_265_246,
            5_396_149_944_436_547_962,
            15_400_337_095_808_520_742,
        ]
    );
}

#[test]
fn seed_sequence_matches_numpy_zero_and_max_u64_seeds() {
    let mut zero = NumpyRng::new(0);
    assert_eq!(
        (0..5).map(|_| zero.next_u64()).collect::<Vec<_>>(),
        vec![
            11_749_869_230_777_074_271,
            4_976_686_463_289_251_617,
            755_828_109_848_996_024,
            304_881_062_738_325_533,
            15_002_187_965_291_974_971,
        ]
    );
    let mut maximum = NumpyRng::new(u64::MAX);
    assert_eq!(
        (0..5).map(|_| maximum.next_u64()).collect::<Vec<_>>(),
        vec![
            12_544_278_110_101_001_871,
            15_593_249_672_699_323_225,
            136_562_751_618_339_402,
            16_501_869_284_920_798_641,
            2_378_988_675_683_841_820,
        ]
    );
}

#[test]
fn poisson_and_bounded_integer_streams_match_numpy_2_2_6() {
    let mut poisson = NumpyRng::new(91);
    assert_eq!(
        (0..8)
            .map(|_| poisson.poisson(3.0).unwrap())
            .collect::<Vec<_>>(),
        vec![5, 4, 4, 2, 2, 4, 3, 6]
    );

    let mut integers = NumpyRng::new(91);
    assert_eq!(
        (0..8)
            .map(|_| integers.integer(3).unwrap())
            .collect::<Vec<_>>(),
        vec![2, 0, 1, 2, 0, 2, 2, 0]
    );
}

#[test]
fn poisson_ptrs_matches_numpy_2_2_6() {
    let expected = [
        (10.0, vec![8, 13, 8, 9, 7, 14, 10, 13, 16, 9, 8, 10]),
        (15.0, vec![19, 14, 11, 20, 15, 19, 22, 14, 13, 18, 16, 9]),
        (50.0, vec![45, 57, 45, 48, 43, 60, 51, 57, 63, 48, 46, 56]),
        (
            256.0,
            vec![245, 273, 246, 251, 240, 278, 258, 271, 286, 252, 247, 270],
        ),
    ];
    for (lambda, expected) in expected {
        let mut rng = NumpyRng::new(91);
        let observed = (0..expected.len())
            .map(|_| rng.poisson(lambda).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(observed, expected, "lambda={lambda}");
    }
}

#[test]
fn singleton_integer_range_does_not_advance_numpy_stream() {
    let mut rng = NumpyRng::new(91);
    assert_eq!(rng.integer(1), Some(0));
    assert_eq!(rng.next_u64(), 5_221_960_716_401_725_447);
}
