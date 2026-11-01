use cnot103_native::{Progress, Sampler};

#[test]
fn sampler_matches_cpython_311_random_shuffle_across_epochs() {
    let mut sampler = Sampler::new(7, 42).unwrap();
    assert_eq!(sampler.next(5).unwrap(), vec![1, 3, 4, 2, 6]);
    assert_eq!(sampler.next(5).unwrap(), vec![0, 5, 3, 6, 2]);
    assert_eq!(sampler.next(5).unwrap(), vec![4, 0, 5, 1, 3]);
    assert_eq!(sampler.next(5).unwrap(), vec![2, 4, 1, 5, 6]);

    let state = sampler.snapshot();
    assert_eq!(state.epochs, 3);
    assert_eq!(state.cursor, 6);
}

#[test]
fn sampler_snapshot_restores_the_exact_next_python_batch() {
    let mut first = Sampler::new(7, 42).unwrap();
    assert_eq!(first.next(5).unwrap(), vec![1, 3, 4, 2, 6]);
    let snapshot = first.snapshot();

    let mut resumed = Sampler::new(7, 999).unwrap();
    resumed.restore(&snapshot).unwrap();
    assert_eq!(resumed.next(5).unwrap(), vec![0, 5, 3, 6, 2]);
    assert_eq!(resumed.snapshot(), {
        let mut uninterrupted = first;
        uninterrupted.next(5).unwrap();
        uninterrupted.snapshot()
    });
}

#[test]
fn sampler_rejects_invalid_resume_state_without_mutating_it() {
    let mut sampler = Sampler::new(5, 0).unwrap();
    let initial = sampler.snapshot();
    let mut invalid = initial.clone();
    invalid.order = vec![0, 1, 1, 3, 4];
    invalid.cursor = 2;
    assert!(sampler.restore(&invalid).is_err());
    assert_eq!(sampler.snapshot(), initial);

    let mut invalid_rng = initial.clone();
    invalid_rng.rng.index = 625;
    assert!(sampler.restore(&invalid_rng).is_err());

    let mut invalid_empty_progress = initial.clone();
    invalid_empty_progress.epochs = 1;
    assert!(sampler.restore(&invalid_empty_progress).is_err());
    assert_eq!(sampler.snapshot(), initial);
}

#[test]
fn sampler_matches_cpython_for_zero_large_and_max_u64_seeds() {
    let cases = [
        (5, 0, 7, vec![2, 1, 0, 4, 3, 0, 2]),
        (12, 999, 11, vec![5, 6, 4, 0, 2, 11, 3, 7, 8, 9, 1]),
        (9, u64::MAX, 8, vec![5, 8, 6, 7, 1, 4, 2, 3]),
    ];
    for (size, seed, count, expected) in cases {
        assert_eq!(
            Sampler::new(size, seed).unwrap().next(count).unwrap(),
            expected
        );
    }
}

#[test]
fn progress_keeps_attempt_accept_reject_counts_consistent() {
    let mut progress = Progress::default();
    progress.record_attempt(true).unwrap();
    progress.record_attempt(false).unwrap();
    progress.record_attempt(true).unwrap();
    assert_eq!(progress.attempted, 3);
    assert_eq!(progress.accepted, 2);
    assert_eq!(progress.rejected, 1);
    assert!(progress.validate());
}

#[test]
fn progress_rejects_overflow_and_inconsistent_counters_without_mutating() {
    let mut overflow = Progress {
        attempted: u64::MAX,
        accepted: u64::MAX,
        rejected: 0,
    };
    let before = overflow;
    assert!(overflow.record_attempt(true).is_err());
    assert_eq!(overflow, before);

    let mut inconsistent = Progress {
        attempted: 3,
        accepted: 1,
        rejected: 1,
    };
    let before = inconsistent;
    assert!(inconsistent.record_attempt(false).is_err());
    assert_eq!(inconsistent, before);
}
