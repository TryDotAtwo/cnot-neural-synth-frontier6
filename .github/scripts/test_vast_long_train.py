import tempfile
import unittest
from pathlib import Path

import vast_long_train as controller


class BudgetTests(unittest.TestCase):
    def test_budget_allows_spend_plus_remaining_hours_and_reserve_under_cap(self):
        self.assertTrue(controller.budget_allows(5.0, 0.25, 52, 2.0))
        self.assertFalse(controller.budget_allows(5.0, 0.26, 52, 2.0))

    def test_budget_rejects_nonfinite_negative_boolean_and_string_values(self):
        invalid = [
            float("nan"),
            float("inf"),
            -0.01,
            True,
            "1.0",
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.assertFalse(controller.budget_allows(value, 0.1, 10, 1.0))
                self.assertFalse(controller.budget_allows(0.0, value, 10, 1.0))
                self.assertFalse(controller.budget_allows(0.0, 0.1, value, 1.0))
                self.assertFalse(controller.budget_allows(0.0, 0.1, 10, value))


class ReportingTests(unittest.TestCase):
    def test_run_state_accepts_only_the_five_allowlisted_states(self):
        allowed = ("not_started", "running", "succeeded", "failed", "unknown")
        for state in allowed:
            with self.subTest(state=state):
                self.assertEqual(controller.classify_run_state({"state": state}), state)
        self.assertEqual(controller.classify_run_state({"state": "training_with_metrics"}), "unknown")
        self.assertEqual(controller.classify_run_state(None), "unknown")

    def test_redacted_report_contains_only_time_and_state(self):
        report = controller.redacted_report("running", "2026-11-20T00:00:00Z")
        self.assertEqual(
            report,
            "checked_at_utc=2026-11-20T00:00:00Z\ntraining_status=running",
        )
        self.assertNotIn("loss", report)
        self.assertNotIn("sha", report)
        self.assertNotIn("quality", report)



class TrainingContractTests(unittest.TestCase):
    EXPECTED_MODEL = {
        "wire_dim": 744,
        "pair_dim": 186,
        "structural_blocks": 8,
        "attention_heads": 8,
        "feedforward_dim": 1488,
        "fusion_blocks": 2,
        "basis_proposal_hidden_dim": 256,
    }

    def test_full_model_contract_and_parameter_count_are_pinned(self):
        self.assertEqual(controller.ORIGINAL_MODEL_CONFIG, self.EXPECTED_MODEL)
        self.assertEqual(controller.ORIGINAL_PARAMETER_COUNT, 136_421_807)
        self.assertTrue(controller.is_original_model(self.EXPECTED_MODEL, 136_421_807))
        smaller = dict(self.EXPECTED_MODEL)
        smaller["structural_blocks"] = 2
        self.assertFalse(controller.is_original_model(smaller, 136_421_807))
        self.assertFalse(controller.is_original_model(self.EXPECTED_MODEL, 136_421_806))

    def test_one_hundred_epochs_resolve_to_exact_attempts_for_batch_four(self):
        self.assertEqual(controller.TRAIN_ROWS_PER_EPOCH, 10_000)
        self.assertEqual(controller.TARGET_EPOCHS, 100)
        self.assertEqual(controller.TRAIN_BATCH_SIZE, 4)
        self.assertEqual(controller.BELLMAN_SAMPLED_NEIGHBORS, 4)
        self.assertEqual(controller.required_attempts_for_epochs(10_000, 4, 100), 250_000)

    def test_epoch_attempt_count_rejects_invalid_or_fractional_schedules(self):
        for values in ((10_000, 0, 100), (10_000, 6, 100), (True, 4, 100), (-1, 4, 100)):
            with self.subTest(values=values):
                self.assertIsNone(controller.required_attempts_for_epochs(*values))


class TargetSelectionTests(unittest.TestCase):
    LABEL = "cnot103-longtrain-epoch100-20261120"

    @staticmethod
    def target(**updates):
        row = {
            "id": 60512345,
            "label": TargetSelectionTests.LABEL,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "actual_status": "running",
        }
        row.update(updates)
        return row

    def test_selects_only_one_exact_run_owned_2x3060(self):
        expected = self.target()
        rows = [expected, self.target(id=60512346, label="other-project")]
        self.assertEqual(controller.classify_target_inventory(rows, self.LABEL), "owned")
        self.assertEqual(controller.select_owned_instance(rows, self.LABEL), expected)

    def test_empty_inventory_is_distinct_from_ambiguous_inventory(self):
        self.assertEqual(controller.classify_target_inventory([], self.LABEL), "absent")
        duplicate = [self.target(), self.target(id=60512346)]
        self.assertEqual(controller.classify_target_inventory(duplicate, self.LABEL), "ambiguous")
        self.assertIsNone(controller.select_owned_instance(duplicate, self.LABEL))

    def test_wrong_gpu_or_malformed_inventory_never_selects_a_target(self):
        wrong_gpu = [self.target(gpu_name="RTX 3060 Ti")]
        wrong_count = [self.target(num_gpus=1)]
        malformed = [self.target(), "not-an-instance-record"]
        self.assertEqual(controller.classify_target_inventory(wrong_gpu, self.LABEL), "mismatch")
        self.assertEqual(controller.classify_target_inventory(wrong_count, self.LABEL), "mismatch")
        self.assertEqual(controller.classify_target_inventory(malformed, self.LABEL), "unknown")
        for rows in (wrong_gpu, wrong_count, malformed):
            self.assertIsNone(controller.select_owned_instance(rows, self.LABEL))

    def test_invalid_run_label_cannot_match_or_become_a_path(self):
        for label in ("../other", "C:/temp", ""):
            with self.subTest(label=label):
                self.assertEqual(controller.classify_target_inventory([], label), "unknown")
                with self.assertRaises(ValueError):
                    controller.run_lock_path(label, Path("root"))


class RunLockTests(unittest.TestCase):
    RUN_KEY = "epoch100-20261120"
    SOURCE_SHA = "a" * 40
    MANIFEST_SHA = "b" * 64

    def test_run_lock_path_is_scoped_under_the_supplied_remote_root(self):
        root = Path("remote-runs")
        self.assertEqual(
            controller.run_lock_path(self.RUN_KEY, root),
            root / self.RUN_KEY / "run.lock",
        )

    def test_run_lock_claim_is_idempotent_for_the_same_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = controller.claim_run_lock(
                root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
            )
            retry = controller.claim_run_lock(
                root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
            )
            self.assertEqual(first, "reserved")
            self.assertEqual(retry, "already_reserved")
            self.assertFalse(controller.may_launch_run(retry))

    def test_concurrent_run_lock_claims_allow_at_most_one_launcher(self):
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ThreadPoolExecutor(max_workers=8) as pool:
                statuses = list(pool.map(
                    lambda _: controller.claim_run_lock(
                        root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
                    ),
                    range(8),
                ))
            self.assertEqual(sum(value == "reserved" for value in statuses), 1)
            self.assertEqual(sum(controller.may_launch_run(value) for value in statuses), 1)

    def test_manifest_or_source_identity_mismatch_never_starts_a_second_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                controller.claim_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
                ),
                "reserved",
            )
            mismatches = (
                controller.claim_run_lock(
                    root, self.RUN_KEY, "c" * 40, self.MANIFEST_SHA
                ),
                controller.claim_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, "d" * 64
                ),
            )
            self.assertEqual(mismatches, ("identity_mismatch", "identity_mismatch"))
            self.assertFalse(any(controller.may_launch_run(value) for value in mismatches))

    def test_corrupt_existing_lock_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = controller.run_lock_path(self.RUN_KEY, root)
            path.parent.mkdir(parents=True)
            path.write_text("not-json", encoding="utf-8")
            status = controller.claim_run_lock(
                root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
            )
            self.assertEqual(status, "unknown")
            self.assertFalse(controller.may_launch_run(status))


    def test_run_lock_state_transitions_keep_retries_read_only_and_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                controller.claim_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
                ),
                "reserved",
            )
            self.assertEqual(
                controller.transition_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA, "running"
                ),
                "updated",
            )
            self.assertEqual(
                controller.claim_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
                ),
                "already_running",
            )
            self.assertEqual(
                controller.transition_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA, "succeeded"
                ),
                "updated",
            )
            self.assertEqual(
                controller.claim_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
                ),
                "already_succeeded",
            )
            self.assertEqual(
                controller.transition_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA, "failed"
                ),
                "invalid_transition",
            )

    def test_run_lock_transition_rejects_manifest_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller.claim_run_lock(
                root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
            )
            self.assertEqual(
                controller.transition_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, "d" * 64, "running"
                ),
                "identity_mismatch",
            )


if __name__ == "__main__":
    unittest.main()
