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

    def test_manifest_or_source_identity_mismatch_never_starts_a_second_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                controller.claim_run_lock(
                    root, self.RUN_KEY, self.SOURCE_SHA, self.MANIFEST_SHA
                ),
                "reserved",
            )
            mismatch = controller.claim_run_lock(
                root, self.RUN_KEY, "c" * 40, self.MANIFEST_SHA
            )
            self.assertEqual(mismatch, "identity_mismatch")
            self.assertFalse(controller.may_launch_run(mismatch))

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


if __name__ == "__main__":
    unittest.main()
