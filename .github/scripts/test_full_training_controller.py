import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import vast_full_training_controller as controller
from vast_full_training_controller import full_run_budget_allows, _summarize


class FullTrainingBudgetTests(unittest.TestCase):
    def test_reserves_48_hours_and_seven_days_of_disk_storage(self):
        self.assertTrue(full_run_budget_allows(1.0, 0.30, 0.06))
        self.assertFalse(full_run_budget_allows(1.0, 0.38, 0.06))
        self.assertFalse(full_run_budget_allows(5.0, 0.30, 0.06))
        self.assertFalse(full_run_budget_allows(1.0, 0.30, 0.06, extra_storage_reserve=4.0))

    def test_rejects_invalid_financial_inputs_closed(self):
        for values in (
            (float("nan"), 0.30, 0.06),
            (-1.0, 0.30, 0.06),
            (1.0, float("inf"), 0.06),
            (1.0, 0.30, -0.01),
            (True, 0.30, 0.06),
        ):
            with self.subTest(values=values):
                self.assertFalse(full_run_budget_allows(*values))


    def test_summary_exports_status_and_cleanup_flag_to_github_step_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}):
                _summarize("2026-11-21T00:00:00Z", "accepted", "started", cleanup_needed=True)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "status=started\ncleanup_needed=yes\n",
            )


    def test_full_run_clock_starts_before_instance_activation(self):
        events = []
        row = {"id": 123}

        def fake_clock():
            events.append("clock")
            return 42.0

        def fake_start_target(*_args):
            events.append("activate")
            return row, "cnot103-test", False, 1.0

        def fake_attach(_row, _run_id, _attempt, _script, _timeout):
            return subprocess.CompletedProcess([], 0, "full_training=launched\\n", ""), "detached", 123

        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1"}),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "_pin_status", return_value=("b" * 40, "c" * 64)),
            patch.object(controller, "_marker", return_value={"resume_existing_label": "cnot103-test"}),
            patch.object(controller.base, "instance_rows", return_value=[row]),
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller.base, "instance_id", return_value=123),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "_start_target", side_effect=fake_start_target),
            patch.object(controller, "_validate_running_budget", return_value=True),
            patch.object(controller, "_launch_script", return_value="remote script"),
            patch.object(controller, "_attach_and_remote", side_effect=fake_attach),
            patch.object(controller, "_summarize"),
            patch.object(controller.time, "time", side_effect=fake_clock),
        ):
            self.assertEqual(controller.start(), 0)

        self.assertLess(events.index("clock"), events.index("activate"))


if __name__ == "__main__":
    unittest.main()
