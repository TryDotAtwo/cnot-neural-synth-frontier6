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
        row = {
            "id": 123,
            "label": "cnot103-test",
            "actual_status": "running",
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
        }

        def fake_clock():
            events.append("clock")
            return 42.0

        def fake_start_target(*_args):
            events.append("activate")
            return row, "cnot103-test", False, 1.0

        def fake_attach(_row, _run_id, _attempt, _script, _timeout):
            return subprocess.CompletedProcess([], 0, "full_training=launched\n", ""), "detached", 123

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

    def test_loading_to_running_is_read_only_and_keeps_the_stable_label(self):
        old_label = "cnot103-r10k-44749514754-a1"
        loading = {
            "id": 123,
            "label": old_label,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "actual_status": "loading",
            "dph_total": 0.20,
            "storage_cost": 0.15,
        }
        running = {**loading, "actual_status": "running"}

        with (
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller.base, "instance_id", return_value=123),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller.base, "instance_rows", return_value=[running]),
            patch.object(controller.base, "api_instance_id") as activate,
            patch.object(controller.base, "offer_inventory") as offers,
            patch.object(controller.base, "create_once") as create,
        ):
            result = controller._start_target(
                [loading], {"resume_existing_label": old_label}, "123", "1"
            )

        self.assertEqual(result, (running, old_label, False, 1.0))
        activate.assert_not_called()
        offers.assert_not_called()
        create.assert_not_called()

    def test_loading_to_stopped_rechecks_budget_and_claims_start_atomically(self):
        old_label = "cnot103-r10k-44749514754-a1"
        run_label = "cnot103-fulltrain-123-a1"
        loading = {
            "id": 123,
            "label": old_label,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "actual_status": "loading",
            "dph_total": 0.20,
            "storage_cost": 0.15,
        }
        stopped = {**loading, "actual_status": "stopped"}
        running = {**stopped, "label": run_label, "actual_status": "running"}

        with (
            patch.object(controller.base, "project_charges", side_effect=[1.0, 1.0]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller.base, "instance_id", return_value=123),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(
                controller.base, "instance_rows",
                side_effect=[[stopped], [stopped], [running]],
            ),
            patch.object(controller, "full_run_budget_allows", return_value=True) as budget_gate,
            patch.object(controller.base, "api_instance_id", return_value=True) as activate,
            patch.object(controller.base, "offer_inventory") as offers,
            patch.object(controller.base, "create_once") as create,
        ):
            result = controller._start_target(
                [loading], {"resume_existing_label": old_label}, "123", "1"
            )

        self.assertEqual(result, (running, run_label, False, 1.0))
        self.assertEqual(activate.call_args.args, (123, "running", run_label))
        self.assertEqual(activate.call_count, 1)
        self.assertEqual(budget_gate.call_count, 2)
        offers.assert_not_called()
        create.assert_not_called()

    def test_wait_fails_closed_if_the_label_moves_to_a_different_instance_id(self):
        old_label = "cnot103-r10k-44749514754-a1"
        loading = {
            "id": 123,
            "label": old_label,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "actual_status": "loading",
            "dph_total": 0.20,
            "storage_cost": 0.15,
        }
        replacement = {**loading, "id": 124, "actual_status": "running"}

        with (
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller.base, "instance_id", side_effect=lambda row: row["id"]),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller.base, "instance_rows", return_value=[replacement]),
            patch.object(controller.base, "api_instance_id") as activate,
        ):
            with self.assertRaises(controller.GateError):
                controller._start_target(
                    [loading], {"resume_existing_label": old_label}, "123", "1"
                )

        activate.assert_not_called()

    def test_cleanup_does_not_stop_a_legacy_target_without_this_run_owner_stamp(self):
        old_label = "cnot103-r10k-44749514754-a1"
        row = {
            "id": 123,
            "label": old_label,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "actual_status": "running",
            "dph_total": 0.20,
            "storage_cost": 0.15,
        }
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "_marker", return_value={"resume_existing_label": old_label}),
            patch.object(controller.base, "instance_rows", return_value=[row]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "_read_remote_owner", return_value=False),
            patch.object(controller, "RemoteSession"),
            patch.object(controller.base, "stop_exact_label") as stop,
            patch.object(controller, "_summarize"),
        ):
            controller.cleanup()

        stop.assert_not_called()

    def test_cleanup_stops_run_claimed_target_without_a_late_step_output(self):
        run_label = "cnot103-fulltrain-123-a1"
        running = {
            "id": 123,
            "label": run_label,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "actual_status": "running",
            "dph_total": 0.20,
            "storage_cost": 0.15,
        }
        stopped = {**running, "actual_status": "stopped"}
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "_marker", return_value={"resume_existing_label": "cnot103-r10k-44749514754-a1"}),
            patch.object(controller.base, "instance_rows", side_effect=[[running], [stopped]]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "_read_remote_status", return_value=("completed", "passed", 100.0, 1.0)),
            patch.object(controller.base, "stop_exact_label", return_value=("verified", "detached")) as stop,
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "full_run_budget_allows", return_value=True),
            patch.object(controller, "_summarize"),
        ):
            self.assertEqual(controller.cleanup(), 0)

        stop.assert_called_once_with(run_label, "123", "1")


    def test_cleanup_stops_legacy_target_after_matching_run_owner_stamp(self):
        old_label = "cnot103-r10k-44749514754-a1"
        running = {
            "id": 123,
            "label": old_label,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "actual_status": "running",
            "dph_total": 0.20,
            "storage_cost": 0.15,
        }
        stopped = {**running, "actual_status": "stopped"}
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "_marker", return_value={"resume_existing_label": old_label}),
            patch.object(controller.base, "instance_rows", side_effect=[[running], [stopped]]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "RemoteSession") as session_factory,
            patch.object(controller, "_read_remote_owner", return_value=True),
            patch.object(controller, "_read_remote_status", return_value=("completed", "passed", 100.0, 1.0)),
            patch.object(controller.base, "stop_exact_label", return_value=("verified", "detached")) as stop,
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "full_run_budget_allows", return_value=True),
            patch.object(controller, "_summarize"),
        ):
            session_factory.return_value.close.return_value = "detached"
            self.assertEqual(controller.cleanup(), 0)

        stop.assert_called_once_with(old_label, "123", "1")

if __name__ == "__main__":
    unittest.main()
