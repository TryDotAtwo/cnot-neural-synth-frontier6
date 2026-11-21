import ast
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import vast_full_training_controller as controller
from vast_full_training_controller import full_run_budget_allows, _summarize

_TEST_TARGET_ENV = patch.dict(os.environ, {"CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"})


def setUpModule():
    _TEST_TARGET_ENV.start()


def tearDownModule():
    _TEST_TARGET_ENV.stop()


class FullTrainingBudgetTests(unittest.TestCase):
    def test_reserves_48_hours_and_seven_days_of_disk_storage(self):
        self.assertTrue(full_run_budget_allows(1.0, 0.30, 0.005))
        self.assertFalse(full_run_budget_allows(1.0, 0.38, 0.005))
        self.assertFalse(full_run_budget_allows(5.0, 0.30, 0.005))
        self.assertFalse(full_run_budget_allows(1.0, 0.30, 0.005, extra_storage_reserve=4.0))

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

    def test_bounded_pilot_budget_can_fit_when_full_run_projection_does_not(self):
        spent = 5.0
        rate = 0.30
        storage = 0.06

        self.assertFalse(full_run_budget_allows(spent, rate, storage))
        self.assertTrue(controller.bootstrap_budget_allows(spent, rate, storage))
        self.assertFalse(controller.bootstrap_budget_allows(19.0, rate, storage))


    def test_summary_exports_status_and_cleanup_flag_to_github_step_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}):
                _summarize("2026-11-21T00:00:00Z", "accepted", "started", cleanup_needed=True)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "status=started\ncleanup_needed=yes\n",
            )

    def test_summary_keeps_stop_verification_separate_from_budget_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}):
                controller._summarize(
                    "2026-11-21T00:00:00Z",
                    "budget_gate",
                    "budget_gate",
                    stop_verification="verified",
                )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "status=budget_gate\nstop_verification=verified\n",
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

        clock_reads = 0

        def fake_clock():
            nonlocal clock_reads
            events.append(f"clock{clock_reads}")
            clock_reads += 1
            return 42.0

        def fake_invoice():
            events.append("invoice")
            return 1.0

        def fake_start_target(*_args, **_kwargs):
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
            patch.object(controller.base, "project_charges", side_effect=fake_invoice),
            patch.object(controller.base, "instance_id", return_value=123),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "_start_target", side_effect=fake_start_target),
            patch.object(controller, "_has_active_supervisor", return_value=False),
            patch.object(controller, "_validate_bootstrap_budget", return_value=True),
            patch.object(controller, "_launch_script", return_value="remote script") as launch,
            patch.object(controller, "_attach_and_remote", side_effect=fake_attach),
            patch.object(controller, "_summarize"),
            patch.object(controller.time, "time", side_effect=fake_clock),
        ):
            self.assertEqual(controller.start(), 0)

        self.assertLess(events.index("clock0"), events.index("activate"))
        self.assertLess(events.index("activate"), events.index("clock1"))
        self.assertLess(events.index("clock1"), events.index("invoice"))
        self.assertEqual(launch.call_args.args[-1], 42.0)

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
            "disk_space": 64,
        }
        running = {**loading, "actual_status": "running"}

        with (
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller.base, "instance_id", return_value=123),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "_validate_bootstrap_budget", return_value=True),
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
            "disk_space": 64,
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
        self.assertEqual(activate.call_args.args, (123, "running"))
        self.assertEqual(activate.call_args.kwargs, {"label": run_label})
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
            "disk_space": 64,
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
            "disk_space": 64,
        }
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1", "CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "CLAIM_VISIBILITY_WAIT_SECONDS", 0),
            patch.object(controller, "_marker", return_value={"resume_existing_label": old_label}),
            patch.object(controller.base, "instance_rows", return_value=[row]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "_read_remote_owner", return_value=False),
            patch.object(controller, "RemoteSession"),
            patch.object(controller.base, "stop_exact_label") as stop,
            patch.object(controller, "_summarize"),
        ):
            self.assertEqual(controller.cleanup(), 0)

        stop.assert_not_called()

    def test_cleanup_is_successful_noop_when_no_run_target_exists(self):
        old_label = "cnot103-r10k-44749514754-a1"
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1", "CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "CLAIM_VISIBILITY_WAIT_SECONDS", 0),
            patch.object(controller, "_marker", return_value={"resume_existing_label": old_label, "approved_instance_id": 123}),
            patch.object(controller.base, "instance_rows", return_value=[]),
            patch.object(controller.base, "stop_exact_label") as stop,
            patch.object(controller, "_summarize") as summarize,
        ):
            self.assertEqual(controller.cleanup(), 0)

        stop.assert_not_called()
        summarize.assert_called_once()
        self.assertEqual(summarize.call_args.args[1:3], ("not_needed", "not_needed"))

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
            "disk_space": 64,
        }
        stopped = {**running, "actual_status": "stopped"}
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1", "CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "_marker", return_value={"resume_existing_label": "cnot103-r10k-44749514754-a1"}),
            patch.object(controller.base, "instance_rows", side_effect=[[running], [stopped]]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "_read_remote_status", return_value=("completed", "passed", None, 1.0, 1_000.0)),
            patch.object(controller.base, "stop_exact_label", return_value=("verified", "detached")) as stop,
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "full_run_budget_allows", return_value=True),
            patch.object(controller, "_summarize"),
        ):
            self.assertEqual(controller.cleanup(), 0)

        stop.assert_called_once_with(run_label, "123", "1", expected_id=123)


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
            "disk_space": 64,
        }
        stopped = {**running, "actual_status": "stopped"}
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1", "CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "CLAIM_VISIBILITY_WAIT_SECONDS", 0),
            patch.object(controller, "CLAIM_VISIBILITY_WAIT_SECONDS", 0),
            patch.object(controller, "_marker", return_value={"resume_existing_label": old_label, "approved_instance_id": 123}),
            patch.object(controller.base, "instance_rows", side_effect=[[running], [stopped]]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "RemoteSession") as session_factory,
            patch.object(controller, "_read_remote_owner", return_value=True),
            patch.object(controller, "_read_remote_status", return_value=("completed", "passed", None, 1.0, 1_000.0)),
            patch.object(controller.base, "stop_exact_label", return_value=("verified", "detached")) as stop,
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "full_run_budget_allows", return_value=True),
            patch.object(controller, "_summarize"),
        ):
            session_factory.return_value.close.return_value = "detached"
            self.assertEqual(controller.cleanup(), 0)

        stop.assert_called_once_with(old_label, "123", "1", expected_id=123)


class InterruptedStartCleanupTests(unittest.TestCase):
    def test_cleanup_recovers_accepted_start_after_job_is_cut_before_outputs(self):
        old_label = "cnot103-r10k-44749514754-a1"
        run_label = "cnot103-fulltrain-123-a1"
        old = {
            "id": 123, "label": old_label, "gpu_name": "RTX 3060", "num_gpus": 2,
            "actual_status": "stopped", "dph_total": 0.20, "storage_cost": 0.15,
            "disk_space": 64,
        }
        running = {**old, "label": run_label, "actual_status": "running"}
        stopped = {**running, "actual_status": "stopped"}

        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1", "CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "_marker", return_value={"resume_existing_label": old_label}),
            patch.object(controller.base, "instance_rows", side_effect=[[old], [running], [stopped]]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "_read_remote_status", return_value=("completed", "passed", None, 1.0, 1_000.0)),
            patch.object(controller.base, "stop_exact_label", return_value=("verified", "detached")) as stop,
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "full_run_budget_allows", return_value=True),
            patch.object(controller, "_summarize"),
            patch.object(controller.time, "monotonic", return_value=0.0),
            patch.object(controller.time, "sleep"),
        ):
            self.assertEqual(controller.cleanup(), 0)

        # The first inventory is stale (pre-PUT); the second observes the
        # atomic run-specific claim. No GITHUB_OUTPUT is present in this job.
        stop.assert_called_once_with(run_label, "123", "1", expected_id=123)


class RemoteStatusReaderTests(unittest.TestCase):
    def test_reader_returns_billing_epoch_and_separate_remaining_forecast(self):
        class Session:
            def run(self, command, timeout):
                return subprocess.CompletedProcess(
                    [], 0,
                    "run_state=baseline\n"
                    "quality_status=measured\n"
                    "started_at_epoch=100.0\n"
                    "budget_baseline=2.5\n"
                    "billing_baseline_epoch=200.0\n"
                    "remaining_work_seconds=3000.0\n"
                    "forecast_total_seconds=3500.0\n"
                    + "forecast_token=" + "a" * 32 + "\n",
                    "",
                )

        result = controller._read_remote_status_full(
            {}, "123", "1", "/workspace/tqp_full_training_123_aaaaaaaa_a1",
            session=Session(),
        )
        self.assertEqual(
            result,
            ("baseline", "measured", 100.0, 2.5, 200.0, 3500.0, 3000.0, "a" * 32),
        )


class ForecastAuthorizationWriterTests(unittest.TestCase):
    def test_authorization_is_written_and_acknowledged_on_the_pinned_session(self):
        class Session:
            def run(self, command, timeout):
                self.command = command
                self.timeout = timeout
                return subprocess.CompletedProcess(
                    [], 0, "full_training=forecast_authorized\n", ""
                )

        session = Session()
        result = controller._write_forecast_authorization(
            session,
            "/workspace/tqp_full_training_123_aaaaaaaa_a1",
            "a" * 32,
            3600.0,
            3500.0,
            1000.0,
        )
        self.assertTrue(result)
        self.assertEqual(session.timeout, 30)
        self.assertIn("forecast-approval.private.json", session.command)
        self.assertIn("authorized_at_epoch", session.command)
        body = session.command.partition("<<'PY'\n")[2].partition("\nPY\n")[0]
        ast.parse(body)

    def test_missing_remote_acknowledgement_is_not_authorization(self):
        class Session:
            def run(self, command, timeout):
                return subprocess.CompletedProcess([], 0, "other_status=ok\n", "")

        self.assertFalse(controller._write_forecast_authorization(
            Session(), "/workspace/tqp_full_training_123_aaaaaaaa_a1",
            "a" * 32, 3600.0, 3500.0, 1000.0,
        ))


class BillingProjectionTests(unittest.TestCase):
    def test_three_hour_loading_is_already_in_the_post_loading_invoice_snapshot(self):
        projected = controller.projected_billable_spend(
            invoice_now=2.0,
            invoice_baseline=2.0,
            billing_baseline_epoch=10_800.0,
            now_epoch=10_800.0,
            instance_rate=0.30,
            invoice_lag_buffer_seconds=0.0,
        )
        self.assertEqual(projected, 2.0)

    def test_newer_invoice_replaces_the_estimate_when_it_has_caught_up(self):
        projected = controller.projected_billable_spend(
            invoice_now=7.0,
            invoice_baseline=5.0,
            billing_baseline_epoch=1_000.0,
            now_epoch=4_600.0,
            instance_rate=0.20,
            invoice_lag_buffer_seconds=0.0,
        )
        self.assertEqual(projected, 7.0)

    def test_lagging_invoice_gets_rate_based_latency_reserve(self):
        projected = controller.projected_billable_spend(
            invoice_now=5.0,
            invoice_baseline=5.0,
            billing_baseline_epoch=1_000.0,
            now_epoch=11_800.0,
            instance_rate=0.20,
            invoice_lag_buffer_seconds=3_600.0,
        )
        self.assertAlmostEqual(projected, 5.8)

    def test_missing_or_future_invoice_snapshot_fails_closed(self):
        self.assertIsNone(controller.projected_billable_spend(1, 1, 2_000, 1_000, 0.2))
        self.assertIsNone(controller.projected_billable_spend(None, 1, 1_000, 2_000, 0.2))


class StorageReservationTests(unittest.TestCase):
    def test_reserves_each_owned_instance_at_its_reported_hourly_rate(self):
        rows = [
            {"id": 123, "label": "cnot103-target", "disk_space": 64, "storage_cost": 0.01},
            {"id": 456, "label": "cnot103-other", "disk_space": 256, "storage_cost": 0.03},
        ]
        with patch.object(controller.base, "instance_id", side_effect=lambda row: row["id"]):
            self.assertAlmostEqual(
                controller._other_storage_reserve(rows, 123),
                0.03 * controller.STORAGE_HOURS,
            )

    def test_missing_allocated_disk_size_fails_closed(self):
        rows = [
            {"id": 456, "label": "cnot103-other", "storage_cost": 0.03},
        ]
        with patch.object(controller.base, "instance_id", side_effect=lambda row: row["id"]):
            self.assertIsNone(controller._other_storage_reserve(rows, 123))


class NewRentalBudgetDiagnosisTests(unittest.TestCase):
    @staticmethod
    def offer(**updates):
        row = {
            "ask_id": 77,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "dph_total": 0.30,
            "storage_cost": 0.15,
            "_quote_allocated_storage_gb": 64,
            "_quote_min_duration_seconds": controller.base.MIN_OFFER_DURATION_SECONDS,
            "disk_space": 512,
        }
        row.update(updates)
        return row

    def test_new_rental_projection_covers_actual_64gb_disk_and_prior_spend(self):
        offer = self.offer()
        self.assertEqual(
            controller.classify_new_rental_budget([offer], [], spent=0.0),
            "within_cap",
        )
        self.assertEqual(
            controller.classify_new_rental_budget([offer], [], spent=4.0),
            "insufficient_remaining_budget",
        )
        expensive = self.offer(dph_total=0.40)
        self.assertEqual(
            controller.classify_new_rental_budget([expensive], [], spent=0.0),
            "run_projection_exceeds_cap",
        )

    def test_new_rental_budget_uses_bootstrap_window_separately(self):
        offer = self.offer(dph_total=0.50)
        self.assertEqual(
            controller.classify_new_rental_budget(
                [offer], [], spent=0.0, active_hours=controller.BOOTSTRAP_HOURS
            ),
            "within_cap",
        )
        self.assertEqual(
            controller.classify_new_rental_budget([offer], [], spent=0.0),
            "run_projection_exceeds_cap",
        )

    def test_new_rental_budget_fails_closed_on_unknown_or_wrong_disk_contract(self):
        self.assertEqual(
            controller.classify_new_rental_budget(
                [self.offer(storage_cost=None)], [], spent=0.0
            ),
            "inputs_unknown",
        )
        self.assertEqual(
            controller.classify_new_rental_budget(
                [self.offer(_quote_allocated_storage_gb=None)], [], spent=0.0
            ),
            "inputs_unknown",
        )
        self.assertEqual(
            controller.classify_new_rental_budget(
                [self.offer(_quote_allocated_storage_gb=32)], [], spent=0.0
            ),
            "inputs_unknown",
        )

    def test_offer_without_allocation_echo_uses_pinned_search_contract(self):
        offer = self.offer()
        self.assertNotIn("allocated_storage", offer)
        self.assertEqual(
            controller.classify_new_rental_budget([offer], [], spent=0.0),
            "within_cap",
        )

    def test_no_eligible_offer_has_its_own_status(self):
        self.assertEqual(
            controller.classify_new_rental_budget([], [], spent=0.0),
            "no_eligible_offer",
        )


    def test_full_run_exact_20_dollar_boundary_and_one_cent_over(self):
        offer = self.offer()
        storage_hourly = controller._offer_storage_hourly_cost(offer)
        self.assertIsNotNone(storage_hourly)
        projected = (
            offer["dph_total"] * (controller.ACTIVE_HOURS + controller.base.STOP_RESERVE_HOURS)
            + storage_hourly * controller.STORAGE_HOURS
            + controller.base.TRANSFER_RESERVE
        )
        exact_spend = controller.BUDGET_CAP - projected
        self.assertEqual(
            controller.classify_new_rental_budget([offer], [], spent=exact_spend),
            "within_cap",
        )
        self.assertEqual(
            controller.classify_new_rental_budget([offer], [], spent=exact_spend + 0.01),
            "insufficient_remaining_budget",
        )

    def test_invalid_allocated_storage_marker_makes_offer_price_unknown(self):
        for allocated in (None, 32, 64.5):
            with self.subTest(allocated=allocated):
                offer = self.offer(_quote_allocated_storage_gb=allocated)
                self.assertEqual(
                    controller.classify_new_rental_budget([offer], [], spent=0.0),
                    "inputs_unknown",
                )

    def test_quote_check_separates_negative_full_projection_from_pilot(self):
        import io
        from contextlib import redirect_stdout

        offer = self.offer(dph_total=0.50)
        output = io.StringIO()
        with (
            patch.object(controller.base, "instance_rows", return_value=[]),
            patch.object(controller.base, "project_charges", return_value=0.0),
            patch.object(controller.base, "offer_quote", return_value=(False, [offer])),
            patch.object(controller.base, "api_instance_id") as update,
            patch.object(controller.base, "create_once") as create,
            patch.object(controller.base, "stop_exact_label") as stop,
            redirect_stdout(output),
        ):
            result = controller.quote_check()

        self.assertEqual(result, 0)
        text = output.getvalue()
        self.assertIn("quote_outcome=negative_classification", text)
        self.assertIn("full_run_budget=run_projection_exceeds_cap", text)
        self.assertIn("bounded_pilot_budget=within_cap", text)
        update.assert_not_called()
        create.assert_not_called()
        stop.assert_not_called()

    def test_quote_check_distinguishes_empty_market_no_eligible_and_unknown(self):
        import io
        from contextlib import redirect_stdout

        for quote, expected in (
            ((True, []), "quote_outcome=empty_market"),
            ((False, []), "quote_outcome=negative_classification"),
        ):
            output = io.StringIO()
            with (
                patch.object(controller.base, "instance_rows", return_value=[]),
                patch.object(controller.base, "project_charges", return_value=0.0),
                patch.object(controller.base, "offer_quote", return_value=quote),
                redirect_stdout(output),
            ):
                self.assertEqual(controller.quote_check(), 0)
            self.assertIn(expected, output.getvalue())

        output = io.StringIO()
        with (
            patch.object(controller.base, "instance_rows", return_value=[]),
            patch.object(controller.base, "project_charges", return_value=0.0),
            patch.object(controller.base, "offer_quote", side_effect=controller.base.GateError("unknown")),
            redirect_stdout(output),
        ):
            self.assertEqual(controller.quote_check(), 1)
        self.assertIn("quote_outcome=inputs_unknown", output.getvalue())

        output = io.StringIO()
        with (
            patch.object(controller.base, "instance_rows", return_value=[]),
            patch.object(controller.base, "project_charges", return_value=0.0),
            patch.object(controller.base, "offer_quote", side_effect=controller.base.TransportFailure("transport")),
            redirect_stdout(output),
        ):
            self.assertEqual(controller.quote_check(), 1)
        self.assertIn("quote_outcome=transport_error", output.getvalue())

    def test_missing_target_budget_path_is_get_only_and_never_quotes(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(controller, "_marker", return_value={
                "resume_existing_label": "cnot103-r10k-123-a1",
            }),
            patch.object(controller.base, "instance_rows", return_value=[]),
            patch.object(controller.base, "project_charges") as charges,
            patch.object(controller.base, "offer_inventory") as offers,
            patch.object(controller.base, "offer_quote") as quote,
            patch.object(controller, "_summarize_budget") as summarize,
            patch.object(controller.base, "api_instance_id") as update,
            patch.object(controller.base, "create_once") as create,
            patch.object(controller.base, "stop_exact_label") as stop,
        ):
            self.assertEqual(controller.budget_check(), 0)

        charges.assert_not_called()
        offers.assert_not_called()
        quote.assert_not_called()
        self.assertEqual(
            summarize.call_args.args[1:],
            ("success", "configuration_missing", "configuration_missing"),
        )
        update.assert_not_called()
        create.assert_not_called()
        stop.assert_not_called()


class BudgetConfigurationDiagnosisTests(unittest.TestCase):
    def test_existing_owned_target_without_private_id_does_not_trigger_rental_path(self):
        target = {
            "id": 123,
            "label": "cnot103-r10k-123-a1",
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "disk_space": 64,
            "storage_cost": 0.005,
            "dph_total": 0.30,
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(controller, "_marker", return_value={
                "resume_existing_label": target["label"],
            }),
            patch.object(controller.base, "instance_rows", return_value=[target]),
            patch.object(controller.base, "project_charges", return_value=0.0),
            patch.object(controller.base, "offer_inventory") as offers,
            patch.object(controller, "_summarize_budget") as summarize,
        ):
            self.assertEqual(controller.budget_check(), 1)

        offers.assert_not_called()
        self.assertEqual(
            summarize.call_args.args[1:],
            ("failed", "configuration_missing", "configuration_missing"),
        )


class FullRunBudgetDiagnosisTests(unittest.TestCase):
    @staticmethod
    def target(**updates):
        row = {
            "id": 123,
            "label": "cnot103-r10k-44749514754-a1",
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "dph_total": 0.30,
            "storage_cost": 0.03,
            "disk_space": 64,
        }
        row.update(updates)
        return row

    def test_budget_diagnostic_uses_full_48_hour_run_projection(self):
        row = self.target()
        self.assertEqual(
            controller.classify_full_run_budget(row, [row], spent=0.0),
            "within_cap",
        )

    def test_budget_diagnostic_distinguishes_prior_spend_from_run_rate(self):
        row = self.target()
        self.assertEqual(
            controller.classify_full_run_budget(row, [row], spent=5.0),
            "insufficient_remaining_budget",
        )
        expensive = self.target(dph_total=0.50)
        self.assertEqual(
            controller.classify_full_run_budget(expensive, [expensive], spent=0.0),
            "run_projection_exceeds_cap",
        )

    def test_budget_diagnostic_fails_closed_when_cost_inputs_are_unknown(self):
        row = self.target(dph_total=None)
        self.assertEqual(
            controller.classify_full_run_budget(row, [row], spent=0.0),
            "inputs_unknown",
        )


    def test_budget_diagnostic_exposes_separate_bootstrap_and_full_run_gates(self):
        row = self.target()
        with (
            patch.object(controller, "_marker", return_value={
                "resume_existing_label": row["label"],
            }),
            patch.object(controller.base, "instance_rows", return_value=[row]),
            patch.object(controller.base, "project_charges", return_value=5.0),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "_summarize_budget") as summarize,
        ):
            self.assertEqual(controller.budget_check(), 0)

        self.assertEqual(
            summarize.call_args.args[2:],
            ("insufficient_remaining_budget", "within_cap"),
        )


    def test_budget_diagnostic_only_reads_and_never_mutates_vast(self):
        row = self.target()
        with (
            patch.object(controller, "_marker", return_value={
                "resume_existing_label": row["label"],
            }),
            patch.object(controller.base, "instance_rows", return_value=[row]),
            patch.object(controller.base, "project_charges", return_value=0.0),
            patch.object(controller, "_summarize_budget") as summarize,
            patch.object(controller.base, "api_instance_id") as update,
            patch.object(controller.base, "create_once") as create,
            patch.object(controller.base, "offer_inventory") as offers,
            patch.object(controller.base, "stop_exact_label") as stop,
        ):
            self.assertEqual(controller.budget_check(), 0)

        summarize.assert_called_once()
        update.assert_not_called()
        create.assert_not_called()
        offers.assert_not_called()
        stop.assert_not_called()

class ForecastMonitoringTests(unittest.TestCase):
    @staticmethod
    def _monitor_result(allow_budget: bool):
        row = {
            "id": 123,
            "label": "cnot103-fulltrain-123-a1",
            "actual_status": "running",
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "dph_total": 0.30,
            "storage_cost": 0.06,
            "disk_space": 64,
        }
        session = object()
        with (
            patch.object(
                controller,
                "_read_remote_status_full",
                return_value=("baseline", "measured", 900.0, 0.0, 900.0, 3600.0, 3500.0, "a" * 32),
            ),
            patch.object(controller.base, "instance_rows", side_effect=[[row], [{**row, "actual_status": "stopped"}]]),
            patch.object(controller.base, "resolve_label", return_value=row),
            patch.object(controller.base, "instance_id", side_effect=lambda value: value["id"]),
            patch.object(controller.base, "project_charges", return_value=0.0),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "full_run_budget_allows", return_value=allow_budget) as budget,
            patch.object(controller, "_write_forecast_authorization", return_value=True) as authorize,
            patch.object(controller.base, "stop_exact_label", return_value=("verified", "detached")) as stop,
            patch.object(controller.time, "monotonic", side_effect=[0.0, 2.0]),
            patch.object(controller.time, "time", return_value=1000.0),
        ):
            result = controller._monitor_window(
                row=row,
                label=row["label"],
                ident=123,
                run_id="123",
                attempt="1",
                root="/workspace/tqp_full_training_123_aaaaaaaa_a1",
                window_seconds=1,
                session=session,
            )

        return result, budget, authorize, stop, row, session

    def test_monitor_authorizes_forecast_before_training_can_continue(self):
        result, budget, authorize, stop, row, session = self._monitor_result(True)

        self.assertEqual(result, (0, "running", "running", "not_attempted"))
        self.assertTrue(budget.called)
        authorize.assert_called_once_with(
            session,
            "/workspace/tqp_full_training_123_aaaaaaaa_a1",
            "a" * 32,
            3600.0,
            3500.0,
            1000.0,
        )
        stop.assert_not_called()

    def test_forecast_delay_near_deadline_does_not_subtract_idle_wait_from_work(self):
        row = {
            "id": 123, "label": "cnot103-fulltrain-123-a1",
            "actual_status": "running", "gpu_name": "RTX 3060", "num_gpus": 2,
            "dph_total": 0.30, "storage_cost": 0.06, "disk_space": 64,
        }
        stopped = {**row, "actual_status": "stopped"}
        started = 100.0
        measured_elapsed = 10 * 3600 + 50 * 60
        remaining_work = 36 * 3600 + 40 * 60
        forecast_total = measured_elapsed + remaining_work
        now_epoch = started + measured_elapsed + 20 * 60
        with (
            patch.object(
                controller,
                "_read_remote_status_full",
                return_value=(
                    "baseline", "measured", started, 0.0, started,
                    forecast_total, remaining_work, "a" * 32,
                ),
            ),
            patch.object(controller.base, "instance_rows", side_effect=[[row], [stopped]]),
            patch.object(controller.base, "resolve_label", return_value=row),
            patch.object(controller.base, "instance_id", side_effect=lambda value: value["id"]),
            patch.object(controller.base, "project_charges", return_value=0.0),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "full_run_budget_allows", return_value=True) as budget,
            patch.object(controller, "_write_forecast_authorization", return_value=True) as authorize,
            patch.object(controller.base, "stop_exact_label", return_value=("verified", "detached")) as stop,
            patch.object(controller.time, "monotonic", side_effect=[0.0, 2.0]),
            patch.object(controller.time, "time", return_value=now_epoch),
        ):
            result = controller._monitor_window(
                row, row["label"], 123, "123", "1",
                "/workspace/tqp_full_training_123_aaaaaaaa_a1", 1, object(),
            )
        self.assertEqual(result, (1, "time_limit", "time_limit", "verified"))
        budget.assert_not_called()
        authorize.assert_not_called()
        stop.assert_called_once_with(row["label"], "123", "1", expected_id=123)

    def test_monitor_stops_exact_target_when_forecast_exceeds_budget(self):
        result, budget, authorize, stop, row, session = self._monitor_result(False)

        self.assertEqual(result, (1, "budget_gate", "budget_gate", "verified"))
        self.assertTrue(budget.called)
        authorize.assert_not_called()
        stop.assert_called_once_with(
            row["label"], "123", "1", expected_id=123
        )


class StableApprovedTargetTests(unittest.TestCase):
    @staticmethod
    def row(ident, label, state):
        return {
            "id": ident, "label": label, "gpu_name": "RTX 3060", "num_gpus": 2,
            "actual_status": state, "dph_total": 0.20, "storage_cost": 0.15,
            "disk_space": 64,
        }

    def test_retry_after_attempt_rename_and_stop_reuses_the_approved_id(self):
        previous_label = "cnot103-fulltrain-100-a1"
        current_label = "cnot103-fulltrain-200-a1"
        target = self.row(123, previous_label, "stopped")
        activated = {**target, "label": current_label, "actual_status": "running"}
        with (
            patch.dict(os.environ, {"CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"}),
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller.base, "instance_id", side_effect=lambda row: row["id"]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "_validate_bootstrap_budget", return_value=True),
            patch.object(controller, "bootstrap_budget_allows", return_value=True),
            patch.object(controller, "_activate_stopped_target", return_value=(activated, current_label)),
            patch.object(controller.base.probe, "observed_state", side_effect=lambda row: "stopped" if row["id"] == 123 else "running"),
            patch.object(controller.base, "offer_inventory", return_value=[{
                "ask_id": "9", "dph_total": 0.20, "storage_cost": 0.15,
                "disk_space": 64,
            }]) as offers,
            patch.object(controller.base, "create_once", return_value={
                "id": 456, "label": current_label, "gpu_name": "RTX 3060",
                "num_gpus": 2, "actual_status": "running",
            }) as create,
        ):
            result = controller._start_target(
                [target],
                {"resume_existing_label": "cnot103-r10k-100-a1"},
                "200",
                "1",
            )

        self.assertEqual(controller.base.instance_id(result[0]), 123)
        self.assertEqual(result[1], current_label)
        offers.assert_not_called()
        create.assert_not_called()

    def test_missing_or_mismatched_approved_target_never_falls_back_to_rent(self):
        label = "cnot103-r10k-100-a1"
        replacement = self.row(456, label, "stopped")
        with (
            patch.dict(os.environ, {"CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"}),
            patch.object(controller.base, "project_charges", return_value=1.0),
            patch.object(controller.base, "instance_id", side_effect=lambda row: row["id"]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller, "_other_storage_reserve", return_value=0.0),
            patch.object(controller, "_validate_bootstrap_budget", return_value=True),
            patch.object(controller, "_activate_stopped_target", return_value=(replacement, label)) as activate,
            patch.object(controller.base, "offer_inventory") as offers,
            patch.object(controller.base, "create_once") as create,
        ):
            with self.assertRaises(controller.GateError):
                controller._start_target(
                    [replacement],
                    {"resume_existing_label": label},
                    "200",
                    "1",
                )

        activate.assert_not_called()
        offers.assert_not_called()
        create.assert_not_called()

    def test_wait_rejects_state_returned_at_or_after_monotonic_deadline(self):
        label = "cnot103-fulltrain-200-a1"
        running = self.row(123, label, "running")
        with (
            patch.object(controller.base, "instance_rows", return_value=[running]),
            patch.object(controller.base, "resolve_label", return_value=running),
            patch.object(controller.base, "instance_id", side_effect=lambda row: row["id"]),
            patch.object(controller.base, "is_2x3060", return_value=True),
            patch.object(controller.base.probe, "observed_state", return_value="running"),
            patch.object(controller.time, "monotonic", side_effect=[9.0, 10.0]),
        ):
            with self.assertRaises(controller.GateError):
                controller._wait_for_exact_instance(123, label, deadline=10.0)

    def test_approved_target_configuration_is_required_before_any_rent_path(self):
        label = "cnot103-r10k-100-a1"
        row = self.row(123, label, "stopped")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(controller.base, "instance_rows", return_value=[row]),
            patch.object(controller.base, "offer_inventory") as offers,
            patch.object(controller.base, "create_once") as create,
        ):
            with self.assertRaises(controller.GateError):
                controller._start_target(
                    [row], {"resume_existing_label": label, "approved_instance_id": 123}, "200", "1"
                )
        offers.assert_not_called()
        create.assert_not_called()

    def test_cleanup_is_unverified_when_post_stop_state_is_still_running(self):
        run_label = "cnot103-fulltrain-123-a1"
        running = {
            "id": 123, "label": run_label, "gpu_name": "RTX 3060", "num_gpus": 2,
            "actual_status": "running", "dph_total": 0.20, "storage_cost": 0.15,
            "disk_space": 64,
        }
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1", "CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "CLAIM_VISIBILITY_WAIT_SECONDS", 0),
            patch.object(controller, "_marker", return_value={"resume_existing_label": "cnot103-r10k-44749514754-a1"}),
            patch.object(controller.base, "instance_rows", side_effect=[[running], [running]]),
            patch.object(controller, "_read_remote_status", return_value=("completed", "passed", None, 1.0, 1_000.0)),
            patch.object(controller.base, "stop_exact_label", return_value=("verified", "detached")) as stop,
            patch.object(controller, "_summarize") as summarize,
        ):
            self.assertEqual(controller.cleanup(), 1)
        stop.assert_called_once_with(run_label, "123", "1", expected_id=123)
        self.assertEqual(summarize.call_args.args[1:3], ("unverified", "unverified"))

    def test_cleanup_is_unverified_when_post_stop_inventory_is_unavailable(self):
        run_label = "cnot103-fulltrain-123-a1"
        running = {
            "id": 123, "label": run_label, "gpu_name": "RTX 3060", "num_gpus": 2,
            "actual_status": "running", "dph_total": 0.20, "storage_cost": 0.15,
            "disk_space": 64,
        }
        api_failure = controller.base.probe.ApiFailure(None)
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "1", "CNOT103_APPROVED_TARGET_INSTANCE_ID": "123"}, clear=True),
            patch.object(controller.base, "run_identity", return_value=("a" * 40, "123", "legacy")),
            patch.object(controller, "CLAIM_VISIBILITY_WAIT_SECONDS", 0),
            patch.object(controller, "_marker", return_value={"resume_existing_label": "cnot103-r10k-44749514754-a1"}),
            patch.object(controller.base, "instance_rows", side_effect=[[running], api_failure]),
            patch.object(controller, "_read_remote_status", return_value=("completed", "passed", None, 1.0, 1_000.0)),
            patch.object(controller.base, "stop_exact_label", return_value=("verified", "detached")) as stop,
            patch.object(controller, "_summarize") as summarize,
        ):
            self.assertEqual(controller.cleanup(), 1)
        stop.assert_called_once_with(run_label, "123", "1", expected_id=123)
        self.assertEqual(summarize.call_args.args[1:3], ("api_failed", "unverified"))

    def test_second_supervisor_is_detected_from_only_an_allowlisted_status(self):
        row = self.row(123, "cnot103-r10k-100-a1", "running")
        with patch.object(
            controller,
            "_attach_and_remote",
            return_value=(
                subprocess.CompletedProcess([], 0, "full_training=supervisor_active\n", ""),
                "detached",
                123,
            ),
        ) as remote:
            self.assertTrue(controller._has_active_supervisor(row, "200", "1"))
        remote.assert_called_once()

    def test_start_capable_workflow_serializes_runs_and_has_finalizer_headroom(self):
        workflow = (
            Path(__file__).resolve().parents[1]
            / "workflows"
            / "vast-full-training.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("group: cnot103-existing-2x3060-target", workflow)
        self.assertTrue("cancel-in-progress: false" in workflow)
        for relative in ("vast-rebuild-remote-dataset.yml", "vast-authorized-2x3060-rebuild.yml"):
            shared = Path(__file__).resolve().parents[1] / "workflows" / relative
            self.assertIn(
                "group: cnot103-existing-2x3060-target",
                shared.read_text(encoding="utf-8"),
            )
        self.assertTrue("timeout-minutes: 300" in workflow)
        self.assertTrue("  finalize:\n    if: always()" in workflow)
        self.assertFalse("needs.start.outputs.cleanup_needed" in workflow)
        launch = controller._launch_script(
            "/workspace/test", "a" * 40, "b" * 40, "c" * 64, 1.0, 0.0,
            2.0, 3.0, "123", "1", 123, 4.0,
        )
        self.assertIn("flock -n 9", launch)
        self.assertIn("supervisor_busy", launch)
        self.assertIn("--billing-baseline-epoch 4.0", launch)


if __name__ == "__main__":
    unittest.main()
