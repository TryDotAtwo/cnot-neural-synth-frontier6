# Budget and lifecycle contract tests for the one-shot Vast controller.
import unittest

import vast_rent_rebuild_dataset as controller


def offer(offer_id, **updates):
    row = {
        "ask_id": offer_id,
        "gpu_name": "RTX 3060",
        "num_gpus": 2,
        "gpu_ram": 12288,
        "compute_cap": 8.6,
        "reliability": 0.99,
        "dph_total": 0.2,
        "storage_cost": 0.15,
        "direct_port_count": 1,
        "rentable": True,
        "verified": True,
    }
    row.update(updates)
    return row


class OfferAndBudgetTests(unittest.TestCase):
    def test_offer_filter_requires_exact_verified_2x3060(self):
        rows = [
            offer(10),
            offer(11, num_gpus=1),
            offer(12, gpu_name="RTX 4060"),
            offer(13, verified=False),
            offer(14, reliability=0.90),
            offer(15, direct_port_count=0),
            offer(16, rentable=False),
            offer(17, gpu_ram=8192),
            offer(18, compute_cap=8.0),
            offer(19, gpu_name="RTX 3060 Ti"),
        ]

        selected = controller.eligible_offers(rows)

        self.assertEqual([row["ask_id"] for row in selected], [10])

    def test_budget_gate_includes_50h_and_transfer_reserve(self):
        self.assertTrue(controller.budget_allows(5.0, 0.24, run_hours=50, reserve=3.0))
        self.assertFalse(controller.budget_allows(5.0, 0.241, run_hours=50, reserve=3.0))

    def test_stopped_storage_and_dataset_rebuild_projection_fit_the_budget(self):
        reserve = controller.total_reserve(0.15)
        self.assertAlmostEqual(reserve, 2.49)
        self.assertEqual(controller.RUN_HOURS, 3.0)
        self.assertTrue(controller.budget_allows(0.0, 0.30, reserve=reserve))
        max_rate = (controller.BUDGET_CAP - reserve) / controller.RUN_HOURS
        self.assertTrue(controller.budget_allows(0.0, max_rate, reserve=reserve))
        self.assertFalse(controller.budget_allows(0.0, max_rate + 0.001, reserve=reserve))

    def test_budget_gate_reports_whether_history_or_offer_price_blocks_rent(self):
        rows = [offer(10)]
        self.assertEqual(controller.budget_gate_reason(0.0, rows), "within_budget")
        self.assertEqual(controller.budget_gate_reason(20.0, rows), "prior_spend")
        reserve = controller.total_reserve(0.15)
        over_budget_rate = (controller.BUDGET_CAP - reserve) / controller.RUN_HOURS + 0.001
        expensive = [offer(11, dph_total=over_budget_rate, storage_cost=0.15)]
        self.assertEqual(controller.budget_gate_reason(0.0, expensive), "offer_cost")

    def test_invalid_costs_fail_closed(self):
        for spent, rate, reserve in [
            (float("nan"), 0.1, 1.0),
            (0.0, float("inf"), 1.0),
            (-0.01, 0.1, 1.0),
            (0.0, -0.1, 1.0),
            (0.0, 0.1, -1.0),
            ("0", 0.1, 1.0),
        ]:
            with self.subTest(spent=spent, rate=rate, reserve=reserve):
                self.assertFalse(controller.budget_allows(spent, rate, run_hours=50, reserve=reserve))

    def test_label_resolution_rejects_duplicates(self):
        expected = {"id": 7, "label": "cnot103-2x3060-rebuild-v1"}
        self.assertEqual(
            controller.resolve_label([expected, {"id": 8, "label": "other"}], expected["label"]),
            expected,
        )
        with self.assertRaises(controller.GateError):
            controller.resolve_label([expected, {"id": 8, "label": expected["label"]}], expected["label"])

    def test_create_timeout_resolves_by_label_without_duplicate_create(self):
        import subprocess
        from unittest.mock import patch

        with patch.object(
            controller, "vast_cli", side_effect=subprocess.TimeoutExpired("vastai", 120)
        ) as create_call, patch.object(controller, "wait_for_label", return_value=None):
            with self.assertRaises(controller.GateError):
                controller.create_once("12345", "cnot103-2x3060-rebuild-v1")

        self.assertEqual(create_call.call_count, 1)

    def test_preflight_targets_only_owned_instances(self):
        owned_id = next(iter(controller.OWNED_INSTANCE_IDS))
        unowned = offer(30, id=99999999, label="other-project")
        labeled = offer(31, id=99999998, label="cnot103-r10k-run-a1")
        legacy_owned = offer(32, id=owned_id, label="legacy-cnot-target")

        selected = controller.probe.target_rows([unowned, labeled, legacy_owned])

        self.assertEqual(selected, [labeled, legacy_owned])

    def test_existing_rebuild_budget_reserves_storage_and_prior_charges(self):
        self.assertTrue(
            hasattr(controller, "existing_rebuild_budget_allows"),
            "existing-target rebuild must use the shared total-spend gate",
        )
        target = offer(33, dph_total=0.2, storage_cost=0.15)
        self.assertTrue(controller.existing_rebuild_budget_allows(target, 16.5))
        self.assertFalse(controller.existing_rebuild_budget_allows(target, 17.0))

    def test_target_scope_ignores_other_2x3060_rows(self):
        wanted = offer(20, label="cnot103-2x3060-rebuild-v1", actual_status="running")
        unrelated = offer(21, label="other-project", actual_status="running")

        selected = controller.target_rows_for_label([wanted, unrelated], wanted["label"])

        self.assertEqual(selected, [wanted])




    def test_project_charges_include_owned_storage_rows_without_metadata_labels(self):
        from unittest.mock import patch

        owned_id = next(iter(controller.OWNED_INSTANCE_IDS))
        unowned_id = 1
        while unowned_id in controller.OWNED_INSTANCE_IDS:
            unowned_id += 1
        records = [
            {
                "description": f"Instance {owned_id} storage charge",
                "amount": 0.25,
            },
            {
                "description": f"Instance {unowned_id} storage charge",
                "amount": 3.0,
                "metadata": {"label": "another-project"},
            },
        ]
        with patch.object(
            controller,
            "vast_cli",
            return_value={"results": records, "next_token": None},
        ):
            self.assertAlmostEqual(controller.project_charges(), 0.25)


class ResumeExistingTargetTests(unittest.TestCase):
    def _environment(self, label):
        return {
            "GITHUB_SHA": "c" * 40,
            "GITHUB_RUN_ID": "99",
            "GITHUB_RUN_ATTEMPT": "1",
            "CNOT103_HOLDOUT_REVISION": "a" * 40,
            "CNOT103_HOLDOUT_SHA256": "b" * 64,
            "CNOT103_RESUME_LABEL": label,
        }

    def test_resume_exact_stopped_target_without_renting_another(self):
        import io
        import os
        from contextlib import redirect_stdout
        from unittest.mock import patch

        label = "cnot103-r10k-44749514754-a1"
        row = offer(40, id=42, label=label, actual_status="stopped")
        output = io.StringIO()
        with (
            patch.dict(os.environ, self._environment(label), clear=True),
            patch.object(controller, "instance_rows", return_value=[row]),
            patch.object(controller, "project_charges", return_value=1.0),
            patch.object(controller, "wait_until_running", return_value=row) as start,
            patch.object(controller, "install_remote_cargo", return_value="ready"),
            patch.object(controller, "scoped_dataset_rebuild", return_value=0) as rebuild,
            patch.object(controller, "create_once") as create,
            patch.object(controller, "offer_inventory") as offers,
            redirect_stdout(output),
        ):
            result = controller.run()

        self.assertEqual(result, 0)
        start.assert_called_once_with(label)
        rebuild.assert_called_once_with(label)
        create.assert_not_called()
        offers.assert_not_called()
        self.assertIn("query_outcome=accepted", output.getvalue())

    def test_resume_target_checks_full_budget_before_starting(self):
        import io
        import os
        from contextlib import redirect_stdout
        from unittest.mock import patch

        label = "cnot103-r10k-44749514754-a1"
        row = offer(41, id=43, label=label, actual_status="stopped")
        output = io.StringIO()
        with (
            patch.dict(os.environ, self._environment(label), clear=True),
            patch.object(controller, "instance_rows", return_value=[row]),
            patch.object(controller, "project_charges", return_value=17.0),
            patch.object(controller, "wait_until_running") as start,
            patch.object(controller, "create_once") as create,
            patch.object(controller, "offer_inventory") as offers,
            redirect_stdout(output),
        ):
            result = controller.run()

        self.assertEqual(result, 1)
        self.assertIn("query_outcome=budget_gate_prior_spend", output.getvalue())
        start.assert_not_called()
        create.assert_not_called()
        offers.assert_not_called()


class HoldoutSourcePinTests(unittest.TestCase):
    def test_missing_or_malformed_pins_fail_closed(self):
        self.assertEqual(controller.holdout_pin_status(None, None), "missing")
        self.assertEqual(controller.holdout_pin_status("a" * 40, None), "missing")
        self.assertEqual(controller.holdout_pin_status("main", "b" * 64), "invalid")
        self.assertEqual(controller.holdout_pin_status("a" * 40, "B" * 64), "invalid")
        self.assertEqual(controller.holdout_pin_status("a" * 40, "b" * 64), "accepted")

    def test_rent_controller_rejects_missing_pin_before_vast_queries(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {
            "GITHUB_SHA": "a" * 40,
            "GITHUB_RUN_ID": "12",
            "GITHUB_RUN_ATTEMPT": "1",
        }, clear=True), patch.object(controller, "instance_rows") as inventory:
            status = controller.run()

        self.assertEqual(status, 1)
        inventory.assert_not_called()


class InstanceControlPayloadTests(unittest.TestCase):
    def test_run_start_claims_unique_label_in_the_same_put(self):
        from unittest.mock import patch

        label = "cnot103-fulltrain-123-a1"
        with (
            patch.object(controller, "api_key", return_value="secret"),
            patch.object(controller.probe, "api_json", return_value={"success": True}) as api,
        ):
            self.assertTrue(controller.api_instance_id(123, "running", label=label))

        api.assert_called_once_with(
            "secret",
            "/instances/123/",
            "PUT",
            {"state": "running", "label": label},
        )

class ObservedStopStateTests(unittest.TestCase):
    def test_requested_transition_is_separate_from_observed_state(self):
        row = {"actual_status": "running", "next_state": "stopped"}

        self.assertEqual(controller.probe.observed_state(row), "running")
        self.assertEqual(controller.probe.requested_state(row), "stopped")

    def test_incomplete_status_does_not_prove_stopped(self):
        row = {"next_state": "stopped"}

        self.assertEqual(controller.probe.observed_state(row), "unknown")
        self.assertEqual(controller.probe.requested_state(row), "stopped")

    def test_stop_loop_does_not_accept_transition_target_as_confirmation(self):
        from unittest.mock import patch

        label = "cnot103-r10k-12-a1"
        row = {
            "id": 123,
            "label": label,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "next_state": "stopped",
        }
        with (
            patch.object(controller, "instance_rows", return_value=[row]),
            patch.object(controller, "remove_run_keys", return_value="not_found"),
            patch.object(controller, "api_instance_id", return_value=True) as stop_request,
            patch.object(controller.time, "monotonic", side_effect=[0.0, 0.0, 301.0]),
            patch.object(controller.time, "sleep"),
        ):
            status, key_status = controller.stop_exact_label(label, "12", "1")

        self.assertEqual(status, "unverified")
        self.assertEqual(key_status, "not_found")
        stop_request.assert_called_once_with(123, "stopped")


    def test_stop_checks_actual_state_once_more_after_its_deadline(self):
        from unittest.mock import patch

        label = "cnot103-r10k-12-a1"
        running = {
            "id": 123,
            "label": label,
            "gpu_name": "RTX 3060",
            "num_gpus": 2,
            "actual_status": "running",
        }
        stopped = {**running, "actual_status": "stopped"}
        with (
            patch.object(controller, "instance_rows", side_effect=[[running], [running], [stopped]]),
            patch.object(controller, "remove_run_keys", return_value="not_found"),
            patch.object(controller, "api_instance_id", return_value=True) as stop_request,
            patch.object(controller.time, "monotonic", side_effect=[0.0, 0.0, 301.0]),
            patch.object(controller.time, "sleep"),
        ):
            status, key_status = controller.stop_exact_label(label, "12", "1")

        self.assertEqual((status, key_status), ("verified", "not_found"))
        stop_request.assert_called_once_with(123, "stopped")

if __name__ == "__main__":
    unittest.main()
