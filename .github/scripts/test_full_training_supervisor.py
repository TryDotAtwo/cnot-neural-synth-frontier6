import unittest

from run_full_training import (
    RunError,
    quality_metrics,
    select_dev_indices,
    validate_training_result,
    validate_feature_coverage,
    validate_feature_coverage_preflight,
    estimate_full_run_seconds,
    forecast_approval_matches,
)


class FullTrainingSupervisorTests(unittest.TestCase):
    def test_holdout_sample_is_deterministic_and_stratified_by_width(self):
        records = [
            {"state": {"n": n}}
            for n in (8, 8, 8, 8, 8, 10, 10, 10, 10, 10)
        ]
        expected = [0, 4, 5, 9]
        self.assertEqual(select_dev_indices(records, per_width=2), expected)
        self.assertEqual(select_dev_indices(records, per_width=2), expected)

    def test_quality_metrics_require_replay_and_the_full_model_input_set(self):
        report = {
            "schema": "cnot103_dev_search_result_v1",
            "feature_model_inputs": 102,
            "training_exposure_audit": {"strict_unseen": True},
            "rows": [
                {"status": "solved", "word": {"physical_cost": 3, "replayed": True}},
                {"status": "depth_limit", "word": None},
            ],
        }
        result = quality_metrics(report)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["solved_count"], 1)
        self.assertEqual(result["physical_costs"], [3])
        broken = dict(report)
        broken["rows"] = [{"status": "solved", "word": {"physical_cost": 3, "replayed": False}}]
        with self.assertRaises(RunError):
            quality_metrics(broken)

    def test_training_result_requires_exact_progress_model_devices_and_audit(self):
        report = {
            "schema": "cnot103_training_result_v1",
            "model_parameters": 136_421_807,
            "feature_model_inputs": 102,
            "cuda_devices": [0, 1],
            "start_progress": {"attempted": 1},
            "end_progress": {"attempted": 2, "accepted": 2, "rejected": 0},
            "last_step": {"accepted": True},
            "exposure_audit": {"strict_unseen": True},
        }
        validate_training_result(report, expected_start=1, expected_end=2)
        report["end_progress"]["attempted"] = 3
        with self.assertRaises(RunError):
            validate_training_result(report, expected_start=1, expected_end=2)


class FeatureCoverageTests(unittest.TestCase):
    @staticmethod
    def coverage_report(panel="train"):
        widths = []
        for width, rows in ((10, 2), (13, 1)):
            features = []
            for index in range(102):
                if width == 10 and index == 0:
                    counts = (rows, 0, 0, 0, 0)
                elif width == 13 and index == 1:
                    counts = (rows, 0, 0, 0, 0)
                elif index == 1 and width == 10:
                    counts = (0, 0, rows, 0, 0)
                elif index == 0 and width == 13:
                    counts = (0, 0, rows, 0, 0)
                else:
                    counts = (0, rows, 0, 0, 0)
                features.append({
                    "id": f"feature-{index}",
                    "role": "candidate",
                    "scope": "matrix",
                    "provider": "native-v25",
                    "not_requested": counts[4],
                    "computed": counts[0],
                    "inapplicable": counts[1],
                    "unsupported": counts[2],
                    "resource_limit": counts[3],
                })
            widths.append({
                "width": width,
                "rows": rows,
                "rejected_rows": 0,
                "features": features,
            })
        return {
            "schema": "cnot103_feature_coverage_v1",
            "input_schema": "runtime_packet_layout_v25",
            "policy_id": "policy-v25-test",
            "manifest_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "panel": panel,
            "batch_size": 1,
            "widths": widths,
        }

    def test_coverage_keeps_102_slots_and_records_status_by_width(self):
        summary = validate_feature_coverage(
            self.coverage_report(),
            expected_manifest_sha256="a" * 64,
            expected_dataset_sha256="b" * 64,
            expected_panel="train",
            expected_rows_by_width={10: 2, 13: 1},
            expected_feature_count=102,
            profile_id="n10-test-support-v1",
            profile_sha256="c" * 64,
            profile_context_widths={"labeled_cuts": [10]},
        )

        self.assertEqual(summary["model_input_slots"], 102)
        self.assertEqual(summary["selected_feature_count"], 102)
        self.assertEqual([row["width"] for row in summary["widths"]], [10, 13])
        self.assertEqual(summary["widths"][0]["features"][0]["computed"], 2)
        self.assertEqual(summary["widths"][1]["features"][0]["unsupported"], 1)
        self.assertEqual(summary["widths"][1]["features"][1]["computed"], 1)

    def test_coverage_rejects_unrequested_features_and_bad_row_accounting(self):
        report = self.coverage_report()
        report["widths"][0]["features"][0]["not_requested"] = 1

        with self.assertRaises(RunError):
            validate_feature_coverage(
                report,
                expected_manifest_sha256="a" * 64,
                expected_dataset_sha256="b" * 64,
                expected_panel="train",
                expected_rows_by_width={10: 2, 13: 1},
                expected_feature_count=102,
                profile_id="n10-test-support-v1",
                profile_sha256="c" * 64,
                profile_context_widths={"labeled_cuts": [10]},
            )


    def test_coverage_reports_computed_values_separately_from_102_input_slots(self):
        summary = validate_feature_coverage(
            self.coverage_report(),
            expected_manifest_sha256="a" * 64,
            expected_dataset_sha256="b" * 64,
            expected_panel="train",
            expected_rows_by_width={10: 2, 13: 1},
            expected_feature_count=102,
            profile_id="n10-test-support-v1",
            profile_sha256="c" * 64,
            profile_context_widths={"labeled_cuts": [10]},
        )

        self.assertEqual(summary["model_input_slots"], 102)
        self.assertEqual(summary["availability_by_width"], [
            {
                "width": 10,
                "rows": 2,
                "model_input_slots": 102,
                "profile_context_keys": ["labeled_cuts"],
                "features_with_any_computed_rows": 1,
                "computed_values": 2,
                "inapplicable_values": 200,
                "unsupported_values": 2,
                "resource_limit_values": 0,
            },
            {
                "width": 13,
                "rows": 1,
                "model_input_slots": 102,
                "profile_context_keys": [],
                "features_with_any_computed_rows": 1,
                "computed_values": 1,
                "inapplicable_values": 100,
                "unsupported_values": 1,
                "resource_limit_values": 0,
            },
        ])


    def _summary_for_preflight(self):
        arguments = {
            "expected_manifest_sha256": "a" * 64,
            "expected_dataset_sha256": "b" * 64,
            "expected_rows_by_width": {10: 2, 13: 1},
            "expected_feature_count": 102,
            "profile_id": "n10-test-support-v1",
            "profile_sha256": "c" * 64,
            "profile_context_widths": {"labeled_cuts": [10]},
        }
        train = validate_feature_coverage(
            self.coverage_report("train"), expected_panel="train", **arguments
        )
        dev = validate_feature_coverage(
            self.coverage_report("dev"), expected_panel="dev", **arguments
        )
        return {
            "schema": "cnot103_feature_coverage_summary_v1",
            "manifest_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "profile_id": "n10-test-support-v1",
            "profile_sha256": "c" * 64,
            "profile_context_widths": {"labeled_cuts": [10]},
            "model_input_slots": 102,
            "panels": {"train": train, "dev": dev},
        }

    def test_pretraining_gate_requires_actual_dataset_and_profile_pins(self):
        summary = self._summary_for_preflight()

        with self.assertRaises(RunError):
            validate_feature_coverage_preflight(
                summary,
                expected_dataset_sha256="d" * 64,
                expected_profile_id="n10-test-support-v1",
                expected_profile_sha256="c" * 64,
                expected_profile_context_widths={"labeled_cuts": [10]},
            )

    def test_pretraining_gate_checks_actual_availability_against_profile_widths(self):
        summary = self._summary_for_preflight()

        self.assertIsNone(validate_feature_coverage_preflight(
            summary,
            expected_dataset_sha256="b" * 64,
            expected_profile_id="n10-test-support-v1",
            expected_profile_sha256="c" * 64,
            expected_profile_context_widths={"labeled_cuts": [10]},
        ))
        summary["panels"]["train"]["availability_by_width"][1]["profile_context_keys"] = [
            "labeled_cuts"
        ]
        with self.assertRaises(RunError):
            validate_feature_coverage_preflight(
                summary,
                expected_dataset_sha256="b" * 64,
                expected_profile_id="n10-test-support-v1",
                expected_profile_sha256="c" * 64,
                expected_profile_context_widths={"labeled_cuts": [10]},
            )

class ForecastGateTests(unittest.TestCase):
    def test_runtime_forecast_reserves_the_final_equal_budget_search(self):
        estimate = estimate_full_run_seconds(
            elapsed_before_full=100.0,
            setup_seconds=2.0,
            update_seconds=0.5,
            remaining_updates=10,
            checkpoint_reserve_seconds=3.0,
            baseline_search_seconds=11.0,
            final_search_reserve_seconds=11.0,
            forecast_safety_seconds=5.0,
        )
        self.assertEqual(estimate, 137.0)

    def test_forecast_approval_must_match_run_token_and_estimate(self):
        approval = {
            "schema": "cnot103_full_training_forecast_approval_v1",
            "authorized": True,
            "forecast_token": "a" * 32,
            "forecast_total_seconds": 1234.0,
        }
        self.assertTrue(forecast_approval_matches(approval, "a" * 32, 1234.0))
        self.assertFalse(forecast_approval_matches(approval, "b" * 32, 1234.0))
        self.assertFalse(forecast_approval_matches(approval, "a" * 32, 1235.0))
        approval["authorized"] = False
        self.assertFalse(forecast_approval_matches(approval, "a" * 32, 1234.0))


if __name__ == "__main__":
    unittest.main()
