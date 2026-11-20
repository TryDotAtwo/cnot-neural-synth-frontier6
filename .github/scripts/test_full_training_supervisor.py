import unittest

from run_full_training import (
    RunError,
    quality_metrics,
    select_dev_indices,
    validate_training_result,
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


if __name__ == "__main__":
    unittest.main()
