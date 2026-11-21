import hashlib
import json
import pathlib
import tempfile
import time
import unittest
from unittest.mock import patch

import run_full_training as training
from run_full_training import (
    Supervisor,
    RunError,
    quality_metrics,
    select_dev_indices,
    validate_training_result,
    validate_feature_coverage,
    validate_feature_coverage_preflight,
    estimate_full_run_seconds,
    estimate_remaining_work_seconds,
    forecast_approval_matches,
    forecast_time_fits,
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
        remaining = estimate_remaining_work_seconds(
            setup_seconds=2.0,
            update_seconds=0.5,
            remaining_updates=10,
            checkpoint_reserve_seconds=3.0,
            baseline_search_seconds=11.0,
            forecast_safety_seconds=5.0,
        )
        self.assertEqual(remaining, 26.0)
        estimate = estimate_full_run_seconds(
            elapsed_before_full=100.0,
            setup_seconds=2.0,
            update_seconds=0.5,
            remaining_updates=10,
            checkpoint_reserve_seconds=3.0,
            baseline_search_seconds=11.0,
            forecast_safety_seconds=5.0,
        )
        self.assertEqual(estimate, 126.0)

    def test_forecast_approval_must_match_run_token_and_estimate(self):
        approval = {
            "schema": "cnot103_full_training_forecast_approval_v1",
            "authorized": True,
            "forecast_token": "a" * 32,
            "forecast_total_seconds": 1234.0,
            "remaining_work_seconds": 1000.0,
            "authorized_at_epoch": 10_000.0,
        }
        self.assertTrue(
            forecast_approval_matches(approval, "a" * 32, 1234.0, 1000.0, 10_005.0)
        )
        self.assertFalse(
            forecast_approval_matches(approval, "b" * 32, 1234.0, 1000.0, 10_005.0)
        )
        self.assertFalse(
            forecast_approval_matches(approval, "a" * 32, 1235.0, 1000.0, 10_005.0)
        )
        self.assertFalse(
            forecast_approval_matches(approval, "a" * 32, 1234.0, 1001.0, 10_005.0)
        )
        self.assertFalse(
            forecast_approval_matches(approval, "a" * 32, 1234.0, 1000.0, 10_361.0)
        )
        approval["authorized"] = False
        self.assertFalse(
            forecast_approval_matches(approval, "a" * 32, 1234.0, 1000.0, 10_005.0)
        )

    def test_twenty_minute_authorization_delay_consumes_remaining_wall_budget(self):
        started = 100.0
        before_delay = started + 10 * 3600
        remaining = (
            training.MAX_RUNTIME_SECONDS
            - (before_delay - started)
            - training.FORECAST_CLEANUP_RESERVE_SECONDS
            - 60
        )
        self.assertTrue(forecast_time_fits(started, remaining, before_delay))
        self.assertFalse(
            forecast_time_fits(started, remaining, before_delay + 20 * 60)
        )


class QualityComparisonTests(unittest.TestCase):
    @staticmethod
    def solved(case_id, cost):
        identity = [1, 2, 4]
        base_gate = [0, 1] if case_id.endswith("a") else [0, 2]
        gates = [base_gate.copy() for _ in range(cost)]
        start = identity.copy()
        start[base_gate[1]] ^= start[base_gate[0]]
        return {
            "case_id": case_id,
            "dimension": 3,
            "width": 3,
            "status": "solved",
            "start_packed_rows": start,
            "end_packed_rows": identity,
            "word": {
                "replayed": True,
                "physical_cost": cost,
                "control_target": gates,
            },
        }

    @staticmethod
    def unsolved(case_id):
        return {
            "case_id": case_id,
            "dimension": 3,
            "width": 3,
            "status": "depth_limit",
            "start_packed_rows": [1, 3, 4] if case_id.endswith("a") else [1, 2, 5],
            "end_packed_rows": None,
            "word": None,
        }

    @staticmethod
    def report(rows, *, search=None, panel="test"):
        return {
            "schema": "cnot103_panel_search_result_v2",
            "panel": panel,
            "panel_digest": "a" * 64,
            "panel_ids": ["case-a", "case-b"],
            "feature_model_inputs": 102,
            "training_exposure_audit": {"strict_unseen": True},
            "search": training.BEAM_CONFIG if search is None else search,
            "head": "qhat",
            "rows": rows,
        }

    def compare(self, baseline_rows, candidate_rows, *, baseline_search=None,
                candidate_search=None):
        return training.compare_quality(
            self.report(baseline_rows, search=baseline_search),
            self.report(candidate_rows, search=candidate_search),
            expected_case_ids=["case-a", "case-b"],
            expected_panel="test",
            expected_panel_digest="a" * 64,
            expected_search=training.BEAM_CONFIG,
        )

    def test_valid_candidate_passes_and_rows_are_paired_by_case_id(self):
        result = self.compare(
            [self.solved("case-a", 3), self.unsolved("case-b")],
            [self.solved("case-b", 1), self.solved("case-a", 1)],
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["baseline_solved_count"], 1)
        self.assertEqual(result["candidate_solved_count"], 2)

    def test_valid_but_worse_candidate_fails(self):
        result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.solved("case-a", 3), self.unsolved("case-b")],
        )
        self.assertEqual(result["status"], "failed")

    def test_previously_solved_case_cannot_become_unsolved(self):
        result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.unsolved("case-a"), self.solved("case-b", 1)],
        )
        self.assertEqual(result["status"], "failed")

    def test_same_solved_count_on_different_cases_fails(self):
        result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.unsolved("case-a"), self.solved("case-b", 3)],
        )
        self.assertEqual(result["status"], "failed")

    def test_zero_solved_baseline_is_inconclusive(self):
        result = self.compare(
            [self.unsolved("case-a"), self.unsolved("case-b")],
            [self.solved("case-a", 1), self.unsolved("case-b")],
        )
        self.assertEqual(result["status"], "inconclusive")

    def test_duplicate_or_missing_case_ids_are_invalid(self):
        duplicate = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-a")],
            [self.solved("case-a", 1), self.unsolved("case-b")],
        )
        missing = self.compare(
            [self.solved("case-a", 1)],
            [self.solved("case-a", 1), self.unsolved("case-b")],
        )
        self.assertEqual(duplicate["status"], "invalid")
        self.assertEqual(missing["status"], "invalid")

    def test_search_budget_mismatch_is_invalid(self):
        other = dict(training.BEAM_CONFIG)
        other["max_depth"] += 1
        result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.solved("case-a", 1), self.unsolved("case-b")],
            candidate_search=other,
        )
        self.assertEqual(result["status"], "invalid")

    def test_replay_and_physical_cost_are_checked_per_original_case(self):
        broken_replay = self.solved("case-a", 1)
        broken_replay["word"]["replayed"] = False
        mismatch = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [broken_replay, self.unsolved("case-b")],
        )
        broken_cost = self.solved("case-a", 1)
        broken_cost["word"]["physical_cost"] = 2
        cost_mismatch = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [broken_cost, self.unsolved("case-b")],
        )
        self.assertEqual(mismatch["status"], "invalid")
        self.assertEqual(cost_mismatch["status"], "invalid")


class CleanBootstrapTests(unittest.TestCase):
    class ReachedCoverage(Exception):
        pass

    def _run_to_coverage(self, *, validation_ok):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        parent = pathlib.Path(temporary.name)
        root = parent / "run"
        source = root / "source"
        source.mkdir(parents=True)
        real_resolve = pathlib.Path.resolve

        def map_workspace(path, *args, **kwargs):
            resolved = real_resolve(path, *args, **kwargs)
            try:
                relative = resolved.relative_to(parent)
            except ValueError:
                return resolved
            return pathlib.Path("/workspace") / relative

        started = time.time()
        with patch.object(pathlib.Path, "resolve", autospec=True, side_effect=map_workspace):
            supervisor = Supervisor(
                root, "a" * 40, "b" * 40, "c" * 64,
                started, 0.0, started + 10_000,
            )

        stages = []
        coverage_calls = []

        def run_stage(name, command, **_kwargs):
            stages.append(name)
            if name == "fetch_holdout":
                (root / training.HOLDOUT_RELATIVE).write_bytes(b"tiny-holdout")
            elif name == "build_frozen_dataset":
                data_path = root / training.DATA_RELATIVE
                candidate_bytes = json.dumps({
                    "panels": {
                        "dev": [{"lineage_id": "dev-case-1", "state": {"n": 2}}],
                        "test": [{"lineage_id": "test-case-1", "state": {"n": 2}}],
                    },
                }, separators=(",", ":")).encode("utf-8")
                data_path.write_bytes(candidate_bytes)
                digest = hashlib.sha256(candidate_bytes).hexdigest()
                (root / training.DATA_SHA_RELATIVE).write_text(digest + "\n", encoding="ascii")
                (root / training.DATA_RECEIPT_RELATIVE).write_text(
                    json.dumps({
                    "schema": "cnot103_dataset_rebuild_receipt_v1",
                    "complete": False,
                    "validation_status": "pending",
                    "dataset_sha256": digest,
                    "dataset_bytes": len(candidate_bytes),
                    "holdout_sqlite_sha256": "c" * 64,
                    "heldout_source_revision": "b" * 40,
                    "builder_source_revision": "a" * 40,
                    "heldout_source_repository": "TryDotAtwo/cnot-neural-synth-frontier6",
                    "heldout_source_filename": "synthetic.sqlite",
                }),
                    encoding="utf-8",
                )
            elif name == "cargo_release_build":
                validator = supervisor._runner("cnot103-validate-data")
                validator.parent.mkdir(parents=True, exist_ok=True)
                validator.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                validator.chmod(0o700)
            elif name == "validate_frozen_dataset":
                if not pathlib.Path(command[0]).is_file():
                    raise AssertionError("Rust validator was invoked before cargo build")
                if not validation_ok:
                    raise training.RunError("synthetic Rust validation rejection")
            elif name == "finalize_frozen_dataset":
                receipt_path = root / training.DATA_RECEIPT_RELATIVE
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt.update({"complete": True, "validation_status": "rust_loader_accepted"})
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            return None

        def coverage(_dataset_sha):
            coverage_calls.append(True)
            raise self.ReachedCoverage()

        with (
            patch.object(supervisor, "_check_source"),
            patch.object(supervisor, "_hardware", return_value={"torch_cuda_devices": 2}),
            patch.object(supervisor, "_free_gib", return_value=100.0),
            patch.object(supervisor, "_ensure_cargo"),
            patch.object(supervisor, "_run", side_effect=run_stage),
            patch.object(supervisor, "_feature_coverage", side_effect=coverage),
        ):
            result = supervisor.run()
        return result, stages, coverage_calls, root

    def test_public_run_builds_validator_then_validates_seals_and_reaches_coverage(self):
        result, stages, coverage, root = self._run_to_coverage(validation_ok=True)

        self.assertEqual(result, 1)  # stopped deliberately at the coverage boundary
        self.assertLess(stages.index("cargo_release_build"), stages.index("validate_frozen_dataset"))
        self.assertLess(stages.index("validate_frozen_dataset"), stages.index("finalize_frozen_dataset"))
        self.assertTrue(coverage)
        receipt = json.loads((root / training.DATA_RECEIPT_RELATIVE).read_text(encoding="utf-8"))
        self.assertTrue(receipt["complete"])

    def test_failed_rust_validation_never_seals_receipt_or_reaches_coverage(self):
        result, stages, coverage, root = self._run_to_coverage(validation_ok=False)

        self.assertEqual(result, 1)
        self.assertLess(stages.index("cargo_release_build"), stages.index("validate_frozen_dataset"))
        self.assertNotIn("finalize_frozen_dataset", stages)
        self.assertFalse(coverage)
        receipt = json.loads((root / training.DATA_RECEIPT_RELATIVE).read_text(encoding="utf-8"))
        self.assertFalse(receipt["complete"])


if __name__ == "__main__":
    unittest.main()
