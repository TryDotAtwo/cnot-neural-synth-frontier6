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
    def test_multigpu_handoff_pins_locked_panel_checkpoint_and_search(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            manifests = root / "manifests"
            reports = root / "reports"
            manifests.mkdir()
            reports.mkdir()
            training_manifest = manifests / "training.json"
            training_manifest.write_text('{"schema":"test"}', encoding="utf-8")

            checkpoint_sha = "a" * 64
            dataset_sha = "d" * 64
            training_result = reports / "training-result.json"
            training_result.write_text(
                json.dumps({
                    "schema": "cnot103_training_result_v1",
                    "end_progress": {"accepted": 8},
                    "checkpoint": {"sha256": checkpoint_sha},
                    "model_parameters": training.MODEL_PARAMETERS,
                    "feature_model_inputs": training.MODEL_INPUTS,
                    "dataset_sha256": dataset_sha,
                }),
                encoding="utf-8",
            )
            indices = [3, 9]
            case_ids = ["test-case-a", "test-case-b"]
            panel_digest = training._panel_digest("test", indices, case_ids)
            search = dict(training.BEAM_CONFIG)
            search_manifest = manifests / "locked-test-search.json"
            search_manifest.write_text(
                json.dumps({
                    "schema": "cnot103_panel_search_v2",
                    "training_manifest": {
                        "path": str(training_manifest),
                        "sha256": hashlib.sha256(training_manifest.read_bytes()).hexdigest(),
                    },
                    "training_result": {
                        "path": str(training_result),
                        "sha256": hashlib.sha256(training_result.read_bytes()).hexdigest(),
                    },
                    "panel": "test",
                    "panel_indices": indices,
                    "panel_ids": case_ids,
                    "panel_digest": panel_digest,
                    "search": search,
                    "head": "qhat",
                }),
                encoding="utf-8",
            )

            kwargs = {
                "root": root,
                "source_revision": "e" * 40,
                "accepted_updates": 8,
                "checkpoint_sha256": checkpoint_sha,
                "training_manifest_path": training_manifest,
                "training_result_path": training_result,
                "search_manifest_path": search_manifest,
                "expected_panel": "test",
                "expected_panel_indices": indices,
                "expected_panel_ids": case_ids,
                "expected_panel_digest": panel_digest,
                "expected_search": search,
                "locked_test_comparison": {"status": "length_gain"},
            }
            handoff = training.build_multigpu_beam_handoff(**kwargs)
            self.assertEqual(handoff["schema"], "cnot103_multigpu_beam_handoff_v1")
            self.assertEqual(handoff["accepted_updates"], 8)
            self.assertEqual(handoff["checkpoint_sha256"], checkpoint_sha)
            self.assertEqual(handoff["dataset_sha256"], dataset_sha)
            self.assertEqual(handoff["model"]["structural_blocks"], 8)
            self.assertEqual(handoff["model"]["parameters"], 136_421_807)
            self.assertEqual(handoff["model"]["input_slots"], 102)
            self.assertEqual(handoff["panel_indices"], indices)
            self.assertEqual(handoff["panel_ids"], case_ids)
            self.assertEqual(handoff["locked_test_status"], "length_gain")
            self.assertEqual(
                handoff["search_manifest"]["sha256"],
                hashlib.sha256(search_manifest.read_bytes()).hexdigest(),
            )

            wrong_checkpoint = dict(kwargs)
            wrong_checkpoint["checkpoint_sha256"] = "b" * 64
            with self.assertRaises(RunError):
                training.build_multigpu_beam_handoff(**wrong_checkpoint)

            changed_search = json.loads(search_manifest.read_text(encoding="utf-8"))
            changed_search["panel_ids"] = list(reversed(case_ids))
            search_manifest.write_text(json.dumps(changed_search), encoding="utf-8")
            with self.assertRaises(RunError):
                training.build_multigpu_beam_handoff(**kwargs)

            outside = dict(kwargs)
            outside["training_manifest_path"] = root.parent / "outside.json"
            (root.parent / "outside.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RunError):
                training.build_multigpu_beam_handoff(**outside)

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
            "feature_execution": "used_for_model_scoring",
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
    CASE_IDS = ["case-a", "case-b"]
    PANEL_INDICES = [3, 8]
    PANEL = "test"
    ROOTS = {
        "case-a": {
            "panel_index": 3,
            "dimension": 3,
            "alphabet_id": "test-cnot-n3",
            "start_packed_rows": [1, 3, 4],
        },
        "case-b": {
            "panel_index": 8,
            "dimension": 3,
            "alphabet_id": "test-cnot-n3",
            "start_packed_rows": [1, 2, 5],
        },
    }

    @staticmethod
    def solved(case_id, cost=1):
        start = [1, 3, 4] if case_id.endswith("a") else [1, 2, 5]
        gate = [0, 1] if case_id.endswith("a") else [0, 2]
        gates = [gate]
        while len(gates) < cost:
            gates.extend([[0, 1], [0, 1]])
        gates = gates[:cost]
        identity = training._identity_packed_rows(3)
        # Keep physical cost honest and replayable for every fixture length.
        if len(gates) % 2 == 0:
            gates.append(gate)
        return {
            "case_id": case_id,
            "panel_index": 3 if case_id == "case-a" else 8,
            "dimension": 3,
            "width": 3,
            "alphabet_id": "test-cnot-n3",
            "wall_seconds": 0.01,
            "status": "solved",
            "start_packed_rows": start,
            "end_packed_rows": identity,
            "word": {
                "replayed": True,
                "physical_cost": len(gates),
                "control_target": gates,
            },
        }

    @staticmethod
    def unsolved(case_id):
        root = QualityComparisonTests.ROOTS[case_id]
        return {
            "case_id": case_id,
            "panel_index": root["panel_index"],
            "dimension": root["dimension"],
            "width": root["dimension"],
            "alphabet_id": root["alphabet_id"],
            "wall_seconds": 0.01,
            "status": "depth_limit",
            "start_packed_rows": root["start_packed_rows"].copy(),
            "end_packed_rows": None,
            "word": None,
        }

    @staticmethod
    def pins(head, char):
        values = {
            "search_manifest_sha256": char * 64,
            "dataset_sha256": "d" * 64,
            "source_revision": "a" * 40,
            "runtime_contract": "native_device_search_v13",
            "training_manifest_sha256": None,
            "training_result_sha256": None,
            "checkpoint_sha256": None,
        }
        if head != "hamming":
            values.update({
                "training_manifest_sha256": "e" * 64,
                "training_result_sha256": "f" * 64,
                "checkpoint_sha256": "9" * 64,
            })
        return values

    @classmethod
    def report(cls, rows, *, search=None, panel="test", head="qhat", manifest_char=None):
        if manifest_char is None:
            manifest_char = "1" if head == "hamming" else "2"
        panel_indices = cls.PANEL_INDICES.copy()
        panel_ids = cls.CASE_IDS.copy()
        pins = cls.pins(head, manifest_char)
        value = {
            "schema": (
                "cnot103_hamming_search_result_v1"
                if head == "hamming"
                else "cnot103_panel_search_result_v2"
            ),
            "panel": panel,
            "panel_digest": training._panel_digest(panel, panel_indices, panel_ids),
            "panel_indices": panel_indices,
            "panel_ids": panel_ids,
            "search_manifest_sha256": pins["search_manifest_sha256"],
            "dataset_sha256": pins["dataset_sha256"],
            "source_revision": pins["source_revision"],
            "runtime_contract": pins["runtime_contract"],
            "panel_roots": [
                {
                    "panel_index": cls.ROOTS[case_id]["panel_index"],
                    "case_id": case_id,
                    "dimension": cls.ROOTS[case_id]["dimension"],
                    "alphabet_id": cls.ROOTS[case_id]["alphabet_id"],
                    "start_packed_rows": cls.ROOTS[case_id]["start_packed_rows"],
                }
                for case_id in cls.CASE_IDS
            ] if head == "hamming" else None,
            "search": training.BEAM_CONFIG if search is None else search,
            "head": head,
            "execution_backend": "cpu_hamming" if head == "hamming" else "cuda_model",
            "rows": rows,
        }
        if head == "qhat":
            value.update({
                "training_manifest_sha256": pins["training_manifest_sha256"],
                "training_result_sha256": pins["training_result_sha256"],
                "checkpoint": {"sha256": pins["checkpoint_sha256"]},
                "feature_model_inputs": 102,
                "feature_execution": "used_for_model_scoring",
                "training_exposure_audit": {"strict_unseen": True},
            })
        return value

    def compare(
        self, baseline_rows, candidate_rows, *,
        baseline_search=None, candidate_search=None,
        baseline_head="qhat", candidate_head="qhat",
        minimum_baseline_solved=1, minimum_baseline_widths=1,
        baseline_manifest_char="1", candidate_manifest_char="2",
    ):
        baseline = self.report(
            baseline_rows, search=baseline_search, head=baseline_head,
            manifest_char=baseline_manifest_char,
        )
        candidate = self.report(
            candidate_rows, search=candidate_search, head=candidate_head,
            manifest_char=candidate_manifest_char,
        )
        return training.compare_quality(
            baseline,
            candidate,
            expected_case_ids=self.CASE_IDS,
            expected_panel_indices=self.PANEL_INDICES,
            expected_case_roots=self.ROOTS,
            expected_panel="test",
            expected_panel_digest=training._panel_digest(
                "test", self.PANEL_INDICES, self.CASE_IDS
            ),
            expected_search=training.BEAM_CONFIG,
            baseline_receipt_pins=self.pins(baseline_head, baseline_manifest_char),
            candidate_receipt_pins=self.pins(candidate_head, candidate_manifest_char),
            baseline_head=baseline_head,
            candidate_head=candidate_head,
            minimum_baseline_solved=minimum_baseline_solved,
            minimum_baseline_widths=minimum_baseline_widths,
        )

    def test_full_baseline_and_shorter_candidate_is_length_gain(self):
        result = self.compare(
            [self.solved("case-a", 3), self.solved("case-b", 3)],
            [self.solved("case-a", 1), self.solved("case-b", 1)],
        )
        self.assertEqual(result["status"], "length_gain")
        self.assertEqual(result["baseline_solved_count"], 2)
        self.assertEqual(result["candidate_solved_count"], 2)
        self.assertAlmostEqual(result["relative_paired_cost_reduction"], 2 / 3)

    def test_same_paired_length_and_new_solution_is_coverage_gain(self):
        result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.solved("case-b", 1), self.solved("case-a", 1)],
        )
        self.assertEqual(result["status"], "coverage_gain")

    def test_exact_quality_equality_is_tie(self):
        result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.solved("case-a", 1), self.unsolved("case-b")],
        )
        self.assertEqual(result["status"], "tie")

    def test_candidate_that_loses_baseline_solution_fails(self):
        result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.unsolved("case-a"), self.solved("case-b", 1)],
        )
        self.assertEqual(result["status"], "failed")

    def test_more_expensive_common_solutions_fail_even_with_more_coverage(self):
        result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.solved("case-a", 3), self.solved("case-b", 1)],
        )
        self.assertEqual(result["status"], "failed")

    def test_empty_or_too_small_baseline_is_inconclusive(self):
        empty = self.compare(
            [self.unsolved("case-a"), self.unsolved("case-b")],
            [self.solved("case-a", 1), self.unsolved("case-b")],
        )
        too_small = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.solved("case-a", 1), self.solved("case-b", 1)],
            minimum_baseline_solved=2,
        )
        self.assertEqual(empty["status"], "inconclusive")
        self.assertEqual(too_small["status"], "inconclusive")

    def test_valid_row_permutation_is_paired_by_id_index_and_original_matrix(self):
        result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.unsolved("case-b"), self.solved("case-a", 1)],
        )
        self.assertEqual(result["status"], "tie")

    def test_declaration_order_must_match_even_when_rows_are_reordered(self):
        baseline = self.report(
            [self.solved("case-a", 1), self.unsolved("case-b")], head="hamming"
        )
        candidate = self.report(
            [self.solved("case-a", 1), self.unsolved("case-b")]
        )
        candidate["panel_ids"] = list(reversed(self.CASE_IDS))
        candidate["panel_indices"] = list(reversed(self.PANEL_INDICES))
        result = training.compare_quality(
            baseline, candidate,
            expected_case_ids=self.CASE_IDS,
            expected_panel_indices=self.PANEL_INDICES,
            expected_case_roots=self.ROOTS,
            expected_panel="test",
            expected_panel_digest=training._panel_digest("test", self.PANEL_INDICES, self.CASE_IDS),
            expected_search=training.BEAM_CONFIG,
            baseline_receipt_pins=self.pins("hamming", "1"),
            candidate_receipt_pins=self.pins("qhat", "2"),
            baseline_head="hamming",
            candidate_head="qhat",
            minimum_baseline_solved=1,
            minimum_baseline_widths=1,
        )
        self.assertEqual(result["status"], "invalid")

    def test_manifest_dataset_checkpoint_case_index_and_root_pins_are_checked(self):
        baseline_rows = [self.solved("case-a", 1), self.unsolved("case-b")]
        candidate_rows = [self.solved("case-a", 1), self.unsolved("case-b")]
        cases = [
            ("search_manifest_sha256", lambda r: r.__setitem__("search_manifest_sha256", "8" * 64)),
            ("dataset_sha256", lambda r: r.__setitem__("dataset_sha256", "7" * 64)),
            ("checkpoint", lambda r: r["checkpoint"].__setitem__("sha256", "6" * 64)),
            ("panel_index", lambda r: r["rows"][0].__setitem__("panel_index", 99)),
            ("case_root", lambda r: r["rows"][0]["start_packed_rows"].__setitem__(0, 2)),
            ("case_id", lambda r: r["rows"][0].__setitem__("case_id", "wrong-case")),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                baseline = self.report(baseline_rows.copy(), head="hamming")
                candidate = self.report(candidate_rows.copy())
                mutate(candidate)
                result = training.compare_quality(
                    baseline, candidate,
                    expected_case_ids=self.CASE_IDS,
                    expected_panel_indices=self.PANEL_INDICES,
                    expected_case_roots=self.ROOTS,
                    expected_panel="test",
                    expected_panel_digest=training._panel_digest(
                        "test", self.PANEL_INDICES, self.CASE_IDS
                    ),
                    expected_search=training.BEAM_CONFIG,
                    baseline_receipt_pins=self.pins("hamming", "1"),
                    candidate_receipt_pins=self.pins("qhat", "2"),
                    baseline_head="hamming",
                    candidate_head="qhat",
                    minimum_baseline_solved=1,
                    minimum_baseline_widths=1,
                )
                self.assertEqual(result["status"], "invalid")

    def test_wrong_scorer_or_matched_budget_is_invalid(self):
        wrong_backend = self.report(
            [self.solved("case-a", 1), self.unsolved("case-b")], head="qhat"
        )
        wrong_backend["execution_backend"] = "cpu_hamming"
        bad = training.compare_quality(
            self.report([self.solved("case-a", 1), self.unsolved("case-b")], head="hamming"),
            wrong_backend,
            expected_case_ids=self.CASE_IDS,
            expected_panel_indices=self.PANEL_INDICES,
            expected_case_roots=self.ROOTS,
            expected_panel="test",
            expected_panel_digest=training._panel_digest("test", self.PANEL_INDICES, self.CASE_IDS),
            expected_search=training.BEAM_CONFIG,
            baseline_receipt_pins=self.pins("hamming", "1"),
            candidate_receipt_pins=self.pins("qhat", "2"),
            baseline_head="hamming",
            candidate_head="qhat",
            minimum_baseline_solved=1,
            minimum_baseline_widths=1,
        )
        other = dict(training.BEAM_CONFIG)
        other["max_depth"] += 1
        mismatched = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [self.solved("case-a", 1), self.unsolved("case-b")],
            candidate_search=other,
        )
        self.assertEqual(bad["status"], "invalid")
        self.assertEqual(mismatched["status"], "invalid")

    def test_independent_replay_and_physical_cost_validation(self):
        broken_replay = self.solved("case-a", 1)
        broken_replay["word"]["replayed"] = False
        broken_cost = self.solved("case-a", 1)
        broken_cost["word"]["physical_cost"] = 2
        replay_result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [broken_replay, self.unsolved("case-b")],
        )
        cost_result = self.compare(
            [self.solved("case-a", 1), self.unsolved("case-b")],
            [broken_cost, self.unsolved("case-b")],
        )
        self.assertEqual(replay_result["status"], "invalid")
        self.assertEqual(cost_result["status"], "invalid")


class BaselineGateTests(unittest.TestCase):
    @staticmethod
    def panel_report(*, solved_by_width, widths=None, invalid_replay=False):
        widths = list(widths or sorted(solved_by_width))
        rows = []
        case_ids = []
        panel_indices = []
        case_roots = {}
        index = 0
        for width in widths:
            solved_count = solved_by_width.get(width, 0)
            for offset in range(2):
                case_id = f"dev-n{width}-{offset}"
                case_ids.append(case_id)
                panel_indices.append(index)
                identity = training._identity_packed_rows(width)
                start = identity.copy()
                start[1] ^= start[0]
                alphabet_id = f"test-cnot-n{width}"
                case_roots[case_id] = {
                    "panel_index": index,
                    "dimension": width,
                    "alphabet_id": alphabet_id,
                    "start_packed_rows": start,
                }
                solved = offset < solved_count
                gate = [0, 1]
                if invalid_replay and solved and not rows:
                    gate = [0, 2]
                rows.append({
                    "case_id": case_id,
                    "panel_index": index,
                    "dimension": width,
                    "width": width,
                    "alphabet_id": alphabet_id,
                    "wall_seconds": 0.01,
                    "status": "solved" if solved else "depth_limit",
                    "start_packed_rows": start,
                    "end_packed_rows": identity if solved else None,
                    "word": ({
                        "replayed": True,
                        "physical_cost": 1,
                        "control_target": [gate],
                    } if solved else None),
                })
                index += 1
        indices = panel_indices.copy()
        ids = case_ids.copy()
        pins = {
            "search_manifest_sha256": "1" * 64,
            "dataset_sha256": "d" * 64,
            "source_revision": "a" * 40,
            "runtime_contract": "native_device_search_v13",
        }
        report = {
            "schema": "cnot103_hamming_search_result_v1",
            "panel": "dev",
            "panel_digest": training._panel_digest("dev", indices, ids),
            "panel_indices": indices,
            "panel_ids": ids,
            "search_manifest_sha256": pins["search_manifest_sha256"],
            "dataset_sha256": pins["dataset_sha256"],
            "source_revision": pins["source_revision"],
            "runtime_contract": pins["runtime_contract"],
            "panel_roots": [
                {
                    "panel_index": case_roots[case_id]["panel_index"],
                    "case_id": case_id,
                    "dimension": case_roots[case_id]["dimension"],
                    "alphabet_id": case_roots[case_id]["alphabet_id"],
                    "start_packed_rows": case_roots[case_id]["start_packed_rows"],
                }
                for case_id in case_ids
            ],
            "search": training.BEAM_CONFIG,
            "head": "hamming",
            "execution_backend": "cpu_hamming",
            "rows": rows,
        }
        return report, case_ids, panel_indices, case_roots, pins, report["panel_digest"]

    def gate(self, report, case_ids, indices, roots, pins, digest):
        return training.evaluate_baseline_gate(
            report,
            expected_case_ids=case_ids,
            expected_panel_indices=indices,
            expected_case_roots=roots,
            expected_panel_digest=digest,
            expected_search=training.BEAM_CONFIG,
            expected_receipt_pins=pins,
            minimum_solved=training.BASELINE_MIN_SOLVED,
            minimum_widths=training.BASELINE_MIN_WIDTHS,
        )

    def test_zero_solved_hamming_baseline_is_inconclusive(self):
        report, ids, indices, roots, pins, digest = self.panel_report(
            solved_by_width={}, widths=[10, 13]
        )
        self.assertEqual(self.gate(report, ids, indices, roots, pins, digest)["status"], "inconclusive")

    def test_trivial_baseline_below_threshold_is_inconclusive(self):
        report, ids, indices, roots, pins, digest = self.panel_report(
            solved_by_width={10: 1, 13: 1}, widths=[10, 13]
        )
        self.assertEqual(self.gate(report, ids, indices, roots, pins, digest)["status"], "inconclusive")

    def test_replay_invalid_hamming_baseline_is_rejected(self):
        report, ids, indices, roots, pins, digest = self.panel_report(
            solved_by_width={10: 1}, widths=[10], invalid_replay=True
        )
        self.assertEqual(self.gate(report, ids, indices, roots, pins, digest)["status"], "invalid")

    def test_frozen_nontrivial_dev_baseline_passes(self):
        widths = [10, 13, 16, 20, 24, 32, 40]
        report, ids, indices, roots, pins, digest = self.panel_report(
            solved_by_width={width: 2 for width in widths}, widths=widths
        )
        result = self.gate(report, ids, indices, roots, pins, digest)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["solved_count"], 14)
        self.assertEqual(result["solved_width_count"], 7)

    def test_baseline_gate_pins_and_root_identity_reject_tampering(self):
        report, ids, indices, roots, pins, digest = self.panel_report(
            solved_by_width={10: 2, 13: 2}, widths=[10, 13]
        )
        report["search_manifest_sha256"] = "9" * 64
        result = self.gate(report, ids, indices, roots, pins, digest)
        self.assertEqual(result["status"], "invalid")

    def test_full_training_refuses_nonpassed_baseline_before_calling_train(self):
        called = []
        with self.assertRaises(training.RunError):
            training.train_after_baseline_gate(
                {"status": "inconclusive"},
                lambda: called.append(True),
            )
        self.assertFalse(called)


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
                        "dev": [{"lineage_id": "dev-case-1", "state": {"n": 2, "alphabet_id": "synthetic-cnot-n2", "packed_rows": [1, 3]}}],
                        "test": [{"lineage_id": "test-case-1", "state": {"n": 2, "alphabet_id": "synthetic-cnot-n2", "packed_rows": [1, 3]}}],
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
            elif name == "cargo_cpu_native_build":
                validator = supervisor._runner("cnot103-validate-data")
                hamming = supervisor._runner("cnot103-hamming-search")
                validator.parent.mkdir(parents=True, exist_ok=True)
                validator.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                validator.chmod(0o700)
                hamming.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                hamming.chmod(0o700)
            elif name == "cargo_release_build":
                return None
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

        def cpu_hamming_search(*_args, **_kwargs):
            stages.append("cpu_hamming_search")
            return {}, {"receipt_pins": {}}, 0.01

        with (
            patch.object(supervisor, "_check_source"),
            patch.object(supervisor, "_hardware", return_value={"torch_cuda_devices": 2}),
            patch.object(supervisor, "_free_gib", return_value=100.0),
            patch.object(supervisor, "_ensure_cargo"),
            patch.object(supervisor, "_run", side_effect=run_stage),
            patch.object(supervisor, "_search", side_effect=cpu_hamming_search),
            patch.object(training, "evaluate_baseline_gate", return_value={"status": "passed"}),
            patch.object(supervisor, "_feature_coverage", side_effect=coverage),
        ):
            result = supervisor.run()
        return result, stages, coverage_calls, root

    def test_public_run_builds_validator_then_validates_seals_and_reaches_coverage(self):
        result, stages, coverage, root = self._run_to_coverage(validation_ok=True)

        self.assertEqual(result, 1)  # stopped deliberately at the coverage boundary
        self.assertLess(stages.index("cargo_cpu_native_build"), stages.index("validate_frozen_dataset"))
        self.assertLess(stages.index("validate_frozen_dataset"), stages.index("finalize_frozen_dataset"))
        self.assertLess(stages.index("cpu_hamming_search"), stages.index("cargo_release_build"))
        self.assertTrue(coverage)
        receipt = json.loads((root / training.DATA_RECEIPT_RELATIVE).read_text(encoding="utf-8"))
        self.assertTrue(receipt["complete"])

    def test_failed_rust_validation_never_seals_receipt_or_reaches_coverage(self):
        result, stages, coverage, root = self._run_to_coverage(validation_ok=False)

        self.assertEqual(result, 1)
        self.assertLess(stages.index("cargo_cpu_native_build"), stages.index("validate_frozen_dataset"))
        self.assertNotIn("cargo_release_build", stages)
        self.assertNotIn("finalize_frozen_dataset", stages)
        self.assertFalse(coverage)
        receipt = json.loads((root / training.DATA_RECEIPT_RELATIVE).read_text(encoding="utf-8"))
        self.assertFalse(receipt["complete"])



class MeasurementCurveTests(unittest.TestCase):
    @staticmethod
    def valid_spec():
        return {
            "schema": "cnot103_measurement_experiment_v3",
            "experiment_id": "learning-curve-complete-dev-v3",
            "bindings": {
                "source": "workflow_head_sha",
                "dataset": "sealed_dataset_sha256",
                "holdout": ["holdout_revision", "holdout_sha256"],
                "panels": "cnot103_panel_cohorts_v1",
            },
            "splits": {
                "policy": "stratified_per_width",
                "cases_per_width": 8,
                "order": ["train", "dev", "test"],
                "locked_test": "after_complete_dev_sweep",
            },
            "model": {
                "parameters": 136_421_807,
                "structural_blocks": 8,
                "input_slots": 102,
                "batch_size": 4,
                "feature_schema": "runtime_packet_layout_v25",
            },
            "speed_pilot": {
                "attempt_cap": 32,
                "mode": "independent_discard_only",
                "requires_all_updates_accepted": True,
            },
            "training": {
                "seed": 202611200001,
                "max_attempts": 512,
                "accepted_milestones": [8, 32, 128, 512],
            },
            "search": training.BEAM_CONFIG.copy(),
            "hardware": {
                "selection": "controller_approved_target",
                "gpu_count": 2,
                "record_actual_devices": True,
            },
            "resource_limits": {
                "max_active_seconds": training.MAX_RUNTIME_SECONDS,
                "max_total_usd": 20.0,
            },
            "quality": {
                "minimum_solved_rate_gain": 0.05,
                "minimum_relative_paired_cost_reduction": 0.05,
            },
        }

    def test_versioned_measurement_spec_is_bounded_and_keeps_the_full_contract(self):
        loader = getattr(training, "load_measurement_spec", None)
        self.assertTrue(callable(loader), "measurement spec loader is missing")
        spec = loader()
        self.assertEqual(spec["schema"], "cnot103_measurement_experiment_v3")
        self.assertEqual(spec["experiment_id"], "learning-curve-complete-dev-v3")
        self.assertEqual(spec["speed_pilot"]["attempt_cap"], 32)
        self.assertEqual(spec["splits"]["locked_test"], "after_complete_dev_sweep")
        historical_spec = (
            pathlib.Path(__file__).resolve().parents[1]
            / "training-experiments"
            / "learning-curve-first-v1.json"
        )
        self.assertEqual(
            json.loads(historical_spec.read_text(encoding="utf-8"))["schema"],
            "cnot103_measurement_experiment_v1",
        )
        preserved_v2 = historical_spec.with_name("learning-curve-complete-dev-v2.json")
        self.assertEqual(
            json.loads(preserved_v2.read_text(encoding="utf-8"))["schema"],
            "cnot103_measurement_experiment_v2",
        )
        self.assertEqual(spec["training"]["max_attempts"], 512)
        self.assertEqual(spec["training"]["accepted_milestones"], [8, 32, 128, 512])
        self.assertNotEqual(spec["training"]["max_attempts"], 250_000)
        self.assertEqual(spec["model"]["parameters"], training.MODEL_PARAMETERS)
        self.assertEqual(spec["model"]["structural_blocks"], training.STRUCTURAL_BLOCKS)
        self.assertEqual(spec["model"]["input_slots"], training.MODEL_INPUTS)
        self.assertEqual(spec["search"], training.BEAM_CONFIG)

    def test_measurement_spec_rejects_bad_milestones_and_independent_resource_caps(self):
        validate = getattr(training, "validate_measurement_spec", None)
        self.assertTrue(callable(validate), "measurement spec validator is missing")
        duplicate = self.valid_spec()
        duplicate["training"]["accepted_milestones"] = [8, 8, 32]
        with self.assertRaises(RunError):
            validate(duplicate)
        over_wall = self.valid_spec()
        over_wall["resource_limits"]["max_active_seconds"] += 1
        with self.assertRaises(RunError):
            validate(over_wall)
        over_spend = self.valid_spec()
        over_spend["resource_limits"]["max_total_usd"] = 20.01
        with self.assertRaises(RunError):
            validate(over_spend)
        wrong_pilot = self.valid_spec()
        wrong_pilot["speed_pilot"]["attempt_cap"] = 500
        with self.assertRaises(RunError):
            validate(wrong_pilot)

    def test_training_progress_delta_reports_attempts_accepts_rejects_and_examples(self):
        delta = getattr(training, "training_progress_delta", None)
        self.assertTrue(callable(delta), "training progress accounting is missing")
        result = delta(
            {"attempted": 2, "accepted": 2, "rejected": 0},
            {"attempted": 8, "accepted": 7, "rejected": 1},
            batch_size=4,
        )
        self.assertEqual(result, {
            "attempted_updates": 6,
            "accepted_updates": 5,
            "rejected_updates": 1,
            "examples_seen": 24,
            "accepted_examples": 20,
            "rejected_examples": 4,
        })

    def test_training_progress_delta_uses_observed_examples_for_a_tail_batch(self):
        delta = training.training_progress_delta(
            {
                "attempted": 0, "accepted": 0, "rejected": 0,
                "examples_seen": 0, "accepted_examples": 0, "rejected_examples": 0,
            },
            {
                "attempted": 2, "accepted": 2, "rejected": 0,
                "examples_seen": 5, "accepted_examples": 5, "rejected_examples": 0,
            },
            batch_size=4,
        )
        self.assertEqual(delta["attempted_updates"], 2)
        self.assertEqual(delta["examples_seen"], 5)
        self.assertEqual(delta["accepted_examples"], 5)

    def test_malformed_measurement_spec_stops_before_dataset_or_optimizer_bootstrap(self):
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.source = pathlib.Path("unused-source")
        supervisor.status = unittest.mock.Mock()
        supervisor._private_failure = unittest.mock.Mock()
        supervisor._check_source = unittest.mock.Mock()
        supervisor._build_dataset = unittest.mock.Mock()
        supervisor._build_cpu_native = unittest.mock.Mock()
        supervisor._build_native = unittest.mock.Mock()
        supervisor._benchmark = unittest.mock.Mock()
        supervisor._hardware = unittest.mock.Mock()
        with patch.object(training, "load_measurement_spec", side_effect=RunError("bad spec")):
            result = supervisor.run()
        self.assertEqual(result, 1)
        supervisor._check_source.assert_called_once_with()
        supervisor._build_dataset.assert_not_called()
        supervisor._build_cpu_native.assert_not_called()
        supervisor._build_native.assert_not_called()
        supervisor._benchmark.assert_not_called()
        supervisor._hardware.assert_not_called()
        supervisor._private_failure.assert_called_once()

    def test_rejected_update_writes_rereadable_partial_receipt_and_withholds_test(self):
        saved = []
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "partial.json"

            def persist_partial(receipt):
                path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
                saved.append(receipt)

            locked_test_calls = []
            result = training.run_measurement_sweep(
                [4],
                initial_progress={"attempted": 0, "accepted": 0, "rejected": 0},
                max_attempts=4,
                train_to=lambda milestone, start: {
                    "start_progress": start,
                    "end_progress": {"attempted": 4, "accepted": 3, "rejected": 1},
                    "checkpoint": {"sha256": "b" * 64},
                    "training_wall_seconds": 1.0,
                },
                search_dev=lambda _report: {"wall_seconds": 0.1},
                compare_dev=lambda _report, _dev: {"status": "tie"},
                persist_receipt=lambda _receipt: self.fail("incomplete milestone must not persist as complete"),
                locked_test=lambda selected: locked_test_calls.append(selected),
                initial_checkpoint={"sha256": "a" * 64, "path": "checkpoints/step-000000.json"},
                persist_partial_receipt=persist_partial,
            )
            reread = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["locked_test"])
        self.assertEqual(locked_test_calls, [])
        self.assertEqual(len(saved), 1)
        self.assertEqual(reread["schema"], "cnot103_learning_curve_partial_v1")
        self.assertEqual(reread["failed_stage"], "training_incomplete")
        self.assertEqual(reread["last_valid_progress"], {"attempted": 0, "accepted": 0, "rejected": 0})
        self.assertEqual(reread["last_valid_checkpoint"]["sha256"], "a" * 64)
        self.assertEqual(reread["observed_progress"], {"attempted": 4, "accepted": 3, "rejected": 1})



    def test_all_dev_checkpoints_run_before_one_locked_test_and_keep_first_gain(self):
        sweep = getattr(training, "run_measurement_sweep", None)
        self.assertTrue(callable(sweep), "measurement sweep runner is missing")
        events = []
        statuses = {8: "length_gain", 32: "coverage_gain", 128: "tie", 512: "no_practical_gain"}
        initial = {"attempted": 2, "accepted": 2, "rejected": 0}
        checkpoints = {milestone: str(milestone % 10) * 64 for milestone in statuses}
        previous_progress = [initial]

        def train_to(milestone, prior_progress):
            events.append(("train", milestone))
            self.assertEqual(prior_progress, previous_progress[0])
            end = {"attempted": milestone, "accepted": milestone, "rejected": 0}
            previous_progress[0] = end
            return {
                "start_progress": prior_progress,
                "end_progress": end,
                "checkpoint": {"sha256": checkpoints[milestone]},
                "training_wall_seconds": float(milestone),
            }

        def search_dev(training_result):
            milestone = training_result["end_progress"]["accepted"]
            events.append(("dev", milestone))
            return {"wall_seconds": 2.0, "solved_count": milestone}

        def compare_dev(training_result, dev_result):
            milestone = training_result["end_progress"]["accepted"]
            events.append(("compare", milestone))
            return {"status": statuses[milestone], "reason": statuses[milestone]}

        def persist(receipt):
            events.append(("persist", receipt["accepted_updates"]))

        def locked_test(selected):
            events.append(("test", selected["accepted_updates"], selected["checkpoint_sha256"]))
            return {"status": "evaluated", "checkpoint_sha256": selected["checkpoint_sha256"]}

        milestones = [8, 32, 128, 512]
        result = sweep(
            milestones,
            initial_progress=initial,
            max_attempts=512,
            train_to=train_to,
            search_dev=search_dev,
            compare_dev=compare_dev,
            persist_receipt=persist,
            locked_test=locked_test,
        )
        self.assertEqual([row["accepted_updates"] for row in result["measurements"]], milestones)
        self.assertEqual(result["schema"], "cnot103_learning_curve_v2")
        self.assertEqual(result["selected"]["accepted_updates"], 8)
        self.assertEqual(result["selected"]["checkpoint_sha256"], checkpoints[8])
        self.assertEqual(result["locked_test"]["status"], "evaluated")
        self.assertEqual(sum(1 for event in events if event[0] == "test"), 1)
        self.assertGreater(events.index(("test", 8, checkpoints[8])), events.index(("persist", 512)))
        self.assertEqual([event[1] for event in events if event[0] == "train"], milestones)
        self.assertEqual([event[1] for event in events if event[0] == "compare"], milestones)
        self.assertEqual(result["final_progress"]["accepted"], 512)

    def test_measurement_sweep_withholds_locked_test_when_no_dev_checkpoint_gains(self):
        sweep = getattr(training, "run_measurement_sweep", None)
        self.assertTrue(callable(sweep), "measurement sweep runner is missing")
        tested = []
        initial = {"attempted": 0, "accepted": 0, "rejected": 0}

        def train_to(milestone, prior_progress):
            end = {"attempted": milestone, "accepted": milestone, "rejected": 0}
            return {
                "start_progress": prior_progress,
                "end_progress": end,
                "checkpoint": {"sha256": str(milestone % 10) * 64},
                "training_wall_seconds": 1.0,
            }

        result = sweep(
            [8, 32],
            initial_progress=initial,
            max_attempts=32,
            train_to=train_to,
            search_dev=lambda report: {"wall_seconds": 1.0},
            compare_dev=lambda report, dev: {"status": "tie"},
            persist_receipt=lambda receipt: None,
            locked_test=lambda selected: tested.append(selected),
        )
        self.assertIsNone(result["selected"])
        self.assertIsNone(result["locked_test"])
        self.assertEqual(tested, [])
        self.assertEqual(len(result["measurements"]), 2)

    def test_remaining_forecast_counts_only_unfinished_work(self):
        estimate = getattr(training, "estimate_measurement_remaining_work_seconds", None)
        self.assertTrue(callable(estimate), "measurement forecast is missing")
        remaining = estimate(
            remaining_attempts=40,
            seconds_per_attempt=2.0,
            future_dev_evaluations=2,
            dev_search_seconds=11.0,
            locked_search_seconds=17.0,
            checkpoint_reserve_seconds=5.0,
            cleanup_reserve_seconds=3.0,
        )
        self.assertEqual(remaining, 127.0)


if __name__ == "__main__":
    unittest.main()
