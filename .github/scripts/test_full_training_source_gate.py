import unittest
from pathlib import Path

from full_training_source_gate import REQUIRED_CHECKS, evaluate_source_gate


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "0123456789abcdef0123456789abcdef01234567"


def valid_checks():
    return {
        name: {"result": "success", "checked_sha": SOURCE_SHA}
        for name in REQUIRED_CHECKS
    }


class SourceGateTests(unittest.TestCase):
    def test_three_exact_successes_are_a_positive_dry_run(self):
        self.assertEqual(("passed", "none"), evaluate_source_gate(SOURCE_SHA, valid_checks()))

    def test_stale_sha_is_rejected(self):
        checks = valid_checks()
        checks["cpu_runtime"]["checked_sha"] = "1123456789abcdef0123456789abcdef01234567"
        self.assertEqual(("failed", "stale_source_sha"), evaluate_source_gate(SOURCE_SHA, checks))

    def test_missing_check_and_missing_sha_are_rejected(self):
        checks = valid_checks()
        checks.pop("cuda_build")
        self.assertEqual(("failed", "missing_check"), evaluate_source_gate(SOURCE_SHA, checks))
        checks = valid_checks()
        checks["manifest"]["checked_sha"] = ""
        self.assertEqual(("failed", "missing_source_sha"), evaluate_source_gate(SOURCE_SHA, checks))

    def test_skipped_and_failed_checks_are_rejected(self):
        for result, reason in (("skipped", "source_check_skipped"),
                               ("failure", "source_check_failed")):
            with self.subTest(result=result):
                checks = valid_checks()
                checks["manifest"]["result"] = result
                self.assertEqual(("failed", reason), evaluate_source_gate(SOURCE_SHA, checks))

    def test_pending_checks_are_rejected(self):
        for result in ("queued", "pending", "in_progress"):
            with self.subTest(result=result):
                checks = valid_checks()
                checks["cuda_build"]["result"] = result
                self.assertEqual(
                    ("failed", "source_check_pending"),
                    evaluate_source_gate(SOURCE_SHA, checks),
                )

    def test_timeout_and_unknown_results_are_rejected(self):
        for result, reason in (("timed_out", "source_check_timeout"),
                               ("cancelled", "source_check_not_successful")):
            with self.subTest(result=result):
                checks = valid_checks()
                checks["cpu_runtime"]["result"] = result
                self.assertEqual(("failed", reason), evaluate_source_gate(SOURCE_SHA, checks))

    def test_parent_wires_all_three_same_run_reusable_checks_to_start(self):
        parent = (ROOT / ".github/workflows/vast-full-training.yml").read_text(encoding="utf-8")
        required_workflows = {
            "source_manifest": "check-full-training-manifest.yml",
            "source_cpu_runtime": "check-full-training-cpu-runtime.yml",
            "source_cuda_build": "check-full-training-cuda-build.yml",
        }
        for job, workflow in required_workflows.items():
            with self.subTest(job=job):
                self.assertIn(f"{job}:", parent)
                self.assertIn(f"uses: ./.github/workflows/{workflow}", parent)
        self.assertGreaterEqual(parent.count("source_sha: ${{ github.sha }}"), 3)
        self.assertIn("needs: [source_manifest, source_cpu_runtime, source_cuda_build]", parent)
        for job in required_workflows:
            self.assertIn(f"needs.{job}.result", parent)
            self.assertIn(f"needs.{job}.outputs.checked_sha", parent)
        self.assertIn("needs.source_preflight.outputs.status == 'passed'", parent)
        self.assertIn("needs: source_preflight", parent)

    def test_each_required_workflow_checks_out_and_returns_the_requested_sha(self):
        workflows = {
            ".github/workflows/check-full-training-manifest.yml": "source",
            ".github/workflows/check-full-training-cpu-runtime.yml": "cpu-torch",
            ".github/workflows/check-full-training-cuda-build.yml": "cuda-build",
        }
        for path, job in workflows.items():
            with self.subTest(path=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                self.assertIn("workflow_call:", text)
                self.assertIn("source_sha:", text)
                self.assertIn("ref: ${{ inputs.source_sha || github.sha }}", text)
                self.assertIn(f"checked_sha: ${{{{ steps.source_revision.outputs.checked_sha }}}}", text)
                self.assertIn("git rev-parse HEAD", text)
                self.assertIn(f"jobs.{job}.outputs.checked_sha", text)

    def test_manifest_ci_runs_the_gate_regressions(self):
        workflow = (ROOT / ".github/workflows/check-full-training-manifest.yml").read_text(encoding="utf-8")
        self.assertIn("test_full_training_source_gate.py", workflow)


if __name__ == "__main__":
    unittest.main()
