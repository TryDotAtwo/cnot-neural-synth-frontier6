import json
import tempfile
import unittest
from pathlib import Path

from cnot_cpu_torch_ci import compile_evidence, gate_receipt


class CpuTorchCiTests(unittest.TestCase):
    def test_compile_receipt_requires_the_expected_executable_artifact(self):
        messages = [
            {
                "reason": "compiler-artifact",
                "target": {"name": "cnot103", "kind": ["lib"], "test": True},
                "executable": "/runner/tmp/cnot103-tests",
            }
        ]
        result = compile_evidence(
            "\n".join(json.dumps(item) for item in messages),
            0,
            "lib_test",
        )
        self.assertEqual(result["status"], "compiled")
        self.assertEqual(result["executable"], "/runner/tmp/cnot103-tests")

    def test_compile_receipt_exposes_only_rustc_code_when_compilation_fails(self):
        message = {
            "reason": "compiler-message",
            "message": {
                "level": "error",
                "code": {"code": "E0425"},
                "message": "private compiler detail is not part of the receipt",
            },
        }
        result = compile_evidence(json.dumps(message), 101, "lib_test")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["rustc_code"], "E0425")
        self.assertNotIn("private compiler detail", json.dumps(result))

    def test_one_failed_check_does_not_hide_other_executed_results(self):
        outcomes = {
            "torch": "success",
            "v13_compile": "compiled",
            "v13": "failed",
            "resume_compile": "compiled",
            "resume": "passed",
            "profile_compile": "compiled",
            "profile": "passed",
        }
        receipt = gate_receipt(outcomes)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["outcomes"]["v13"], "failed")
        self.assertEqual(receipt["outcomes"]["resume"], "passed")
        self.assertEqual(receipt["outcomes"]["profile"], "passed")

    def test_missing_outcome_keeps_the_overall_gate_closed(self):
        receipt = gate_receipt({"torch": "success", "v13_compile": "compiled", "v13": "passed"})
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["outcomes"]["resume"], "not_run")


if __name__ == "__main__":
    unittest.main()
