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
                "spans": [
                    {
                        "file_name": "native/src/device_search/runtime.rs",
                        "line_start": 861,
                        "is_primary": True,
                    }
                ],
            },
        }
        result = compile_evidence(json.dumps(message), 101, "lib_test")
        self.assertEqual(result["status"], "compile_failed_runtime_tests")
        self.assertEqual(result["rustc_code"], "E0425")
        self.assertNotIn("private compiler detail", json.dumps(result))

    def test_compile_receipt_walks_child_and_macro_expansion_spans(self):
        message = {
            "reason": "compiler-message",
            "message": {
                "level": "error",
                "code": {"code": "E0599"},
                "message": "compiler detail remains private",
                "spans": [],
                "children": [
                    {
                        "level": "note",
                        "message": "nested compiler detail remains private",
                        "spans": [
                            {
                                "file_name": "/runner/work/project/native/src/device_search.rs",
                                "line_start": 88,
                                "is_primary": True,
                            }
                        ],
                    }
                ],
            },
        }
        result = compile_evidence(json.dumps(message), 101, "lib_test")
        self.assertEqual(result["status"], "compile_failed_device_search")
        self.assertEqual(result["rustc_code"], "E0599")
        self.assertNotIn("compiler detail", json.dumps(result))

    def test_unresolved_name_without_a_source_span_has_a_safe_class(self):
        message = {
            "reason": "compiler-message",
            "message": {
                "level": "error",
                "code": {"code": "E0425"},
                "message": "cannot find private_name in this scope",
                "spans": [],
            },
        }
        result = compile_evidence(json.dumps(message), 101, "lib_test")
        self.assertEqual(result["status"], "compile_failed_unresolved_name")
        self.assertEqual(result["rustc_code"], "E0425")
        self.assertNotIn("private_name", json.dumps(result))

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
