import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
