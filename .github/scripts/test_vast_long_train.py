import unittest

import vast_long_train as controller


class BudgetTests(unittest.TestCase):
    def test_budget_allows_spend_plus_remaining_hours_and_reserve_under_cap(self):
        self.assertTrue(controller.budget_allows(5.0, 0.25, 52, 2.0))
        self.assertFalse(controller.budget_allows(5.0, 0.26, 52, 2.0))

    def test_budget_rejects_nonfinite_negative_boolean_and_string_values(self):
        invalid = [
            float("nan"),
            float("inf"),
            -0.01,
            True,
            "1.0",
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.assertFalse(controller.budget_allows(value, 0.1, 10, 1.0))
                self.assertFalse(controller.budget_allows(0.0, value, 10, 1.0))
                self.assertFalse(controller.budget_allows(0.0, 0.1, value, 1.0))
                self.assertFalse(controller.budget_allows(0.0, 0.1, 10, value))


class ReportingTests(unittest.TestCase):
    def test_run_state_accepts_only_the_five_allowlisted_states(self):
        allowed = ("not_started", "running", "succeeded", "failed", "unknown")
        for state in allowed:
            with self.subTest(state=state):
                self.assertEqual(controller.classify_run_state({"state": state}), state)
        self.assertEqual(controller.classify_run_state({"state": "training_with_metrics"}), "unknown")
        self.assertEqual(controller.classify_run_state(None), "unknown")

    def test_redacted_report_contains_only_time_and_state(self):
        report = controller.redacted_report("running", "2026-11-20T00:00:00Z")
        self.assertEqual(
            report,
            "checked_at_utc=2026-11-20T00:00:00Z\ntraining_status=running",
        )
        self.assertNotIn("loss", report)
        self.assertNotIn("sha", report)
        self.assertNotIn("quality", report)


if __name__ == "__main__":
    unittest.main()
