# Red phase: these contract tests intentionally precede the controller implementation.
import unittest

import vast_rent_rebuild_dataset as controller


def offer(offer_id, **updates):
    row = {
        "ask_id": offer_id,
        "gpu_name": "RTX 3060",
        "num_gpus": 2,
        "gpu_ram": 12288,
        "reliability": 0.99,
        "dph_total": 0.2,
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
        ]

        selected = controller.eligible_offers(rows)

        self.assertEqual([row["ask_id"] for row in selected], [10])

    def test_budget_gate_includes_50h_and_transfer_reserve(self):
        self.assertTrue(controller.budget_allows(5.0, 0.24, run_hours=50, reserve=3.0))
        self.assertFalse(controller.budget_allows(5.0, 0.241, run_hours=50, reserve=3.0))

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


if __name__ == "__main__":
    unittest.main()
