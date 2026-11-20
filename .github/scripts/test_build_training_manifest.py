import json
import tempfile
import unittest
from pathlib import Path

from build_training_manifest import (
    FEATURE_SCHEMA,
    MODEL_INPUT_COUNT,
    TARGET_ATTEMPTS,
    WIDTHS,
    _self_test_profile,
    build_manifest,
    context_widths,
    validate_profile,
)


class TrainingManifestTests(unittest.TestCase):
    def setUp(self):
        self.profile = _self_test_profile()
        self.profile_bytes = json.dumps(self.profile, sort_keys=True).encode("utf-8")

    def test_uses_all_width_alphabets_and_full_training_contract(self):
        manifest, provenance = build_manifest(
            self.profile,
            self.profile_bytes,
            "/workspace/train/frozen_train_10k.json",
            "b" * 64,
            "/workspace/train/checkpoints",
            False,
        )
        self.assertEqual([item["dimension"] for item in manifest["alphabets"]], list(WIDTHS))
        for alphabet in manifest["alphabets"]:
            n = alphabet["dimension"]
            self.assertEqual(len(alphabet["gates"]), n * (n - 1))
            self.assertEqual(len(alphabet["costs"]), n * (n - 1))
        self.assertEqual(manifest["training"]["target_attempts"], 250_000)
        self.assertEqual(manifest["training"]["attempts"], 250_000)
        self.assertEqual(manifest["training"]["cuda_devices"], [0, 1])
        self.assertEqual(manifest["training"]["batch_size"], 4)
        self.assertEqual(
            manifest["model"],
            {
                "wire_dim": 744,
                "pair_dim": 186,
                "structural_blocks": 8,
                "attention_heads": 8,
                "feedforward_dim": 1488,
                "fusion_blocks": 2,
                "basis_proposal_hidden_dim": 256,
            },
        )
        self.assertEqual(manifest["feature_policy"]["schema_id"], FEATURE_SCHEMA)
        self.assertEqual(manifest["objective"]["sampled_neighbor_count"], 4)
        self.assertEqual(provenance["model_input_slots"], MODEL_INPUT_COUNT)
        self.assertEqual(provenance["target_attempts"], TARGET_ATTEMPTS)
        self.assertEqual(provenance["context_widths"]["selected_residual_widths"], [3])

    def test_context_profile_is_verbatim_and_reports_sparse_widths(self):
        contexts = validate_profile(self.profile)
        self.assertEqual(contexts, self.profile["feature_contexts"])
        widths = context_widths(contexts)
        self.assertEqual(widths["labeled_cuts"], [3])
        self.assertNotIn(10, widths["labeled_cuts"])
        self.assertEqual(widths["i10_max_n"], [3])

    def test_rejects_unpinned_or_malformed_profile(self):
        bad = json.loads(json.dumps(self.profile))
        bad["source"]["revision"] = "main"
        with self.assertRaises(ValueError):
            validate_profile(bad)
        bad = json.loads(json.dumps(self.profile))
        bad["feature_contexts"]["max_n"] = 10
        with self.assertRaises(ValueError):
            validate_profile(bad)

    def test_rejects_invalid_data_identity(self):
        with self.assertRaises(ValueError):
            build_manifest(
                self.profile,
                self.profile_bytes,
                "/workspace/train/data.json",
                "not-a-sha",
                "/workspace/train/checkpoints",
                False,
            )

    def test_resume_preserves_absolute_epoch_ceiling(self):
        manifest, _ = build_manifest(
            self.profile,
            self.profile_bytes,
            "/workspace/train/data.json",
            "b" * 64,
            "/workspace/train/checkpoints",
            True,
        )
        self.assertTrue(manifest["resume"])
        self.assertEqual(manifest["training"]["target_attempts"], 250_000)
        self.assertEqual(manifest["training"]["attempts"], 250_000)

    def test_cli_writes_manifest_and_private_provenance(self):
        manifest, provenance = build_manifest(
            self.profile,
            self.profile_bytes,
            "/workspace/train/data.json",
            "b" * 64,
            "/workspace/train/checkpoints",
            False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            sidecar = path.with_suffix(".json.provenance.json")
            from build_training_manifest import _write_private_json

            _write_private_json(path, manifest)
            _write_private_json(sidecar, provenance)
            self.assertEqual(json.loads(path.read_text())["training"]["target_attempts"], 250_000)
            self.assertEqual(json.loads(sidecar.read_text())["profile_id"], "self-test")


if __name__ == "__main__":
    unittest.main()
