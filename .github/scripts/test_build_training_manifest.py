import copy
import hashlib
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
    validate_profile_source,
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
            attempts=250_000,
            target_attempts=TARGET_ATTEMPTS,
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
        interval = manifest["training"]["checkpoint_every_attempts"]
        self.assertGreater(interval, 0)
        checkpoint_count = (250_000 + interval - 1) // interval
        self.assertLessEqual(checkpoint_count, 10)
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
        self.assertEqual(provenance["profile_source_repository"], self.profile["source"]["repository"])
        self.assertEqual(provenance["profile_source_sha256"], self.profile["source"]["sha256"])
        self.assertEqual(provenance["context_widths"]["selected_residual_widths"], [8])

    def test_bounded_invocations_keep_the_absolute_250k_target(self):
        initial, initial_provenance = build_manifest(
            self.profile, self.profile_bytes,
            "/workspace/train/frozen_train_10k.json", "b" * 64,
            "/workspace/train/checkpoints", False,
            attempts=1, target_attempts=TARGET_ATTEMPTS,
        )
        resumed, resumed_provenance = build_manifest(
            self.profile, self.profile_bytes,
            "/workspace/train/frozen_train_10k.json", "b" * 64,
            "/workspace/train/checkpoints", True,
            attempts=249_998, target_attempts=TARGET_ATTEMPTS,
        )
        self.assertEqual(initial["training"]["attempts"], 1)
        self.assertEqual(initial["training"]["target_attempts"], 250_000)
        self.assertEqual(initial_provenance["invocation_attempts"], 1)
        self.assertEqual(resumed["training"]["attempts"], 249_998)
        self.assertEqual(resumed["training"]["target_attempts"], 250_000)
        self.assertEqual(resumed_provenance["invocation_attempts"], 249_998)
        with self.assertRaises(ValueError):
            build_manifest(
                self.profile, self.profile_bytes,
                "/workspace/train/frozen_train_10k.json", "b" * 64,
                "/workspace/train/checkpoints", True,
                attempts=250_001, target_attempts=250_000,
            )
        with self.assertRaises(ValueError):
            build_manifest(
                self.profile, self.profile_bytes,
                "/workspace/train/frozen_train_10k.json", "b" * 64,
                "/workspace/train/checkpoints", True,
                attempts=0, target_attempts=250_000,
            )

    def test_context_profile_is_verbatim_and_reports_sparse_widths(self):
        contexts = validate_profile(self.profile)
        self.assertEqual(contexts, self.profile["feature_contexts"])
        widths = context_widths(contexts)
        self.assertEqual(widths["labeled_cuts"], [3])
        self.assertNotIn(10, widths["labeled_cuts"])
        self.assertEqual(widths["i10_max_n"], [3])

    def test_rejects_boolean_alphabet_dimension(self):
        bad = copy.deepcopy(self.profile)
        bad["feature_contexts"]["alphabet"]["dimension"] = True
        with self.assertRaises(ValueError):
            validate_profile(bad)

    def test_rejects_selected_residual_alphabet_outside_objective_widths(self):
        bad = copy.deepcopy(self.profile)
        n = 3
        gates = [
            {"control": control, "target": target}
            for control in range(n)
            for target in range(n)
            if control != target
        ]
        bad["feature_contexts"]["alphabet"] = {
            "dimension": n,
            "id": f"all_to_all_unit_cnot:{n}",
            "gates": gates,
            "costs": [1] * len(gates),
        }
        with self.assertRaises(ValueError):
            validate_profile(bad)

    def test_rejects_additional_residual_alphabet_with_noncanonical_costs(self):
        bad = copy.deepcopy(self.profile)
        alphabet = copy.deepcopy(bad["feature_contexts"]["alphabet"])
        alphabet["costs"][0] = 2
        bad["feature_contexts"]["selected_residual_additional_contexts"] = [{
            "alphabet": alphabet,
            "gates": [{"control": 0, "target": 1}],
        }]
        with self.assertRaises(ValueError):
            validate_profile(bad)

    def test_rejects_missing_or_malformed_source_sha_and_repository(self):
        bad = copy.deepcopy(self.profile)
        bad["source"]["sha256"] = "main"
        with self.assertRaises(ValueError):
            validate_profile(bad)
        bad = copy.deepcopy(self.profile)
        bad["source"]["repository"] = "another/repository"
        with self.assertRaises(ValueError):
            validate_profile(bad)

    def test_profile_source_requires_exact_repository_revision_path_and_sha(self):
        source = b"immutable source artifact"
        digest = hashlib.sha256(source).hexdigest()
        self.profile["source"]["sha256"] = digest

        validate_profile_source(
            self.profile,
            source_fetcher=lambda revision, path: (
                source if revision == "a" * 40 and path == "native/tests/support/feature_contexts.rs" else b""
            ),
        )
        with self.assertRaises(ValueError):
            validate_profile_source(
                self.profile,
                source_fetcher=lambda *_args: b"changed source artifact",
            )

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

    def test_default_manifest_is_a_bounded_smoke_and_full_history_is_explicit(self):
        manifest, provenance = build_manifest(
            self.profile,
            self.profile_bytes,
            "/workspace/train/frozen_train_10k.json",
            "b" * 64,
            "/workspace/train/checkpoints",
            False,
        )
        self.assertEqual(manifest["training"]["attempts"], 2)
        self.assertEqual(manifest["training"]["target_attempts"], 2)
        self.assertEqual(provenance["invocation_attempts"], 2)
        self.assertEqual(provenance["target_attempts"], 2)

        historical, _ = build_manifest(
            self.profile,
            self.profile_bytes,
            "/workspace/train/frozen_train_10k.json",
            "b" * 64,
            "/workspace/train/checkpoints",
            False,
            attempts=250_000,
            target_attempts=TARGET_ATTEMPTS,
        )
        self.assertEqual(historical["training"]["attempts"], TARGET_ATTEMPTS)
        self.assertEqual(historical["training"]["target_attempts"], TARGET_ATTEMPTS)

    def test_resume_preserves_bounded_smoke_ceiling(self):
        manifest, _ = build_manifest(
            self.profile,
            self.profile_bytes,
            "/workspace/train/data.json",
            "b" * 64,
            "/workspace/train/checkpoints",
            True,
        )
        self.assertTrue(manifest["resume"])
        self.assertEqual(manifest["training"]["target_attempts"], 2)
        self.assertEqual(manifest["training"]["attempts"], 2)

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
            self.assertEqual(json.loads(path.read_text())["training"]["target_attempts"], 2)
            self.assertEqual(json.loads(sidecar.read_text())["profile_id"], "self-test")


    def test_one_and_two_gpu_manifests_are_explicit_and_invalid_counts_fail(self):
        for gpu_count, expected_devices in ((1, [0]), (2, [0, 1])):
            manifest, provenance = build_manifest(
                self.profile,
                self.profile_bytes,
                "/workspace/train/frozen_train_10k.json",
                "b" * 64,
                "/workspace/train/checkpoints",
                False,
                attempts=1,
                target_attempts=1,
                gpu_count=gpu_count,
            )
            self.assertEqual(manifest["training"]["cuda_devices"], expected_devices)
            self.assertEqual(provenance["gpu_count"], gpu_count)

        for invalid_count in (0, 3, True):
            with self.assertRaises(ValueError):
                build_manifest(
                    self.profile,
                    self.profile_bytes,
                    "/workspace/train/frozen_train_10k.json",
                    "b" * 64,
                    "/workspace/train/checkpoints",
                    False,
                    attempts=1,
                    target_attempts=1,
                    gpu_count=invalid_count,
                )


    def test_cli_self_test_accepts_one_gpu(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "single-gpu.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("build_training_manifest.py")),
                    "--self-test",
                    "--gpu-count",
                    "1",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("manifest_build=pass", result.stdout)
            manifest = json.loads(output.read_text())
            provenance = json.loads(
                output.with_suffix(output.suffix + ".provenance.json").read_text()
            )
            self.assertEqual(manifest["training"]["cuda_devices"], [0])
            self.assertEqual(provenance["gpu_count"], 1)


if __name__ == "__main__":
    unittest.main()
