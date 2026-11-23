import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_training_manifest import BATCH_SIZE, FEATURE_SCHEMA, MODEL_ARCHITECTURE, MODEL_INPUT_COUNT
from run_full_training import MODEL_INPUTS, MODEL_PARAMETERS, STRUCTURAL_BLOCKS, RunError
from run_molab_single_gpu_smoke import run_molab_single_gpu_smoke


class MolabSingleGpuSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.profile = self.root / "profile.json"
        self.profile.write_text("synthetic profile", encoding="utf-8")
        self.dataset = self.root / "sealed-train.json"
        self.dataset.write_bytes(b"synthetic pinned training bytes")
        self.trainer = self.root / "cnot103-train"
        self.trainer.write_bytes(b"synthetic native executable")
        self.checkpoint_dir = self.root / "fresh-checkpoints"
        self.source_sha = "a" * 40
        self.dataset_sha = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        self.builder_calls = []
        self.runner_calls = []

    def tearDown(self):
        self.temp.cleanup()

    def _manifest_builder(self, profile, dataset, dataset_sha, checkpoint, resume, output):
        self.builder_calls.append((resume, output))
        manifest = {
            "schema": "cnot103_training_run_v1",
            "dataset": {"path": str(dataset), "sha256": dataset_sha},
            "checkpoint_dir": str(checkpoint),
            "resume": resume,
            "training": {
                "attempts": 1,
                "target_attempts": 2,
                "cuda_devices": [0],
                "batch_size": BATCH_SIZE,
            },
            "model": MODEL_ARCHITECTURE.copy(),
            "feature_policy": {"schema_id": FEATURE_SCHEMA},
        }
        serialized = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        provenance = {
            "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
            "dataset_sha256": dataset_sha,
            "manifest_sha256": hashlib.sha256(serialized).hexdigest(),
            "model_input_slots": MODEL_INPUT_COUNT,
            "gpu_count": 1,
            "invocation_attempts": 1,
            "target_attempts": 2,
        }
        return manifest, provenance

    def _native_runner(self, trainer, manifest_path, log_dir, step, repo_root):
        self.runner_calls.append((step, manifest_path, log_dir, repo_root))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        start = step - 1
        end = step
        checkpoint_dir = Path(manifest["checkpoint_dir"])
        checkpoint_bytes = f"synthetic-checkpoint-step-{end}".encode("utf-8")
        sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        filename = f"step-{end:012d}-{sha256}.qcp"
        (checkpoint_dir / filename).write_bytes(checkpoint_bytes)
        checkpoint = {
            "version": 1,
            "file": filename,
            "sha256": sha256,
            "contract_sha256": "c" * 64,
            "progress": {"attempted": end, "accepted": end, "rejected": 0},
        }
        (checkpoint_dir / "last.json").write_text(
            json.dumps(checkpoint, sort_keys=True), encoding="utf-8"
        )
        return {
            "schema": "cnot103_training_result_v1",
            "model_parameters": MODEL_PARAMETERS,
            "feature_model_inputs": MODEL_INPUTS,
            "cuda_devices": [0],
            "start_progress": {"attempted": start},
            "end_progress": {"attempted": end, "accepted": end, "rejected": 0},
            "last_step": {"accepted": True},
            "exposure_audit": {"strict_unseen": True},
            "checkpoint": checkpoint,
        }

    def _run(self, **overrides):
        args = {
            "profile_path": self.profile,
            "dataset_path": self.dataset,
            "dataset_sha256": self.dataset_sha,
            "checkpoint_dir": self.checkpoint_dir,
            "trainer_path": self.trainer,
            "source_sha": self.source_sha,
            "repo_root": self.repo,
            "source_revision": lambda _root: self.source_sha,
            "manifest_builder": self._manifest_builder,
            "native_runner": self._native_runner,
        }
        args.update(overrides)
        return run_molab_single_gpu_smoke(**args)

    def test_full_model_one_gpu_runs_fresh_step_then_fresh_process_resume(self):
        receipt_path = self._run()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["schema"], "cnot103_molab_single_gpu_smoke_v1")
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["cuda_devices"], [0])
        self.assertEqual(receipt["model_parameters"], MODEL_PARAMETERS)
        self.assertEqual(receipt["feature_model_inputs"], 102)
        self.assertEqual(receipt["structural_blocks"], 8)
        self.assertEqual(receipt["steps"][0]["start_attempted"], 0)
        self.assertEqual(receipt["steps"][0]["end_attempted"], 1)
        self.assertEqual(receipt["steps"][1]["start_attempted"], 1)
        self.assertEqual(receipt["steps"][1]["end_attempted"], 2)
        self.assertTrue(all(step["strict_unseen"] for step in receipt["steps"]))
        self.assertTrue(all(step["checkpoint"]["sha256"] for step in receipt["steps"]))
        self.assertEqual([resume for resume, _path in self.builder_calls], [False, True])
        self.assertEqual([step for step, *_rest in self.runner_calls], [1, 2])
        first_manifest = json.loads(
            (receipt_path.parent / "step-1.manifest.json").read_text(encoding="utf-8")
        )
        second_manifest = json.loads(
            (receipt_path.parent / "step-2.manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(first_manifest["resume"])
        self.assertTrue(second_manifest["resume"])
        self.assertEqual(first_manifest["training"]["cuda_devices"], [0])
        self.assertEqual(second_manifest["training"]["attempts"], 1)

    def test_dataset_digest_mismatch_fails_before_creating_checkpoint_or_running(self):
        with self.assertRaises(RunError):
            self._run(dataset_sha256="b" * 64)
        self.assertFalse(self.checkpoint_dir.exists())
        self.assertEqual(self.builder_calls, [])
        self.assertEqual(self.runner_calls, [])

    def test_source_sha_mismatch_fails_before_manifest_or_training(self):
        with self.assertRaises(RunError):
            self._run(source_sha="b" * 40)
        self.assertFalse(self.checkpoint_dir.exists())
        self.assertEqual(self.builder_calls, [])
        self.assertEqual(self.runner_calls, [])

    def test_existing_checkpoint_directory_is_preserved_and_rejected(self):
        self.checkpoint_dir.mkdir()
        marker = self.checkpoint_dir / "do-not-overwrite"
        marker.write_text("existing owner data", encoding="utf-8")
        with self.assertRaises(RunError):
            self._run()
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing owner data")
        self.assertEqual(self.runner_calls, [])

    def test_changed_model_architecture_is_rejected_before_native_training(self):
        def wrong_architecture_builder(*args):
            manifest, provenance = self._manifest_builder(*args)
            manifest["model"]["wire_dim"] = 32
            serialized = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
            provenance["manifest_sha256"] = hashlib.sha256(serialized).hexdigest()
            return manifest, provenance

        with self.assertRaises(RunError):
            self._run(manifest_builder=wrong_architecture_builder)
        self.assertFalse(self.checkpoint_dir.exists())
        self.assertEqual(self.runner_calls, [])

    def test_input_slot_provenance_mismatch_is_rejected_before_native_training(self):
        def wrong_slot_builder(*args):
            manifest, provenance = self._manifest_builder(*args)
            provenance["model_input_slots"] = MODEL_INPUT_COUNT - 1
            return manifest, provenance

        with self.assertRaises(RunError):
            self._run(manifest_builder=wrong_slot_builder)
        self.assertFalse(self.checkpoint_dir.exists())
        self.assertEqual(self.runner_calls, [])

    def test_two_gpu_manifest_is_rejected_before_native_training(self):
        def bad_manifest_builder(*args):
            manifest, provenance = self._manifest_builder(*args)
            manifest["training"]["cuda_devices"] = [0, 1]
            return manifest, provenance

        with self.assertRaises(RunError):
            self._run(manifest_builder=bad_manifest_builder)
        self.assertFalse(self.checkpoint_dir.exists())
        self.assertEqual(self.runner_calls, [])

    def test_two_gpu_report_is_rejected_and_not_counted_as_a_smoke_step(self):
        def wrong_gpu_runner(*args):
            report = self._native_runner(*args)
            report["cuda_devices"] = [0, 1]
            return report

        with self.assertRaises(RunError):
            self._run(native_runner=wrong_gpu_runner)
        receipt_path = self.checkpoint_dir / ".molab-single-gpu-smoke" / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["steps"], [])

    def test_checkpoint_bytes_must_match_reported_sha(self):
        def corrupted_runner(*args):
            report = self._native_runner(*args)
            report["checkpoint"]["sha256"] = "d" * 64
            return report

        with self.assertRaises(RunError):
            self._run(native_runner=corrupted_runner)
        receipt_path = self.checkpoint_dir / ".molab-single-gpu-smoke" / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["steps"], [])


if __name__ == "__main__":
    unittest.main()
