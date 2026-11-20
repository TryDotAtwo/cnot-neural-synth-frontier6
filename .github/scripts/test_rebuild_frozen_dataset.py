import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rebuild_frozen_dataset as builder


def root_record(n, rows, source, seed, lineage):
    return {
        "state": builder.state_wire(n, rows),
        "source": source,
        "seed": seed,
        "lineage_id": lineage,
    }


def make_source_receipt(path, revision):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    return {
        "schema": "cnot103_public_holdout_source_v1",
        "repository": "TryDotAtwo/quantum-generator",
        "filename": "full_shared_native_holdout_20261001.sqlite",
        "revision": revision,
        "sha256": digest,
        "bytes": size,
        "expected_sha256": digest,
        "expected_bytes": size,
    }


class FrozenDatasetBuilderTests(unittest.TestCase):
    def test_rejects_one_cnot_holdout_neighbor_and_retries_that_record(self):
        dev = [root_record(3, builder.identity(3), "dev", 1, "dev-1")]
        test = [root_record(8, builder.identity(8), "test", 2, "test-2")]

        one_gate = builder.identity(3)
        builder.apply_gate(one_gate, 0, 1)
        two_gates = builder.identity(3)
        builder.apply_gate(two_gates, 0, 1)
        builder.apply_gate(two_gates, 1, 2)
        rejected = root_record(3, one_gate, "train", 3, "train-3")
        accepted = root_record(3, two_gates, "train", 4, "train-4")

        with (
            patch.object(builder, "WIDTHS", (3,)),
            patch.object(builder, "TRAIN_COUNT", 1),
            patch.object(builder, "DEV_COUNT", 1),
            patch.object(builder, "TEST_COUNT", 1),
            patch.object(builder, "generate_record", side_effect=[rejected, accepted]) as generate,
        ):
            body, _ = builder.build_body(dev, test, 202611190001)

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(body["panels"]["train"][0]["seed"], 4)

    def test_walk_lengths_use_an_independent_deterministic_stream(self):
        observed = []

        def record_walk_length(rng, n, rows, length):
            observed.append(length)
            gate = (0, 1)
            builder.apply_gate(rows, *gate)
            return [gate]

        positions = range(0, 42 * 42, 42)
        with patch.object(builder, "walk_component", side_effect=record_walk_length):
            for position in positions:
                builder.generate_record(position, 202611190001)

        self.assertEqual(len(observed), len(positions))
        self.assertTrue(set(observed).issubset({8, 16, 24}))
        self.assertGreater(len(set(observed)), 1)

    def test_hf_provenance_is_distinct_and_receipt_stays_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            data_dir = root / "data"
            input_dir.mkdir()
            data_dir.mkdir()
            holdout = input_dir / "full_shared_native_holdout_20261001.sqlite"
            holdout.write_bytes(b"synthetic holdout bytes")
            hf_revision = "a" * 40
            builder_revision = "b" * 40
            source_receipt = make_source_receipt(holdout, hf_revision)
            (input_dir / "holdout-source.receipt.json").write_text(
                json.dumps(source_receipt), encoding="utf-8"
            )
            output = data_dir / "frozen_train_10k.json"
            receipt = data_dir / "build.receipt.json"
            args = type("Args", (), {
                "experiment_root": str(root),
                "holdout_sqlite": str(holdout),
                "output": str(output),
                "receipt": str(receipt),
                "source_revision": builder_revision,
                "builder_source_revision": builder_revision,
                "seed": 202611190001,
            })()
            body = {"provenance": {}}

            with (
                patch.object(builder, "read_holdout", return_value=([], [])),
                patch.object(builder, "build_body", return_value=(body, {})),
            ):
                builder.run(args)

            dataset = json.loads(output.read_text(encoding="utf-8"))
            build_receipt = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(dataset["provenance"]["heldout_source_revision"], hf_revision)
            self.assertEqual(dataset["provenance"]["builder_source_revision"], builder_revision)
            self.assertFalse(build_receipt["complete"])
            self.assertEqual(build_receipt["validation_status"], "pending")

    def test_rejects_holdout_receipt_that_does_not_match_sqlite_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            data_dir = root / "data"
            input_dir.mkdir()
            data_dir.mkdir()
            holdout = input_dir / "full_shared_native_holdout_20261001.sqlite"
            holdout.write_bytes(b"synthetic holdout bytes")
            source_receipt = make_source_receipt(holdout, "a" * 40)
            source_receipt["sha256"] = "0" * 64
            (input_dir / "holdout-source.receipt.json").write_text(
                json.dumps(source_receipt), encoding="utf-8"
            )
            args = type("Args", (), {
                "experiment_root": str(root),
                "holdout_sqlite": str(holdout),
                "output": str(data_dir / "frozen_train_10k.json"),
                "receipt": str(data_dir / "build.receipt.json"),
                "source_revision": "b" * 40,
                "builder_source_revision": "b" * 40,
                "seed": 202611190001,
            })()

            with self.assertRaises(builder.BuildError):
                builder.run(args)


    def test_finalizer_requires_matching_dataset_sha_and_records_loader_acceptance(self):
        self.assertTrue(
            hasattr(builder, "finalize_dataset"),
            "the builder must expose a post-validator finalization step",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            output = data_dir / "frozen_train_10k.json"
            output.write_bytes(b"canonical dataset")
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            output.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")
            receipt = data_dir / "build.receipt.json"
            receipt.write_text(json.dumps({
                "schema": "cnot103_dataset_rebuild_receipt_v1",
                "complete": False,
                "validation_status": "pending",
                "dataset_sha256": digest,
                "dataset_bytes": output.stat().st_size,
            }), encoding="utf-8")

            with self.assertRaises(builder.BuildError):
                builder.finalize_dataset(root, output, receipt, "0" * 64)
            self.assertFalse(json.loads(receipt.read_text(encoding="utf-8"))["complete"])

            builder.finalize_dataset(root, output, receipt, digest)
            completed = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertTrue(completed["complete"])
            self.assertEqual(completed["validation_status"], "rust_loader_accepted")


if __name__ == "__main__":
    unittest.main()
