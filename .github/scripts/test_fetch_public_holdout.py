import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import fetch_public_holdout as fetcher


class PublicHoldoutPinTests(unittest.TestCase):
    def test_expected_revision_and_sha_are_strictly_validated(self):
        with self.assertRaises(fetcher.FetchError):
            fetcher.validate_expected_pin("main", "a" * 64)
        with self.assertRaises(fetcher.FetchError):
            fetcher.validate_expected_pin("a" * 40, "A" * 64)

    def test_download_and_receipt_use_the_exact_revision_path_and_sha(self):
        source = b"pinned holdout test bytes"
        expected_sha = hashlib.sha256(source).hexdigest()
        revision = "a" * 40
        seen = {}

        def metadata(revision_arg):
            seen["metadata_revision"] = revision_arg
            return len(source), expected_sha

        def download(path, revision_arg, size, sha256):
            seen.update(revision=revision_arg, size=size, sha256=sha256)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fetcher.run(root, revision, expected_sha, metadata_getter=metadata, downloader=download)
            receipt = json.loads(
                (root / "input" / "holdout-source.receipt.json").read_text(encoding="utf-8")
            )

        self.assertEqual(seen["metadata_revision"], revision)
        self.assertEqual(seen["revision"], revision)
        self.assertEqual(seen["sha256"], expected_sha)
        self.assertEqual(receipt["revision"], revision)
        self.assertEqual(receipt["filename"], fetcher.FILENAME)
        self.assertEqual(receipt["sha256"], expected_sha)
        self.assertEqual(receipt["expected_sha256"], expected_sha)

    def test_metadata_sha_mismatch_stops_before_artifact_download(self):
        revision = "a" * 40
        downloaded = False

        def download(*_args):
            nonlocal downloaded
            downloaded = True

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(fetcher.FetchError):
                fetcher.run(
                    Path(temporary),
                    revision,
                    "b" * 64,
                    metadata_getter=lambda _revision: (10, "c" * 64),
                    downloader=download,
                )

        self.assertFalse(downloaded)

    def test_actual_artifact_is_rehashed_before_receipt_creation(self):
        source = b"expected bytes"
        revision = "a" * 40
        expected_sha = hashlib.sha256(source).hexdigest()

        def download(path, *_args):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wrong artifact")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(fetcher.FetchError):
                fetcher.run(
                    root,
                    revision,
                    expected_sha,
                    metadata_getter=lambda _revision: (len(source), expected_sha),
                    downloader=download,
                )
            self.assertFalse((root / "input" / "holdout-source.receipt.json").exists())

    def test_current_pin_discovery_binds_metadata_to_exact_revision(self):
        revision = "d" * 40
        expected_sha = "e" * 64
        seen = []

        def metadata(revision_arg):
            seen.append(revision_arg)
            return 123, expected_sha

        self.assertEqual(
            fetcher.discover_current_pin(
                revision_getter=lambda: revision,
                metadata_getter=metadata,
            ),
            (revision, expected_sha),
        )
        self.assertEqual(seen, [revision])

    def test_current_pin_discovery_rejects_invalid_metadata_digest(self):
        with self.assertRaises(fetcher.FetchError):
            fetcher.discover_current_pin(
                revision_getter=lambda: "d" * 40,
                metadata_getter=lambda _revision: (123, "not-a-sha"),
            )

    def test_github_environment_receives_only_validated_pin_variables(self):
        revision = "d" * 40
        expected_sha = "e" * 64
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / "github-env"
            fetcher.write_github_environment_pin(env_file, revision, expected_sha)
            self.assertEqual(
                env_file.read_text(encoding="utf-8").splitlines(),
                [
                    f"CNOT103_HOLDOUT_REVISION={revision}",
                    f"CNOT103_HOLDOUT_SHA256={expected_sha}",
                ],
            )


if __name__ == "__main__":
    unittest.main()
