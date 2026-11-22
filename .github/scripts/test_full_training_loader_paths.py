import pathlib
import subprocess
import tempfile
import unittest

from full_training_loader_paths import (
    loader_validation_required,
    loader_validation_required_for_paths,
    loader_validation_required_for_range,
)


class FullTrainingLoaderPathTests(unittest.TestCase):
    def test_unrelated_sources_and_descriptions_do_not_compile_the_loader(self):
        self.assertFalse(
            loader_validation_required_for_paths(
                [
                    ".github/workflows/unrelated.yml",
                    ".github/scripts/unrelated.py",
                    "docs/cnot-model.md",
                    "native/README.md",
                ]
            )
        )

    def test_native_sources_and_cargo_configuration_compile_the_loader(self):
        self.assertTrue(
            loader_validation_required_for_paths(
                [
                    "native/src/frozen_data.rs",
                    "native/tests/fixtures/manifest.json",
                    "native/Cargo.toml",
                    "Cargo.lock",
                    ".cargo/config.toml",
                ]
            )
        )

    def test_explicit_reusable_source_pin_forces_full_check_when_caller_is_push(self):
        self.assertTrue(
            loader_validation_required(
                source_sha_input="a" * 40,
                event_name="push",
                before="invalid",
                after="invalid",
            )
        )

    def test_manual_full_validation_is_not_path_scoped(self):
        self.assertTrue(
            loader_validation_required(
                source_sha_input="",
                event_name="workflow_dispatch",
                before="invalid",
                after="invalid",
            )
        )

    def test_native_to_docs_rename_still_detects_deleted_native_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "CI test"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "ci@example.invalid"],
                check=True,
            )
            source = root / "native" / "src" / "lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text("pub fn synthetic_fixture() {}\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "native/src/lib.rs"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "native base"],
                check=True,
            )
            before = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            destination = root / "docs" / "example.md"
            destination.parent.mkdir()
            subprocess.run(
                ["git", "-C", str(root), "mv", "native/src/lib.rs", "docs/example.md"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "move native source to docs"],
                check=True,
            )
            after = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            self.assertTrue(loader_validation_required_for_range(before, after, cwd=root))


if __name__ == "__main__":
    unittest.main()
