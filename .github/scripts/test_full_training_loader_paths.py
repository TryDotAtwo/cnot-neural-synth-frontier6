import unittest

from full_training_loader_paths import loader_validation_required_for_paths


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


if __name__ == "__main__":
    unittest.main()
