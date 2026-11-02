import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from cnot103.layout import FEATURE_LAYOUT


class FeatureLayoutSourceContractTests(unittest.TestCase):
    def test_h07_keeps_the_pinned_unimplemented_placeholder_shape(self):
        # The pinned feature schema marks H07 unimplemented with group/1 as a
        # placeholder. It does not define a global six-channel model input.
        self.assertEqual(FEATURE_LAYOUT["H07"], ("group", 1))


if __name__ == "__main__":
    unittest.main()
