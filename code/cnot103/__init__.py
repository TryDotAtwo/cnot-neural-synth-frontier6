"""Clean, strict contracts for the 103-feature CNOT neural synthesizer."""

from .schema import FEATURE_IDS, FEATURE_ID_MANIFEST_ID
from .layout import FEATURE_LAYOUT, FEATURE_LAYOUT_ID
from .contracts import FeatureEntry, FeaturePacket, FeatureStatus, IncompleteFeaturePacket
from .compat_v3 import from_research_v3, require_research_v3_complete
from .runtime import forward_complete_103

__all__ = [
    "FEATURE_IDS",
    "FEATURE_ID_MANIFEST_ID",
    "FEATURE_LAYOUT",
    "FEATURE_LAYOUT_ID",
    "FeatureEntry",
    "FeaturePacket",
    "FeatureStatus",
    "IncompleteFeaturePacket",
    "from_research_v3",
    "require_research_v3_complete",
    "forward_complete_103",
]
