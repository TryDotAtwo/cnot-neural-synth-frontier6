"""Strict entry point for declaring a v3 inference as a complete-103 pass."""

from __future__ import annotations

from typing import Any

from .compat_v3 import require_research_v3_complete


def forward_complete_103(model: Any, batch: Any, legacy_packet: Any) -> Any:
    """Run the legacy model only after every feature is computed for every row.

    Partial packets remain useful for diagnostics and research, but this entry
    point rejects them so a selected-ID count cannot be reported as full input
    coverage.
    """

    require_research_v3_complete(legacy_packet)
    return model(batch, legacy_packet)
