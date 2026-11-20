#!/usr/bin/env python3
"""Budget and status primitives for restartable long training on Vast."""

from __future__ import annotations

import math
import re
from typing import Any

BUDGET_CAP = 20.0
_ALLOWED_STATES = frozenset({"not_started", "running", "succeeded", "failed", "unknown"})
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z$"
)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def budget_allows(
    spent_usd: object,
    hourly_rate: object,
    remaining_hours: object,
    reserve_usd: object,
) -> bool:
    """Return whether past spend plus the full remaining run fits the $20 cap."""
    spent = _finite_number(spent_usd)
    rate = _finite_number(hourly_rate)
    hours = _finite_number(remaining_hours)
    reserve = _finite_number(reserve_usd)
    if (
        spent is None
        or rate is None
        or hours is None
        or reserve is None
        or spent < 0
        or rate <= 0
        or hours <= 0
        or reserve < 0
    ):
        return False
    return spent + rate * hours + reserve <= BUDGET_CAP + 1e-9


def classify_run_state(remote_status: object) -> str:
    """Reduce a remote state record to the small public status allowlist."""
    if not isinstance(remote_status, dict):
        return "unknown"
    state = remote_status.get("state")
    return state if isinstance(state, str) and state in _ALLOWED_STATES else "unknown"


def redacted_report(state: object, checked_at: object) -> str:
    """Format only the UTC check time and an allowlisted operational state."""
    safe_state = state if isinstance(state, str) and state in _ALLOWED_STATES else "unknown"
    safe_time = checked_at if isinstance(checked_at, str) and _UTC_TIMESTAMP.fullmatch(checked_at) else "unavailable"
    return f"checked_at_utc={safe_time}\ntraining_status={safe_state}"
