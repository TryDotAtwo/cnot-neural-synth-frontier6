#!/usr/bin/env python3
"""Fail-closed controls for restartable long training on Vast."""

from __future__ import annotations

import json
import math
import os
import pathlib
import tempfile
import re
from typing import Any

BUDGET_CAP = 20.0
REMOTE_RUN_ROOT = pathlib.Path("/workspace/tqp_long_training")
_ALLOWED_STATES = frozenset({"not_started", "running", "succeeded", "failed", "unknown"})
_VAST_STATES = frozenset({
    "running", "stopped", "starting", "stopping", "scheduling",
    "loading", "created", "offline", "exited", "error",
})
_RUN_KEY = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_RUN_LABEL = re.compile(r"^cnot103-longtrain-[a-z0-9][a-z0-9-]{0,63}$")
_SOURCE_SHA = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_SHA = re.compile(r"^[0-9a-f]{64}$")
_LOCK_SCHEMA = "cnot103_long_train_lock_v1"
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


def _is_exact_2x3060(row: dict[str, Any]) -> bool:
    name = row.get("gpu_name")
    count = row.get("num_gpus", row.get("gpu_count"))
    if not isinstance(name, str) or isinstance(count, bool) or not isinstance(count, int):
        return False
    normalized = name.casefold().replace(" ", "").replace("_", "")
    return "rtx3060" in normalized and "rtx3060ti" not in normalized and count == 2


def classify_target_inventory(rows: object, label: object) -> str:
    """Return an allowlisted decision for an exact labeled 2x RTX 3060 target."""
    if not isinstance(label, str) or not _RUN_LABEL.fullmatch(label):
        return "unknown"
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return "unknown"
    matches = [row for row in rows if row.get("label") == label]
    if not matches:
        return "absent"
    if len(matches) != 1:
        return "ambiguous"
    row = matches[0]
    instance_id = row.get("id")
    if (
        isinstance(instance_id, bool)
        or not isinstance(instance_id, int)
        or instance_id <= 0
        or not _is_exact_2x3060(row)
    ):
        return "mismatch"
    remote_state = row.get("actual_status") or row.get("cur_state") or row.get("next_state")
    if not isinstance(remote_state, str) or remote_state.strip().casefold() not in _VAST_STATES:
        return "unknown"
    return "owned"


def select_owned_instance(rows: object, label: object) -> dict[str, Any] | None:
    """Select only the unique, well-formed instance carrying this exact run label."""
    if classify_target_inventory(rows, label) != "owned":
        return None
    assert isinstance(rows, list)
    assert isinstance(label, str)
    return next(row for row in rows if isinstance(row, dict) and row.get("label") == label)


def _valid_run_identity(run_key: object, source_sha: object, manifest_sha: object) -> bool:
    return (
        isinstance(run_key, str)
        and _RUN_KEY.fullmatch(run_key) is not None
        and isinstance(source_sha, str)
        and _SOURCE_SHA.fullmatch(source_sha) is not None
        and isinstance(manifest_sha, str)
        and _MANIFEST_SHA.fullmatch(manifest_sha) is not None
    )


def run_lock_path(run_key: object, root: pathlib.Path | str = REMOTE_RUN_ROOT) -> pathlib.Path:
    if not isinstance(run_key, str) or _RUN_KEY.fullmatch(run_key) is None:
        raise ValueError("invalid run key")
    return pathlib.Path(root) / run_key / "run.lock"


def _read_lock(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema") != _LOCK_SCHEMA:
        return None
    if value.get("state") not in {"reserved", "running", "succeeded", "failed"}:
        return None
    return value


def claim_run_lock(
    root: pathlib.Path | str,
    run_key: object,
    source_sha: object,
    manifest_sha: object,
) -> str:
    """Atomically reserve this run identity; retries never authorize a second launch."""
    if not _valid_run_identity(run_key, source_sha, manifest_sha):
        return "unknown"
    try:
        path = run_lock_path(run_key, root)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.parent.is_symlink():
            return "unknown"
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        existing = _read_lock(path)
        if existing is None:
            return "unknown"
        if (
            existing.get("run_key") != run_key
            or existing.get("source_sha") != source_sha
            or existing.get("manifest_sha") != manifest_sha
        ):
            return "identity_mismatch"
        return {
            "reserved": "already_reserved",
            "running": "already_running",
            "succeeded": "already_succeeded",
            "failed": "already_failed",
        }[existing["state"]]
    except (OSError, ValueError):
        return "unknown"

    record = {
        "schema": _LOCK_SCHEMA,
        "run_key": run_key,
        "source_sha": source_sha,
        "manifest_sha": manifest_sha,
        "state": "reserved",
    }
    try:
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with os.fdopen(descriptor, "wb") as lock_file:
            lock_file.write(payload)
            lock_file.flush()
            os.fsync(lock_file.fileno())
    except OSError:
        return "unknown"
    return "reserved"


def transition_run_lock(
    root: pathlib.Path | str,
    run_key: object,
    source_sha: object,
    manifest_sha: object,
    next_state: object,
) -> str:
    """Atomically record one monotone state change for the exact pinned run."""
    if not _valid_run_identity(run_key, source_sha, manifest_sha):
        return "unknown"
    if next_state not in {"running", "succeeded", "failed"}:
        return "unknown"
    try:
        path = run_lock_path(run_key, root)
        if path.parent.is_symlink() or path.is_symlink():
            return "unknown"
        existing = _read_lock(path)
    except (OSError, ValueError):
        return "unknown"
    if existing is None:
        return "unknown"
    if (
        existing.get("run_key") != run_key
        or existing.get("source_sha") != source_sha
        or existing.get("manifest_sha") != manifest_sha
    ):
        return "identity_mismatch"

    current_state = existing["state"]
    if current_state == next_state:
        return "unchanged"
    allowed = {
        "reserved": {"running", "failed"},
        "running": {"succeeded", "failed"},
        "succeeded": set(),
        "failed": set(),
    }
    if next_state not in allowed[current_state]:
        return "invalid_transition"

    existing["state"] = next_state
    temporary_path: pathlib.Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".run-lock-", dir=path.parent
        )
        temporary_path = pathlib.Path(temporary_name)
        payload = json.dumps(existing, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with os.fdopen(descriptor, "wb") as lock_file:
            lock_file.write(payload)
            lock_file.flush()
            os.fsync(lock_file.fileno())
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        return "unknown"
    return "updated"


def may_launch_run(lock_status: object) -> bool:
    """Only the invocation that atomically created the reservation may launch."""
    return lock_status == "reserved"
