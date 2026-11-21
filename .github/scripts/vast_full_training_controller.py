#!/usr/bin/env python3
"""Budget-gated Vast lifecycle controller for one reproducible full CNOT run.

All public output is reduced to an allowlisted operation outcome. Detailed
training data, checkpoints, logs, and quality reports stay on the Vast disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone

import vast_rent_rebuild_dataset as base

BUDGET_CAP = 20.0
ACTIVE_HOURS = 48.0
STORAGE_HOURS = 168.0
DISK_GB = 64
MAX_RUNTIME_SECONDS = 47 * 3600 + 45 * 60
MAX_STOP_SECONDS = 300
BUDGET_POLL_SECONDS = 5 * 60
MONITOR_SECONDS = 5 * 60 * 60 + 30 * 60
POLL_SECONDS = 60
STARTUP_WAIT_SECONDS = 3 * 60 * 60
BOOTSTRAP_HOURS = 8.0
BOOTSTRAP_SECONDS = int(BOOTSTRAP_HOURS * 3600)
FORECAST_APPROVAL_WAIT_SECONDS = 20 * 60
FORECAST_APPROVAL_POLL_SECONDS = 5
CLAIM_VISIBILITY_WAIT_SECONDS = 2 * 60
CLAIM_VISIBILITY_POLL_SECONDS = 5
PROFILE_PATH = ".github/training-profiles/n10-test-support-v1.json"
MARKER_PATH = pathlib.Path(__file__).resolve().parents[1] / "paid-runs" / "2026-11-20-full-training.json"

OUTCOMES = {
    "accepted", "running", "completed", "failed", "time_limit", "budget_gate",
    "configuration_missing", "source_pin_missing", "source_pin_invalid",
    "target_unavailable", "target_busy", "target_gpu_mismatch", "target_ambiguous",
    "workspace_exists", "remote_launch_failed", "remote_state_unavailable",
    "api_failed", "controller_failed", "unverified", "not_needed",
}
STATUSES = {
    "accepted", "running", "completed", "failed", "time_limit", "budget_gate",
    "started", "stopped", "unverified", "not_needed", "unavailable",
}
REMOTE_STATES = {
    "starting", "preparing", "benchmarking", "resuming", "baseline",
    "training", "final_quality", "completed", "failed", "budget_gate",
    "time_limit", "unavailable",
}
QUALITY_STATES = {"not_ready", "measured", "passed", "failed", "unavailable"}


class GateError(RuntimeError):
    pass



def _finite(value: object) -> float | None:
    return base.finite_number(value)


def _storage_reserve(storage_cost: object, hours: float = STORAGE_HOURS) -> float | None:
    monthly = _finite(storage_cost)
    reserve_hours = _finite(hours)
    if monthly is None or reserve_hours is None or monthly < 0 or reserve_hours < 0:
        return None
    return monthly * DISK_GB * reserve_hours / 720.0


def full_run_budget_allows(
    spent: object,
    rate: object,
    storage_cost: object,
    active_hours: object = ACTIVE_HOURS,
    storage_hours: object = STORAGE_HOURS,
    extra_storage_reserve: object = 0.0,
) -> bool:
    """Reserve the full run, seven retained-storage days, and transfer overhead."""
    spent_value = _finite(spent)
    rate_value = _finite(rate)
    active = _finite(active_hours)
    extra = _finite(extra_storage_reserve)
    storage = _storage_reserve(storage_cost, _finite(storage_hours))
    if (
        spent_value is None or rate_value is None or active is None
        or extra is None or storage is None
        or spent_value < 0 or rate_value < 0 or active < 0 or extra < 0
    ):
        return False
    return (
        spent_value + rate_value * active + storage + base.TRANSFER_RESERVE + extra
        <= BUDGET_CAP + 1e-9
    )



def bootstrap_budget_allows(
    spent: object,
    rate: object,
    storage_cost: object,
    active_hours: object = BOOTSTRAP_HOURS,
    storage_hours: object = STORAGE_HOURS,
    extra_storage_reserve: object = 0.0,
) -> bool:
    """Reserve only the bounded preflight/pilot window plus verified stop headroom."""
    active = _finite(active_hours)
    if active is None or active < 0:
        return False
    return full_run_budget_allows(
        spent,
        rate,
        storage_cost,
        active_hours=active + (MAX_STOP_SECONDS + BUDGET_POLL_SECONDS) / 3600.0,
        storage_hours=storage_hours,
        extra_storage_reserve=extra_storage_reserve,
    )


def _bootstrap_remaining_hours(started_at_epoch: float | None) -> float:
    if started_at_epoch is None:
        return BOOTSTRAP_HOURS
    started = _finite(started_at_epoch)
    if started is None or started <= 0:
        return 0.0
    elapsed = max(0.0, time.time() - started)
    return max(0.0, BOOTSTRAP_HOURS - elapsed / 3600.0)


def run_workspace(run_id: str, revision: str, attempt: str) -> str:
    if (
        not re.fullmatch(r"[0-9]{1,20}", run_id)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not re.fullmatch(r"[0-9]{1,10}", attempt)
    ):
        raise GateError("run identity is invalid")
    return f"/workspace/tqp_full_training_{run_id}_{revision[:8]}_a{attempt}"


def _run_identity() -> tuple[str, str, str, str]:
    revision, run_id, _legacy_label = base.run_identity()
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not re.fullmatch(r"[0-9]{1,10}", attempt):
        raise GateError("run attempt is invalid")
    return revision, run_id, attempt, run_workspace(run_id, revision, attempt)


def _marker() -> dict:
    try:
        value = json.loads(MARKER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise GateError("paid-run marker is missing or invalid") from None
    if (
        not isinstance(value, dict)
        or value.get("schema") != "cnot103_full_training_v1"
        or value.get("max_total_usd") != BUDGET_CAP
        or value.get("max_active_hours") != ACTIVE_HOURS
        or value.get("disk_gb") != DISK_GB
        or value.get("preserve_remote_disk") is not True
    ):
        raise GateError("paid-run marker does not match the full-run contract")
    old_label = value.get("resume_existing_label")
    if not isinstance(old_label, str) or not re.fullmatch(
        r"cnot103-r10k-[0-9]{1,20}-a[0-9]{1,10}", old_label
    ):
        raise GateError("approved existing target label is invalid")
    return value


def _fresh_label(run_id: str, attempt: str) -> str:
    return f"cnot103-fulltrain-{run_id}-a{attempt}"


def _other_storage_reserve(rows: list[dict], exclude_id: int | None) -> float | None:
    total = 0.0
    for row in rows:
        ident = base.instance_id(row)
        if ident is not None and ident == exclude_id:
            continue
        label = str(row.get("label") or "").casefold()
        if not (label.startswith("cnot103") or ident in base.OWNED_INSTANCE_IDS):
            continue
        reserve = _storage_reserve(row.get("storage_cost"))
        if reserve is None:
            return None
        total += reserve
    return total


def _labels_for_run(rows: list[dict], old_label: str, fresh_label: str) -> tuple[dict | None, str | None]:
    old = base.resolve_label(rows, old_label)
    fresh = base.resolve_label(rows, fresh_label)
    if old is not None and fresh is not None:
        raise GateError("target labels are ambiguous")
    if old is not None:
        return old, old_label
    if fresh is not None:
        return fresh, fresh_label
    return None, None


def _pin_status() -> tuple[str, str]:
    revision = os.environ.get("CNOT103_HOLDOUT_REVISION", "")
    sha256 = os.environ.get("CNOT103_HOLDOUT_SHA256", "")
    status = base.holdout_pin_status(revision, sha256)
    if status != "accepted":
        raise GateError("pinned holdout metadata is invalid")
    return revision, sha256


def _new_private_key(run_id: str, attempt: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, str]:
    root = pathlib.Path(tempfile.mkdtemp(prefix="vast-full-training-"))
    key_path = root / "temporary"
    known_hosts = root / "known_hosts"
    generated = subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C",
         f"gha-rust-{run_id}-{attempt}", "-f", str(key_path)],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if generated.returncode != 0:
        for child in root.iterdir():
            child.unlink(missing_ok=True)
        root.rmdir()
        raise GateError("temporary SSH key generation failed")
    public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()
    return root, key_path, known_hosts, public_key


def _remove_local_key(root: pathlib.Path, key_path: pathlib.Path) -> None:
    for name in ("temporary", "temporary.pub", "known_hosts"):
        try:
            (root / name).unlink(missing_ok=True)
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def _launch_script(
    root: str,
    source_revision: str,
    holdout_revision: str,
    holdout_sha256: str,
    started_at_epoch: float,
    budget_baseline: float,
    bootstrap_deadline_epoch: float,
    run_id: str,
    attempt: str,
    instance_id: int,
) -> str:
    q = shlex.quote
    source = root + "/source"
    return f"""set -euo pipefail
ROOT={q(root)}
SOURCE={q(source)}
if [ -e "$ROOT" ] || [ -L "$ROOT" ]; then
  printf 'full_training=workspace_exists\\n'
  exit 10
fi
mkdir -m 700 "$ROOT"
if ! git clone --quiet --filter=blob:none --no-checkout https://github.com/TryDotAtwo/cnot-neural-synth-frontier6.git "$SOURCE" >/dev/null 2>&1; then
  printf 'full_training=source_failed\\n'
  exit 11
fi
if ! git -C "$SOURCE" fetch --quiet --depth=1 origin {q(source_revision)} >/dev/null 2>&1; then
  printf 'full_training=source_failed\\n'
  exit 11
fi
if ! git -C "$SOURCE" checkout --quiet --detach {q(source_revision)} >/dev/null 2>&1; then
  printf 'full_training=source_failed\\n'
  exit 11
fi
python3 - "$ROOT/owner.private.json" <<'PY'
import json
import os
import sys
value = {{
    "schema": "cnot103_full_training_owner_v1",
    "run_id": {run_id!r},
    "attempt": {attempt!r},
    "source_revision": {source_revision!r},
    "instance_id": {instance_id},
    "workspace": {root!r},
}}
descriptor = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as output:
    json.dump(value, output, sort_keys=True, separators=(",", ":"))
    output.write("\\n")
    output.flush()
    os.fsync(output.fileno())
PY
nohup setsid python3 {q(source + "/.github/scripts/run_full_training.py")} \\
  --root {q(root)} \\
  --source-revision {q(source_revision)} \\
  --holdout-revision {q(holdout_revision)} \\
  --holdout-sha256 {q(holdout_sha256)} \\
  --started-at-epoch {q(str(started_at_epoch))} \\
  --budget-baseline {q(str(budget_baseline))} \\
  --bootstrap-deadline-epoch {q(str(bootstrap_deadline_epoch))} \\
  >"$ROOT/supervisor.private.log" 2>&1 </dev/null &
PID=$!
(umask 077; printf '%s\\n' "$PID" >"$ROOT/supervisor.pid")
sleep 3
kill -0 "$PID" 2>/dev/null || {{ printf 'full_training=launch_failed\\n'; exit 12; }}
printf 'full_training=launched\\n'
"""

def _attach_and_remote(
    row: dict, run_id: str, attempt: str, script: str, timeout_seconds: int
) -> tuple[subprocess.CompletedProcess[str] | None, str, int]:
    ident = base.instance_id(row)
    if ident is None:
        raise GateError("target id is invalid")
    host, port = base.endpoint(row)
    temp_root, key_path, known_hosts, public_key = _new_private_key(run_id, attempt)
    attached = False
    key_cleanup = "not_needed"
    proc = None
    try:
        if base.probe.find_attached_key_id(base.api_key(), ident, public_key) is not None:
            raise GateError("temporary SSH key collision")
        base.attach_key(ident, public_key)
        attached = True
        proc = base.remote_script(host, port, key_path, script, timeout_seconds)
    finally:
        if attached:
            key_cleanup = base.detach_key(ident, public_key)
        _remove_local_key(temp_root, key_path)
    return proc, key_cleanup, ident




class RemoteSession:
    def __init__(self, row: dict, run_id: str, attempt: str):
        self.ident = base.instance_id(row)
        if self.ident is None:
            raise GateError("target id is invalid")
        self.host, self.port = base.endpoint(row)
        self.temp_root, self.key_path, self.known_hosts, self.public_key = _new_private_key(run_id, attempt)
        self.attached = False
        self.closed = False
        try:
            if base.probe.find_attached_key_id(base.api_key(), self.ident, self.public_key) is not None:
                raise GateError("temporary SSH key collision")
            base.attach_key(self.ident, self.public_key)
            self.attached = True
        except Exception:
            _remove_local_key(self.temp_root, self.key_path)
            raise

    def run(self, script: str, timeout_seconds: int) -> subprocess.CompletedProcess[str] | None:
        if self.closed:
            raise GateError("remote session is closed")
        return base.remote_script(
            self.host, self.port, self.key_path, script, timeout_seconds
        )

    def close(self) -> str:
        if self.closed:
            return "not_found"
        self.closed = True
        result = "not_found"
        if self.attached:
            result = base.detach_key(self.ident, self.public_key)
        _remove_local_key(self.temp_root, self.key_path)
        return result


def _wait_for_exact_instance(ident: int, label: str, deadline: float) -> dict:
    """Poll GET inventory for one exact ID without starting or stopping it."""
    while True:
        rows = base.instance_rows()
        labeled = base.resolve_label(rows, label)
        if labeled is not None and base.instance_id(labeled) != ident:
            raise GateError("target instance id changed while waiting")
        row = next((item for item in rows if base.instance_id(item) == ident), None)
        if row is not None:
            if row.get("label") != label:
                raise GateError("target label changed while waiting")
            state = base.probe.observed_state(row)
            if state in {"running", "stopped", "exited"}:
                if not base.is_2x3060(row):
                    raise GateError("target GPU inventory does not match")
                return row
            if state in {"error", "offline"}:
                raise GateError("target entered an unusable state")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GateError("target startup wait reached its three-hour deadline")
        time.sleep(min(POLL_SECONDS, remaining))


def _activate_stopped_target(
    target: dict,
    current_label: str,
    fresh_label: str,
    deadline: float,
    started_at_epoch: float | None = None,
) -> tuple[dict, str]:
    ident = base.instance_id(target)
    if ident is None:
        raise GateError("exact target id is invalid")
    rows = base.instance_rows()
    current = base.resolve_label(rows, current_label)
    if current is None or base.instance_id(current) != ident:
        raise GateError("exact target identity changed before start")
    if current.get("label") != current_label or not base.is_2x3060(current):
        raise GateError("exact target label or GPU inventory changed before start")
    state = base.probe.observed_state(current)
    if state not in {"running", "stopped", "exited"}:
        current = _wait_for_exact_instance(ident, current_label, deadline)
        state = base.probe.observed_state(current)
    if state == "running":
        return current, current_label
    if state not in {"stopped", "exited"}:
        raise GateError("target did not reach a startable state")
    if current_label != fresh_label and base.resolve_label(rows, fresh_label) is not None:
        raise GateError("run-specific start label is already occupied")
    spend = base.project_charges()
    if not _validate_bootstrap_budget(
        current, spend, rows, ident,
        active_hours=_bootstrap_remaining_hours(started_at_epoch),
    ):
        raise GateError("the bounded pilot budget gate is closed before start")
    # Vast accepts state and label in one instance update. That label is the
    # recoverable proof that this run requested the start, even if Actions dies
    # before Python can publish any step output.
    if not base.api_instance_id(ident, "running", label=fresh_label):
        raise GateError("exact target start was not confirmed")
    started = _wait_for_exact_instance(ident, fresh_label, deadline)
    if base.probe.observed_state(started) != "running":
        raise GateError("target did not reach running after its one start request")
    return started, fresh_label


def _start_target(
    rows: list[dict], marker: dict, run_id: str, attempt: str,
    started_at_epoch: float | None = None,
) -> tuple[dict, str, bool, float]:
    deadline = time.monotonic() + STARTUP_WAIT_SECONDS
    fresh_label = _fresh_label(run_id, attempt)
    old_label = marker["resume_existing_label"]
    target, label = _labels_for_run(rows, old_label, fresh_label)
    spent = base.project_charges()
    bootstrap_hours = _bootstrap_remaining_hours(started_at_epoch)
    if target is not None:
        ident = base.instance_id(target)
        if ident is None or not base.is_2x3060(target):
            raise GateError("target GPU inventory does not match")
        if not _validate_bootstrap_budget(
            target, spent, rows, ident, active_hours=bootstrap_hours
        ):
            raise GateError("the bounded pilot budget gate is closed")
        state = base.probe.observed_state(target)
        if state not in {"running", "stopped", "exited"}:
            target = _wait_for_exact_instance(ident, label, deadline)
            state = base.probe.observed_state(target)
        if state in {"stopped", "exited"}:
            target, label = _activate_stopped_target(
                target, label, fresh_label, deadline, started_at_epoch
            )
        elif state != "running":
            raise GateError("exact target did not reach a usable state")
        return target, label, False, spent

    transitions = [
        row for row in rows
        if base.is_2x3060(row)
        and base.probe.observed_state(row) not in {"stopped", "exited", "error", "offline"}
    ]
    if transitions:
        raise GateError("another 2x3060 target is active or transitioning")
    extra_storage = _other_storage_reserve(rows, None)
    if extra_storage is None:
        raise GateError("existing storage cost is unknown")
    offers = base.offer_inventory()
    selected = next(
        (
            offer for offer in offers
            if bootstrap_budget_allows(
                spent,
                offer.get("dph_total"),
                offer.get("storage_cost"),
                active_hours=bootstrap_hours,
                extra_storage_reserve=extra_storage,
            )
        ),
        None,
    )
    if selected is None:
        raise GateError("no verified 2x3060 offer fits the bounded pilot budget")
    offer_id = str(selected.get("ask_id") or selected.get("id") or selected.get("offer_id") or "")
    if not re.fullmatch(r"[0-9]{1,20}", offer_id):
        raise GateError("selected offer identity is invalid")
    target = base.create_once(offer_id, fresh_label)
    ident = base.instance_id(target)
    if ident is None or target.get("label") != fresh_label or not base.is_2x3060(target):
        raise GateError("new target identity or GPU inventory is invalid")
    state = base.probe.observed_state(target)
    if state not in {"running", "stopped", "exited"}:
        target = _wait_for_exact_instance(ident, fresh_label, deadline)
        state = base.probe.observed_state(target)
    if state in {"stopped", "exited"}:
        target, _label = _activate_stopped_target(
            target, fresh_label, fresh_label, deadline, started_at_epoch
        )
    elif state != "running":
        raise GateError("new target did not reach a usable state")
    return target, fresh_label, True, spent

def classify_full_run_budget(row: object, rows: object, spent: object) -> str:
    """Classify the pinned full-run budget without exposing financial values."""
    if not isinstance(row, dict):
        return "target_missing"
    if not base.is_2x3060(row):
        return "target_mismatch"
    ident = base.instance_id(row)
    spend_value = _finite(spent)
    rate = _finite(row.get("dph_total"))
    storage_monthly = _finite(row.get("storage_cost"))
    extra = _other_storage_reserve(rows, ident) if isinstance(rows, list) and ident else None
    if (
        ident is None or spend_value is None or rate is None or storage_monthly is None
        or extra is None or spend_value < 0 or rate < 0
        or storage_monthly < 0 or extra < 0
    ):
        return "inputs_unknown"
    if full_run_budget_allows(
        spend_value, rate, storage_monthly, extra_storage_reserve=extra
    ):
        return "within_cap"
    if full_run_budget_allows(
        0.0, rate, storage_monthly, extra_storage_reserve=extra
    ):
        return "insufficient_remaining_budget"
    return "run_projection_exceeds_cap"


def classify_bootstrap_budget(
    row: object,
    rows: object,
    spent: object,
    active_hours: object = BOOTSTRAP_HOURS,
) -> str:
    """Classify the bounded pilot reserve without exposing financial values."""
    if not isinstance(row, dict):
        return "target_missing"
    if not base.is_2x3060(row):
        return "target_mismatch"
    ident = base.instance_id(row)
    spend_value = _finite(spent)
    rate = _finite(row.get("dph_total"))
    storage_monthly = _finite(row.get("storage_cost"))
    extra = _other_storage_reserve(rows, ident) if isinstance(rows, list) and ident else None
    active = _finite(active_hours)
    if (
        ident is None or spend_value is None or rate is None or storage_monthly is None
        or extra is None or active is None or spend_value < 0 or rate < 0
        or storage_monthly < 0 or extra < 0 or active < 0
    ):
        return "inputs_unknown"
    if bootstrap_budget_allows(
        spend_value, rate, storage_monthly, active_hours=active,
        extra_storage_reserve=extra,
    ):
        return "within_cap"
    if bootstrap_budget_allows(
        0.0, rate, storage_monthly, active_hours=active,
        extra_storage_reserve=extra,
    ):
        return "insufficient_remaining_budget"
    return "run_projection_exceeds_cap"



def _validate_running_budget(
    row: dict, spend: float, rows: list[dict], exclude_id: int | None = None
) -> bool:
    extra = _other_storage_reserve(rows, exclude_id)
    return extra is not None and full_run_budget_allows(
        spend,
        row.get("dph_total"),
        row.get("storage_cost"),
        extra_storage_reserve=extra,
    )


def _validate_bootstrap_budget(
    row: dict,
    spend: float,
    rows: list[dict],
    exclude_id: int | None = None,
    active_hours: float = BOOTSTRAP_HOURS,
) -> bool:
    extra = _other_storage_reserve(rows, exclude_id)
    return extra is not None and bootstrap_budget_allows(
        spend,
        row.get("dph_total"),
        row.get("storage_cost"),
        active_hours=active_hours,
        extra_storage_reserve=extra,
    )


def start() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome, status = "controller_failed", "unavailable"
    run_id = attempt = ""
    try:
        started_at_epoch = time.time()
        revision, run_id, attempt, root = _run_identity()
        holdout_revision, holdout_sha256 = _pin_status()
        marker = _marker()
        rows = base.instance_rows()
        row, label, _created, _spend_before = _start_target(
            rows, marker, run_id, attempt, started_at_epoch
        )
        ident = base.instance_id(row)
        if ident is None or not base.is_2x3060(row):
            outcome, status = "target_gpu_mismatch", "failed"
            raise GateError("running target identity or GPU differs")
        running_rows = base.instance_rows()
        current = base.resolve_label(running_rows, label)
        if (
            current is None or base.instance_id(current) != ident
            or base.probe.observed_state(current) != "running"
        ):
            outcome, status = "target_busy", "failed"
            raise GateError("exact target changed before remote launch")
        row = current
        spend_now = base.project_charges()
        if not _validate_bootstrap_budget(
            row, spend_now, running_rows, ident,
            active_hours=_bootstrap_remaining_hours(started_at_epoch),
        ):
            outcome, status = "budget_gate", "budget_gate"
            raise GateError("post-start bounded pilot budget recheck failed")
        script = _launch_script(
            root,
            revision,
            holdout_revision,
            holdout_sha256,
            started_at_epoch,
            spend_now,
            started_at_epoch + BOOTSTRAP_SECONDS,
            run_id,
            attempt,
            ident,
        )
        proc, key_cleanup, _ = _attach_and_remote(row, run_id, attempt, script, 180)
        if (
            proc is None or proc.returncode != 0
            or "full_training=launched" not in proc.stdout.splitlines()
            or key_cleanup not in {"detached", "not_found"}
        ):
            outcome, status = "remote_launch_failed", "failed"
            raise GateError("remote training supervisor launch failed")
        outcome, status = "accepted", "started"
    except base.probe.ApiFailure:
        outcome, status = "api_failed", "failed"
    except GateError as error:
        if outcome == "controller_failed":
            message = str(error)
            if "budget" in message:
                outcome, status = "budget_gate", "budget_gate"
            elif "GPU" in message or "3060" in message:
                outcome, status = "target_gpu_mismatch", "failed"
            elif "stopped" in message or "active" in message or "transition" in message or "wait" in message:
                outcome, status = "target_busy", "failed"
            elif "pin" in message or "holdout" in message:
                outcome, status = "source_pin_invalid", "failed"
            elif "configuration" in message:
                outcome, status = "configuration_missing", "failed"
            else:
                outcome, status = "target_unavailable", "failed"
    except Exception:
        outcome, status = "controller_failed", "failed"
    try:
        _summarize(checked, outcome, status)
    except Exception:
        return 1
    return 0 if outcome == "accepted" else 1

def _read_remote_status_full(
    row: dict, run_id: str, attempt: str, root: str,
    session: RemoteSession | None = None,
) -> tuple[str, str, float | None, float | None, float | None, str | None]:
    q = shlex.quote
    remote = f"""set -euo pipefail
FILE={q(root + "/status.private.json")}
if [ ! -r "$FILE" ]; then
  printf 'run_state=unavailable\\n'
  exit 0
fi
python3 - "$FILE" <<'PY'
import json
import math
import re
import sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    value = {{}}
state = value.get("state")
if state not in {sorted(REMOTE_STATES)!r}:
    state = "unavailable"
quality = value.get("quality_status")
if quality not in {sorted(QUALITY_STATES)!r}:
    quality = "unavailable"
def number(key):
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
        return None
    return float(item)
print("run_state=" + state)
print("quality_status=" + quality)
for name in ("started_at_epoch", "budget_baseline", "instance_rate", "storage_cost"):
    item = number(name)
    if item is not None:
        print(name + "=" + str(item))
forecast = number("forecast_total_seconds")
if forecast is not None and forecast > 0:
    print("forecast_total_seconds=" + str(forecast))
token = value.get("forecast_token")
if isinstance(token, str) and re.fullmatch(r"[0-9a-f]{32}", token):
    print("forecast_token=" + token)
PY
"""
    if session is None:
        proc, key_cleanup, _ident = _attach_and_remote(row, run_id, attempt, remote, 30)
        if proc is None or proc.returncode != 0 or key_cleanup not in {"detached", "not_found"}:
            return "unavailable", "unavailable", None, None, None, None
    else:
        proc = session.run(remote, 30)
        if proc is None or proc.returncode != 0:
            return "unavailable", "unavailable", None, None, None, None
    values = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {
            "run_state", "quality_status", "started_at_epoch",
            "budget_baseline", "instance_rate", "storage_cost",
            "forecast_total_seconds", "forecast_token",
        }:
            values[key] = value
    state = values.get("run_state", "unavailable")
    quality = values.get("quality_status", "unavailable")

    def parsed_number(key):
        try:
            value = float(values[key])
            return value if math.isfinite(value) else None
        except (KeyError, ValueError):
            return None

    token = values.get("forecast_token")
    if token is not None and not re.fullmatch(r"[0-9a-f]{32}", token):
        token = None
    return (
        state,
        quality,
        parsed_number("started_at_epoch"),
        parsed_number("budget_baseline"),
        parsed_number("forecast_total_seconds"),
        token,
    )


def _read_remote_status(
    row: dict, run_id: str, attempt: str, root: str,
    session: RemoteSession | None = None,
) -> tuple[str, str, float | None, float | None]:
    values = _read_remote_status_full(row, run_id, attempt, root, session=session)
    return values[0], values[1], values[2], values[3]


def _write_forecast_authorization(
    session: RemoteSession, root: str, token: str, forecast_total_seconds: float
) -> bool:
    seconds = _finite(forecast_total_seconds)
    if not re.fullmatch(r"[0-9a-f]{32}", token) or seconds is None or seconds <= 0:
        return False
    q = shlex.quote
    path = root + "/forecast-approval.private.json"
    remote = f"""set -euo pipefail
python3 - {q(path)} {q(token)} {q(str(seconds))} <<'PY'
import json
import math
import os
import sys
path, token, raw_seconds = sys.argv[1:]
seconds = float(raw_seconds)
if not math.isfinite(seconds) or seconds <= 0:
    raise SystemExit(2)
value = {{
    "schema": "cnot103_full_training_forecast_approval_v1",
    "authorized": True,
    "forecast_token": token,
    "forecast_total_seconds": seconds,
}}
try:
    existing = json.load(open(path, encoding="utf-8"))
except FileNotFoundError:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(value, output, sort_keys=True, separators=(",", ":"))
        output.write("\\n")
        output.flush()
        os.fsync(output.fileno())
except Exception:
    raise SystemExit(3)
else:
    if (
        existing.get("schema") != value["schema"]
        or existing.get("authorized") is not True
        or existing.get("forecast_token") != token
        or existing.get("forecast_total_seconds") != seconds
    ):
        raise SystemExit(4)
print("full_training=forecast_authorized")
PY
"""
    proc = session.run(remote, 30)
    return (
        proc is not None
        and proc.returncode == 0
        and "full_training=forecast_authorized" in proc.stdout.splitlines()
    )



def _read_remote_owner(
    row: dict,
    run_id: str,
    attempt: str,
    root: str,
    source_revision: str,
    session: RemoteSession | None = None,
) -> bool:
    ident = base.instance_id(row)
    if ident is None:
        return False
    q = shlex.quote
    remote = f"""set -euo pipefail
FILE={q(root + "/owner.private.json")}
python3 - "$FILE" <<'PY'
import json
import sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    value = {{}}
expected = {{
    "schema": "cnot103_full_training_owner_v1",
    "run_id": {run_id!r},
    "attempt": {attempt!r},
    "source_revision": {source_revision!r},
    "instance_id": {ident},
    "workspace": {root!r},
}}
print("owner_verified=" + str(all(value.get(key) == item for key, item in expected.items())).lower())
PY
"""
    try:
        if session is None:
            proc, key_cleanup, exact_id = _attach_and_remote(
                row, run_id, attempt, remote, 30
            )
            if (
                proc is None or proc.returncode != 0 or exact_id != ident
                or key_cleanup not in {"detached", "not_found"}
            ):
                return False
        else:
            proc = session.run(remote, 30)
            if proc is None or proc.returncode != 0:
                return False
        return "owner_verified=true" in proc.stdout.splitlines()
    except Exception:
        return False

def _find_run_target(rows: list[dict], marker: dict, run_id: str, attempt: str) -> tuple[dict | None, str | None]:
    old_label = marker["resume_existing_label"]
    fresh_label = _fresh_label(run_id, attempt)
    return _labels_for_run(rows, old_label, fresh_label)


def _find_cleanup_target(
    marker: dict, run_id: str, attempt: str
) -> tuple[dict | None, str | None]:
    """Wait for a run-specific label to become visible before falling back.

    A start/create API response can be accepted just before its Actions process
    is killed. Inventory may briefly still show the pre-start label, so cleanup
    polls GET-only for the unique claim before considering the shared legacy
    label. A legacy label never grants stop ownership by itself.
    """
    old_label = marker["resume_existing_label"]
    fresh_label = _fresh_label(run_id, attempt)
    deadline = time.monotonic() + CLAIM_VISIBILITY_WAIT_SECONDS
    while True:
        rows = base.instance_rows()
        old = base.resolve_label(rows, old_label)
        fresh = base.resolve_label(rows, fresh_label)
        if old is not None and fresh is not None:
            raise GateError("run cleanup labels are ambiguous")
        if fresh is not None:
            return fresh, fresh_label
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return (old, old_label) if old is not None else (None, None)
        time.sleep(min(CLAIM_VISIBILITY_POLL_SECONDS, remaining))

def _monitor_window(
    row: dict,
    label: str,
    ident: int,
    run_id: str,
    attempt: str,
    root: str,
    window_seconds: int,
    session: RemoteSession,
) -> tuple[int, str, str]:
    host_deadline = time.monotonic() + max(1, min(int(window_seconds), MONITOR_SECONDS))
    next_budget_check = 0.0
    authorized_forecast_token: str | None = None

    def stop_for_gate(outcome: str) -> tuple[int, str, str]:
        stop_status, _key_status = base.stop_exact_label(
            label, run_id, attempt, expected_id=ident
        )
        if stop_status not in {"verified", "not_needed"}:
            return 1, "unverified", "unverified"
        return 1, outcome, outcome

    while True:
        (
            state,
            quality,
            started_at,
            budget_baseline,
            forecast_total_seconds,
            forecast_token,
        ) = _read_remote_status_full(
            row, run_id, attempt, root, session=session
        )
        if state == "completed":
            if quality == "passed":
                return 0, "completed", "completed"
            return 1, "failed", "failed"
        if state == "budget_gate":
            return 1, "budget_gate", "budget_gate"
        if state == "time_limit":
            return 1, "time_limit", "time_limit"
        if state == "failed":
            return 1, "failed", "failed"
        if state == "unavailable":
            return 1, "remote_state_unavailable", "unavailable"

        now_monotonic = time.monotonic()
        if (
            (forecast_total_seconds is None) != (forecast_token is None)
            or (
                forecast_token is not None
                and not re.fullmatch(r"[0-9a-f]{32}", forecast_token)
            )
        ):
            return stop_for_gate("unverified")

        forecast_pending_authorization = (
            forecast_token is not None
            and forecast_token != authorized_forecast_token
        )
        if now_monotonic >= next_budget_check or forecast_pending_authorization:
            current_rows = base.instance_rows()
            current = base.resolve_label(current_rows, label)
            if (
                current is None
                or base.instance_id(current) != ident
                or base.probe.observed_state(current) != "running"
            ):
                return 1, "unverified", "unverified"
            spend_now = base.project_charges()
            rate = _finite(current.get("dph_total"))
            extra_storage = _other_storage_reserve(current_rows, ident)
            if (
                rate is None or extra_storage is None
                or started_at is None or budget_baseline is None
            ):
                return stop_for_gate("budget_gate")
            elapsed_seconds = max(0.0, time.time() - started_at)
            elapsed_hours = elapsed_seconds / 3600.0
            projected_spend = max(spend_now, budget_baseline + rate * elapsed_hours)
            if forecast_total_seconds is None:
                if elapsed_seconds >= BOOTSTRAP_SECONDS:
                    return stop_for_gate("time_limit")
                remaining_hours = (
                    (BOOTSTRAP_SECONDS - elapsed_seconds) / 3600.0
                    + (MAX_STOP_SECONDS + BUDGET_POLL_SECONDS) / 3600.0
                )
            else:
                if (
                    forecast_total_seconds > MAX_RUNTIME_SECONDS
                    or elapsed_seconds >= forecast_total_seconds
                ):
                    return stop_for_gate("time_limit")
                remaining_hours = (
                    (forecast_total_seconds - elapsed_seconds) / 3600.0
                    + (MAX_STOP_SECONDS + BUDGET_POLL_SECONDS) / 3600.0
                )
            if not full_run_budget_allows(
                projected_spend,
                rate,
                current.get("storage_cost"),
                active_hours=remaining_hours,
                extra_storage_reserve=extra_storage,
            ):
                return stop_for_gate("budget_gate")
            if forecast_token is not None:
                if not _write_forecast_authorization(
                    session, root, forecast_token, forecast_total_seconds
                ):
                    return stop_for_gate("unverified")
                authorized_forecast_token = forecast_token
            row = current
            next_budget_check = now_monotonic + BUDGET_POLL_SECONDS
        if now_monotonic >= host_deadline:
            return 0, "running", "running"
        time.sleep(min(POLL_SECONDS, max(1.0, host_deadline - now_monotonic)))



def monitor(window_seconds: int = MONITOR_SECONDS) -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome, status = "controller_failed", "unavailable"
    session = None
    result_code = 1
    try:
        revision, run_id, attempt, root = _run_identity()
        marker = _marker()
        rows = base.instance_rows()
        row, label = _find_run_target(rows, marker, run_id, attempt)
        if row is None or label is None or not base.is_2x3060(row):
            outcome, status = "target_unavailable", "unavailable"
            raise GateError("exact full-training target is unavailable")
        if base.probe.observed_state(row) != "running":
            outcome, status = "target_busy", "unverified"
            raise GateError("exact full-training target is not running")
        ident = base.instance_id(row)
        if ident is None:
            outcome, status = "target_unavailable", "unavailable"
            raise GateError("target identity is unavailable")
        session = RemoteSession(row, run_id, attempt)
        if not _read_remote_owner(
            row, run_id, attempt, root, revision, session=session
        ):
            outcome, status = "target_unavailable", "unavailable"
            raise GateError("remote workspace ownership is not verified")
        result_code, outcome, status = _monitor_window(
            row, label, ident, run_id, attempt, root, window_seconds, session
        )
    except base.probe.ApiFailure:
        outcome, status = "api_failed", "unverified"
    except (GateError, base.GateError):
        if outcome == "controller_failed":
            outcome, status = "target_unavailable", "unavailable"
    except Exception:
        outcome, status = "controller_failed", "unavailable"
    finally:
        if session is not None:
            key_cleanup = session.close()
            if key_cleanup not in {"detached", "not_found"} and result_code == 0:
                outcome, status, result_code = "api_failed", "unverified", 1
    _summarize(checked, outcome, status)
    return result_code

def budget_check() -> int:
    """Read-only, redacted classification for the pinned full-training target."""
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome = "failed"
    budget_status = "unavailable"
    bootstrap_status = "unavailable"
    try:
        marker = _marker()
        rows = base.instance_rows()
        targets = base.target_rows_for_label(rows, marker["resume_existing_label"])
        if not targets:
            budget_status = bootstrap_status = "target_missing"
        elif len(targets) != 1:
            budget_status = bootstrap_status = "target_ambiguous"
        else:
            spend = base.project_charges()
            budget_status = classify_full_run_budget(targets[0], rows, spend)
            bootstrap_status = classify_bootstrap_budget(targets[0], rows, spend)
        outcome = "success"
    except Exception:
        outcome, budget_status, bootstrap_status = "failed", "unavailable", "unavailable"
    _summarize_budget(checked, outcome, budget_status, bootstrap_status)
    return 0 if outcome == "success" else 1



def cleanup() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome, status = "controller_failed", "unverified"
    state, quality = "unavailable", "unavailable"
    session = None
    try:
        revision, run_id, attempt, root = _run_identity()
        marker = _marker()
        row, label = _find_cleanup_target(marker, run_id, attempt)
        if row is None or label is None:
            outcome, status = "not_needed", "not_needed"
        elif not base.is_2x3060(row):
            outcome, status = "target_gpu_mismatch", "unverified"
        else:
            ident = base.instance_id(row)
            if ident is None:
                outcome, status = "unverified", "unverified"
            else:
                old_label = marker["resume_existing_label"]
                run_claimed_label = _fresh_label(run_id, attempt)
                actual_state = base.probe.observed_state(row)
                owned = label == run_claimed_label
                if label == old_label:
                    # A shared historical label proves no ownership. Only the
                    # run-specific private owner stamp proves this run launched
                    # work on an already-running resumable target.
                    if actual_state == "running":
                        session = RemoteSession(row, run_id, attempt)
                        owned = _read_remote_owner(
                            row, run_id, attempt, root, revision, session=session
                        )
                        if owned:
                            state, quality, started_at, budget_baseline = _read_remote_status(
                                row, run_id, attempt, root, session=session
                            )
                        else:
                            outcome, status = "not_needed", "not_needed"
                    else:
                        outcome, status = "not_needed", "not_needed"
                elif actual_state == "running":
                    state, quality, started_at, budget_baseline = _read_remote_status(
                        row, run_id, attempt, root
                    )
                else:
                    started_at = None
                    budget_baseline = None
                if session is not None:
                    key_cleanup = session.close()
                    session = None
                    if key_cleanup not in {"detached", "not_found"}:
                        state, quality = "unavailable", "unavailable"
                if owned:
                    rate = _finite(row.get("dph_total"))
                    stop_status, _key_status = base.stop_exact_label(
                        label, run_id, attempt, expected_id=ident
                    )
                    if stop_status not in {"verified", "not_needed"}:
                        outcome, status = "unverified", "unverified"
                    else:
                        final_rows = base.instance_rows()
                        final_row = base.resolve_label(final_rows, label)
                        if (
                            final_row is None
                            or base.instance_id(final_row) != ident
                            or base.probe.observed_state(final_row) not in {"stopped", "exited"}
                        ):
                            outcome, status = "unverified", "unverified"
                        else:
                            spend = base.project_charges()
                            extra = _other_storage_reserve(final_rows, ident)
                            elapsed_hours = (
                                max(0.0, (time.time() - started_at) / 3600.0)
                                if started_at is not None else 0.0
                            )
                            baseline = budget_baseline if budget_baseline is not None else spend
                            projected_spend = max(spend, baseline + (rate or 0.0) * elapsed_hours)
                            if rate is None or extra is None or not full_run_budget_allows(
                                projected_spend,
                                rate,
                                final_row.get("storage_cost"),
                                active_hours=0.0,
                                extra_storage_reserve=extra,
                            ):
                                outcome, status = "budget_gate", "budget_gate"
                            elif state != "completed" or quality != "passed":
                                outcome, status = (
                                    "time_limit" if state == "time_limit" else "failed"
                                ), "failed"
                            else:
                                outcome, status = "accepted", "stopped"
    except base.probe.ApiFailure:
        outcome, status = "api_failed", "unverified"
    except (GateError, base.GateError):
        outcome, status = "unverified", "unverified"
    except Exception:
        outcome, status = "controller_failed", "unverified"
    finally:
        if session is not None:
            session.close()
    _summarize(checked, outcome, status)
    return 0 if outcome in {"accepted", "not_needed"} else 1

def _summarize(
    checked: str,
    outcome: str,
    status: str,
    cleanup_needed: bool | None = None,
) -> None:
    if outcome not in OUTCOMES:
        outcome = "controller_failed"
    if status not in STATUSES:
        status = "unavailable"
    if cleanup_needed is not None and type(cleanup_needed) is not bool:
        raise GateError("cleanup output flag is invalid")
    print(f"checked_at={checked}")
    print(f"query_outcome={outcome}")
    print(f"status={status}")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        try:
            with open(output_path, "a", encoding="utf-8") as output:
                output.write(f"status={status}\n")
                if cleanup_needed is not None:
                    value = "yes" if cleanup_needed else "no"
                    output.write(f"cleanup_needed={value}\n")
        except OSError:
            raise GateError("controller step output could not be written") from None
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as summary:
                summary.write(
                    "## Full CNOT training controller\\n"
                    f"- checked_at: {checked}\\n"
                    f"- query_outcome: {outcome}\\n"
                    f"- status: {status}\\n"
                )
        except OSError:
            pass


def _summarize_budget(
    checked: str, outcome: str, budget_status: str, bootstrap_status: str = "unavailable"
) -> None:
    allowed_statuses = {
        "within_cap", "insufficient_remaining_budget", "run_projection_exceeds_cap",
        "inputs_unknown", "target_missing", "target_ambiguous", "target_mismatch",
        "unavailable",
    }
    if outcome not in {"success", "failed"}:
        outcome = "failed"
    if budget_status not in allowed_statuses:
        budget_status = "unavailable"
    if bootstrap_status not in allowed_statuses:
        bootstrap_status = "unavailable"
    print(f"checked_at={checked}")
    print(f"query_outcome={outcome}")
    print(f"training_budget={budget_status}")
    print(f"bootstrap_budget={bootstrap_status}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as summary:
                summary.write(
                    "## GET-only full-training budget check\n"
                    f"- checked_at: {checked}\n"
                    f"- query_outcome: {outcome}\n"
                    f"- training_budget: {budget_status}\n"
                    f"- bootstrap_budget: {bootstrap_status}\n"
                )
        except OSError:
            pass



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("start", "monitor", "cleanup", "budget-check"))
    parser.add_argument("--window-seconds", type=int, default=MONITOR_SECONDS)
    args = parser.parse_args()
    if args.mode == "start":
        return start()
    if args.mode == "monitor":
        return monitor(args.window_seconds)
    if args.mode == "cleanup":
        return cleanup()
    return budget_check()


if __name__ == "__main__":
    raise SystemExit(main())
