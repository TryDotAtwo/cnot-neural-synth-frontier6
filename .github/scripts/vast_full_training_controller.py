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
BILLING_LAG_BUFFER_SECONDS = 60 * 60
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
QUALITY_STATES = {"not_ready", "measured", "passed", "failed", "inconclusive", "invalid", "unavailable"}


class GateError(RuntimeError):
    pass



def _finite(value: object) -> float | None:
    return base.finite_number(value)


def projected_billable_spend(
    invoice_now: object,
    invoice_baseline: object,
    billing_baseline_epoch: object,
    now_epoch: object,
    instance_rate: object,
    invoice_lag_buffer_seconds: object = BILLING_LAG_BUFFER_SECONDS,
) -> float | None:
    """Use a same-time invoice baseline plus elapsed billing and lag reserve."""
    current = _finite(invoice_now)
    baseline = _finite(invoice_baseline)
    baseline_epoch = _finite(billing_baseline_epoch)
    now = _finite(now_epoch)
    rate = _finite(instance_rate)
    lag = _finite(invoice_lag_buffer_seconds)
    if (
        current is None or baseline is None or baseline_epoch is None or now is None
        or rate is None or lag is None or current < 0 or baseline < 0 or rate < 0
        or lag < 0 or now < baseline_epoch
    ):
        return None
    projected = baseline + rate * (now - baseline_epoch + lag) / 3600.0
    if not math.isfinite(projected):
        return None
    return max(current, projected)


def full_run_time_fits(
    started_at_epoch: object,
    remaining_work_seconds: object,
    now_epoch: object,
    cleanup_reserve_seconds: object = MAX_STOP_SECONDS + BUDGET_POLL_SECONDS,
) -> bool:
    started = _finite(started_at_epoch)
    remaining = _finite(remaining_work_seconds)
    now = _finite(now_epoch)
    cleanup = _finite(cleanup_reserve_seconds)
    if (
        started is None or remaining is None or now is None or cleanup is None
        or started <= 0 or remaining < 0 or cleanup < 0 or now < started
    ):
        return False
    return now - started + remaining + cleanup <= MAX_RUNTIME_SECONDS


def _storage_reserve(storage_cost_per_hour: object, hours: float = STORAGE_HOURS) -> float | None:
    """Reserve the instance API's hourly storage charge for each retained hour."""
    hourly = _finite(storage_cost_per_hour)
    reserve_hours = _finite(hours)
    if hourly is None or reserve_hours is None or hourly < 0 or reserve_hours < 0:
        return None
    return hourly * reserve_hours


def _storage_reserve_for_row(row: object, hours: float = STORAGE_HOURS) -> float | None:
    if not isinstance(row, dict):
        return None
    disk_gb = _finite(row.get("disk_space"))
    if disk_gb is None or disk_gb <= 0:
        return None
    return _storage_reserve(row.get("storage_cost"), hours)


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


def _approved_instance_id(_marker: dict) -> int:
    # Stable identity is private deployment configuration, never a repository marker.
    secret = os.environ.get("CNOT103_APPROVED_TARGET_INSTANCE_ID", "").strip()
    if not re.fullmatch(r"[0-9]{1,20}", secret):
        raise GateError("approved target configuration is missing or invalid")
    return int(secret)


def _row_for_approved_target(
    rows: list[dict], marker: dict
) -> tuple[dict | None, str | None]:
    ident = _approved_instance_id(marker)
    matches = [row for row in rows if base.instance_id(row) == ident]
    if len(matches) > 1:
        raise GateError("approved target identity is ambiguous")
    if not matches:
        return None, None
    row = matches[0]
    label = row.get("label")
    allowed_label = (
        isinstance(label, str)
        and (
            label == marker["resume_existing_label"]
            or re.fullmatch(r"cnot103-fulltrain-[0-9]{1,20}-a[0-9]{1,10}", label)
        )
    )
    if not allowed_label or not base.is_2x3060(row):
        raise GateError("approved target identity or GPU inventory mismatches")
    same_label = [item for item in rows if item.get("label") == label]
    if len(same_label) != 1 or base.instance_id(same_label[0]) != ident:
        raise GateError("approved target label is ambiguous")
    return row, label


def _other_storage_reserve(rows: list[dict], exclude_id: int | None) -> float | None:
    total = 0.0
    for row in rows:
        ident = base.instance_id(row)
        if ident is not None and ident == exclude_id:
            continue
        label = str(row.get("label") or "").casefold()
        if not (label.startswith("cnot103") or ident in base.OWNED_INSTANCE_IDS):
            continue
        reserve = _storage_reserve_for_row(row)
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


def _has_active_supervisor(
    row: dict, run_id: str, attempt: str
) -> bool:
    remote = r"""set -euo pipefail
python3 - <<'PY'
import glob
import os
active = False
for pid_path in glob.glob("/workspace/tqp_full_training_*/supervisor.pid"):
    try:
        pid = int(open(pid_path, encoding="ascii").read().strip())
        if pid <= 0:
            continue
        os.kill(pid, 0)
        command = open(f"/proc/{{pid}}/cmdline", "rb").read()
    except (OSError, ValueError):
        continue
    if b"run_full_training.py" in command:
        active = True
        break
print("full_training=supervisor_active" if active else "full_training=no_supervisor")
PY
"""
    proc, key_cleanup, _ident = _attach_and_remote(row, run_id, attempt, remote, 30)
    if proc is None or proc.returncode != 0 or key_cleanup not in {"detached", "not_found"}:
        raise GateError("existing supervisor ownership could not be checked")
    statuses = set(proc.stdout.splitlines())
    if "full_training=supervisor_active" in statuses:
        return True
    if "full_training=no_supervisor" in statuses:
        return False
    raise GateError("existing supervisor ownership check was inconclusive")


def _launch_script(
    root: str,
    source_revision: str,
    holdout_revision: str,
    holdout_sha256: str,
    started_at_epoch: float,
    budget_baseline: float,
    bootstrap_deadline_epoch: float,
    startup_deadline_epoch: float,
    run_id: str,
    attempt: str,
    instance_id: int,
    billing_baseline_epoch: float | None = None,
) -> str:
    q = shlex.quote
    source = root + "/source"
    billing_epoch = started_at_epoch if billing_baseline_epoch is None else billing_baseline_epoch
    return f"""set -euo pipefail
ROOT={q(root)}
SOURCE={q(source)}
STARTUP_DEADLINE={q(str(startup_deadline_epoch))}
exec 9>/workspace/.cnot103-full-training-launch.lock
if ! flock -n 9; then
  printf 'full_training=supervisor_busy\\n'
  exit 13
fi
python3 - "$ROOT" "$STARTUP_DEADLINE" <<'PY'
import glob
import os
import sys
import time
root, raw_deadline = sys.argv[1:]
try:
    deadline = float(raw_deadline)
except ValueError:
    raise SystemExit(14)
if not (time.time() < deadline):
    print("full_training=start_deadline")
    raise SystemExit(14)
for pid_path in glob.glob("/workspace/tqp_full_training_*/supervisor.pid"):
    owner_root = os.path.dirname(pid_path)
    if owner_root == root:
        continue
    try:
        raw_pid = open(pid_path, encoding="ascii").read().strip()
        pid = int(raw_pid)
        if pid <= 0:
            continue
        os.kill(pid, 0)
        command = open(f"/proc/{{pid}}/cmdline", "rb").read()
    except (OSError, ValueError):
        continue
    if b"run_full_training.py" in command:
        print("full_training=supervisor_busy")
        raise SystemExit(13)
PY
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
python3 - "$STARTUP_DEADLINE" <<'PY'
import sys
import time
try:
    deadline = float(sys.argv[1])
except ValueError:
    raise SystemExit(14)
if time.time() >= deadline:
    print("full_training=start_deadline")
    raise SystemExit(14)
PY
nohup setsid python3 {q(source + "/.github/scripts/run_full_training.py")} \\
  --root {q(root)} \\
  --source-revision {q(source_revision)} \\
  --holdout-revision {q(holdout_revision)} \\
  --holdout-sha256 {q(holdout_sha256)} \\
  --started-at-epoch {q(str(started_at_epoch))} \\
  --budget-baseline {q(str(budget_baseline))} \\
  --billing-baseline-epoch {q(str(billing_epoch))} \\
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
        if time.monotonic() >= deadline:
            raise GateError("target startup wait reached its three-hour deadline")
        rows = base.instance_rows()
        # A blocking GET may return after the deadline; never accept its state.
        if time.monotonic() >= deadline:
            raise GateError("target startup wait reached its three-hour deadline")
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
    if time.monotonic() >= deadline:
        raise GateError("target startup wait reached its three-hour deadline")
    ident = base.instance_id(target)
    if ident is None:
        raise GateError("exact target id is invalid")
    rows = base.instance_rows()
    if time.monotonic() >= deadline:
        raise GateError("target startup wait reached its three-hour deadline")
    current = next((item for item in rows if base.instance_id(item) == ident), None)
    labeled = base.resolve_label(rows, current_label)
    if labeled is not None and base.instance_id(labeled) != ident:
        raise GateError("exact target label moved to another instance")
    if current is None or current.get("label") != current_label or not base.is_2x3060(current):
        raise GateError("exact target identity, label, or GPU changed before start")
    state = base.probe.observed_state(current)
    if state not in {"running", "stopped", "exited"}:
        current = _wait_for_exact_instance(ident, current_label, deadline)
        state = base.probe.observed_state(current)
    if state == "running":
        return current, current_label
    if state not in {"stopped", "exited"}:
        raise GateError("target did not reach a startable state")
    if time.monotonic() >= deadline:
        raise GateError("target startup wait reached its three-hour deadline")
    if current_label != fresh_label:
        existing_fresh = base.resolve_label(rows, fresh_label)
        if existing_fresh is not None:
            raise GateError("run-specific start label is already occupied")
    spend = base.project_charges()
    if not _validate_bootstrap_budget(
        current, spend, rows, ident,
        active_hours=_bootstrap_remaining_hours(started_at_epoch),
    ):
        raise GateError("the bounded pilot budget gate is closed before start")
    # Recheck the monotonic deadline immediately before the only start mutation.
    if time.monotonic() >= deadline:
        raise GateError("target startup wait reached its three-hour deadline")
    if not base.api_instance_id(ident, "running", label=fresh_label):
        raise GateError("exact target start was not confirmed")
    started = _wait_for_exact_instance(ident, fresh_label, deadline)
    if base.probe.observed_state(started) != "running":
        raise GateError("target did not reach running after its one start request")
    return started, fresh_label


def _start_target(
    rows: list[dict], marker: dict, run_id: str, attempt: str,
    started_at_epoch: float | None = None,
    deadline: float | None = None,
) -> tuple[dict, str, bool, float]:
    if deadline is None:
        deadline = time.monotonic() + STARTUP_WAIT_SECONDS
    if time.monotonic() >= deadline:
        raise GateError("target startup wait reached its three-hour deadline")
    target, label = _row_for_approved_target(rows, marker)
    if target is None or label is None:
        raise GateError("approved existing target is missing")
    ident = base.instance_id(target)
    if ident is None or not base.is_2x3060(target):
        raise GateError("target GPU inventory does not match")
    spent = base.project_charges()
    bootstrap_hours = _bootstrap_remaining_hours(started_at_epoch)
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
            target, label, _fresh_label(run_id, attempt), deadline, started_at_epoch
        )
    elif state != "running":
        raise GateError("exact target did not reach a usable state")
    return target, label, False, spent



def _offer_storage_hourly_cost(row: object) -> float | None:
    """Convert the offer's monthly per-GB tariff for the requested 64 GB disk."""
    if not isinstance(row, dict):
        return None
    monthly_per_gb = _finite(row.get("storage_cost"))
    disk_gb = _finite(row.get("allocated_storage"))
    if disk_gb is None:
        disk_gb = _finite(row.get("disk_space"))
    if (
        monthly_per_gb is None or disk_gb is None
        or monthly_per_gb < 0 or disk_gb != DISK_GB
    ):
        return None
    hourly = monthly_per_gb * disk_gb / 720.0
    return hourly if math.isfinite(hourly) else None


def classify_new_rental_budget(
    offers: object,
    rows: object,
    spent: object,
    active_hours: object = ACTIVE_HOURS,
) -> str:
    """Classify a GET-only new-rental projection without exposing prices."""
    if not isinstance(offers, list) or not isinstance(rows, list):
        return "inputs_unknown"
    if not offers:
        return "no_eligible_offer"
    spend_value = _finite(spent)
    active = _finite(active_hours)
    extra = _other_storage_reserve(rows, exclude_id=None)
    if (
        spend_value is None or spend_value < 0 or active is None or active < 0
        or extra is None or extra < 0
    ):
        return "inputs_unknown"

    saw_valid_offer = False
    saw_unknown_offer = False
    fits_without_prior_spend = False
    for offer in offers:
        if not isinstance(offer, dict) or not base.is_2x3060(offer):
            saw_unknown_offer = True
            continue
        rate = _finite(offer.get("dph_total"))
        storage_hourly = _offer_storage_hourly_cost(offer)
        if rate is None or rate <= 0 or storage_hourly is None:
            saw_unknown_offer = True
            continue
        saw_valid_offer = True
        if full_run_budget_allows(
            spend_value,
            rate,
            storage_hourly,
            active_hours=active,
            extra_storage_reserve=extra,
        ):
            return "within_cap"
        if full_run_budget_allows(
            0.0,
            rate,
            storage_hourly,
            active_hours=active,
            extra_storage_reserve=extra,
        ):
            fits_without_prior_spend = True

    if saw_unknown_offer or not saw_valid_offer:
        return "inputs_unknown" if saw_unknown_offer else "no_eligible_offer"
    if fits_without_prior_spend:
        return "insufficient_remaining_budget"
    return "run_projection_exceeds_cap"


def classify_full_run_budget(row: object, rows: object, spent: object) -> str:
    """Classify the pinned full-run budget without exposing financial values."""
    if not isinstance(row, dict):
        return "target_missing"
    if not base.is_2x3060(row):
        return "target_mismatch"
    ident = base.instance_id(row)
    spend_value = _finite(spent)
    rate = _finite(row.get("dph_total"))
    storage_hourly = _finite(row.get("storage_cost"))
    storage_reserve = _storage_reserve_for_row(row)
    extra = _other_storage_reserve(rows, ident) if isinstance(rows, list) and ident else None
    if (
        ident is None or spend_value is None or rate is None or storage_hourly is None
        or storage_reserve is None or extra is None or spend_value < 0 or rate < 0
        or storage_hourly < 0 or extra < 0
    ):
        return "inputs_unknown"
    if full_run_budget_allows(
        spend_value, rate, storage_hourly, extra_storage_reserve=extra
    ):
        return "within_cap"
    if full_run_budget_allows(
        0.0, rate, storage_hourly, extra_storage_reserve=extra
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
    storage_hourly = _finite(row.get("storage_cost"))
    storage_reserve = _storage_reserve_for_row(row)
    extra = _other_storage_reserve(rows, ident) if isinstance(rows, list) and ident else None
    active = _finite(active_hours)
    if (
        ident is None or spend_value is None or rate is None or storage_hourly is None
        or storage_reserve is None or extra is None or active is None or spend_value < 0
        or rate < 0 or storage_hourly < 0 or extra < 0 or active < 0
    ):
        return "inputs_unknown"
    if bootstrap_budget_allows(
        spend_value, rate, storage_hourly, active_hours=active,
        extra_storage_reserve=extra,
    ):
        return "within_cap"
    if bootstrap_budget_allows(
        0.0, rate, storage_hourly, active_hours=active,
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
    return (
        _storage_reserve_for_row(row) is not None
        and extra is not None
        and bootstrap_budget_allows(
            spend,
            row.get("dph_total"),
            row.get("storage_cost"),
            active_hours=active_hours,
            extra_storage_reserve=extra,
        )
    )


def start() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome, status = "controller_failed", "unavailable"
    run_id = attempt = ""
    try:
        started_at_epoch = time.time()
        startup_deadline = time.monotonic() + STARTUP_WAIT_SECONDS
        revision, run_id, attempt, root = _run_identity()
        holdout_revision, holdout_sha256 = _pin_status()
        marker = _marker()
        rows = base.instance_rows()
        row, label, _created, _spend_before = _start_target(
            rows, marker, run_id, attempt, started_at_epoch,
            deadline=startup_deadline,
        )
        ident = base.instance_id(row)
        if ident is None or ident != _approved_instance_id(marker) or not base.is_2x3060(row):
            outcome, status = "target_gpu_mismatch", "failed"
            raise GateError("running target identity or GPU differs")
        running_rows = base.instance_rows()
        current, current_label = _row_for_approved_target(running_rows, marker)
        if (
            current is None or current_label != label
            or base.instance_id(current) != ident
            or base.probe.observed_state(current) != "running"
        ):
            outcome, status = "target_busy", "unverified"
            raise GateError("exact target changed before remote launch")
        row = current
        billing_baseline_epoch = time.time()
        spend_now = base.project_charges()
        if not _validate_bootstrap_budget(
            row, spend_now, running_rows, ident,
            active_hours=_bootstrap_remaining_hours(started_at_epoch),
        ):
            outcome, status = "budget_gate", "budget_gate"
            raise GateError("post-start bounded pilot budget recheck failed")
        if _has_active_supervisor(row, run_id, attempt):
            outcome, status = "target_busy", "unverified"
            raise GateError("an existing training supervisor already owns this target")
        if time.monotonic() >= startup_deadline:
            outcome, status = "target_busy", "unverified"
            raise GateError("target startup wait reached its three-hour deadline")
        startup_deadline_epoch = time.time() + max(
            0.0, startup_deadline - time.monotonic()
        )
        script = _launch_script(
            root,
            revision,
            holdout_revision,
            holdout_sha256,
            started_at_epoch,
            spend_now,
            started_at_epoch + BOOTSTRAP_SECONDS,
            startup_deadline_epoch,
            run_id,
            attempt,
            ident,
            billing_baseline_epoch,
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
) -> tuple[str, str, float | None, float | None, float | None, float | None, float | None, str | None]:
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
for name in ("started_at_epoch", "budget_baseline", "billing_baseline_epoch", "remaining_work_seconds", "instance_rate", "storage_cost"):
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
            return "unavailable", "unavailable", None, None, None, None, None, None
    else:
        proc = session.run(remote, 30)
        if proc is None or proc.returncode != 0:
            return "unavailable", "unavailable", None, None, None, None, None, None
    values = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {
            "run_state", "quality_status", "started_at_epoch",
            "budget_baseline", "billing_baseline_epoch", "remaining_work_seconds",
            "instance_rate", "storage_cost", "forecast_total_seconds", "forecast_token",
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
        parsed_number("billing_baseline_epoch"),
        parsed_number("forecast_total_seconds"),
        parsed_number("remaining_work_seconds"),
        token,
    )


def _read_remote_status(
    row: dict, run_id: str, attempt: str, root: str,
    session: RemoteSession | None = None,
) -> tuple[str, str, float | None, float | None, float | None]:
    values = _read_remote_status_full(row, run_id, attempt, root, session=session)
    return values[0], values[1], values[2], values[3], values[4]


def _write_forecast_authorization(
    session: RemoteSession,
    root: str,
    token: str,
    forecast_total_seconds: float,
    remaining_work_seconds: float | None = None,
    authorized_at_epoch: float | None = None,
) -> bool:
    seconds = _finite(forecast_total_seconds)
    remaining = _finite(remaining_work_seconds)
    authorized_at = time.time() if authorized_at_epoch is None else _finite(authorized_at_epoch)
    if (
        not re.fullmatch(r"[0-9a-f]{32}", token)
        or seconds is None or seconds <= 0
        or remaining is None or remaining <= 0
        or authorized_at is None or authorized_at <= 0
    ):
        return False
    q = shlex.quote
    path = root + "/forecast-approval.private.json"
    remote = f"""set -euo pipefail
python3 - {q(path)} {q(token)} {q(str(seconds))} {q(str(remaining))} {q(str(authorized_at))} <<'PY'
import json
import math
import os
import stat
import sys
path, token, raw_seconds, raw_remaining, raw_authorized_at = sys.argv[1:]
seconds, remaining, authorized_at = map(float, (raw_seconds, raw_remaining, raw_authorized_at))
if not all(math.isfinite(x) for x in (seconds, remaining, authorized_at)) or seconds <= 0 or remaining <= 0 or authorized_at <= 0:
    raise SystemExit(2)
value = {{
    "schema": "cnot103_full_training_forecast_approval_v1",
    "authorized": True,
    "forecast_token": token,
    "forecast_total_seconds": seconds,
    "remaining_work_seconds": remaining,
    "authorized_at_epoch": authorized_at,
}}
try:
    existing = json.load(open(path, encoding="utf-8"))
except FileNotFoundError:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
except Exception:
    raise SystemExit(3)
else:
    if (
        not isinstance(existing, dict)
        or existing.get("schema") != value["schema"]
        or existing.get("authorized") is not True
        or existing.get("forecast_token") != token
        or existing.get("forecast_total_seconds") != seconds
        or existing.get("remaining_work_seconds") != remaining
    ):
        raise SystemExit(4)
    try:
        if not stat.S_ISREG(os.stat(path, follow_symlinks=False).st_mode):
            raise SystemExit(5)
        descriptor = os.open(
            path, os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError:
        raise SystemExit(5)
with os.fdopen(descriptor, "w", encoding="utf-8") as output:
    json.dump(value, output, sort_keys=True, separators=(",", ":"))
    output.write("\\n")
    output.flush()
    os.fsync(output.fileno())
print("full_training=forecast_authorized")
PY
"""
    try:
        proc = session.run(remote, 30)
    except Exception:
        return False
    return (
        proc is not None
        and proc.returncode == 0
        and "full_training=forecast_authorized" in getattr(proc, "stdout", "").splitlines()
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

def _find_run_target(
    rows: list[dict], marker: dict, run_id: str, attempt: str
) -> tuple[dict | None, str | None]:
    """Resolve the configured stable instance ID; labels are only consistency checks."""
    return _row_for_approved_target(rows, marker)


def _find_cleanup_target(
    marker: dict, run_id: str, attempt: str
) -> tuple[dict | None, str | None]:
    """GET-only recovery by stable ID; a current-attempt label proves a start claim."""
    ident = _approved_instance_id(marker)
    fresh_label = _fresh_label(run_id, attempt)
    deadline = time.monotonic() + CLAIM_VISIBILITY_WAIT_SECONDS
    while True:
        rows = base.instance_rows()
        fresh = base.resolve_label(rows, fresh_label)
        if fresh is not None and base.instance_id(fresh) != ident:
            raise GateError("run cleanup label points to a different instance")
        row, label = _row_for_approved_target(rows, marker)
        if row is not None and label == fresh_label:
            return row, label
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return row, label
        time.sleep(min(CLAIM_VISIBILITY_POLL_SECONDS, remaining))


def _confirm_stopped_instance(ident: int, label: str) -> bool:
    try:
        rows = base.instance_rows()
        current = next((item for item in rows if base.instance_id(item) == ident), None)
        labeled = base.resolve_label(rows, label)
        return (
            current is not None
            and current.get("label") == label
            and (labeled is None or base.instance_id(labeled) == ident)
            and base.probe.observed_state(current) in {"stopped", "exited"}
        )
    except Exception:
        return False


def _monitor_window(
    row: dict,
    label: str,
    ident: int,
    run_id: str,
    attempt: str,
    root: str,
    window_seconds: int,
    session: RemoteSession,
) -> tuple[int, str, str, str]:
    host_deadline = time.monotonic() + max(1, min(int(window_seconds), MONITOR_SECONDS))
    next_budget_check = 0.0
    authorized_forecast_token: str | None = None

    def stop_for_gate(outcome: str) -> tuple[int, str, str, str]:
        stop_status, _key_status = base.stop_exact_label(
            label, run_id, attempt, expected_id=ident
        )
        if stop_status not in {"verified", "not_needed"}:
            return 1, "unverified", "unverified", "unverified"
        if not _confirm_stopped_instance(ident, label):
            return 1, "unverified", "unverified", "unverified"
        return 1, outcome, outcome, "verified"

    while True:
        (
            state,
            quality,
            started_at,
            budget_baseline,
            billing_baseline_epoch,
            forecast_total_seconds,
            remaining_work_seconds,
            forecast_token,
        ) = _read_remote_status_full(
            row, run_id, attempt, root, session=session
        )
        if state == "completed":
            if quality == "passed":
                return 0, "completed", "completed", "not_attempted"
            return 1, "failed", "failed", "not_attempted"
        if state == "budget_gate":
            return 1, "budget_gate", "budget_gate", "not_attempted"
        if state == "time_limit":
            return 1, "time_limit", "time_limit", "not_attempted"
        if state == "failed":
            return 1, "failed", "failed", "not_attempted"
        if state == "unavailable":
            return 1, "remote_state_unavailable", "unavailable", "not_attempted"

        now_monotonic = time.monotonic()
        if (
            (forecast_total_seconds is None) != (forecast_token is None)
            or (forecast_token is None) != (remaining_work_seconds is None)
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
                return 1, "unverified", "unverified", "not_attempted"
            spend_now = base.project_charges()
            now_epoch = time.time()
            rate = _finite(current.get("dph_total"))
            extra_storage = _other_storage_reserve(current_rows, ident)
            projected_spend = projected_billable_spend(
                spend_now,
                budget_baseline,
                billing_baseline_epoch,
                now_epoch,
                rate,
            )
            if (
                rate is None or extra_storage is None or projected_spend is None
                or _storage_reserve_for_row(current) is None or started_at is None
            ):
                return stop_for_gate("budget_gate")
            elapsed_seconds = max(0.0, now_epoch - started_at)
            cleanup_reserve_seconds = MAX_STOP_SECONDS + BUDGET_POLL_SECONDS
            if forecast_total_seconds is None:
                if elapsed_seconds >= BOOTSTRAP_SECONDS:
                    return stop_for_gate("time_limit")
                remaining_hours = (
                    (BOOTSTRAP_SECONDS - elapsed_seconds + cleanup_reserve_seconds) / 3600.0
                )
            else:
                if (
                    remaining_work_seconds is None
                    or forecast_total_seconds > MAX_RUNTIME_SECONDS
                    or not full_run_time_fits(
                        started_at,
                        remaining_work_seconds,
                        now_epoch,
                        cleanup_reserve_seconds,
                    )
                ):
                    return stop_for_gate("time_limit")
                remaining_hours = (
                    (remaining_work_seconds + cleanup_reserve_seconds) / 3600.0
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
                    session,
                    root,
                    forecast_token,
                    forecast_total_seconds,
                    remaining_work_seconds,
                    now_epoch,
                ):
                    return stop_for_gate("unverified")
                authorized_forecast_token = forecast_token
            row = current
            next_budget_check = now_monotonic + BUDGET_POLL_SECONDS
        if now_monotonic >= host_deadline:
            return 0, "running", "running", "not_attempted"
        time.sleep(min(POLL_SECONDS, max(1.0, host_deadline - now_monotonic)))


def monitor(window_seconds: int = MONITOR_SECONDS) -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome, status = "controller_failed", "unavailable"
    session = None
    result_code = 1
    stop_verification = "not_attempted"
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
        result_code, outcome, status, stop_verification = _monitor_window(
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
    _summarize(checked, outcome, status, stop_verification=stop_verification)
    return result_code

def budget_check() -> int:
    """Read-only, redacted classification for the full-run target or rental offers."""
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome = "failed"
    budget_status = "unavailable"
    bootstrap_status = "unavailable"
    try:
        marker = _marker()
        rows = base.instance_rows()
        try:
            target, _label = _row_for_approved_target(rows, marker)
        except GateError as exc:
            if str(exc) != "approved target configuration is missing or invalid":
                raise
            owned_3060 = any(
                base.is_2x3060(row)
                and (
                    base.instance_id(row) in base.OWNED_INSTANCE_IDS
                    or str(row.get("label") or "").casefold().startswith("cnot103")
                )
                for row in rows
                if isinstance(row, dict)
            )
            if owned_3060:
                raise
            spend = base.project_charges()
            offers = base.offer_inventory()
            budget_status = classify_new_rental_budget(offers, rows, spend)
            bootstrap_status = classify_new_rental_budget(
                offers, rows, spend, active_hours=BOOTSTRAP_HOURS
            )
        else:
            if target is None:
                budget_status = bootstrap_status = "target_missing"
            else:
                spend = base.project_charges()
                budget_status = classify_full_run_budget(target, rows, spend)
                bootstrap_status = classify_bootstrap_budget(target, rows, spend)
        outcome = "success"
    except GateError as exc:
        reason = str(exc)
        if reason == "approved target configuration is missing or invalid":
            classified = "configuration_missing"
        elif reason in {
            "approved target identity is ambiguous",
            "approved target label is ambiguous",
        }:
            classified = "target_ambiguous"
        elif reason == "approved target identity or GPU inventory mismatches":
            classified = "target_mismatch"
        else:
            classified = "unavailable"
        outcome, budget_status, bootstrap_status = "failed", classified, classified
    except Exception:
        outcome, budget_status, bootstrap_status = "failed", "unavailable", "unavailable"
    _summarize_budget(checked, outcome, budget_status, bootstrap_status)
    return 0 if outcome == "success" else 1



def cleanup() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome, status = "controller_failed", "unverified"
    state, quality = "unavailable", "unavailable"
    stop_verification = "not_attempted"
    session = None
    try:
        revision, run_id, attempt, root = _run_identity()
        marker = _marker()
        row, label = _find_cleanup_target(marker, run_id, attempt)
        if row is None or label is None:
            outcome, status, stop_verification = "not_needed", "not_needed", "not_needed"
        elif not base.is_2x3060(row):
            outcome, status = "target_gpu_mismatch", "unverified"
        else:
            ident = base.instance_id(row)
            if ident is None:
                outcome, status = "unverified", "unverified"
            else:
                run_claimed_label = _fresh_label(run_id, attempt)
                actual_state = base.probe.observed_state(row)
                owned = label == run_claimed_label
                started_at = None
                budget_baseline = None
                billing_baseline_epoch = None
                if not owned and actual_state == "running":
                    # Any non-current label, including a previous attempt's
                    # label, needs the private run-and-workspace owner stamp.
                    session = RemoteSession(row, run_id, attempt)
                    owned = _read_remote_owner(
                        row, run_id, attempt, root, revision, session=session
                    )
                    if owned:
                        (
                            state, quality, started_at, budget_baseline,
                            billing_baseline_epoch,
                        ) = _read_remote_status(row, run_id, attempt, root, session=session)
                elif not owned:
                    outcome, status, stop_verification = "not_needed", "not_needed", "not_needed"
                if owned and actual_state == "running" and session is None:
                    (
                        state, quality, started_at, budget_baseline,
                        billing_baseline_epoch,
                    ) = _read_remote_status(row, run_id, attempt, root)
                if session is not None:
                    key_cleanup = session.close()
                    session = None
                    if key_cleanup not in {"detached", "not_found"}:
                        state, quality = "unavailable", "unavailable"
                if not owned:
                    outcome, status, stop_verification = "not_needed", "not_needed", "not_needed"
                if owned:
                    stop_verification = "unverified"
                    rate = _finite(row.get("dph_total"))
                    stop_status, _key_status = base.stop_exact_label(
                        label, run_id, attempt, expected_id=ident
                    )
                    if stop_status not in {"verified", "not_needed"}:
                        outcome, status = "unverified", "unverified"
                    else:
                        final_rows = base.instance_rows()
                        final_row = next(
                            (item for item in final_rows if base.instance_id(item) == ident),
                            None,
                        )
                        labeled_row = base.resolve_label(final_rows, label)
                        if (
                            final_row is None
                            or final_row.get("label") != label
                            or (labeled_row is not None and base.instance_id(labeled_row) != ident)
                            or base.probe.observed_state(final_row) not in {"stopped", "exited"}
                        ):
                            outcome, status = "unverified", "unverified"
                        else:
                            stop_verification = "verified"
                            spend = base.project_charges()
                            billing_check_epoch = time.time()
                            extra = _other_storage_reserve(final_rows, ident)
                            projected_spend = projected_billable_spend(
                                spend,
                                budget_baseline,
                                billing_baseline_epoch,
                                billing_check_epoch,
                                rate,
                            )
                            if (
                                rate is None or extra is None or projected_spend is None
                                or _storage_reserve_for_row(final_row) is None
                                or not full_run_budget_allows(
                                    projected_spend,
                                    rate,
                                    final_row.get("storage_cost"),
                                    active_hours=0.0,
                                    extra_storage_reserve=extra,
                                )
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
        if stop_verification != "not_needed":
            stop_verification = "unverified"
    except (GateError, base.GateError):
        outcome, status = "unverified", "unverified"
        if stop_verification != "not_needed":
            stop_verification = "unverified"
    except Exception:
        outcome, status = "controller_failed", "unverified"
        if stop_verification != "not_needed":
            stop_verification = "unverified"
    finally:
        if session is not None:
            session.close()
    _summarize(checked, outcome, status, stop_verification=stop_verification)
    return 0 if outcome in {"accepted", "not_needed"} else 1
def _summarize(
    checked: str,
    outcome: str,
    status: str,
    cleanup_needed: bool | None = None,
    stop_verification: str | None = None,
) -> None:
    if outcome not in OUTCOMES:
        outcome = "controller_failed"
    if status not in STATUSES:
        status = "unavailable"
    if cleanup_needed is not None and type(cleanup_needed) is not bool:
        raise GateError("cleanup output flag is invalid")
    allowed_stop_values = {"verified", "unverified", "not_needed", "not_attempted"}
    if stop_verification is not None and stop_verification not in allowed_stop_values:
        stop_verification = "unverified"
    print(f"checked_at={checked}")
    print(f"query_outcome={outcome}")
    print(f"status={status}")
    if stop_verification is not None:
        print(f"stop_verification={stop_verification}")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        try:
            with open(output_path, "a", encoding="utf-8") as output:
                output.write(f"status={status}\n")
                if cleanup_needed is not None:
                    value = "yes" if cleanup_needed else "no"
                    output.write(f"cleanup_needed={value}\n")
                if stop_verification is not None:
                    output.write(f"stop_verification={stop_verification}\n")
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
                if stop_verification is not None:
                    summary.write(f"- stop_verification: {stop_verification}\\n")
        except OSError:
            pass


def _summarize_budget(
    checked: str, outcome: str, budget_status: str, bootstrap_status: str = "unavailable"
) -> None:
    allowed_statuses = {
        "within_cap", "insufficient_remaining_budget", "run_projection_exceeds_cap",
        "inputs_unknown", "no_eligible_offer", "configuration_missing", "target_missing",
        "target_ambiguous", "target_mismatch", "unavailable",
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
