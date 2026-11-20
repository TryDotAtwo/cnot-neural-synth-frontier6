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
MONITOR_SECONDS = 5 * 60 * 60 + 30 * 60
POLL_SECONDS = 60
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


START_REQUESTED_LABEL: str | None = None


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
) -> str:
    q = shlex.quote
    source = root + "/source"
    command = (
        "nohup setsid python3 " + q(source + "/.github/scripts/run_full_training.py")
        + " --root " + q(root)
        + " --source-revision " + q(source_revision)
        + " --holdout-revision " + q(holdout_revision)
        + " --holdout-sha256 " + q(holdout_sha256)
        + " --started-at-epoch " + q(str(started_at_epoch))
        + " --budget-baseline " + q(str(budget_baseline))
    )
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
nohup setsid python3 {q(source + "/.github/scripts/run_full_training.py")} \\
  --root {q(root)} \\
  --source-revision {q(source_revision)} \\
  --holdout-revision {q(holdout_revision)} \\
  --holdout-sha256 {q(holdout_sha256)} \\
  --started-at-epoch {q(str(started_at_epoch))} \\
  --budget-baseline {q(str(budget_baseline))} \\
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


def _start_target(
    rows: list[dict], marker: dict, run_id: str, attempt: str
) -> tuple[dict, str, bool, float]:
    fresh_label = _fresh_label(run_id, attempt)
    old_label = marker["resume_existing_label"]
    target, label = _labels_for_run(rows, old_label, fresh_label)
    spent = base.project_charges()
    global START_REQUESTED_LABEL
    if target is not None:
        if not base.is_2x3060(target):
            raise GateError("target GPU inventory does not match")
        state = base.probe.observed_state(target)
        if state not in {"stopped", "exited"}:
            raise GateError("exact target is not safely stopped")
        ident = base.instance_id(target)
        if ident is None:
            raise GateError("exact target id is invalid")
        extra_storage = _other_storage_reserve(rows, ident)
        rate = _finite(target.get("dph_total"))
        storage_cost = target.get("storage_cost")
        if extra_storage is None or not full_run_budget_allows(
            spent, rate, storage_cost, extra_storage_reserve=extra_storage
        ):
            raise GateError("the full-run budget gate is closed")
        START_REQUESTED_LABEL = label or old_label
        if not base.api_instance_id(ident, "running"):
            raise GateError("exact target start was not confirmed")
        running = base.wait_until_running(label or old_label)
        return running, label or old_label, False, spent

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
            if full_run_budget_allows(
                spent,
                offer.get("dph_total"),
                offer.get("storage_cost"),
                extra_storage_reserve=extra_storage,
            )
        ),
        None,
    )
    if selected is None:
        raise GateError("no verified 2x3060 offer fits the full-run budget")
    offer_id = str(selected.get("ask_id") or selected.get("id") or selected.get("offer_id") or "")
    if not re.fullmatch(r"[0-9]{1,20}", offer_id):
        raise GateError("selected offer identity is invalid")
    START_REQUESTED_LABEL = fresh_label
    base.CREATE_ATTEMPTED = False
    row = base.create_once(offer_id, fresh_label)
    running = base.wait_until_running(fresh_label)
    return running, fresh_label, True, spent


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


def start() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome, status = "controller_failed", "unavailable"
    label = None
    run_id = attempt = ""
    global START_REQUESTED_LABEL
    START_REQUESTED_LABEL = None
    attached_target_id = None
    try:
        revision, run_id, attempt, root = _run_identity()
        holdout_revision, holdout_sha256 = _pin_status()
        marker = _marker()
        rows = base.instance_rows()
        row, label, _created, _spend_before = _start_target(rows, marker, run_id, attempt)
        ident = base.instance_id(row)
        if ident is None or not base.is_2x3060(row):
            outcome, status = "target_gpu_mismatch", "failed"
            raise GateError("running target identity or GPU differs")
        attached_target_id = ident
        running_rows = base.instance_rows()
        started_at_epoch = time.time()
        spend_now = base.project_charges()
        if not _validate_running_budget(row, spend_now, running_rows, ident):
            outcome, status = "budget_gate", "budget_gate"
            raise GateError("post-start full-run budget recheck failed")
        script = _launch_script(
            root,
            revision,
            holdout_revision,
            holdout_sha256,
            started_at_epoch,
            spend_now,
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
            elif "stopped" in message or "active" in message or "transition" in message:
                outcome, status = "target_busy", "failed"
            elif "pin" in message or "holdout" in message:
                outcome, status = "source_pin_invalid", "failed"
            elif "configuration" in message:
                outcome, status = "configuration_missing", "failed"
            else:
                outcome, status = "target_unavailable", "failed"
    except Exception:
        outcome, status = "controller_failed", "failed"
    finally:
        cleanup_label = START_REQUESTED_LABEL or label
        if outcome != "accepted" and cleanup_label and run_id and attempt:
            try:
                base.stop_exact_label(cleanup_label, run_id, attempt)
            except Exception:
                pass
    _summarize(checked, outcome, status)
    return 0 if outcome == "accepted" else 1


def _read_remote_status(
    row: dict, run_id: str, attempt: str, root: str,
    session: RemoteSession | None = None,
) -> tuple[str, str, float | None, float | None]:
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
PY
"""
    if session is None:
        proc, key_cleanup, _ident = _attach_and_remote(row, run_id, attempt, remote, 30)
        if proc is None or proc.returncode != 0 or key_cleanup not in {"detached", "not_found"}:
            return "unavailable", "unavailable", None, None
    else:
        proc = session.run(remote, 30)
        if proc is None or proc.returncode != 0:
            return "unavailable", "unavailable", None, None
    values = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {
            "run_state", "quality_status", "started_at_epoch",
            "budget_baseline", "instance_rate", "storage_cost",
        }:
            values[key] = value
    state = values.get("run_state", "unavailable")
    quality = values.get("quality_status", "unavailable")
    parsed = []
    for key in ("started_at_epoch", "budget_baseline", "instance_rate", "storage_cost"):
        try:
            parsed.append(float(values[key]))
        except (KeyError, ValueError):
            parsed.append(None)
    return state, quality, parsed[0], parsed[1]


def _find_run_target(rows: list[dict], marker: dict, run_id: str, attempt: str) -> tuple[dict | None, str | None]:
    old_label = marker["resume_existing_label"]
    fresh_label = _fresh_label(run_id, attempt)
    return _labels_for_run(rows, old_label, fresh_label)


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
    while True:
        state, quality, started_at, budget_baseline = _read_remote_status(
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
        if time.monotonic() >= next_budget_check:
            current_rows = base.instance_rows()
            current = base.resolve_label(current_rows, label)
            if current is None or base.instance_id(current) != ident:
                return 1, "unverified", "unverified"
            spend_now = base.project_charges()
            rate = _finite(current.get("dph_total"))
            extra_storage = _other_storage_reserve(current_rows, ident)
            if rate is None or extra_storage is None or started_at is None:
                base.stop_exact_label(label, run_id, attempt)
                return 1, "budget_gate", "budget_gate"
            elapsed_hours = max(0.0, (time.time() - started_at) / 3600.0)
            baseline = budget_baseline if budget_baseline is not None else spend_now
            projected_spend = max(spend_now, baseline + rate * elapsed_hours)
            remaining_hours = max(
                0.0, (MAX_RUNTIME_SECONDS / 3600.0) - elapsed_hours
            )
            if not full_run_budget_allows(
                projected_spend,
                rate,
                current.get("storage_cost"),
                active_hours=remaining_hours + 0.25,
                extra_storage_reserve=extra_storage,
            ):
                base.stop_exact_label(label, run_id, attempt)
                return 1, "budget_gate", "budget_gate"
            row = current
            next_budget_check = time.monotonic() + 15 * 60
        if time.monotonic() >= host_deadline:
            return 0, "running", "running"
        time.sleep(min(POLL_SECONDS, max(1.0, host_deadline - time.monotonic())))


def monitor(window_seconds: int = MONITOR_SECONDS) -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome, status = "controller_failed", "unavailable"
    session = None
    result_code = 1
    try:
        _revision, run_id, attempt, root = _run_identity()
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


def cleanup() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome, status = "controller_failed", "unverified"
    state, quality = "unavailable", "unavailable"
    try:
        _revision, run_id, attempt, root = _run_identity()
        marker = _marker()
        rows = base.instance_rows()
        row, label = _find_run_target(rows, marker, run_id, attempt)
        if row is None or label is None:
            outcome, status = "not_needed", "not_needed"
            _summarize(checked, outcome, status)
            return 1
        if not base.is_2x3060(row):
            outcome, status = "target_gpu_mismatch", "unverified"
            _summarize(checked, outcome, status)
            return 1
        started_at = None
        budget_baseline = None
        rate = _finite(row.get("dph_total"))
        if base.probe.observed_state(row) == "running":
            try:
                state, quality, started_at, budget_baseline = _read_remote_status(
                    row, run_id, attempt, root
                )
            except Exception:
                state, quality = "unavailable", "unavailable"
        stop_status, _key_status = base.stop_exact_label(label, run_id, attempt)
        if stop_status not in {"verified", "not_needed"}:
            outcome, status = "unverified", "unverified"
            _summarize(checked, outcome, status)
            return 1
        final_rows = base.instance_rows()
        final_row = base.resolve_label(final_rows, label)
        if final_row is not None and base.probe.observed_state(final_row) not in {"stopped", "exited"}:
            outcome, status = "unverified", "unverified"
            _summarize(checked, outcome, status)
            return 1
        spend = base.project_charges()
        if final_row is not None:
            extra = _other_storage_reserve(final_rows, base.instance_id(final_row))
            elapsed_hours = max(0.0, (time.time() - started_at) / 3600.0) if started_at else 0.0
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
                _summarize(checked, outcome, status)
                return 1
        if state != "completed" or quality != "passed":
            outcome, status = ("time_limit" if state == "time_limit" else "failed"), "failed"
            _summarize(checked, outcome, status)
            return 1
        outcome, status = "accepted", "stopped"
    except base.probe.ApiFailure:
        outcome, status = "api_failed", "unverified"
    except GateError:
        outcome, status = "unverified", "unverified"
    except Exception:
        outcome, status = "controller_failed", "unverified"
    _summarize(checked, outcome, status)
    return 0 if outcome == "accepted" else 1


def _summarize(checked: str, outcome: str, status: str) -> None:
    if outcome not in OUTCOMES:
        outcome = "controller_failed"
    if status not in STATUSES:
        status = "unavailable"
    print(f"checked_at={checked}")
    print(f"query_outcome={outcome}")
    print(f"status={status}")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("start", "monitor", "cleanup"))
    parser.add_argument("--window-seconds", type=int, default=MONITOR_SECONDS)
    args = parser.parse_args()
    if args.mode == "start":
        return start()
    if args.mode == "monitor":
        return monitor(args.window_seconds)
    return cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
