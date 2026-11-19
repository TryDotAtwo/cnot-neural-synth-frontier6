#!/usr/bin/env python3
"""Budget-gated, one-shot 2x RTX 3060 rental for rebuilding the 10k dataset."""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import vast_probe_existing_2x3060 as probe  # noqa: E402
import vast_rebuild_remote_dataset as dataset  # noqa: E402

IMAGE = "pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel"
DISK_GB = "64"
BUDGET_CAP = 20.0
RUN_HOURS = 55.0
TRANSFER_RESERVE = 3.0
MIN_RELIABILITY = 0.95
MIN_GPU_MEMORY_MB = 12_000
MIN_COMPUTE_CAPABILITY = 8.6
MAX_LABEL_WAIT_SECONDS = 1_200
MAX_INSTANCE_START_SECONDS = 900
MAX_STOP_SECONDS = 300
ALLOWED_OUTCOMES = {
    "accepted", "configuration_missing", "run_identity_invalid",
    "label_collision", "existing_2x3060_present", "offer_gate",
    "budget_gate", "create_uncertain", "instance_start_failed",
    "target_gpu_mismatch", "toolchain_failed", "dataset_failed",
    "api_failed", "controller_failed",
}
OWNED_INSTANCE_IDS = {
    60488802, 60491053, 60509722, 60522214, 60585616, 60586973, 60588069,
    60621205, 60622406, 60626596, 60642345, 60646895, 60654590, 60658156,
    60660708, 60667533, 60674887, 60694337, 60695154, 60729505, 60745475,
    60749603, 60917228, 60919132, 60932086, 61081857,
}


class GateError(RuntimeError):
    pass


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def truth(value: object) -> bool:
    return value is True or value == 1 or (
        isinstance(value, str) and value.strip().casefold() == "true"
    )


def budget_allows(spent: object, rate: object, run_hours: object = RUN_HOURS,
                  reserve: object = TRANSFER_RESERVE) -> bool:
    spent_value = finite_number(spent)
    rate_value = finite_number(rate)
    hours_value = finite_number(run_hours)
    reserve_value = finite_number(reserve)
    if (
        spent_value is None or rate_value is None or hours_value is None
        or reserve_value is None or spent_value < 0 or rate_value < 0
        or hours_value <= 0 or reserve_value < 0
    ):
        return False
    return spent_value + rate_value * hours_value + reserve_value <= BUDGET_CAP + 1e-9


def eligible_offers(rows: object) -> list[dict]:
    if not isinstance(rows, list):
        raise GateError("offer inventory has an unknown shape")
    eligible = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        gpu_name = str(row.get("gpu_name") or "").casefold().replace("_", "").replace(" ", "")
        gpu_count = finite_number(row.get("num_gpus"))
        memory = finite_number(row.get("gpu_ram"))
        capability = finite_number(row.get("compute_cap"))
        reliability = finite_number(row.get("reliability"))
        rate = finite_number(row.get("dph_total"))
        ports = finite_number(row.get("direct_port_count"))
        offer_id = row.get("ask_id") or row.get("id") or row.get("offer_id")
        offer_id_text = str(offer_id)
        verified = row.get("verified")
        checks = (
            "rtx3060" in gpu_name,
            gpu_count == 2,
            memory is not None and memory >= MIN_GPU_MEMORY_MB,
            capability is not None and capability >= MIN_COMPUTE_CAPABILITY,
            reliability is not None and reliability >= MIN_RELIABILITY,
            truth(row.get("rentable")),
            ports is not None and ports >= 1,
            rate is not None and rate > 0,
            re.fullmatch(r"[0-9]{1,20}", offer_id_text) is not None,
            verified is None or truth(verified),
        )
        if all(checks):
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            float(row["dph_total"]),
            -float(row["reliability"]),
            int(row.get("ask_id") or row.get("id") or row.get("offer_id")),
        )
    )
    return eligible


def resolve_label(rows: object, label: str) -> dict | None:
    if not isinstance(rows, list):
        raise GateError("instance inventory has an unknown shape")
    matches = [
        row for row in rows
        if isinstance(row, dict) and row.get("label") == label
    ]
    if len(matches) > 1:
        raise GateError("stable label is ambiguous")
    return matches[0] if matches else None


def target_rows_for_label(rows: object, label: str) -> list[dict]:
    if not isinstance(rows, list):
        raise GateError("instance inventory has an unknown shape")
    return [
        row for row in rows
        if isinstance(row, dict) and row.get("label") == label
    ]


def instance_rows() -> list[dict]:
    payload = probe.api_json(api_key(), "/instances/")
    rows = payload if isinstance(payload, list) else payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise GateError("instance inventory has an unknown shape")
    return rows


def api_key() -> str:
    value = os.environ.get("VAST_API_KEY", "")
    if not value:
        raise GateError("Vast configuration is missing")
    return value


def vast_cli(args: list[str], timeout: int = 90) -> object:
    command = ["vastai", "--api-key", api_key(), *args]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
        if result.returncode != 0:
            raise GateError("Vast query did not succeed")
        return json.loads(result.stdout.decode("utf-8"))
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError, json.JSONDecodeError):
        raise GateError("Vast query could not be verified") from None


def project_charges() -> float:
    start = date(2026, 11, 13)
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    args = [
        "show", "invoices-v1", "--charges", "--charge-type", "instance",
        "--start-date", start.isoformat(), "--end-date", end.isoformat(),
        "--limit", "100", "--raw",
    ]
    records = []
    token = None
    for _ in range(100):
        page_args = args + (["--next-token", token] if token else [])
        page = vast_cli(page_args, timeout=70)
        if not isinstance(page, dict) or not isinstance(page.get("results"), list):
            raise GateError("Vast billing response has an unknown shape")
        records.extend(page["results"])
        token = page.get("next_token")
        if token is None or token == "":
            break
        if not isinstance(token, str):
            raise GateError("Vast billing pagination token is invalid")
    else:
        raise GateError("Vast billing pagination did not finish")

    total = 0.0
    for row in records:
        if not isinstance(row, dict):
            raise GateError("Vast billing row has an invalid shape")
        description = str(row.get("description") or "")
        match = re.match(r"^Instance ([0-9]+) Charges(?:\s|$)", description)
        metadata = row.get("metadata")
        label = str(metadata.get("label") or "").lower() if isinstance(metadata, dict) else ""
        if not ((match and int(match.group(1)) in OWNED_INSTANCE_IDS) or label.startswith("cnot103")):
            continue
        amount = finite_number(row.get("amount"))
        if amount is None:
            raise GateError("Vast billing row has an invalid amount")
        total += amount
    if not math.isfinite(total) or total < 0:
        raise GateError("Vast project charges are invalid")
    return total


def is_2x3060(row: dict) -> bool:
    name = str(row.get("gpu_name") or "").casefold()
    count = row.get("num_gpus", row.get("gpu_count"))
    return (
        "rtx" in name and "3060" in name
        and isinstance(count, int) and not isinstance(count, bool) and count == 2
    )


def offer_inventory() -> list[dict]:
    rows = vast_cli([
        "search", "offers", "gpu_name=RTX_3060 num_gpus=2 verified=true",
        "--order=dph_total", "--limit=100", "--raw",
    ])
    if isinstance(rows, dict):
        rows = rows.get("offers", rows.get("results"))
    return eligible_offers(rows)


def run_identity() -> tuple[str, str, str]:
    revision = os.environ.get("GITHUB_SHA", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not re.fullmatch(r"[0-9]{1,20}", run_id)
        or not re.fullmatch(r"[0-9]{1,10}", attempt)
    ):
        raise GateError("run identity is invalid")
    label = f"cnot103-r10k-{run_id}-a{attempt}"
    return revision, run_id, label


def instance_id(row: dict | None) -> int | None:
    if not isinstance(row, dict):
        return None
    value = row.get("id")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,20}", value):
        return int(value)
    return None


def wait_for_label(label: str, attempts: int = 40, delay: float = 15,
                   expected_id: int | None = None) -> dict | None:
    for index in range(attempts):
        row = resolve_label(instance_rows(), label)
        if row is not None:
            found_id = instance_id(row)
            if expected_id is not None and found_id != expected_id:
                raise GateError("created id and stable label do not match")
            return row
        if index + 1 < attempts:
            time.sleep(delay)
    return None


def created_id(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("new_instance_id", "instance_id", "id", "new_contract"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        if isinstance(value, str) and re.fullmatch(r"[0-9]{1,20}", value):
            return int(value)
    return None


CREATE_ATTEMPTED = False


def create_once(offer_id: str, label: str) -> dict:
    global CREATE_ATTEMPTED
    if CREATE_ATTEMPTED:
        raise GateError("rental create was already attempted")
    CREATE_ATTEMPTED = True
    try:
        payload = vast_cli([
            "create", "instance", str(offer_id),
            "--image", IMAGE, "--disk", DISK_GB, "--ssh", "--direct",
            "--label", label, "--cancel-unavail", "--raw",
        ], timeout=120)
        expected_id = created_id(payload)
    except (GateError, OSError, subprocess.TimeoutExpired):
        expected_id = None
    row = wait_for_label(label, attempts=80, delay=15, expected_id=expected_id)
    if row is None:
        raise GateError("single rental request was not resolved by its stable label")
    return row


def api_instance_id(instance: int, state: str) -> bool:
    response = probe.api_json(
        api_key(), f"/instances/{instance}/", "PUT", {"state": state}
    )
    return isinstance(response, dict) and response.get("success") is True


def wait_until_running(label: str) -> dict:
    deadline = time.monotonic() + MAX_INSTANCE_START_SECONDS
    start_requested = False
    while time.monotonic() < deadline:
        row = resolve_label(instance_rows(), label)
        if row is None:
            time.sleep(10)
            continue
        current_state = probe.state(row)
        if current_state == "running":
            if not is_2x3060(row):
                raise GateError("created target GPU inventory does not match")
            return row
        if current_state in {"stopped", "exited"} and not start_requested:
            ident = instance_id(row)
            if ident is None or not api_instance_id(ident, "running"):
                raise GateError("new target could not be started")
            start_requested = True
        if current_state in {"error", "offline"}:
            raise GateError("new target entered an unusable state")
        time.sleep(10)
    raise GateError("new target did not reach running state")


def attach_key(instance: int, public_key: str) -> None:
    if probe.find_attached_key_id(api_key(), instance, public_key) is not None:
        raise GateError("temporary SSH key already exists")
    response = probe.api_json(
        api_key(), f"/instances/{instance}/ssh", "POST", {"ssh_key": public_key}
    )
    if not isinstance(response, dict) or response.get("success") is not True:
        raise GateError("temporary SSH key attachment was not confirmed")
    if probe.find_attached_key_id(api_key(), instance, public_key) is None:
        raise GateError("temporary SSH key attachment was not confirmed")


def detach_key(instance: int, public_key: str) -> str:
    try:
        key_id = probe.find_attached_key_id(api_key(), instance, public_key)
        if key_id is None:
            return "not_found"
        response = probe.api_json(
            api_key(), f"/instances/{instance}/ssh/{key_id}", "DELETE"
        )
        return "detached" if isinstance(response, dict) and response.get("success") is True else "failed"
    except (probe.ApiFailure, GateError, ValueError):
        return "failed"


def endpoint(row: dict) -> tuple[str, int]:
    host = row.get("ssh_host")
    port = row.get("ssh_port")
    if not isinstance(host, str) or not re.fullmatch(r"ssh[0-9]+\.vast\.ai", host.casefold()):
        raise GateError("remote SSH endpoint is unavailable")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise GateError("remote SSH endpoint is unavailable")
    return host, port


def remote_script(host: str, port: int, key_path: pathlib.Path,
                  script: str, timeout_seconds: int) -> subprocess.CompletedProcess[str] | None:
    known_hosts = key_path.parent / "known_hosts"
    command = [
        "ssh", "-i", str(key_path), "-p", str(port),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=8",
        "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        f"root@{host}",
        f"timeout --signal=TERM --kill-after=30 {timeout_seconds}s bash -s",
    ]
    try:
        return subprocess.run(
            command, input=script, capture_output=True, text=True,
            timeout=timeout_seconds + 120, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def install_remote_cargo(row: dict, run_id: str, revision: str, attempt: str) -> str:
    ident = instance_id(row)
    if ident is None:
        return "failed"
    host, port = endpoint(row)
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="vast-rustup-"))
    key_path = temp_root / "temporary"
    public_key = ""
    attached = False
    result = "failed"
    try:
        generated = subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C",
             f"gha-rust-{run_id}-{attempt}", "-f", str(key_path)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if generated.returncode != 0:
            return "failed"
        public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()
        attach_key(ident, public_key)
        attached = True
        bootstrap_root = f"/workspace/tqp-rust-bootstrap_{run_id}_{revision[:8]}_a{attempt}"
        remote = f"""set -euo pipefail
if command -v cargo >/dev/null 2>&1; then
  printf 'rust_toolchain=ready\\n'
  exit 0
fi
ROOT='{bootstrap_root}'
if [ -e "$ROOT" ] || [ -L "$ROOT" ]; then
  printf 'rust_toolchain=workspace_exists\\n'
  exit 10
fi
mkdir -m 700 "$ROOT"
exec 3>&1
exec >>"$ROOT/bootstrap.private.log" 2>&1
finish() {{
  code=$?
  trap - EXIT
  if [ "$code" -eq 0 ]; then status=ready; else status=failed; fi
  printf 'rust_toolchain=%s\\n' "$status" >&3
}}
trap finish EXIT
if ! command -v curl >/dev/null 2>&1; then exit 11; fi
curl --fail --silent --show-error --retry 3 --retry-delay 2 https://sh.rustup.rs -o "$ROOT/rustup-init.sh"
sh "$ROOT/rustup-init.sh" -y --profile minimal --default-toolchain stable
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version
"""
        proc = remote_script(host, port, key_path, remote, 1_200)
        if proc is not None and proc.returncode == 0 and "rust_toolchain=ready" in proc.stdout.splitlines():
            result = "ready"
    except (GateError, OSError, subprocess.TimeoutExpired, probe.ApiFailure):
        result = "failed"
    finally:
        if attached and public_key:
            cleanup = detach_key(ident, public_key)
            if cleanup != "detached":
                result = "failed"
        for name in ("temporary", "temporary.pub", "known_hosts"):
            try:
                (temp_root / name).unlink(missing_ok=True)
            except OSError:
                result = "failed"
        try:
            temp_root.rmdir()
        except OSError:
            result = "failed"
    return result


def scoped_dataset_rebuild(label: str) -> int:
    original_probe = probe.target_rows
    original_dataset = dataset.target_rows

    def select_only_label(payload: object) -> list[dict]:
        rows = original_probe(payload)
        return target_rows_for_label(rows, label)

    probe.target_rows = select_only_label
    dataset.target_rows = select_only_label
    try:
        return dataset.main()
    finally:
        probe.target_rows = original_probe
        dataset.target_rows = original_dataset


def remove_run_keys(instance: int, run_id: str, attempt: str) -> str:
    comments = {f"gha-rust-{run_id}-{attempt}", f"gha-dataset-{run_id}-{attempt}"}
    try:
        response = probe.api_json(api_key(), f"/instances/{instance}/ssh")
        if not isinstance(response, dict):
            return "failed"
        keys = response.get("ssh_keys")
        if isinstance(keys, str):
            try:
                keys = json.loads(keys)
            except json.JSONDecodeError:
                return "failed"
        if not isinstance(keys, list):
            return "failed"
        found = []
        for item in keys:
            if not isinstance(item, dict):
                continue
            public = item.get("public_key") or item.get("ssh_key") or item.get("key")
            key_id = item.get("id")
            if (
                isinstance(public, str) and any(public.rstrip().endswith(comment) for comment in comments)
                and isinstance(key_id, int) and not isinstance(key_id, bool) and key_id > 0
            ):
                found.append(key_id)
        for key_id in found:
            response = probe.api_json(
                api_key(), f"/instances/{instance}/ssh/{key_id}", "DELETE"
            )
            if not isinstance(response, dict) or response.get("success") is not True:
                return "failed"
        return "detached" if found else "not_found"
    except (probe.ApiFailure, GateError, ValueError):
        return "failed"


def stop_exact_label(label: str, run_id: str, attempt: str) -> tuple[str, str]:
    try:
        row = resolve_label(instance_rows(), label)
    except (GateError, probe.ApiFailure):
        return "unverified", "failed"
    if row is None:
        return "not_needed", "not_needed"
    ident = instance_id(row)
    if ident is None or row.get("label") != label:
        return "unverified", "failed"
    key_status = remove_run_keys(ident, run_id, attempt)
    current = probe.state(row)
    if current not in {"stopped", "exited"}:
        try:
            api_instance_id(ident, "stopped")
        except (GateError, probe.ApiFailure):
            pass
    deadline = time.monotonic() + MAX_STOP_SECONDS
    while time.monotonic() < deadline:
        try:
            current_row = resolve_label(instance_rows(), label)
        except (GateError, probe.ApiFailure):
            return "unverified", key_status
        if current_row is None:
            return "unverified", key_status
        if instance_id(current_row) != ident:
            return "unverified", key_status
        if probe.state(current_row) in {"stopped", "exited"}:
            return "verified", key_status
        time.sleep(5)
    return "unverified", key_status


def summarize(checked: str, outcome: str, status: str) -> None:
    if outcome not in ALLOWED_OUTCOMES:
        outcome = "controller_failed"
    if status not in {"accepted", "failed", "not_needed", "verified", "unverified"}:
        status = "failed"
    print(f"checked_at={checked}")
    print(f"query_outcome={outcome}")
    print(f"status={status}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as summary:
                summary.write(
                    "## Remote 10k dataset rebuild\n"
                    f"- checked_at: {checked}\n"
                    f"- query_outcome: {outcome}\n"
                    f"- status: {status}\n"
                )
        except OSError:
            pass


def run() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outcome = "controller_failed"
    status = "failed"
    try:
        revision, run_id, label = run_identity()
        attempt = os.environ["GITHUB_RUN_ATTEMPT"]
        if resolve_label(instance_rows(), label) is not None:
            outcome = "label_collision"
            raise GateError("stable label already exists")
        before = instance_rows()
        if any(is_2x3060(row) for row in before):
            outcome = "existing_2x3060_present"
            raise GateError("an RTX 3060 pair already exists")
        spent = project_charges()
        offers = offer_inventory()
        if not offers:
            outcome = "offer_gate"
            raise GateError("no verified 2x3060 offer is available")
        selected = next(
            (row for row in offers if budget_allows(spent, row["dph_total"])),
            None,
        )
        if selected is None:
            outcome = "budget_gate"
            raise GateError("the $20 budget gate is closed")
        offer_id = str(selected.get("ask_id") or selected.get("id") or selected.get("offer_id"))
        row = create_once(offer_id, label)
        row = wait_until_running(label)
        rate = finite_number(row.get("dph_total"))
        spent_after_create = project_charges()
        if rate is None or not budget_allows(spent_after_create, rate):
            outcome = "budget_gate"
            raise GateError("post-create budget recheck failed")
        if not is_2x3060(row):
            outcome = "target_gpu_mismatch"
            raise GateError("created GPU shape changed")
        toolchain = install_remote_cargo(row, run_id, revision, attempt)
        if toolchain != "ready":
            outcome = "toolchain_failed"
            raise GateError("remote Rust toolchain bootstrap failed")
        code = scoped_dataset_rebuild(label)
        if code != 0:
            outcome = "dataset_failed"
            raise GateError("remote dataset build did not pass")
        outcome = "accepted"
        status = "accepted"
    except probe.ApiFailure as error:
        outcome = "api_failed"
        status = "failed"
    except GateError as error:
        if outcome == "controller_failed":
            outcome = "controller_failed"
        status = "failed"
    except Exception:
        outcome = "controller_failed"
        status = "failed"
    summarize(checked, outcome, status)
    return 0 if outcome == "accepted" else 1


def cleanup() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        _revision, run_id, label = run_identity()
        attempt = os.environ["GITHUB_RUN_ATTEMPT"]
        stop_status, key_status = stop_exact_label(label, run_id, attempt)
        outcome = "accepted" if stop_status in {"verified", "not_needed"} and key_status in {"detached", "not_found", "not_needed"} else "controller_failed"
        summarize(checked, outcome, stop_status)
        print(f"temporary_ssh_key={key_status}")
        return 0 if outcome == "accepted" else 1
    except Exception:
        summarize(checked, "controller_failed", "unverified")
        return 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        raise SystemExit(run())
    if mode == "cleanup":
        raise SystemExit(cleanup())
    raise SystemExit(2)
