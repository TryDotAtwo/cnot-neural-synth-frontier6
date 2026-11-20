#!/usr/bin/env python3
"""Redacted, reversible SSH preflight for the sole already-running 2x RTX 3060."""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = "https://console.vast.ai/api/v0"
MAX_JSON_BYTES = 4 * 1024 * 1024
GPU_COMMAND = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
DISK_COMMAND = "df -Pk /workspace"
ALLOWED_STATES = {"running", "stopped", "starting", "stopping", "scheduling", "loading", "created", "offline", "exited", "error"}

OWNED_INSTANCE_IDS = {
    60488802, 60491053, 60509722, 60522214, 60585616, 60586973, 60588069,
    60621205, 60622406, 60626596, 60642345, 60646895, 60654590, 60658156,
    60660708, 60667533, 60674887, 60694337, 60695154, 60729505, 60745475,
    60749603, 60917228, 60919132, 60932086, 61081857,
}


class ApiFailure(RuntimeError):
    def __init__(self, status: int | None):
        super().__init__("api request failed")
        self.status = status


def api_json(api_key: str, path: str, method: str = "GET", payload: object | None = None) -> object:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json", "User-Agent": "TheQuantumProject-GitHub-Actions/1.0"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(f"{API}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=25) as response:
            raw = response.read(MAX_JSON_BYTES + 1)
    except HTTPError as error:
        raise ApiFailure(error.code) from None
    except (URLError, TimeoutError, OSError):
        raise ApiFailure(None) from None
    if len(raw) > MAX_JSON_BYTES:
        raise ApiFailure(None)
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ApiFailure(None) from None


def is_project_owned(row: dict) -> bool:
    instance_id = row.get("id")
    known_id = (
        isinstance(instance_id, int)
        and not isinstance(instance_id, bool)
        and instance_id in OWNED_INSTANCE_IDS
    )
    label = row.get("label")
    known_label = (
        isinstance(label, str)
        and re.fullmatch(r"(?i)cnot103[a-z0-9_.-]{1,120}", label) is not None
    )
    return known_id or known_label


def observed_state(row: dict) -> str:
    """Return a confirmed current state; next_state is only a requested transition."""
    value = row.get("actual_status") or row.get("cur_state")
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower()
    return value if value in ALLOWED_STATES else "unknown"


def requested_state(row: dict) -> str:
    """Return the requested transition target without treating it as observation."""
    value = row.get("next_state")
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower()
    return value if value in ALLOWED_STATES else "unknown"


def state(row: dict) -> str:
    """Compatibility alias for call sites that mean observed state."""
    return observed_state(row)


def target_rows(payload: object) -> list[dict]:
    rows = payload if isinstance(payload, list) else payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("invalid inventory")
    return [
        row for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("gpu_name"), str)
        and "3060" in row["gpu_name"].casefold()
        and isinstance(row.get("num_gpus", row.get("gpu_count")), int)
        and not isinstance(row.get("num_gpus", row.get("gpu_count")), bool)
        and row.get("num_gpus", row.get("gpu_count")) == 2
        and is_project_owned(row)
    ]


def ssh_key_blob(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    fields = value.strip().split()
    return fields[1] if len(fields) >= 2 and fields[0].startswith("ssh-") else None


def find_attached_key_id(api_key: str, instance_id: int, public_key: str) -> int | None:
    response = api_json(api_key, f"/instances/{instance_id}/ssh")
    if not isinstance(response, dict):
        return None
    keys = response.get("ssh_keys")
    if isinstance(keys, str):
        try:
            keys = json.loads(keys)
        except json.JSONDecodeError:
            return None
    if not isinstance(keys, list):
        return None
    wanted = ssh_key_blob(public_key)
    for item in keys:
        if not isinstance(item, dict) or ssh_key_blob(item.get("public_key") or item.get("ssh_key") or item.get("key")) != wanted:
            continue
        key_id = item.get("id")
        if isinstance(key_id, int) and not isinstance(key_id, bool) and key_id > 0:
            return key_id
    return None


def classify_ssh_failure(stderr: str) -> str:
    message = stderr.casefold()
    if "permission denied" in message:
        return "ssh_auth_failed"
    if "connection timed out" in message or "operation timed out" in message:
        return "ssh_timeout"
    if "connection refused" in message:
        return "ssh_connection_refused"
    if "no route to host" in message or "network is unreachable" in message:
        return "ssh_network_unreachable"
    if "could not resolve hostname" in message or "temporary failure in name resolution" in message:
        return "ssh_dns_failure"
    if "host key verification failed" in message or "remote host identification has changed" in message:
        return "ssh_host_key_failure"
    if "kex_exchange_identification" in message or "connection reset by peer" in message:
        return "ssh_handshake_failed"
    return "ssh_rejected"


def run_ssh(host: str, port: int, key_path: pathlib.Path, known_hosts: pathlib.Path, command: str) -> tuple[subprocess.CompletedProcess[str] | None, str]:
    argv = [
        "ssh", "-i", str(key_path), "-p", str(port),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=8",
        "-o", "ConnectionAttempts=1",
        f"root@{host}", command,
    ]
    failure_class = "ssh_failed"
    for attempt in range(4):
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=18, check=False)
        except subprocess.TimeoutExpired:
            result = None
            failure_class = "ssh_timeout"
        except FileNotFoundError:
            result = None
            failure_class = "ssh_client_unavailable"
        except OSError:
            result = None
            failure_class = "ssh_local_error"
        if result is not None and result.returncode == 0:
            return result, "connected"
        if result is not None:
            failure_class = classify_ssh_failure(result.stderr or "")
        if attempt < 3:
            time.sleep(3)
    return None, failure_class


def classify_gpu_output(raw: str) -> str:
    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    return "verified_2x3060" if len(rows) == 2 and all("3060" in line.casefold() for line in rows) else "gpu_inventory_mismatch"


def classify_disk_output(raw: str) -> str:
    available = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 5 and all(part.isdigit() for part in fields[1:4]):
            available.append(int(fields[3]))
    if not available:
        return "unknown"
    return "adequate" if max(available) >= 40 * 1024 * 1024 else "low"


def safe_budget_projection(rows: list[dict]) -> str:
    running = [row for row in rows if state(row) == "running"]
    if len(running) != 1:
        return "unknown"
    rate = running[0].get("dph_total")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
        return "unknown"
    return "within_48h_cap" if rate * 48 <= 19.0 else "above_48h_cap"


def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    inventory = "query_failed"
    ssh_status = "not_attempted"
    gpu = "unknown"
    disk = "unknown"
    budget = "unknown"
    cleanup = "not_needed"
    temp_dir: pathlib.Path | None = None
    attach_attempted = False
    api_key = os.environ.get("VAST_API_KEY", "")
    instance_id: int | None = None
    public_key = ""

    try:
        if not api_key:
            inventory = "configuration_missing"
        else:
            rows = target_rows(api_json(api_key, "/instances/"))
            budget = safe_budget_projection(rows)
            running = [row for row in rows if state(row) == "running"]
            if len(running) != 1:
                inventory = "running_target_ambiguous"
            else:
                candidate = running[0].get("id")
                if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate <= 0:
                    inventory = "target_id_unavailable"
                else:
                    instance_id = candidate
                    inventory = "verified_single_running_target"
                    host = running[0].get("ssh_host")
                    port = running[0].get("ssh_port")
                    if not isinstance(host, str) or not re.fullmatch(r"ssh[0-9]+\.vast\.ai", host.casefold()):
                        ssh_status = "endpoint_unavailable"
                    elif isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                        ssh_status = "endpoint_unavailable"
                    else:
                        temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="vast-ssh-"))
                        key_path = temp_dir / "temporary"
                        known_hosts = temp_dir / "known_hosts"
                        generated = subprocess.run(
                            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"gha-{os.environ.get('GITHUB_RUN_ID', 'run')}", "-f", str(key_path)],
                            capture_output=True, text=True, timeout=15, check=False,
                        )
                        if generated.returncode != 0:
                            ssh_status = "key_generation_failed"
                        else:
                            public_key = (key_path.with_suffix(".pub")).read_text(encoding="utf-8").strip()
                            if find_attached_key_id(api_key, instance_id, public_key) is not None:
                                ssh_status = "key_collision"
                            else:
                                attach_attempted = True
                                attached = api_json(api_key, f"/instances/{instance_id}/ssh", "POST", {"ssh_key": public_key})
                                if not isinstance(attached, dict) or attached.get("success") is not True:
                                    ssh_status = "key_attach_failed"
                                else:
                                    ssh_status = "attached"
                                    if find_attached_key_id(api_key, instance_id, public_key) is None:
                                        ssh_status = "key_attach_unconfirmed"
                                    else:
                                        gpu_result, ssh_status = run_ssh(host, port, key_path, known_hosts, GPU_COMMAND)
                                        if gpu_result is not None:
                                            gpu = classify_gpu_output(gpu_result.stdout)
                                            disk_result, disk_ssh_status = run_ssh(host, port, key_path, known_hosts, DISK_COMMAND)
                                            if disk_result is None:
                                                disk = "unknown"
                                                ssh_status = disk_ssh_status
                                            else:
                                                disk = classify_disk_output(disk_result.stdout)
    except ApiFailure as error:
        if error.status in {401, 403}:
            inventory = "api_auth_failed"
        else:
            inventory = "api_query_failed"
    except (ValueError, OSError, subprocess.TimeoutExpired):
        if inventory == "verified_single_running_target" and ssh_status == "not_attempted":
            ssh_status = "preflight_failed"
        elif inventory == "query_failed":
            inventory = "query_failed"

    finally:
        if attach_attempted and api_key and instance_id is not None and public_key:
            try:
                key_id = find_attached_key_id(api_key, instance_id, public_key)
                if key_id is None:
                    cleanup = "not_found"
                else:
                    detached = api_json(api_key, f"/instances/{instance_id}/ssh/{key_id}", "DELETE")
                    cleanup = "detached" if isinstance(detached, dict) and detached.get("success") is True else "failed"
            except (ApiFailure, ValueError):
                cleanup = "failed"
        if temp_dir is not None:
            for name in ("temporary", "temporary.pub", "known_hosts"):
                try:
                    (temp_dir / name).unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    print(f"checked_at={checked}")
    print(f"inventory={inventory}")
    print(f"ssh={ssh_status}")
    print(f"gpu={gpu}")
    print(f"remote_disk={disk}")
    print(f"budget_projection_48h={budget}")
    print(f"temporary_ssh_key={cleanup}")
    try:
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as summary:
                summary.write(
                    "## Existing 2x3060 preflight\n"
                    f"- checked_at: {checked}\n"
                    f"- inventory: {inventory}\n"
                    f"- ssh: {ssh_status}\n"
                    f"- gpu: {gpu}\n"
                    f"- remote_disk: {disk}\n"
                    f"- budget_projection_48h: {budget}\n"
                    f"- temporary_ssh_key: {cleanup}\n"
                )
    except OSError:
        pass

    ok = (
        inventory == "verified_single_running_target"
        and ssh_status == "connected"
        and gpu == "verified_2x3060"
        and disk == "adequate"
        and budget == "within_48h_cap"
        and cleanup in {"detached", "not_needed"}
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
