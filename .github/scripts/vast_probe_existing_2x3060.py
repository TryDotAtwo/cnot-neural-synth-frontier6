#!/usr/bin/env python3
"""Read-only preflight for the sole already-running 2x RTX 3060 target."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

API = "https://console.vast.ai/api/v0"
GPU_COMMAND = "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"
DISK_COMMAND = "df -Pk /workspace"
ALLOWED_STATES = {"running", "stopped", "starting", "stopping", "scheduling", "loading", "created", "offline", "exited", "error"}
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 64 * 1024


def state(row: dict) -> str:
    value = row.get("actual_status") or row.get("cur_state") or row.get("next_state")
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower()
    return value if value in ALLOWED_STATES else "unknown"


def read_json(url: str, api_key: str, method: str = "GET", body: bytes | None = None) -> object:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "TheQuantumProject-GitHub-Actions/1.0",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, headers=headers, data=body, method=method)
    with urlopen(request, timeout=25) as response:
        payload = response.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("response too large")
    return json.loads(payload)


def target_rows(payload: object) -> list[dict]:
    rows = payload if isinstance(payload, list) else payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("invalid inventory")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        gpu = row.get("gpu_name")
        count = row.get("num_gpus", row.get("gpu_count"))
        if (
            isinstance(gpu, str)
            and "3060" in gpu.casefold()
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count == 2
        ):
            result.append(row)
    return result


def read_remote_output(result_url: str) -> str:
    parsed = urlparse(result_url)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or not (host == "s3.amazonaws.com" or host.endswith(".amazonaws.com") or host.endswith(".vast.ai"))
    ):
        raise ValueError("untrusted result host")
    request = Request(result_url, headers={"User-Agent": "TheQuantumProject-GitHub-Actions/1.0"})
    last_error = None
    for _ in range(15):
        try:
            with urlopen(request, timeout=10) as response:
                raw = response.read(MAX_RESULT_BYTES + 1)
            if len(raw) > MAX_RESULT_BYTES:
                raise ValueError("result too large")
            return raw.decode("utf-8", errors="strict")
        except HTTPError as error:
            last_error = error.code
            if error.code not in {404, 409, 425, 429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError, UnicodeDecodeError, OSError):
            last_error = "transient"
        time.sleep(2)
    raise RuntimeError(f"result unavailable: {last_error}")


def classify_gpu_output(raw: str) -> str:
    rows = [line.strip() for line in raw.splitlines() if line.strip() and "," in line]
    gpu_rows = [line for line in rows if "3060" in line.casefold()]
    return "verified_2x3060" if len(gpu_rows) == 2 and len(rows) == 2 else "gpu_inventory_mismatch"


def classify_disk_output(raw: str) -> str:
    numeric_rows = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 5 and all(part.isdigit() for part in fields[1:4]):
            numeric_rows.append(fields)
    if not numeric_rows:
        return "unknown"
    free_kib = max(int(row[3]) for row in numeric_rows)
    return "adequate" if free_kib >= 40 * 1024 * 1024 else "low"


def execute_remote(api_key: str, instance_id: int, command: str) -> str:
    if len(command.encode("utf-8")) > 512:
        raise ValueError("command too long")
    body = json.dumps({"command": command}, separators=(",", ":")).encode("utf-8")
    queued = read_json(f"{API}/instances/command/{instance_id}", api_key, "PUT", body)
    if not isinstance(queued, dict) or queued.get("success") is not True or not isinstance(queued.get("result_url"), str):
        raise RuntimeError("remote command failed")
    return read_remote_output(queued["result_url"])


def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    status = "query_failed"
    disk = "unknown"
    budget = "unknown"
    phase = "inventory"
    try:
        api_key = os.environ.get("VAST_API_KEY", "")
        if not api_key:
            status = "configuration_missing"
        else:
            matches = target_rows(read_json(f"{API}/instances/", api_key))
            running = [row for row in matches if state(row) == "running"]
            if len(running) != 1:
                status = "running_target_ambiguous"
            else:
                hourly = running[0].get("dph_total")
                if isinstance(hourly, (int, float)) and not isinstance(hourly, bool) and math.isfinite(hourly) and hourly >= 0:
                    budget = "within_48h_cap" if hourly * 48 <= 19.0 else "above_48h_cap"
                instance_id = running[0].get("id")
                if not isinstance(instance_id, int) or isinstance(instance_id, bool) or instance_id <= 0:
                    status = "target_id_unavailable"
                else:
                    phase = "execute_gpu"
                    gpu_output = execute_remote(api_key, instance_id, GPU_COMMAND)
                    status = classify_gpu_output(gpu_output)
                    if status == "verified_2x3060":
                        phase = "execute_disk"
                        disk_output = execute_remote(api_key, instance_id, DISK_COMMAND)
                        disk = classify_disk_output(disk_output)
    except HTTPError as error:
        if phase.startswith("execute_"):
            if error.code == 400:
                try:
                    detail = json.loads(error.read(4096))
                    kind = detail.get("error") if isinstance(detail, dict) else None
                except Exception:
                    kind = None
                status = {
                    "invalid_args": "execute_invalid_args",
                    "invalid_container_id": "execute_no_container_id",
                    "invalid_container": "execute_container_unavailable",
                }.get(kind, "execute_bad_request")
            else:
                status = {
                    401: "execute_auth_rejected",
                    403: "execute_permission_denied",
                    404: "execute_endpoint_unavailable",
                    413: "execute_payload_rejected",
                    429: "execute_rate_limited",
                }.get(error.code, "execute_http_error")
        elif phase == "inventory":
            status = {
                401: "inventory_auth_rejected",
                403: "inventory_permission_denied",
                429: "inventory_rate_limited",
            }.get(error.code, "inventory_http_error")
        else:
            status = "result_http_error"
    except URLError:
        status = "execute_network_error" if phase.startswith("execute_") else "inventory_network_error"
    except TimeoutError:
        status = "execute_timeout" if phase.startswith("execute_") else "inventory_timeout"
    except RuntimeError:
        status = "remote_command_failed" if phase.startswith("execute_") else "probe_failed"
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        status = "remote_response_invalid" if phase.startswith("execute_") else "inventory_invalid"
    except Exception:
        status = "probe_failed"
    print(f"checked_at_utc={checked}")
    print(f"remote_probe={status}")
    print(f"remote_disk={disk}")
    print(f"budget_projection_48h={budget}")
    return 0 if status == "verified_2x3060" else 1

if __name__ == "__main__":
    raise SystemExit(main())
