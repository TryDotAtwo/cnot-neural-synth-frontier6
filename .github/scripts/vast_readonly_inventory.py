#!/usr/bin/env python3
"""Read-only, redacted inventory check for an existing owned 2x RTX 3060."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ALLOWED_STATES = {"running", "stopped", "starting", "stopping", "scheduling", "loading", "created", "offline", "exited", "error"}


def state(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower()
    return value if value in ALLOWED_STATES else "unknown"


def inventory() -> str:
    api_key = os.environ.get("VAST_API_KEY", "")
    if not api_key:
        return "configuration_missing"
    request = Request(
        "https://console.vast.ai/api/v0/instances/",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json", "User-Agent": "TheQuantumProject-GitHub-Actions/1.0"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read())
    except HTTPError:
        return "query_failed"
    except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
        return "query_failed"
    rows = payload if isinstance(payload, list) else payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return "query_failed"

    matches = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        gpu = row.get("gpu_name")
        count = row.get("num_gpus", row.get("gpu_count"))
        if (
            isinstance(label, str)
            and re.fullmatch(r"(?i)cnot103[a-z0-9_.-]{1,120}", label)
            and isinstance(gpu, str)
            and "3060" in gpu.casefold()
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count == 2
        ):
            matches.append(state(row.get("actual_status") or row.get("cur_state") or row.get("next_state")))

    if len(matches) > 1:
        return "multiple_targets"
    if not matches:
        return "no_existing_target"
    return f"existing_target_{matches[0]}"


def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        status = inventory()
    except Exception:
        status = "query_failed"
    print(f"checked_at_utc={checked}")
    print(f"vast_inventory={status}")
    return 0 if status not in {"configuration_missing", "query_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
