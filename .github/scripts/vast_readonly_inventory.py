#!/usr/bin/env python3
"""Read-only, redacted inventory check for an existing owned 2x RTX 3060."""

# Trigger a fresh GET-only check after an unverified cleanup result.

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

from vast_probe_existing_2x3060 import OWNED_INSTANCE_IDS
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ALLOWED_STATES = {"running", "stopped", "starting", "stopping", "scheduling", "loading", "created", "offline", "exited", "error"}


def state(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower()
    return value if value in ALLOWED_STATES else "unknown"


def inventory() -> tuple[str, str]:
    api_key = os.environ.get("VAST_API_KEY", "")
    if not api_key:
        return "configuration_missing", "unknown"

    def fetch_json(url: str) -> object:
        request = Request(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json", "User-Agent": "TheQuantumProject-GitHub-Actions/1.0"},
            method="GET",
        )
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read())

    try:
        payload = fetch_json("https://console.vast.ai/api/v0/instances/")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError, OSError):
        return "query_failed", "unknown"
    rows = payload if isinstance(payload, list) else payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return "query_failed", "unknown"

    matches = []
    project_matches = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        gpu = row.get("gpu_name")
        count = row.get("num_gpus", row.get("gpu_count"))
        if (
            not isinstance(gpu, str)
            or "3060" not in gpu.casefold()
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count != 2
        ):
            continue
        instance_id = row.get("id")
        instance_state = state(row.get("actual_status") or row.get("cur_state") or row.get("next_state"))
        matches.append((instance_id, instance_state))
        label = row.get("label")
        owned_id = (
            isinstance(instance_id, int)
            and not isinstance(instance_id, bool)
            and instance_id in OWNED_INSTANCE_IDS
        )
        owned_label = (
            isinstance(label, str)
            and re.fullmatch(r"(?i)cnot103[a-z0-9_.-]{1,120}", label) is not None
        )
        if owned_id or owned_label:
            project_matches.append((instance_id, instance_state))

    def state_summary(entries: list[tuple[object, str]], empty: str) -> str:
        states = [instance_state for _, instance_state in entries]
        if not states:
            return empty
        if len(states) == 1:
            return f"existing_target_{states[0]}"
        counts = {}
        for value in states:
            counts[value] = counts.get(value, 0) + 1
        return "multiple_targets_" + "_".join(f"{name}{counts[name]}" for name in sorted(counts))

    inventory_status = state_summary(matches, "no_existing_target")
    project_status = state_summary(project_matches, "no_project_target")

    try:
        volume_payload = fetch_json("https://console.vast.ai/api/v0/volumes")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError, OSError):
        return project_status, "query_failed"
    volume_rows = volume_payload.get("volumes") if isinstance(volume_payload, dict) else None
    if not isinstance(volume_rows, list):
        return project_status, "invalid_response"
    if not volume_rows:
        return project_status, "no_volumes"

    attached_ids = set()
    has_instance_lists = False
    for volume in volume_rows:
        if not isinstance(volume, dict) or not isinstance(volume.get("instances"), list):
            continue
        has_instance_lists = True
        for attached in volume["instances"]:
            candidates = []
            if isinstance(attached, dict):
                candidates = [attached.get(key) for key in ("id", "instance_id", "contract_id")]
            elif isinstance(attached, int) and not isinstance(attached, bool):
                candidates = [attached]
            for candidate in candidates:
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    attached_ids.add(candidate)
    if not has_instance_lists:
        return inventory_status, "unknown"

    target_ids = {instance_id for instance_id, _ in project_matches if isinstance(instance_id, int) and not isinstance(instance_id, bool)}
    running_ids = {instance_id for instance_id, instance_state in project_matches if instance_state == "running" and isinstance(instance_id, int) and not isinstance(instance_id, bool)}
    exited_ids = {instance_id for instance_id, instance_state in project_matches if instance_state == "exited" and isinstance(instance_id, int) and not isinstance(instance_id, bool)}
    running_attached = bool(running_ids & attached_ids)
    exited_attached = bool(exited_ids & attached_ids)
    if running_attached and exited_attached:
        return project_status, "running_and_exited_attached"
    if running_attached:
        return project_status, "running_attached"
    if exited_attached:
        return project_status, "exited_attached"
    if target_ids & attached_ids:
        return project_status, "other_3060_state_attached"
    if attached_ids:
        return project_status, "other_instances_attached"
    return project_status, "detached_volumes_present"

def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        status, volume_status = inventory()
    except Exception:
        status, volume_status = "query_failed", "unknown"
    print(f"checked_at_utc={checked}")
    print(f"vast_inventory={status}")
    print(f"vast_volume_attachment={volume_status}")
    return 0 if status not in {"configuration_missing", "query_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
