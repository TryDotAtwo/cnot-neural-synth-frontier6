"""Destroy project-owned Vast instances and their unambiguous project volumes."""
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, ".github/scripts")
import vast_qhat4_driver as driver  # noqa: E402

VAST_API = "https://console.vast.ai/api/v0"
PROJECT_PREFIX = "cnot103"
REQUEST_INTERVAL = 0.45
MAX_TARGETS = 50


def bucket(count):
    if count == 0:
        return "none"
    if count == 1:
        return "one"
    if count <= 5:
        return "two_to_five"
    return "six_plus"


def safe_id(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[0-9]{1,20}", text):
            return int(text)
        match = re.fullmatch(r"V\.([0-9]{1,20})", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def is_project_instance(row):
    ident = safe_id(row.get("id"))
    label = row.get("label")
    labelled = isinstance(label, str) and label.casefold().startswith(PROJECT_PREFIX)
    return labelled or ident in driver.OWNED_IDS


def list_instances():
    rows = driver.vast(["show", "instances", "--raw"], timeout=90)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("instance_inventory_unverified")
    return rows


def api_request(method, path, payload=None):
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        raise RuntimeError("configuration_missing")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{VAST_API}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = None
            return response.status, body
    except urllib.error.HTTPError as error:
        try:
            raw = error.read()
            body = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            body = None
        return error.code, body
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None


def list_volumes():
    status, payload = api_request("GET", "/volumes")
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError("volume_inventory_unverified")
    rows = payload.get("volumes")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("volume_inventory_unverified")
    return rows


def volume_id(row):
    for key in ("id", "volume_id", "contract_id"):
        ident = safe_id(row.get(key))
        if ident is not None:
            return ident
    return None


def volume_instance_refs(row):
    refs = set()
    unknown = False
    raw = row.get("instances")
    if raw is None:
        return refs, unknown
    if not isinstance(raw, list):
        return refs, True
    for item in raw:
        if isinstance(item, dict):
            ident = safe_id(item.get("id"))
            if ident is None:
                ident = safe_id(item.get("instance_id"))
            if ident is None:
                unknown = True
            else:
                refs.add(ident)
        else:
            ident = safe_id(item)
            if ident is None:
                unknown = True
            else:
                refs.add(ident)
    return refs, unknown


def volume_has_project_name(row):
    for key in ("label", "name", "volume_name", "description"):
        value = row.get(key)
        if isinstance(value, str) and value.strip().casefold().startswith(PROJECT_PREFIX):
            return True
    return False


def volume_is_project_owned(row, project_instance_ids):
    refs, _unknown = volume_instance_refs(row)
    return volume_has_project_name(row) or bool(refs & project_instance_ids)


def delete_instance(ident):
    status, body = api_request("DELETE", f"/instances/{ident}")
    time.sleep(REQUEST_INTERVAL)
    if status == 404:
        return True
    return status == 200 and isinstance(body, dict) and body.get("success") is True


def instance_absent(ident):
    status, body = api_request("GET", f"/instances/{ident}")
    time.sleep(REQUEST_INTERVAL)
    if status == 404:
        return True
    if status == 200:
        return False
    return None


def delete_volume(ident):
    status, body = api_request("DELETE", "/volumes", {"id": ident})
    time.sleep(REQUEST_INTERVAL)
    if status == 404:
        return True
    return status == 200 and isinstance(body, dict) and body.get("success") is True


def wait_instances_absent(instance_ids, timeout=300):
    deadline = time.monotonic() + timeout
    last = {ident: None for ident in instance_ids}
    while time.monotonic() < deadline:
        for ident in instance_ids:
            last[ident] = instance_absent(ident)
        if all(value is True for value in last.values()):
            return True
        time.sleep(10)
    return False


def main():
    print("check_time_utc=" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    try:
        rows = list_instances()
        project_rows = [row for row in rows if is_project_instance(row)]
        instance_ids = {safe_id(row.get("id")) for row in project_rows}
        if None in instance_ids:
            print("cleanup_gate=unverified")
            print("ownership_scan=unverified")
            return 1
        if len(instance_ids) > MAX_TARGETS:
            print("cleanup_gate=unverified")
            print("ownership_scan=target_cap_exceeded")
            return 1

        volumes = list_volumes()
    except Exception as error:
        category = str(error) if str(error) in {
            "configuration_missing", "instance_inventory_unverified",
            "volume_inventory_unverified",
        } else "query_failed"
        print("ownership_scan=" + category)
        print("cleanup_gate=unverified")
        return 1

    project_volumes = []
    volume_ambiguous = False
    project_instance_ids = set(instance_ids)
    for row in volumes:
        refs, unknown_refs = volume_instance_refs(row)
        labelled = volume_has_project_name(row)
        linked = bool(refs & project_instance_ids)
        unrelated_refs = refs - project_instance_ids
        if (labelled or linked) and (unknown_refs or unrelated_refs):
            volume_ambiguous = True
            continue
        if not (labelled or linked):
            continue
        ident = volume_id(row)
        if ident is None:
            volume_ambiguous = True
            continue
        project_volumes.append(ident)

    print("ownership_scan=pass")
    print("owned_instance_bucket=" + bucket(len(instance_ids)))
    print("owned_volume_bucket=" + bucket(len(project_volumes)))
    if volume_ambiguous:
        print("shared_or_unidentified_volume=present")
    else:
        print("shared_or_unidentified_volume=none")

    instance_requests_ok = True
    for ident in sorted(instance_ids):
        if not delete_instance(ident):
            instance_requests_ok = False

    instances_gone = wait_instances_absent(instance_ids) if instance_ids else True
    volume_requests_ok = True
    for ident in sorted(set(project_volumes)):
        # Do not delete a volume that may still be attached to an instance.
        if not instances_gone:
            volume_requests_ok = False
            continue
        if not delete_volume(ident):
            volume_requests_ok = False

    try:
        final_instances = [row for row in list_instances() if is_project_instance(row)]
        final_volumes = list_volumes()
        remaining_volume_rows = [
            row for row in final_volumes
            if volume_is_project_owned(row, project_instance_ids)
            or volume_id(row) in set(project_volumes)
        ]
        instances_empty = not final_instances
        volumes_empty = not remaining_volume_rows and not volume_ambiguous
    except Exception:
        instances_empty = False
        volumes_empty = False

    print("instance_delete=" + ("verified" if instances_empty else "unverified"))
    print("volume_delete=" + (
        "verified" if volumes_empty and project_volumes
        else "not_present" if volumes_empty
        else "unverified"
    ))
    print("post_delete_instances=" + ("empty" if instances_empty else "remaining"))
    print("post_delete_volumes=" + ("empty" if volumes_empty else "remaining_or_unverified"))

    complete = (
        instance_requests_ok and volume_requests_ok
        and instances_empty and volumes_empty
    )
    print("cleanup_gate=" + ("pass" if complete else "partial"))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
