"""Reissue a stop for the uniquely labeled Qhat4 RTX 3060 and verify state."""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

LABEL = "cnot103-qhat4-training-v1"
ALLOWED = {"created", "loading", "running", "stopping", "stopped", "exited", "offline", "unknown", "error"}
VAST_API = "https://console.vast.ai/api/v0"


def query_rows(key):
    proc = subprocess.run(
        ["vastai", "--api-key", key, "show", "instances", "--raw"],
        capture_output=True,
        timeout=90,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError("status query failed")
    rows = json.loads(proc.stdout.decode("utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("status response shape invalid")
    return [row for row in rows if isinstance(row, dict)]


def safe_status(row):
    actual = row.get("actual_status")
    intended = row.get("intended_status")
    return (
        actual if isinstance(actual, str) and actual in ALLOWED else "other",
        intended if isinstance(intended, str) and intended in ALLOWED else "other",
    )


def stopped(row):
    return row.get("actual_status") in ("stopped", "exited") and row.get("intended_status") == "stopped"


def request_stop(key, ident):
    proc = subprocess.run(
        ["vastai", "--api-key", key, "stop", "instance", str(ident)],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if proc.returncode == 0:
        return True
    request = urllib.request.Request(
        f"{VAST_API}/instances/{ident}/",
        data=json.dumps({"state": "stopped"}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response.read()
        return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def main():
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        print("status_probe=configuration_missing")
        return 1
    try:
        matches = [row for row in query_rows(key) if row.get("label") == LABEL]
    except (OSError, subprocess.TimeoutExpired, RuntimeError, UnicodeDecodeError, json.JSONDecodeError):
        print("status_probe=query_failed")
        print("instance_stop=unverified")
        return 1
    if len(matches) != 1:
        print("target_found=false")
        print("instance_stop=unverified")
        return 1

    row = matches[0]
    gpu = str(row.get("gpu_name") or "").casefold().replace("_", "").replace(" ", "")
    if "rtx" not in gpu or "3060" not in gpu:
        print("target_identity=unverified")
        print("instance_stop=unverified")
        return 1

    actual, intended = safe_status(row)
    print("target_found=true")
    if stopped(row):
        print(f"actual_status={actual}")
        print(f"intended_status={intended}")
        print("instance_stop=verified")
        return 0
    if intended != "stopped":
        print(f"actual_status={actual}")
        print(f"intended_status={intended}")
        print("instance_stop=not_requested")
        return 1

    ident = row.get("id")
    if isinstance(ident, bool) or not str(ident).isdigit():
        print("instance_stop=unverified")
        return 1
    if not request_stop(key, ident):
        print("instance_stop=unverified")
        return 1
    print("stop_request=sent")

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            rows = query_rows(key)
            current = [item for item in rows if str(item.get("id")) == str(ident)]
        except (OSError, subprocess.TimeoutExpired, RuntimeError, UnicodeDecodeError, json.JSONDecodeError):
            current = []
        if len(current) == 1:
            row = current[0]
            if row.get("label") != LABEL:
                print("target_identity=unverified")
                print("instance_stop=unverified")
                return 1
            actual, intended = safe_status(row)
            if stopped(row):
                print(f"actual_status={actual}")
                print(f"intended_status={intended}")
                print("instance_stop=verified")
                return 0
        time.sleep(10)

    print(f"actual_status={actual}")
    print(f"intended_status={intended}")
    print("instance_stop=pending")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
