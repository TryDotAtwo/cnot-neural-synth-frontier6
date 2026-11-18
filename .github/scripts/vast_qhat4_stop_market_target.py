"""Stop and verify only the uniquely labeled Qhat4 marketplace RTX 3060."""
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

LABEL = "cnot103-qhat4-training-v1"
API = "https://console.vast.ai/api/v0"


def fail(message):
    raise RuntimeError(message)


def cli_json(args, timeout=60):
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        fail("required cleanup configuration is missing")
    try:
        proc = subprocess.run(
            ["vastai", "--api-key", key, *args, "--raw"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        fail("Vast cleanup query was not confirmed")
    if proc.returncode:
        fail("Vast cleanup query did not succeed")
    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail("Vast cleanup response could not be parsed")


def rows():
    value = cli_json(["show", "instances"])
    if not isinstance(value, list):
        fail("Vast instance list has an unknown shape")
    return [item for item in value if isinstance(item, dict)]


def is_3060(row):
    name = str(row.get("gpu_name") or "").casefold().replace("_", "").replace(" ", "")
    return "rtx" in name and "3060" in name


def stopped(row):
    return row.get("actual_status") in ("stopped", "exited") and row.get("intended_status") == "stopped"


def stop(instance_id):
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        fail("required cleanup configuration is missing")
    request = urllib.request.Request(
        f"{API}/instances/{instance_id}/",
        data=json.dumps({"state": "stopped"}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            status_code = response.status
            body = response.read()
    except urllib.error.HTTPError as error:
        status_code = error.code
        try:
            body = error.read()
        except OSError:
            body = b""
    except (urllib.error.URLError, TimeoutError):
        print("stop_api_result=unconfirmed")
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    accepted = (
        200 <= status_code < 300
        and isinstance(payload, dict)
        and payload.get("success") is True
    )
    print("stop_api_result=accepted" if accepted else "stop_api_result=rejected")
    return accepted


def instance_detail(instance_id):
    matches = [item for item in rows() if str(item.get("id")) == str(instance_id)]
    if len(matches) != 1:
        fail("Vast target did not resolve uniquely in the instance list")
    return matches[0]


def main():
    matches = [item for item in rows() if item.get("label") == LABEL]
    if len(matches) != 1:
        print("cleanup_target=ambiguous_or_missing")
        print("instance_stop=unverified")
        return 1
    row = matches[0]
    instance_id = row.get("id")
    if not re.fullmatch(r"[0-9]{1,20}", str(instance_id)) or not is_3060(row):
        print("cleanup_target=identity_mismatch")
        print("instance_stop=unverified")
        return 1

    print("cleanup_target=verified")
    if not stopped(row):
        stop(instance_id)
        deadline = time.monotonic() + 180
        last_actual = "unknown"
        last_intended = "unknown"
        for attempt in range(19):
            current = instance_detail(instance_id)
            if current.get("label") != LABEL or not is_3060(current):
                print("cleanup_target=identity_mismatch")
                print("instance_stop=unverified")
                return 1
            actual = current.get("actual_status")
            intended = current.get("intended_status")
            allowed = {"running", "stopped", "exited", "starting", "stopping", "created", "loading", "unknown", "offline", "error"}
            last_actual = actual if actual in allowed else "other"
            last_intended = intended if intended in allowed else "other"
            if stopped(current):
                print("instance_stop=verified")
                return 0
            if time.monotonic() >= deadline or attempt == 18:
                break
            time.sleep(10)
        print(f"stop_last_state=actual:{last_actual},intended:{last_intended}")
        print("instance_stop=unverified")
        return 1

    print("instance_stop=verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("cleanup_target=failed")
        print("instance_stop=unverified")
        raise SystemExit(1)
