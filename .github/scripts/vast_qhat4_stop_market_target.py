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
    key = os.environ["VAST_API_KEY"].strip()
    try:
        proc = subprocess.run(
            ["vastai", "--api-key", key, "stop", "instance", str(instance_id), "--raw"],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        proc = None
    if proc is not None and proc.returncode == 0:
        return

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
            response.read()
    except (urllib.error.URLError, TimeoutError):
        fail("Vast stop request was not confirmed")


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
        while time.monotonic() < deadline:
            matches = [item for item in rows() if str(item.get("id")) == str(instance_id)]
            if len(matches) == 1 and matches[0].get("label") == LABEL and is_3060(matches[0]):
                if stopped(matches[0]):
                    print("instance_stop=verified")
                    return 0
            time.sleep(10)
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
