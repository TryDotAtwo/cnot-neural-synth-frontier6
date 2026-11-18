"""Read-only status probe for the unique Qhat4 marketplace RTX 3060."""
import json
import os
import subprocess
import sys

LABEL = "cnot103-qhat4-training-v1"
ALLOWED = {"created", "loading", "running", "stopping", "stopped", "exited", "offline", "unknown", "error"}


def main():
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        print("status_probe=configuration_missing")
        return 1
    try:
        proc = subprocess.run(
            ["vastai", "--api-key", key, "show", "instances", "--raw"],
            capture_output=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        print("status_probe=query_unconfirmed")
        return 1
    if proc.returncode:
        print("status_probe=query_failed")
        return 1
    try:
        rows = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("status_probe=response_invalid")
        return 1
    if not isinstance(rows, list):
        print("status_probe=response_shape_invalid")
        return 1

    matches = [row for row in rows if isinstance(row, dict) and row.get("label") == LABEL]
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

    actual = row.get("actual_status")
    intended = row.get("intended_status")
    actual_safe = actual if actual in ALLOWED else "other"
    intended_safe = intended if intended in ALLOWED else "other"
    print("target_found=true")
    print(f"actual_status={actual_safe}")
    print(f"intended_status={intended_safe}")
    stopped = actual in ("stopped", "exited") and intended == "stopped"
    print("instance_stop=verified" if stopped else "instance_stop=pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
