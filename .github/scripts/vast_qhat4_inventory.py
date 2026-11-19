"""Read-only, redacted inventory of project-owned Vast instances."""
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, ".github/scripts")
import vast_qhat4_driver as driver  # noqa: E402

ALLOWED = {"created", "loading", "running", "stopping", "stopped", "exited", "offline", "unknown", "error"}


def bucket(count):
    if count == 0:
        return "none"
    if count == 1:
        return "one"
    if count <= 5:
        return "two_to_five"
    return "six_plus"


def query_rows(key):
    proc = subprocess.run(
        ["vastai", "--api-key", key, "show", "instances", "--raw"],
        capture_output=True,
        timeout=90,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError("query_failed")
    rows = json.loads(proc.stdout.decode("utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("response_invalid")
    return rows


def main():
    print("check_time_utc=" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        print("query_outcome=configuration_missing")
        return 1
    try:
        rows = query_rows(key)
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        outcome = "response_invalid" if str(error) == "response_invalid" else "query_failed"
        print("query_outcome=" + outcome)
        return 1

    owned = []
    for row in rows:
        try:
            ident = int(row.get("id"))
        except (TypeError, ValueError):
            ident = None
        label = row.get("label")
        project_label = isinstance(label, str) and label.casefold().startswith("cnot103")
        if project_label or ident in driver.OWNED_IDS:
            owned.append(row)

    actual = [row.get("actual_status") if isinstance(row.get("actual_status"), str) else "other" for row in owned]
    intended = [row.get("intended_status") if isinstance(row.get("intended_status"), str) else "other" for row in owned]
    known_actual = [value if value in ALLOWED else "other" for value in actual]
    stopped = sum(
        1 for row in owned
        if row.get("actual_status") in ("stopped", "exited")
        and row.get("intended_status") == "stopped"
    )
    not_stopped = len(owned) - stopped
    print("query_outcome=success")
    print("owned_instance_bucket=" + bucket(len(owned)))
    print("owned_not_stopped_bucket=" + bucket(not_stopped))
    print("owned_stopped_bucket=" + bucket(stopped))
    print("owned_loading=" + ("present" if "loading" in known_actual else "none"))
    print("owned_running=" + ("present" if "running" in known_actual else "none"))

    legacy = [row for row in owned if row.get("label") == "cnot103-qhat4-training-v1"]
    if len(legacy) != 1:
        print("qhat4_v1_target=" + ("none" if not legacy else "ambiguous"))
    else:
        state = legacy[0].get("actual_status")
        print("qhat4_v1_target=" + (state if isinstance(state, str) and state in ALLOWED else "other"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
