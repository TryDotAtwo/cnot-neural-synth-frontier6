"""Budgeted, single-instance RTX 3060 runner for the full Qhat4 epoch gate."""
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

LABEL = "cnot103-qhat4-training-v2"
IMAGE = "pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel"
DISK_GB = "64"
OFFER_QUERY = (
    "gpu_name=RTX_3060 num_gpus=1 verified=true"
)

TARGET_CONFIGURED = False
TARGET_CREATED = False
CREATE_ATTEMPTED = False
MARKETPLACE_GATE = "fail"
STOP_GATE = "not_needed"
KEY_CLEANUP_GATE = "not_needed"
REMOVE_KEY_RESULT = None


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truth(value):
    return value is True or value == 1 or (
        isinstance(value, str) and value.strip().casefold() == "true"
    )


def status_stopped(row):
    return (
        row.get("actual_status") in ("stopped", "exited")
        and row.get("intended_status") == "stopped"
    )


def is_3060(row):
    name = str(row.get("gpu_name") or "").casefold().replace("_", "").replace(" ", "")
    return "rtx" in name and "3060" in name


def instance_rows():
    rows = driver.vast(
        ["show", "instances", "--raw"], timeout=90
    )
    if not isinstance(rows, list):
        raise driver.GateError("Vast instance list has an unknown shape")
    return [row for row in rows if isinstance(row, dict)]


def configure_target(instance_id):
    global TARGET_CONFIGURED
    if isinstance(instance_id, bool) or not re.fullmatch(r"[0-9]{1,20}", str(instance_id)):
        raise driver.GateError("created target id is invalid")
    driver.INSTANCE_ID = int(instance_id)
    driver.INSTANCE_LABELS = {LABEL}
    driver.REQUESTED_INSTANCE_ID = str(instance_id)
    driver.SELECTED_INSTANCE_ID = ""
    driver.SELECTED_INSTANCE_LABEL = ""
    driver.VAST_ROOT = driver.pathlib.Path(
        f"/workspace/qhat4-recovery-{driver.run_tag()}"
    )
    os.environ["VAST_INSTANCE_ID"] = str(instance_id)
    TARGET_CONFIGURED = True


def resolve_label(expected_id=None, attempts=18, delay=10):
    for attempt in range(attempts):
        rows = [row for row in instance_rows() if row.get("label") == LABEL]
        if len(rows) > 1:
            raise driver.GateError("stable marketplace label is ambiguous")
        if rows:
            row = rows[0]
            ident = row.get("id")
            if expected_id is not None and str(ident) != str(expected_id):
                raise driver.GateError("created id and stable label do not match")
            if not is_3060(row):
                raise driver.GateError("stable label does not identify an RTX 3060")
            return row
        if attempt + 1 < attempts:
            time.sleep(delay)
    return None


def eligible_offers():
    rows = driver.vast(
        [
            "search", "offers", OFFER_QUERY,
            "--order=dph_total", "--limit=100", "--raw",
        ],
        timeout=90,
    )
    if isinstance(rows, dict):
        rows = rows.get("offers", rows.get("results"))
    if not isinstance(rows, list):
        raise driver.GateError("offer search response has an unknown shape")

    source = driver.get_instance(
        driver.SOURCE_INSTANCE_ID, driver.SOURCE_INSTANCE_LABELS
    )
    if source.get("actual_status") not in ("stopped", "exited") or source.get("intended_status") != "stopped":
        raise driver.GateError("pinned source RTX 3060 is not stopped")

    eligible = []
    rejected = {}
    for row in rows:
        if not isinstance(row, dict):
            rejected["row_shape"] = rejected.get("row_shape", 0) + 1
            continue
        name = str(row.get("gpu_name") or "").casefold().replace("_", "").replace(" ", "")
        gpu_count = number(row.get("num_gpus"))
        memory = number(row.get("gpu_ram"))
        capability = number(row.get("compute_cap"))
        reliability = number(row.get("reliability"))
        rate = number(row.get("dph_total"))
        ports = number(row.get("direct_port_count"))
        offer_id = row.get("ask_id") or row.get("id") or row.get("offer_id")
        checks = (
            ("gpu_name", "rtx" in name and "3060" in name),
            ("gpu_count", gpu_count == 1),
            ("gpu_ram", memory is not None and memory >= 12000),
            ("compute_cap", capability is not None and capability >= 8.6),
            ("reliability", reliability is not None and reliability >= 0.95),
            ("rentable", truth(row.get("rentable"))),
            ("direct_port", ports is not None and ports >= 1),
            ("rate", rate is not None and rate > 0),
            ("offer_id", re.fullmatch(r"[0-9]{1,20}", str(offer_id)) is not None),
        )
        failed = next((name for name, passed in checks if not passed), None)
        if failed:
            rejected[failed] = rejected.get(failed, 0) + 1
            continue
        # The query itself requires verified=true. Reject an explicit false value,
        # while allowing the raw CLI response to omit the filtered field.
        verified = row.get("verified")
        if verified is False or verified == 0 or (
            isinstance(verified, str) and verified.strip().casefold() == "false"
        ):
            rejected["verified"] = rejected.get("verified", 0) + 1
            continue
        same_machine = (
            source.get("machine_id") is not None
            and row.get("machine_id") is not None
            and str(source.get("machine_id")) == str(row.get("machine_id"))
        )
        source_host, offer_host = source.get("host_id"), row.get("host_id")
        source_location = source.get("geolocation")
        offer_location = row.get("geolocation")
        same_local_network = (
            source_host is not None and offer_host is not None
            and str(source_host) == str(offer_host)
            and isinstance(source_location, str) and isinstance(offer_location, str)
            and source_location.strip().casefold() == offer_location.strip().casefold()
        )
        local_copy = same_machine or same_local_network
        eligible.append((rate, -reliability, int(offer_id), local_copy))

    if not eligible:
        summary = ",".join(f"{key}:{rejected[key]}" for key in sorted(rejected)) or "none"
        print(f"offer_filter_summary=rows:{len(rows)} eligible:0 rejected:{summary}")
        raise driver.GateError("no verified RTX 3060 offer meets the hardware gates")
    eligible.sort(
        key=lambda item: (
            3.5 * item[0] + (1.5 if item[3] else 5.0),
            item[1],
            item[2],
        )
    )
    return eligible


def create_once(offer_id):
    global CREATE_ATTEMPTED, TARGET_CREATED
    command = [
        "vastai", "--api-key", driver.env_value("VAST_API_KEY"),
        "create", "instance", str(offer_id),
        "--image", IMAGE, "--disk", DISK_GB, "--ssh", "--direct",
        "--label", LABEL, "--cancel-unavail", "--raw",
    ]
    CREATE_ATTEMPTED = True
    try:
        proc = subprocess.run(
            command, capture_output=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        print("instance_create=response_uncertain")
        row = resolve_label()
        if row is None:
            raise driver.GateError("single marketplace create request was not confirmed") from None
        configure_target(row.get("id"))
        TARGET_CREATED = True
        print("marketplace_target=created")
        return row

    try:
        payload = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None

    reported_id = None
    if isinstance(payload, dict):
        candidate = payload.get("new_contract", payload.get("id"))
        if re.fullmatch(r"[0-9]{1,20}", str(candidate)):
            reported_id = str(candidate)
            configure_target(reported_id)
            TARGET_CREATED = True

    response_accepted = (
        proc.returncode == 0
        and isinstance(payload, dict)
        and payload.get("success") is True
    )
    if response_accepted:
        print("instance_create=accepted")
    else:
        print("instance_create=response_uncertain")

    row = resolve_label(expected_id=reported_id)
    if row is None:
        raise driver.GateError("created instance did not resolve by stable label")
    if not TARGET_CONFIGURED:
        configure_target(row.get("id"))
    TARGET_CREATED = True
    print("marketplace_target=created")
    return row


def stop_by_id():
    ident = str(driver.INSTANCE_ID)
    try:
        driver.vast(["stop", "instance", ident], timeout=60)
        return
    except driver.GateError:
        pass

    request = urllib.request.Request(
        f"{driver.VAST_API}/instances/{ident}/",
        data=json.dumps({"state": "stopped"}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {driver.env_value('VAST_API_KEY')}",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError):
        raise driver.GateError("owned instance stop request was not confirmed") from None


def api_instance():
    return driver.get_api_instance()


def wait_stopped(timeout=180, interval=10, reissue_if_running=False):
    deadline = time.monotonic() + timeout
    stop_reissued = False
    while time.monotonic() < deadline:
        row = api_instance()
        if row.get("label") != LABEL or not is_3060(row):
            raise driver.GateError("marketplace target identity changed")
        if status_stopped(row):
            return True
        if (
            reissue_if_running and not stop_reissued
            and row.get("actual_status") == "running"
            and row.get("intended_status") == "stopped"
        ):
            stop_by_id()
            stop_reissued = True
        time.sleep(interval)
    return False


def stop_new_instance_before_copy(row):
    if status_stopped(row):
        return
    stop_by_id()
    if not wait_stopped():
        raise driver.GateError("new instance did not stop before input copy")


def select_target():
    matches = [row for row in instance_rows() if row.get("label") == LABEL]
    if len(matches) > 1:
        raise driver.GateError("stable marketplace label is ambiguous")
    if not matches:
        offers = eligible_offers()
        spent = driver.project_charges()
        selected = next(
            (
                offer for offer in offers
                if spent + 3.5 * offer[0] + (1.5 if offer[3] else 5.0) < 20.0
            ),
            None,
        )
        if selected is None:
            print("creation_budget_gate=fail")
            raise driver.GateError("the $20 budget gate did not pass before instance creation")
        rate, _negative_reliability, offer_id, _local_copy = selected
        print("creation_budget_gate=pass")
        row = create_once(offer_id)
        stop_new_instance_before_copy(row)
        row = api_instance()
        if not status_stopped(row):
            raise driver.GateError("new instance was not verified stopped before input copy")
        print("marketplace_target=created_stopped")
        return row

    row = matches[0]
    if not is_3060(row):
        raise driver.GateError("stable label does not identify an RTX 3060")
    if status_stopped(row):
        configure_target(row.get("id"))
        print("marketplace_target=reused")
        return row

    if (
        row.get("actual_status") not in {"loading", "stopping", "running"}
        or row.get("intended_status") != "stopped"
    ):
        raise driver.GateError("existing stable-label RTX 3060 is not safely stopping")

    configure_target(row.get("id"))
    print("marketplace_target=waiting_for_stopped")
    if not wait_stopped(timeout=7200, interval=30, reissue_if_running=True):
        raise driver.GateError("existing stable-label RTX 3060 did not reach stopped within wait window")
    row = api_instance()
    print("marketplace_target=reused")
    return row


def install_runtime_flag_wrappers():
    original_start = driver.mark_started
    original_key = driver.mark_key_attached
    original_remove = driver.remove_ephemeral_key

    def mark_started():
        os.environ["VAST_STARTED_BY_WORKFLOW"] = "true"
        original_start()

    def mark_key_attached():
        os.environ["VAST_EPHEMERAL_KEY_ATTACHED"] = "true"
        original_key()

    def remove_ephemeral_key(state):
        global REMOVE_KEY_RESULT
        REMOVE_KEY_RESULT = bool(original_remove(state))
        return REMOVE_KEY_RESULT

    driver.mark_started = mark_started
    driver.mark_key_attached = mark_key_attached
    driver.remove_ephemeral_key = remove_ephemeral_key


def finalize():
    global STOP_GATE, KEY_CLEANUP_GATE
    if TARGET_CONFIGURED:
        started = os.environ.get("VAST_STARTED_BY_WORKFLOW") == "true"
        key_attached = os.environ.get("VAST_EPHEMERAL_KEY_ATTACHED") == "true"
        try:
            if started or key_attached:
                driver.stop_if_owned()
            else:
                current = api_instance()
                if not status_stopped(current):
                    stop_by_id()
                    wait_stopped()

            current = api_instance()
            STOP_GATE = "verified" if status_stopped(current) else "unverified"
        except Exception:
            STOP_GATE = "unverified"

        if not key_attached:
            KEY_CLEANUP_GATE = "not_needed"
        elif REMOVE_KEY_RESULT is True:
            KEY_CLEANUP_GATE = "pass"
        elif REMOVE_KEY_RESULT is False:
            KEY_CLEANUP_GATE = "unverified"
        else:
            KEY_CLEANUP_GATE = "not_checked"

    try:
        driver.cleanup_runner_key()
    except Exception:
        if KEY_CLEANUP_GATE == "pass":
            KEY_CLEANUP_GATE = "unverified"

    if TARGET_CONFIGURED:
        print(f"instance_stop={STOP_GATE}")
    print(f"ephemeral_key_cleanup={KEY_CLEANUP_GATE}")
    driver.write_output({
        "marketplace_gate": MARKETPLACE_GATE,
        "instance_stop": STOP_GATE,
        "ephemeral_key_cleanup": KEY_CLEANUP_GATE,
    })


def safe_failure_code(error):
    message = str(error).casefold()
    if "instance list has an unknown shape" in message:
        return "instance_list_shape"
    if "stable marketplace label is ambiguous" in message:
        return "stable_label_ambiguous"
    if "is not stopped" in message:
        return "target_not_stopped"
    if "does not identify an rtx 3060" in message:
        return "target_gpu_mismatch"
    if "budget" in message:
        return "budget_gate"
    if "offer" in message:
        return "offer_gate"
    if "create" in message:
        return "create_gate"
    if "copy" in message:
        return "copy_gate"
    if "start" in message or "ssh" in message:
        return "start_gate"
    if "remote" in message or "dataset" in message or "trainer" in message:
        return "training_gate"
    if "instance" in message or "state" in message:
        return "instance_state"
    return "operational"


def main():
    global MARKETPLACE_GATE
    os.environ.setdefault("VAST_STARTED_BY_WORKFLOW", "false")
    os.environ.setdefault("VAST_EPHEMERAL_KEY_ATTACHED", "false")
    install_runtime_flag_wrappers()

    try:
        select_target()
        driver.preflight()
        driver.start()
        state = driver.get_instance()
        if state.get("actual_status") != "running" or state.get("intended_status") != "running":
            raise driver.GateError("selected RTX 3060 did not reach running state")
        driver.run_experiment(state)
        MARKETPLACE_GATE = "pass"
        print("marketplace_gate=pass")
    except Exception as error:
        category = driver.failure_category(error)
        if category not in {
            "copy_network", "fallback_path", "data_copy", "vast_query", "budget",
            "vast_start", "ssh", "remote_preflight", "instance_state", "operational",
        }:
            category = "operational"
        print("marketplace_gate=fail")
        print(f"failure_category={category}")
        print(f"runner_failure={safe_failure_code(error)}")
    finally:
        finalize()

    if MARKETPLACE_GATE != "pass" or STOP_GATE != "verified":
        return 1
    if KEY_CLEANUP_GATE not in ("pass", "not_needed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
