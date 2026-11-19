#!/usr/bin/env python3
"""Summarize private Rust diagnostics without moving raw logs off the Vast disk."""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from vast_probe_existing_2x3060 import (
    ApiFailure,
    api_json,
    find_attached_key_id,
    safe_budget_projection,
    state,
    target_rows,
)

MAX_REMOTE_SECONDS = 45
REMOTE_STATUS = "not_started"
REMOTE_BUILD_CLASS = "unavailable"


def remote_summary(host: str, port: int, key_path: pathlib.Path, known_hosts: pathlib.Path) -> str | None:
    global REMOTE_STATUS, REMOTE_BUILD_CLASS
    REMOTE_STATUS = "ssh_failed"
    code = r"""
import json, pathlib, re, sys
root = pathlib.Path("/workspace/tqp_private_build_diagnostics")
candidates = sorted(root.glob("run-*-attempt-*.jsonl"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
candidate = candidates[0] if candidates else None
if candidate is None:
    print("private_diagnostic_sites=missing")
    raise SystemExit(0)
def area(path):
    p = "/" + path.replace("\\", "/").lstrip("/")
    if p.endswith("/run_config/execution.rs"): return "run_exec_root"
    if p.endswith("/run_config.rs"): return "run_config_root"
    if "/run_config/execution/" in p:
        name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name in {"artifacts","coverage","evaluation","feature_runtime","field","h07","quality","search"}:
            return "run_exec_" + name
    if "/run_config/manifest/" in p:
        name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name in {"mod","parser","validation","tests"}: return "run_manifest_" + name
    if p.endswith("/nn_training.rs"): return "nn_training_root"
    if "/nn_training/trainer/" in p:
        name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name in {"device","step","checkpoint","data_parallel","microbatch"}: return "nn_trainer_" + name
    if p.endswith("/nn_training/trainer.rs"): return "nn_trainer_root"
    if p.endswith("/training_session.rs"): return "training_session_root"
    if "/training_session/" in p:
        name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name in {"device","lifecycle","run","validation"}: return "training_session_" + name
    return "other"
pairs = set()
non_json_lines = 0
cargo_missing = False
for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        non_json_lines += 1
        lowered = line.casefold()
        if "cargo: command not found" in lowered or "bash: cargo: not found" in lowered:
            cargo_missing = True
        continue
    if item.get("reason") != "compiler-message":
        continue
    message = item.get("message")
    if not isinstance(message, dict) or message.get("level") != "error":
        continue
    code = message.get("code")
    code = code.get("code") if isinstance(code, dict) else None
    if not isinstance(code, str) or not re.fullmatch(r"E[0-9]{4}", code):
        code = "EOTHER"
    spans = message.get("spans")
    primary = next((s for s in spans if isinstance(s, dict) and s.get("is_primary")), None) if isinstance(spans, list) else None
    filename = primary.get("file_name", "") if primary else ""
    line_number = primary.get("line_start", 0) if primary else 0
    line_number = line_number if isinstance(line_number, int) and line_number > 0 else 0
    pairs.add(code + "@" + area(filename) + ":" + str(line_number))
if not pairs:
    print("private_diagnostic_sites=none")
else:
    print("private_diagnostic_sites=" + "+".join(sorted(pairs)[:6]))
if pairs:
    failure_class = "compiler_errors"
elif cargo_missing:
    failure_class = "cargo_not_found"
elif non_json_lines:
    failure_class = "unclassified_text"
else:
    failure_class = "no_compiler_errors"
print("private_build_failure_class=" + failure_class)
"""
    argv = [
        "ssh", "-i", str(key_path), "-p", str(port),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=8",
        "-o", "ConnectionAttempts=1",
        f"root@{host}", "bash -s",
    ]
    remote = "set -euo pipefail\npython3 - <<'PY'\n" + code + "\nPY\n"
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(4):
        try:
            result = subprocess.run(
                argv, input=remote, capture_output=True, text=True,
                timeout=MAX_REMOTE_SECONDS, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0:
            break
        if result is not None and result.returncode != 255:
            return None
        if attempt < 3:
            time.sleep(3)
    if result is None:
        return None
    if result.returncode != 0:
        REMOTE_STATUS = "remote_failed"
        error_text = (result.stderr or "").casefold()
        if "syntaxerror" in error_text or "invalid syntax" in error_text:
            REMOTE_BUILD_CLASS = "summarizer_syntax_error"
        elif "filenotfounderror" in error_text or "no such file" in error_text:
            REMOTE_BUILD_CLASS = "remote_file_issue"
        elif "permission denied" in error_text:
            REMOTE_BUILD_CLASS = "remote_permission_denied"
        elif "python3: not found" in error_text or "command not found" in error_text:
            REMOTE_BUILD_CLASS = "remote_tool_missing"
        elif "attributeerror" in error_text or "typeerror" in error_text:
            REMOTE_BUILD_CLASS = "remote_parse_error"
        else:
            REMOTE_BUILD_CLASS = "remote_execution_error"
        return None
    for line in result.stdout.splitlines():
        if line.startswith("private_build_failure_class="):
            value = line.partition("=")[2]
            if value not in {"compiler_errors", "cargo_not_found", "unclassified_text", "no_compiler_errors"}:
                REMOTE_STATUS = "marker_invalid"
                return None
            REMOTE_BUILD_CLASS = value
            continue
        if not line.startswith("private_diagnostic_sites="):
            continue
        value = line.partition("=")[2]
        if value in {"missing", "none"}:
            REMOTE_STATUS = "ok"
            return value
        if re.fullmatch(r"E[0-9]{4}@[a-z0-9_]+:[0-9]+(?:\+E[0-9]{4}@[a-z0-9_]+:[0-9]+){0,5}", value):
            REMOTE_STATUS = "ok"
            return value
        REMOTE_STATUS = "marker_invalid"
        return None
    REMOTE_STATUS = "marker_missing"
    return None


AREA_FILES = {
    "run_config_root": "native/src/run_config.rs",
    "run_manifest_mod": "native/src/run_config/manifest/mod.rs",
    "run_manifest_parser": "native/src/run_config/manifest/parser.rs",
    "run_manifest_validation": "native/src/run_config/manifest/validation.rs",
    "run_exec_artifacts": "native/src/run_config/execution/artifacts.rs",
    "run_exec_coverage": "native/src/run_config/execution/coverage.rs",
    "run_exec_evaluation": "native/src/run_config/execution/evaluation.rs",
    "run_exec_feature_runtime": "native/src/run_config/execution/feature_runtime.rs",
    "run_exec_field": "native/src/run_config/execution/field.rs",
    "run_exec_h07": "native/src/run_config/execution/h07.rs",
    "run_exec_quality": "native/src/run_config/execution/quality.rs",
    "run_exec_search": "native/src/run_config/execution/search.rs",
    "run_exec_root": "native/src/run_config/execution.rs",
    "nn_training_root": "native/src/nn_training.rs",
    "nn_trainer_root": "native/src/nn_training/trainer.rs",
    "nn_trainer_device": "native/src/nn_training/trainer/device.rs",
    "nn_trainer_step": "native/src/nn_training/trainer/step.rs",
    "nn_trainer_checkpoint": "native/src/nn_training/trainer/checkpoint.rs",
    "nn_trainer_microbatch": "native/src/nn_training/trainer/device/microbatch.rs",
    "nn_trainer_data_parallel": "native/src/nn_training/trainer/device/data_parallel.rs",
    "training_session_root": "native/src/training_session.rs",
    "training_session_run": "native/src/training_session/run.rs",
}
SAFE_FUNCTIONS = {
    "execute_benchmark_manifest", "execute_run_manifest", "audit_json", "parse_run_manifest", "validate", "new",
    "build", "run_device_updates", "run_device_block",
    "run_device_microbatch_updates", "run_device_microbatch_block",
    "run_device_data_parallel_block", "bind_replica_parameters",
    "sync_replica", "accumulate_replica",
}


def summarize_sites(value: str, workspace: pathlib.Path) -> str:
    if value in {"missing", "none"}:
        return value
    results = []
    for item in value.split("+")[:6]:
        match = re.fullmatch(r"(E[0-9]{4})@([a-z0-9_]+):([0-9]+)", item)
        if not match:
            return "unavailable"
        code, area, line_text = match.groups()
        relative = AREA_FILES.get(area)
        function = "module"
        if relative:
            source = workspace / relative
            try:
                lines = source.read_text(encoding="utf-8").splitlines()
                line_number = int(line_text)
                for line in lines[:max(0, min(line_number, len(lines)))]:
                    found = re.match(
                        r"\s*(?:(?:pub(?:\([^)]*\))?)\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)",
                        line,
                    )
                    if found:
                        function = found.group(1)
            except (OSError, ValueError):
                function = "source_unavailable"
        if function not in SAFE_FUNCTIONS and function not in {"module", "source_unavailable"}:
            function = "function_other"
        results.append(code + "_" + area + "_" + function)
    return "+".join(results)


def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary = "unavailable"
    cleanup = "not_needed"
    temp_dir: pathlib.Path | None = None
    attach_attempted = False
    public_key = ""
    instance_id: int | None = None
    api_key = os.environ.get("VAST_API_KEY", "")

    try:
        if not api_key:
            summary = "configuration_missing"
            raise RuntimeError()
        rows = target_rows(api_json(api_key, "/instances/"))
        running = [row for row in rows if state(row) == "running"]
        if len(running) != 1:
            summary = "target_ambiguous"
            raise RuntimeError()
        if safe_budget_projection(rows) != "within_48h_cap":
            summary = "budget_gate_closed"
            raise RuntimeError()

        target = running[0]
        candidate = target.get("id")
        host = target.get("ssh_host")
        port = target.get("ssh_port")
        if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate <= 0:
            summary = "target_unavailable"
            raise RuntimeError()
        if not isinstance(host, str) or not re.fullmatch(r"ssh[0-9]+\.vast\.ai", host.casefold()):
            summary = "endpoint_unavailable"
            raise RuntimeError()
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            summary = "endpoint_unavailable"
            raise RuntimeError()
        instance_id = candidate

        temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="vast-summary-"))
        key_path = temp_dir / "temporary"
        known_hosts = temp_dir / "known_hosts"
        generated = subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "gha-summary-" + os.environ.get("GITHUB_RUN_ID", "run"), "-f", str(key_path)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if generated.returncode != 0:
            summary = "key_generation_failed"
            raise RuntimeError()
        public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()
        if find_attached_key_id(api_key, instance_id, public_key) is not None:
            summary = "key_collision"
            raise RuntimeError()
        attach_attempted = True
        attached = api_json(api_key, f"/instances/{instance_id}/ssh", "POST", {"ssh_key": public_key})
        if not isinstance(attached, dict) or attached.get("success") is not True:
            summary = "key_attach_failed"
            raise RuntimeError()
        if find_attached_key_id(api_key, instance_id, public_key) is None:
            summary = "key_attach_unconfirmed"
            raise RuntimeError()

        value = remote_summary(host, port, key_path, known_hosts)
        summary = summarize_sites(value, pathlib.Path(os.environ.get("GITHUB_WORKSPACE", "."))) if value else "unavailable"

    except ApiFailure as error:
        summary = "api_auth_failed" if error.status in {401, 403} else "api_failed"
    except (OSError, subprocess.TimeoutExpired):
        summary = "controller_failed"
    except RuntimeError:
        pass
    except Exception:
        summary = "controller_failed"
    finally:
        if attach_attempted and api_key and instance_id is not None and public_key:
            try:
                key_id = find_attached_key_id(api_key, instance_id, public_key)
                if key_id is None:
                    cleanup = "not_found"
                else:
                    detached = api_json(api_key, f"/instances/{instance_id}/ssh/{key_id}", "DELETE")
                    cleanup = "detached" if isinstance(detached, dict) and detached.get("success") is True else "failed"
            except (ApiFailure, ValueError):
                cleanup = "failed"
        if temp_dir is not None:
            for name in ("temporary", "temporary.pub", "known_hosts"):
                try:
                    (temp_dir / name).unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    if summary not in {"missing", "none", "unavailable", "configuration_missing", "run_identity_invalid",
                       "target_ambiguous", "budget_gate_closed", "target_unavailable", "endpoint_unavailable",
                       "key_generation_failed", "key_collision", "key_attach_failed", "key_attach_unconfirmed",
                       "api_auth_failed", "api_failed", "controller_failed"} and not re.fullmatch(
                           r"E[0-9]{4}_[a-z0-9_]+(?:\+E[0-9]{4}_[a-z0-9_]+){0,5}", summary
                       ):
        summary = "unavailable"
    print(f"checked_at={checked}")
    print(f"private_diagnostic_remote_status={REMOTE_STATUS}")
    print(f"private_build_failure_class={REMOTE_BUILD_CLASS}")
    print(f"private_diagnostic_summary={summary}")
    print(f"temporary_ssh_key={cleanup}")
    return 0 if cleanup in {"detached", "not_needed"} and summary != "unavailable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
