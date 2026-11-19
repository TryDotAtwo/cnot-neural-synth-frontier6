#!/usr/bin/env python3
"""Store private Rust diagnostics on the existing Vast disk, never in Actions logs."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import shutil
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

REMOTE_DIR = "/workspace/tqp_private_build_diagnostics"
MAX_BYTES = 64 * 1024 * 1024


def ssh(host: str, port: int, key_path: pathlib.Path, known_hosts: pathlib.Path, command: str, timeout: int = 20) -> subprocess.CompletedProcess[str] | None:
    argv = [
        "ssh", "-i", str(key_path), "-p", str(port),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=8",
        "-o", "ConnectionAttempts=1",
        f"root@{host}", command,
    ]
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(4):
        try:
            last = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            last = None
        if last is not None and last.returncode != 255:
            return last
        if attempt < 3:
            time.sleep(3)
    return last


def ensure_ssh_tools() -> bool:
    if all(shutil.which(name) for name in ("ssh", "ssh-keygen", "scp")):
        return True
    if not shutil.which("apt-get"):
        return False
    try:
        updated = subprocess.run(
            ["apt-get", "update", "-qq"],
            capture_output=True, text=True, timeout=240, check=False,
        )
        if updated.returncode != 0:
            return False
        installed = subprocess.run(
            ["apt-get", "install", "-y", "-qq", "openssh-client"],
            capture_output=True, text=True, timeout=240, check=False,
        )
        return installed.returncode == 0 and all(
            shutil.which(name) for name in ("ssh", "ssh-keygen", "scp")
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


AREA_FILES = {
    "run_config_root": "native/src/run_config.rs",
    "run_exec_root": "native/src/run_config/execution.rs",
    "run_exec_artifacts": "native/src/run_config/execution/artifacts.rs",
    "run_exec_coverage": "native/src/run_config/execution/coverage.rs",
    "run_exec_evaluation": "native/src/run_config/execution/evaluation.rs",
    "run_exec_feature_runtime": "native/src/run_config/execution/feature_runtime.rs",
    "run_exec_field": "native/src/run_config/execution/field.rs",
    "run_exec_h07": "native/src/run_config/execution/h07.rs",
    "run_exec_quality": "native/src/run_config/execution/quality.rs",
    "run_exec_search": "native/src/run_config/execution/search.rs",
    "run_manifest_mod": "native/src/run_config/manifest/mod.rs",
    "run_manifest_parser": "native/src/run_config/manifest/parser.rs",
    "run_manifest_validation": "native/src/run_config/manifest/validation.rs",
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


def diagnostic_area(path: str) -> str:
    p = "/" + path.replace("\\", "/").lstrip("/")
    if p.endswith("/run_config/execution.rs"):
        return "run_exec_root"
    if p.endswith("/run_config.rs"):
        return "run_config_root"
    if "/run_config/execution/" in p:
        name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name in {"artifacts", "coverage", "evaluation", "feature_runtime", "field", "h07", "quality", "search"}:
            return "run_exec_" + name
    if "/run_config/manifest/" in p:
        name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name in {"mod", "parser", "validation"}:
            return "run_manifest_" + name
    if p.endswith("/nn_training.rs"):
        return "nn_training_root"
    if p.endswith("/nn_training/trainer.rs"):
        return "nn_trainer_root"
    if "/nn_training/trainer/" in p:
        name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name in {"device", "step", "checkpoint", "data_parallel", "microbatch"}:
            return "nn_trainer_" + name
    if p.endswith("/training_session.rs"):
        return "training_session_root"
    if "/training_session/" in p:
        name = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name in {"device", "lifecycle", "run", "validation"}:
            return "training_session_" + name
    return "other"


def diagnostic_summary(source: pathlib.Path, workspace: pathlib.Path) -> str:
    pairs: set[str] = set()
    saw_error = False
    try:
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict) or item.get("reason") != "compiler-message":
                continue
            message = item.get("message")
            if not isinstance(message, dict) or message.get("level") != "error":
                continue
            saw_error = True
            code = message.get("code")
            code = code.get("code") if isinstance(code, dict) else None
            if not isinstance(code, str) or not re.fullmatch(r"E[0-9]{4}", code):
                code = "EOTHER"
            spans = message.get("spans")
            primary = next((span for span in spans if isinstance(span, dict) and span.get("is_primary")), None) if isinstance(spans, list) else None
            filename = primary.get("file_name", "") if primary else ""
            line_number = primary.get("line_start", 0) if primary else 0
            line_number = line_number if isinstance(line_number, int) and line_number > 0 else 0
            area = diagnostic_area(filename)
            function = "module"
            relative = AREA_FILES.get(area)
            if relative:
                try:
                    lines = (workspace / relative).read_text(encoding="utf-8").splitlines()
                    pattern = re.compile(r"\s*(?:(?:pub(?:\([^)]*\))?)\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)")
                    for source_line in lines[:max(0, min(line_number, len(lines)))]:
                        found = pattern.match(source_line)
                        if found:
                            function = found.group(1)
                except (OSError, ValueError):
                    function = "source_unavailable"
            if function not in SAFE_FUNCTIONS and function not in {"module", "source_unavailable"}:
                function = "function_other"
            pairs.add(code + "_" + area + "_" + function)
    except OSError:
        return "unavailable"
    if not saw_error:
        return "no_compiler_diagnostics"
    if not pairs:
        return "unavailable"
    result = "+".join(sorted(pairs)[:6])
    return result if re.fullmatch(r"E[0-9]{4}_[a-z0-9_]+(?:\+E[0-9]{4}_[a-z0-9_]+){0,5}", result) else "unavailable"


def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result = "not_stored"
    cleanup = "not_needed"
    diagnostic = "unavailable"
    temp_dir: pathlib.Path | None = None
    attach_attempted = False
    public_key = ""
    instance_id: int | None = None
    api_key = os.environ.get("VAST_API_KEY", "")

    try:
        source = pathlib.Path(os.environ.get("PRIVATE_DIAGNOSTIC_PATH", ""))
        if source.is_file() and 0 < source.stat().st_size <= MAX_BYTES:
            diagnostic = diagnostic_summary(source, pathlib.Path(os.environ.get("GITHUB_WORKSPACE", ".")))
        if not api_key or not source.is_file() or source.stat().st_size == 0 or source.stat().st_size > MAX_BYTES:
            result = "input_unavailable"
            raise RuntimeError()
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
        if not run_id.isdigit() or not attempt.isdigit():
            result = "run_identity_invalid"
            raise RuntimeError()

        rows = target_rows(api_json(api_key, "/instances/"))
        running = [row for row in rows if state(row) == "running"]
        if len(running) != 1:
            result = "target_ambiguous"
            raise RuntimeError()
        if safe_budget_projection(rows) != "within_48h_cap":
            result = "budget_gate_closed"
            raise RuntimeError()

        target = running[0]
        candidate = target.get("id")
        host = target.get("ssh_host")
        port = target.get("ssh_port")
        if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate <= 0:
            result = "target_unavailable"
            raise RuntimeError()
        if not isinstance(host, str) or not re.fullmatch(r"ssh[0-9]+\.vast\.ai", host.casefold()):
            result = "endpoint_unavailable"
            raise RuntimeError()
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            result = "endpoint_unavailable"
            raise RuntimeError()
        instance_id = candidate
        if not ensure_ssh_tools():
            result = "ssh_tools_unavailable"
            raise RuntimeError()

        remote_path = f"{REMOTE_DIR}/run-{run_id}-attempt-{attempt}.jsonl"
        temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="vast-private-log-"))
        key_path = temp_dir / "temporary"
        known_hosts = temp_dir / "known_hosts"
        generated = subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"gha-private-log-{run_id}-{attempt}", "-f", str(key_path)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if generated.returncode != 0:
            result = "key_generation_failed"
            raise RuntimeError()
        public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()
        if find_attached_key_id(api_key, instance_id, public_key) is not None:
            result = "key_collision"
            raise RuntimeError()

        attach_attempted = True
        attached = api_json(api_key, f"/instances/{instance_id}/ssh", "POST", {"ssh_key": public_key})
        if not isinstance(attached, dict) or attached.get("success") is not True:
            result = "key_attach_failed"
            raise RuntimeError()
        if find_attached_key_id(api_key, instance_id, public_key) is None:
            result = "key_attach_unconfirmed"
            raise RuntimeError()

        prepare = ssh(host, port, key_path, known_hosts, f"umask 077 && mkdir -p {REMOTE_DIR} && test ! -e {remote_path}")
        if prepare is None or prepare.returncode == 255:
            result = "connection_failed"
            raise RuntimeError()
        if prepare.returncode != 0:
            result = "remote_path_unavailable"
            raise RuntimeError()

        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        copied = subprocess.run(
            [
                "scp", "-q", "-i", str(key_path), "-P", str(port),
                "-o", "BatchMode=yes",
                "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={known_hosts}",
                "-o", "GlobalKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=8",
                str(source), f"root@{host}:{remote_path}",
            ],
            capture_output=True, text=True, timeout=90, check=False,
        )
        if copied.returncode != 0:
            result = "upload_failed"
            raise RuntimeError()

        verified = ssh(host, port, key_path, known_hosts, f"sha256sum {remote_path}")
        if verified is None or verified.returncode != 0:
            result = "verification_failed"
            raise RuntimeError()
        match = re.match(r"^([0-9a-f]{64})\s", verified.stdout)
        if not match or match.group(1) != digest:
            result = "verification_failed"
            raise RuntimeError()
        result = "stored"

    except ApiFailure as error:
        result = "api_auth_failed" if error.status in {401, 403} else "api_failed"
    except (OSError, subprocess.TimeoutExpired):
        result = "transfer_failed"
    except RuntimeError:
        pass
    except Exception:
        result = "transfer_failed"
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

    print(f"checked_at={checked}")
    if diagnostic not in {"no_compiler_diagnostics", "unavailable"} and not re.fullmatch(r"E[0-9]{4}_[a-z0-9_]+(?:\+E[0-9]{4}_[a-z0-9_]+){0,5}", diagnostic):
        diagnostic = "unavailable"
    print(f"private_build_diagnostic={result}")
    print(f"private_build_diagnostic_summary={diagnostic}")
    print(f"temporary_ssh_key={cleanup}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as summary:
                summary.write(
                    "## Private build diagnostic transfer\n"
                    f"- checked_at: {checked}\n"
                    f"- private_build_diagnostic: {result}\n"
                    f"- temporary_ssh_key: {cleanup}\n"
                )
        except OSError:
            pass
    return 0 if result == "stored" and cleanup in {"detached", "not_needed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
