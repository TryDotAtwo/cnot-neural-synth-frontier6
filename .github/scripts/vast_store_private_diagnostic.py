#!/usr/bin/env python3
"""Store private Rust diagnostics on the existing Vast disk, never in Actions logs."""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import subprocess
import tempfile
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
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result = "not_stored"
    cleanup = "not_needed"
    temp_dir: pathlib.Path | None = None
    attach_attempted = False
    public_key = ""
    instance_id: int | None = None
    api_key = os.environ.get("VAST_API_KEY", "")

    try:
        source = pathlib.Path(os.environ.get("PRIVATE_DIAGNOSTIC_PATH", ""))
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
        if prepare is None or prepare.returncode != 0:
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
    print(f"private_build_diagnostic={result}")
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
