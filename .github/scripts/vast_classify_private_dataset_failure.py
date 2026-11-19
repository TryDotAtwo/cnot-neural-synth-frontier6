#!/usr/bin/env python3
"""Classify one private dataset failure on the existing Vast disk."""

from __future__ import annotations

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
from vast_rebuild_remote_dataset import remote_bash


RUN_ID = "44569035847"
SOURCE_SHA8 = "01008f40"
ATTEMPT = "2"
DETAIL_CLASSES = {
    "private_log_missing",
    "disk_full",
    "toolchain_unavailable",
    "network_failure",
    "holdout_panels_incomplete",
    "holdout_database_invalid",
    "holdout_record_invalid",
    "holdout_certificate_invalid",
    "holdout_identity_or_symmetry_overlap",
    "generated_certificate_replay_failed",
    "train_root_generation_exhausted",
    "frozen_data_validation_failed",
    "validator_build_failed",
    "artifact_conflict",
    "other_generation_validation_failure",
    "target_not_running",
    "target_ambiguous",
    "budget_gate_closed",
    "endpoint_unavailable",
    "key_generation_failed",
    "key_attach_failed",
    "key_attach_unconfirmed",
    "connection_failed",
    "ssh_timeout_or_connection_failure",
    "remote_read_failed",
    "api_auth_failed",
    "api_failed",
    "controller_failed",
    "unavailable",
}
REMOTE_CLASSIFIER = f"""set -euo pipefail
ROOT='/workspace/tqp_remote_dataset_{RUN_ID}_{SOURCE_SHA8}_a{ATTEMPT}'
LOG="$ROOT/remote-dataset.private.log"
BUILDER_LOG="$ROOT/dataset-builder.private.log"
if [ ! -r "$LOG" ]; then
  detail=private_log_missing
elif grep -Eiq 'no space left|disk quota exceeded' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=disk_full
elif grep -Eiq 'command not found|cargo: not found' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=toolchain_unavailable
elif grep -Eiq 'HTTPError|urlopen error|connection reset|could not resolve|timed out' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=network_failure
elif grep -Fq 'holdout does not contain the pinned 1792+1792 panels' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_panels_incomplete
elif grep -Eiq 'holdout database has no roots table|holdout roots table has an unsupported schema|holdout database could not be read' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_database_invalid
elif grep -Eiq 'holdout reduction word does not replay|holdout certificate|holdout word certificate|holdout certificate endpoints' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_certificate_invalid
elif grep -Eiq 'holdout record|holdout state|holdout gate' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_record_invalid
elif grep -Eiq 'dev/test .*repeats|finite-symmetry views overlap|finite-symmetry view set is incomplete' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_identity_or_symmetry_overlap
elif grep -Fq 'generated reduction certificate failed replay' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=generated_certificate_replay_failed
elif grep -Fq 'could not generate a unique train root within the retry bound' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=train_root_generation_exhausted
elif grep -Fq 'frozen_data_validation=failed' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=frozen_data_validation_failed
elif grep -Eiq 'could not compile|failed to compile' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=validator_build_failed
elif grep -Eiq 'output path is outside|refusing to overwrite|destination appeared during the write|preserving an existing' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=artifact_conflict
else
  detail=other_generation_validation_failure
fi
printf 'dataset_failure_detail=%s\\n' "$detail"
"""


def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    detail = "unavailable"
    cleanup = "not_needed"
    temp_dir: pathlib.Path | None = None
    instance_id: int | None = None
    public_key = ""
    attach_attempted = False
    api_key = os.environ.get("VAST_API_KEY", "")

    try:
        if not api_key:
            detail = "api_auth_failed"
            raise RuntimeError()
        rows = target_rows(api_json(api_key, "/instances/"))
        running = [row for row in rows if state(row) == "running"]
        if not running:
            detail = "target_not_running"
            raise RuntimeError()
        if len(running) != 1:
            detail = "target_ambiguous"
            raise RuntimeError()
        if safe_budget_projection(rows) != "within_48h_cap":
            detail = "budget_gate_closed"
            raise RuntimeError()

        target = running[0]
        candidate = target.get("id")
        host = target.get("ssh_host")
        port = target.get("ssh_port")
        if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate <= 0:
            detail = "endpoint_unavailable"
            raise RuntimeError()
        if not isinstance(host, str) or not re.fullmatch(r"ssh[0-9]+\.vast\.ai", host.casefold()):
            detail = "endpoint_unavailable"
            raise RuntimeError()
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            detail = "endpoint_unavailable"
            raise RuntimeError()
        instance_id = candidate

        temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="vast-private-dataset-classifier-"))
        key_path = temp_dir / "temporary"
        known_hosts = temp_dir / "known_hosts"
        generated = subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"gha-dataset-classifier-{RUN_ID}-{ATTEMPT}", "-f", str(key_path)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if generated.returncode != 0:
            detail = "key_generation_failed"
            raise RuntimeError()
        public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()

        attach_attempted = True
        attached = api_json(api_key, f"/instances/{instance_id}/ssh", "POST", {"ssh_key": public_key})
        if not isinstance(attached, dict) or attached.get("success") is not True:
            detail = "key_attach_failed"
            raise RuntimeError()
        if find_attached_key_id(api_key, instance_id, public_key) is None:
            detail = "key_attach_unconfirmed"
            raise RuntimeError()

        proc = remote_bash(host, port, key_path, known_hosts, REMOTE_CLASSIFIER)
        if proc is None:
            detail = "ssh_timeout_or_connection_failure"
        elif proc.returncode == 255:
            detail = "connection_failed"
        elif proc.returncode == 0:
            found = next(
                (
                    line.partition("=")[2]
                    for line in proc.stdout.splitlines()
                    if line.startswith("dataset_failure_detail=")
                    and line.partition("=")[2] in DETAIL_CLASSES
                ),
                None,
            )
            detail = found or "remote_read_failed"
        else:
            detail = "remote_read_failed"
    except ApiFailure as error:
        detail = "api_auth_failed" if error.status in {401, 403} else "api_failed"
    except (OSError, subprocess.TimeoutExpired):
        detail = "controller_failed"
    except RuntimeError:
        pass
    except Exception:
        detail = "controller_failed"
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

    if detail not in DETAIL_CLASSES:
        detail = "unavailable"
    if cleanup not in {"detached", "not_needed", "not_found", "failed"}:
        cleanup = "failed"
    print(f"checked_at_utc={checked}")
    print(f"dataset_failure_detail={detail}")
    print(f"temporary_ssh_key={cleanup}")
    return 0 if detail in DETAIL_CLASSES and cleanup in {"detached", "not_needed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
