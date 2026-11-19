#!/usr/bin/env python3
"""Classify one private dataset failure on the existing Vast disk."""

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
    classify_ssh_failure,
    find_attached_key_id,
    safe_budget_projection,
    state,
    target_rows,
)
import vast_rebuild_remote_dataset as remote_dataset

# Bound this read-only classifier even if the remote shell stalls.
remote_dataset.MAX_REMOTE_SECONDS = 90
remote_dataset.REMOTE_COMMAND_SECONDS = 70


RUN_ID = "44605523186"
SOURCE_SHA8 = "c99333b8"
ATTEMPT = "1"
DETAIL_CLASSES = {
    "private_log_missing",
    "disk_full",
    "toolchain_unavailable",
    "network_failure",
    "holdout_panels_incomplete",
    "holdout_database_invalid",
    "holdout_database_missing",
    "holdout_database_empty",
    "holdout_database_no_roots_table",
    "holdout_roots_schema_unsupported",
    "holdout_database_read_error",
    "holdout_database_readable",
    "holdout_sqlite_corruption",
    "holdout_sqlite_io_error",
    "holdout_sqlite_busy",
    "holdout_sqlite_query_error",
    "holdout_record_invalid",
    "holdout_record_metadata_invalid",
    "holdout_split_invalid",
    "holdout_matrix_invalid",
    "generated_record_generation_failed",
    "builder_arguments_invalid",
    "holdout_certificate_invalid",
    "holdout_identity_or_symmetry_overlap",
    "generated_certificate_replay_failed",
    "train_root_generation_exhausted",
    "frozen_data_validation_failed",
    "validator_build_failed",
    "validator_compile_error",
    "validator_build_network_error",
    "validator_build_failed_unclassified",
    "artifact_conflict",
    "other_generation_validation_failure",
    "builder_domain_error_unclassified",
    "builder_io_error",
    "builder_exception_other",
    "dataset_artifacts_incomplete",
    "dataset_outputs_missing",
    "validator_failure_unclassified",
    "validator_recheck_passed",
    "validator_recheck_failed",
    "validator_recheck_missing_marker",
    "validator_runtime_error",
    "validator_binary_missing",
    "dataset_checksum_sidecar_invalid",
    "target_not_running",
    "target_ambiguous",
    "budget_gate_closed",
    "ssh_auth_failed",
    "ssh_timeout",
    "ssh_connection_refused",
    "ssh_network_unreachable",
    "ssh_dns_failure",
    "ssh_host_key_failure",
    "ssh_handshake_failed",
    "ssh_rejected",
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
DB="$ROOT/input/full_shared_native_holdout_20261001.sqlite"
DATA="$ROOT/data/frozen_train_10k.json"
SHA="$ROOT/data/frozen_train_10k.sha256"
RECEIPT="$ROOT/data/build.receipt.json"
VALIDATOR_BIN="$ROOT/cargo-target/debug/cnot103-validate-data"
if [ ! -r "$LOG" ]; then
  detail=private_log_missing
elif [ ! -e "$DB" ]; then
  detail=holdout_database_missing
elif [ ! -s "$DB" ]; then
  detail=holdout_database_empty
elif grep -Eiq 'no space left|disk quota exceeded' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=disk_full
elif grep -Eiq 'command not found|cargo: not found' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=toolchain_unavailable
elif grep -Eiq 'HTTPError|urlopen error|connection reset|could not resolve|timed out' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=network_failure
elif grep -Fq 'holdout contains a singular matrix' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_matrix_invalid
elif grep -Fq 'holdout does not contain the pinned 1792+1792 panels' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_panels_incomplete
elif grep -Fq 'holdout database has no roots table' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_database_no_roots_table
elif grep -Fq 'holdout roots table has an unsupported schema' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_roots_schema_unsupported
elif grep -Fq 'holdout database could not be read' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail="$(python3 - "$DB" <<'PY'
import pathlib
import sqlite3
import sys

path = pathlib.Path(sys.argv[1])
connection = None
if not path.is_file():
    detail = "holdout_database_missing"
elif path.stat().st_size == 0:
    detail = "holdout_database_empty"
else:
    try:
        connection = sqlite3.connect("file:" + path.as_posix() + "?mode=ro", uri=True)
        tables = tuple(row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        if "roots" not in tables:
            detail = "holdout_database_no_roots_table"
        else:
            columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(roots)"))
            required = ("split", "epoch", "position", "record_json")
            if any(name not in columns for name in required):
                detail = "holdout_roots_schema_unsupported"
            else:
                connection.execute("SELECT split, record_json FROM roots WHERE split IN ('dev','test') ORDER BY split, epoch, position").fetchall()
                detail = "holdout_database_readable"
    except sqlite3.Error as error:
        code = getattr(error, "sqlite_errorname", "")
        if code in ("SQLITE_CORRUPT", "SQLITE_NOTADB"):
            detail = "holdout_sqlite_corruption"
        elif code == "SQLITE_IOERR":
            detail = "holdout_sqlite_io_error"
        elif code in ("SQLITE_BUSY", "SQLITE_LOCKED"):
            detail = "holdout_sqlite_busy"
        else:
            detail = "holdout_sqlite_query_error"
    finally:
        if connection is not None:
            connection.close()
print(detail)
PY
)"
elif grep -Eiq 'holdout reduction word does not replay|holdout certificate|holdout word certificate|holdout certificate endpoints' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_certificate_invalid
elif grep -Eiq 'holdout record|holdout state|holdout gate' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_record_invalid
elif grep -Eiq 'holdout exact-distance field|holdout exact-evidence field' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_record_metadata_invalid
elif grep -Fq 'holdout contains an unsupported split' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=holdout_split_invalid
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
elif grep -Eiq 'invalid random range|generated gate is outside the CNOT alphabet' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=generated_record_generation_failed
elif grep -Eiq 'run directory does not exist|output file names are invalid|remote rebuild arguments are incomplete|seed must fit|source revision must be a lowercase commit SHA' "$LOG" "$BUILDER_LOG" 2>/dev/null; then
  detail=builder_arguments_invalid
elif [ -s "$BUILDER_LOG" ] && grep -Eq '^(BuildError):' "$BUILDER_LOG"; then
  detail=builder_domain_error_unclassified
elif [ -s "$BUILDER_LOG" ] && grep -Eq '^(OSError|PermissionError|FileNotFoundError):' "$BUILDER_LOG"; then
  detail=builder_io_error
elif [ -s "$BUILDER_LOG" ]; then
  detail=builder_exception_other
elif [ -s "$DATA" ] && [ -s "$SHA" ] && [ -s "$RECEIPT" ]; then
  DATA_SHA="$(tr -d '\\r\\n' < "$SHA")"
  if ! printf '%s' "$DATA_SHA" | grep -Eq '^[0-9a-f]{64}$'; then
    detail=dataset_checksum_sidecar_invalid
  elif [ ! -x "$VALIDATOR_BIN" ]; then
    CARGO_LOG="$ROOT/classifier-cargo.private.log"
    if ! command -v cargo >/dev/null 2>&1; then
      detail=toolchain_unavailable
    elif CARGO_TARGET_DIR="$ROOT/cargo-target" cargo build --quiet --manifest-path "$ROOT/source/native/Cargo.toml" --no-default-features --bin cnot103-validate-data >"$CARGO_LOG" 2>&1; then
      if [ ! -x "$VALIDATOR_BIN" ]; then
        detail=validator_binary_missing
      elif "$VALIDATOR_BIN" "$DATA" "$DATA_SHA" >"$ROOT/classifier-validation.private.log" 2>&1; then
        if grep -Fq 'frozen_data_validation=accepted' "$ROOT/classifier-validation.private.log"; then
          detail=validator_recheck_passed
        else
          detail=validator_recheck_missing_marker
        fi
      elif grep -Fq 'frozen_data_validation=failed' "$ROOT/classifier-validation.private.log"; then
        detail=validator_recheck_failed
      else
        detail=validator_runtime_error
      fi
    elif grep -Eiq 'failed to download|failed to get|could not connect|connection reset|could not resolve|timed out|HTTP.*(502|503|504)' "$CARGO_LOG"; then
      detail=validator_build_network_error
    elif grep -Eiq 'error\[E[0-9]{4}\]|could not compile|failed to compile' "$CARGO_LOG"; then
      detail=validator_compile_error
    elif grep -Eiq 'no space left|disk quota exceeded' "$CARGO_LOG"; then
      detail=disk_full
    else
      detail=validator_build_failed_unclassified
    fi
  elif "$VALIDATOR_BIN" "$DATA" "$DATA_SHA" >"$ROOT/classifier-validation.private.log" 2>&1; then
    if grep -Fq 'frozen_data_validation=accepted' "$ROOT/classifier-validation.private.log"; then
      detail=validator_recheck_passed
    else
      detail=validator_recheck_missing_marker
    fi
  elif grep -Fq 'frozen_data_validation=failed' "$ROOT/classifier-validation.private.log"; then
    detail=validator_recheck_failed
  elif grep -Eiq 'error while loading shared libraries|command not found|permission denied' "$ROOT/classifier-validation.private.log"; then
    detail=validator_runtime_error
  else
    detail=validator_recheck_failed
  fi
elif [ -e "$DATA" ] || [ -e "$SHA" ] || [ -e "$RECEIPT" ]; then
  if [ ! -x "$VALIDATOR_BIN" ]; then
    detail=validator_binary_missing
  else
    detail=dataset_artifacts_incomplete
  fi
else
  detail=dataset_outputs_missing
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
        time.sleep(12)

        proc = remote_dataset.remote_bash(host, port, key_path, known_hosts, REMOTE_CLASSIFIER)
        if proc is None:
            detail = "ssh_timeout_or_connection_failure"
        elif proc.returncode == 255:
            detail = classify_ssh_failure(proc.stderr or "")
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
