#!/usr/bin/env python3
"""Rebuild and validate the frozen 10k dataset on the existing Vast disk."""

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

MAX_REMOTE_SECONDS = 7_800
REMOTE_COMMAND_SECONDS = 7_500


def remote_bash(host: str, port: int, key_path: pathlib.Path, known_hosts: pathlib.Path, script: str) -> subprocess.CompletedProcess[str] | None:
    argv = [
        "ssh", "-i", str(key_path), "-p", str(port),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=8",
        "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        f"root@{host}", f"timeout --signal=TERM --kill-after=30 {REMOTE_COMMAND_SECONDS}s bash -s",
    ]
    try:
        return subprocess.run(
            argv, input=script, capture_output=True, text=True,
            timeout=MAX_REMOTE_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result = "not_started"
    cleanup = "not_needed"
    temp_dir: pathlib.Path | None = None
    attach_attempted = False
    public_key = ""
    instance_id: int | None = None
    api_key = os.environ.get("VAST_API_KEY", "")

    try:
        source_revision = os.environ.get("GITHUB_SHA", "")
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
        if not api_key:
            result = "configuration_missing"
            raise RuntimeError()
        if not re.fullmatch(r"[0-9a-f]{40}", source_revision) or not run_id.isdigit() or not attempt.isdigit():
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

        temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="vast-dataset-"))
        key_path = temp_dir / "temporary"
        known_hosts = temp_dir / "known_hosts"
        generated = subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"gha-dataset-{run_id}-{attempt}", "-f", str(key_path)],
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

        remote = f"""set -euo pipefail
if [ -x "$HOME/.cargo/bin/cargo" ]; then export PATH="$HOME/.cargo/bin:$PATH"; fi
ROOT='/workspace/tqp_remote_dataset_{run_id}_{source_revision[:8]}_a{attempt}'
if [ -e "$ROOT" ] || [ -L "$ROOT" ]; then
  printf 'dataset_build=workspace_exists\\n'
  exit 10
fi
if ! mkdir -m 700 "$ROOT"; then
  printf 'dataset_build=workspace_creation_failed\\n'
  exit 11
fi
exec 3>&1
exec >>"$ROOT/remote-dataset.private.log" 2>&1
if ! command -v cargo >/dev/null 2>&1; then
  printf 'dataset_build=toolchain_unavailable\n'
  exit 12
fi
finish() {{
  code=$?
  trap - EXIT
  if [ "$code" -eq 0 ]; then
    status=accepted
  else
    status=failed
  fi
  printf 'dataset_build=%s\\n' "$status" >"$ROOT/dataset-status.private" || true
  printf 'dataset_build=%s\\n' "$status" >&3
}}
trap finish EXIT
available_kb="$(df -Pk /workspace | awk 'NR==2 {print $4}')"
case "$available_kb" in
  *[!0-9]*|'') exit 22 ;;
esac
if [ "$available_kb" -lt 41943040 ]; then exit 23; fi
SOURCE="$ROOT/source"
DATA="$ROOT/data/frozen_train_10k.json"
mkdir -p "$ROOT/data"
git clone --quiet --filter=blob:none --no-checkout https://github.com/TryDotAtwo/cnot-neural-synth-frontier6.git "$SOURCE"
git -C "$SOURCE" fetch --quiet --depth=1 origin '{source_revision}'
git -C "$SOURCE" checkout --quiet --detach '{source_revision}'
python3 "$SOURCE/.github/scripts/fetch_public_holdout.py" --experiment-root "$ROOT"
python3 "$SOURCE/.github/scripts/rebuild_frozen_dataset.py" \\
  --experiment-root "$ROOT" \\
  --holdout-sqlite "$ROOT/input/full_shared_native_holdout_20261001.sqlite" \\
  --output "$DATA" \\
  --receipt "$ROOT/data/build.receipt.json" \\
  --source-revision '{source_revision}' \\
  --seed 202611190001
DATA_SHA="$(tr -d '\\r\\n' <"$ROOT/data/frozen_train_10k.sha256")"
case "$DATA_SHA" in
  *[!0-9a-f]*|'') exit 20 ;;
esac
if [ "$(expr length "$DATA_SHA")" -ne 64 ]; then exit 21; fi
CARGO_TARGET_DIR="$ROOT/cargo-target" cargo run --quiet \\
  --manifest-path "$SOURCE/native/Cargo.toml" \\
  --no-default-features --bin cnot103-validate-data -- "$DATA" "$DATA_SHA"
"""
        proc = remote_bash(host, port, key_path, known_hosts, remote)
        if proc is None:
            result = "remote_timeout_or_connection_failure"
        else:
            allowed = {
                "accepted", "failed", "workspace_exists",
                "workspace_creation_failed", "toolchain_unavailable",
            }
            status = next(
                (
                    line.partition("=")[2]
                    for line in proc.stdout.splitlines()
                    if line.startswith("dataset_build=")
                    and line.partition("=")[2] in allowed
                ),
                None,
            )
            if proc.returncode == 0 and status == "accepted":
                result = "accepted"
            elif status == "workspace_exists":
                result = "workspace_exists"
            elif status == "workspace_creation_failed":
                result = "workspace_creation_failed"
            elif status == "toolchain_unavailable":
                result = "toolchain_unavailable"
            else:
                result = "remote_failed"

    except ApiFailure as error:
        result = "api_auth_failed" if error.status in {401, 403} else "api_failed"
    except (OSError, subprocess.TimeoutExpired):
        result = "controller_failed"
    except RuntimeError:
        pass
    except Exception:
        result = "controller_failed"
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
    print(f"dataset_build={result}")
    print(f"temporary_ssh_key={cleanup}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as summary:
                summary.write(
                    "## Remote dataset rebuild\n"
                    f"- checked_at: {checked}\n"
                    f"- dataset_build: {result}\n"
                    f"- temporary_ssh_key: {cleanup}\n"
                )
        except OSError:
            pass
    return 0 if result == "accepted" and cleanup in {"detached", "not_needed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
