#!/usr/bin/env python3
"""Remote full CNOT training supervisor. All detailed artifacts stay on Vast."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

TARGET_ATTEMPTS = 250_000
PILOT_ATTEMPTS = 500
BATCH_SIZE = 4
MODEL_PARAMETERS = 136_421_807
MODEL_INPUTS = 102
STRUCTURAL_BLOCKS = 8
CHECKPOINT_INTERVAL = 25_000
EXPECTED_CHECKPOINT_FILES = 12
CHECKPOINT_FREE_GIB = 24
INITIAL_FREE_GIB = 40
MAX_RUNTIME_SECONDS = 47 * 3600 + 45 * 60
FORECAST_SAFETY_SECONDS = 15 * 60
CHECKPOINT_WRITE_RESERVE_SECONDS = 30 * 60
PROFILE_RELATIVE = ".github/training-profiles/n10-test-support-v1.json"
DATA_RELATIVE = "data/frozen_train_10k.json"
DATA_SHA_RELATIVE = "data/frozen_train_10k.sha256"
DATA_RECEIPT_RELATIVE = "data/build.receipt.json"
HOLDOUT_RELATIVE = "input/full_shared_native_holdout_20261001.sqlite"
BEAM_CONFIG = {
    "beam_width": 32,
    "shortlist": 32,
    "max_depth": 64,
    "max_expanded": 16,
    "inference_batch": 32,
}


class RunError(RuntimeError):
    pass


class TimeLimit(RunError):
    pass


def select_dev_indices(records: object, per_width: int = 8) -> list[int]:
    if not isinstance(records, list) or not records:
        raise RunError("dev panel is empty or malformed")
    if type(per_width) is not int or per_width < 1:
        raise RunError("per-width holdout sample must be positive")
    groups: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RunError("dev record is malformed")
        state = record.get("state")
        width = state.get("n") if isinstance(state, dict) else None
        if type(width) is not int or width < 1:
            raise RunError("dev width is malformed")
        groups.setdefault(width, []).append(index)
    selected: list[int] = []
    for width in sorted(groups):
        candidates = groups[width]
        count = min(per_width, len(candidates))
        if count == 1:
            positions = [(len(candidates) - 1) // 2]
        else:
            positions = [
                slot * (len(candidates) - 1) // (count - 1)
                for slot in range(count)
            ]
        selected.extend(candidates[position] for position in positions)
    if not selected or len(selected) != len(set(selected)):
        raise RunError("holdout index selection is empty or duplicated")
    return sorted(selected)


def quality_metrics(report: object) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema") != "cnot103_dev_search_result_v1":
        raise RunError("beam-search report schema is invalid")
    if report.get("feature_model_inputs") != MODEL_INPUTS:
        raise RunError("beam search did not use all 102 model inputs")
    audit = report.get("training_exposure_audit")
    if not isinstance(audit, dict) or audit.get("strict_unseen") is not True:
        raise RunError("beam search exposure audit failed")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RunError("beam-search holdout result is empty")
    costs: list[int] = []
    solved = 0
    statuses = {"solved", "depth_limit", "resource_limit", "exhausted"}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in statuses:
            raise RunError("beam-search row is malformed")
        word = row.get("word")
        if row["status"] == "solved":
            if not isinstance(word, dict) or word.get("replayed") is not True:
                raise RunError("a solved word lacks successful replay")
            cost = word.get("physical_cost")
            if type(cost) is not int or cost < 0:
                raise RunError("a solved word has an invalid physical cost")
            solved += 1
            costs.append(cost)
        elif word is not None:
            raise RunError("an unsolved row unexpectedly contains a word")
    return {
        "case_count": len(rows),
        "solved_count": solved,
        "solved_rate": solved / len(rows),
        "physical_costs": costs,
        "median_physical_cost": statistics.median(costs) if costs else None,
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted(statuses)
        },
    }


def validate_training_result(
    report: object, expected_start: int, expected_end: int
) -> None:
    if not isinstance(report, dict) or report.get("schema") != "cnot103_training_result_v1":
        raise RunError("training report schema is invalid")
    if (
        report.get("model_parameters") != MODEL_PARAMETERS
        or report.get("feature_model_inputs") != MODEL_INPUTS
        or report.get("cuda_devices") != [0, 1]
    ):
        raise RunError("training model, input, or GPU contract differs")
    start = report.get("start_progress")
    end = report.get("end_progress")
    audit = report.get("exposure_audit")
    if (
        not isinstance(start, dict) or start.get("attempted") != expected_start
        or not isinstance(end, dict) or end.get("attempted") != expected_end
        or end.get("accepted") != expected_end
        or end.get("rejected") != 0
        or not isinstance(audit, dict) or audit.get("strict_unseen") is not True
    ):
        raise RunError("training progress or exposure audit differs")
    last_step = report.get("last_step")
    if not isinstance(last_step, dict) or last_step.get("accepted") is not True:
        raise RunError("last optimizer update was rejected")


def validate_benchmark(report: object, attempts: int = PILOT_ATTEMPTS) -> tuple[float, float]:
    if not isinstance(report, dict) or report.get("schema") != "cnot103_discard_only_benchmark_v2":
        raise RunError("benchmark report schema is invalid")
    if (
        report.get("model_parameters") != MODEL_PARAMETERS
        or report.get("feature_model_inputs") != MODEL_INPUTS
        or report.get("cuda_devices") != [0, 1]
        or report.get("attempts") != attempts
        or report.get("batch_size") != BATCH_SIZE
    ):
        raise RunError("benchmark did not run the requested full model workload")
    audit = report.get("exposure_audit")
    last_step = report.get("last_step")
    if not isinstance(audit, dict) or audit.get("strict_unseen") is not True:
        raise RunError("benchmark exposure audit failed")
    if not isinstance(last_step, dict) or last_step.get("accepted") is not True:
        raise RunError("benchmark optimizer update was rejected")
    update_seconds = report.get("update_seconds")
    setup_seconds = report.get("setup_seconds")
    if (
        isinstance(update_seconds, bool) or not isinstance(update_seconds, (int, float))
        or not math.isfinite(update_seconds) or update_seconds <= 0
        or isinstance(setup_seconds, bool) or not isinstance(setup_seconds, (int, float))
        or not math.isfinite(setup_seconds) or setup_seconds < 0
    ):
        raise RunError("benchmark timing is invalid")
    return float(update_seconds), float(setup_seconds)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_private_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise RunError("a private artifact partial already exists")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_new_private_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RunError("refusing to overwrite a private result")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise RunError("preserving an existing private partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            raise RunError("private result destination appeared during write")
        os.rename(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class Supervisor:
    def __init__(
        self,
        root: pathlib.Path,
        source_revision: str,
        holdout_revision: str,
        holdout_sha256: str,
        started_at_epoch: float,
        budget_baseline: float,
    ):
        self.root = root
        self.source = root / "source"
        self.source_revision = source_revision
        self.holdout_revision = holdout_revision
        self.holdout_sha256 = holdout_sha256
        self.started_at_epoch = started_at_epoch
        self.budget_baseline = budget_baseline
        self.status_path = root / "status.private.json"
        self.events_path = root / "events.private.jsonl"
        self.error_path = root / "error.private.json"
        self.private_report_path = root / "final-result.private.json"
        self.current_state = "starting"
        self.quality_status = "not_ready"
        self.deadline = started_at_epoch + MAX_RUNTIME_SECONDS
        self.environment = self._environment()
        self._prepare_directories()

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        cargo_home = pathlib.Path.home() / ".cargo"
        env["CARGO_HOME"] = str(cargo_home)
        env["RUSTUP_HOME"] = str(pathlib.Path.home() / ".rustup")
        env["PATH"] = str(cargo_home / "bin") + os.pathsep + env.get("PATH", "")
        env["CNOT103_CUDA_ARCHITECTURES"] = "86"
        env["CUDA_HOME"] = "/usr/local/cuda"
        env["LIBTORCH_USE_PYTORCH"] = "1"
        torch_lib = "/opt/conda/lib/python3.11/site-packages/torch/lib"
        env["LD_LIBRARY_PATH"] = torch_lib + ":" + "/usr/local/cuda/lib64"
        env["CARGO_TARGET_DIR"] = str(self.root / "cargo-target")
        return env

    def _prepare_directories(self) -> None:
        if not self.root.is_dir() or not self.source.is_dir():
            raise RunError("remote workspace or pinned source is missing")
        resolved_root = self.root.resolve()
        try:
            resolved_root.relative_to(pathlib.Path("/workspace").resolve())
        except ValueError:
            raise RunError("remote workspace escaped /workspace") from None
        for name in ("data", "input", "logs", "manifests", "reports", "checkpoints"):
            path = self.root / name
            if path.exists():
                raise RunError("preserving an existing run artifact directory")
            path.mkdir(mode=0o700, parents=False)
        if self.status_path.exists() or self.events_path.exists():
            raise RunError("preserving existing supervisor status")

    def status(self, state: str, quality: str | None = None) -> None:
        allowed_states = {
            "starting", "preparing", "benchmarking", "resuming", "baseline",
            "training", "final_quality", "completed", "failed", "budget_gate", "time_limit",
        }
        allowed_quality = {"not_ready", "measured", "passed", "failed"}
        if state not in allowed_states:
            state = "failed"
        if quality is not None:
            if quality not in allowed_quality:
                quality = "failed"
            self.quality_status = quality
        self.current_state = state
        value = {
            "schema": "cnot103_full_training_status_v1",
            "state": state,
            "quality_status": self.quality_status,
            "updated_at_epoch": time.time(),
            "started_at_epoch": self.started_at_epoch,
            "budget_baseline": self.budget_baseline,
        }
        _atomic_private_json(self.status_path, value)
        event = {"at_utc": _now(), "state": state, "quality_status": self.quality_status}
        fd = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as output:
            output.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def check_time(self) -> None:
        if time.time() >= self.deadline:
            raise TimeLimit("48-hour wall-time guard reached")

    def _runner(self, name: str) -> pathlib.Path:
        return self.root / "cargo-target" / "release" / name

    def _run(
        self,
        name: str,
        command: list[str],
        *,
        state: str,
        json_stdout: bool = False,
    ) -> dict[str, Any] | None:
        self.check_time()
        self.status(state)
        stdout_path = self.root / "reports" / f"{name}.stdout.json"
        stderr_path = self.root / "logs" / f"{name}.stderr.log"
        if stdout_path.exists() or stderr_path.exists():
            raise RunError("preserving existing stage output")
        timeout = max(1, int(self.deadline - time.time()))
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            os.chmod(stdout_path, 0o600)
            os.chmod(stderr_path, 0o600)
            try:
                result = subprocess.run(
                    command,
                    cwd=self.source,
                    env=self.environment,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise TimeLimit(f"{name} exceeded the 48-hour wall-time guard") from None
        if result.returncode != 0:
            raise RunError(f"{name} failed with exit code {result.returncode}")
        if not json_stdout:
            return None
        try:
            value = json.loads(stdout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RunError(f"{name} did not emit a valid private JSON report") from None
        if not isinstance(value, dict):
            raise RunError(f"{name} report is not a JSON object")
        return value

    def _ensure_cargo(self) -> None:
        cargo = self.environment["CARGO_HOME"] + "/bin/cargo"
        if pathlib.Path(cargo).is_file():
            return
        if shutil.which("cargo", path=self.environment["PATH"]):
            return
        self.status("preparing")
        log_path = self.root / "logs" / "rustup-install.private.log"
        if log_path.exists():
            raise RunError("preserving existing toolchain log")
        with log_path.open("xb") as log:
            os.chmod(log_path, 0o600)
            if shutil.which("curl") is None or shutil.which("sh") is None:
                raise RunError("remote Rust bootstrap tools are unavailable")
            download = subprocess.run(
                [
                    "curl", "--fail", "--silent", "--show-error", "--retry", "3",
                    "--retry-delay", "2", "https://sh.rustup.rs", "-o",
                    str(self.root / "rustup-init.sh"),
                ],
                env=self.environment,
                stdout=log,
                stderr=log,
                timeout=min(180, max(1, int(self.deadline - time.time()))),
                check=False,
            )
            if download.returncode != 0:
                raise RunError("remote Rust bootstrap download failed")
            install = subprocess.run(
                [
                    "sh", str(self.root / "rustup-init.sh"), "-y",
                    "--profile", "minimal", "--default-toolchain", "stable",
                    "--no-modify-path",
                ],
                env=self.environment,
                stdout=log,
                stderr=log,
                timeout=min(1200, max(1, int(self.deadline - time.time()))),
                check=False,
            )
            if install.returncode != 0:
                raise RunError("remote Rust toolchain install failed")
        pathlib.Path(self.root / "rustup-init.sh").unlink(missing_ok=True)
        if not pathlib.Path(cargo).is_file():
            raise RunError("remote cargo executable is unavailable")

    def _hardware(self) -> dict[str, Any]:
        self.check_time()
        command = [
            "nvidia-smi", "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        if result.returncode != 0:
            raise RunError("NVIDIA device query failed")
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 2:
            raise RunError("exactly two CUDA devices are required")
        for line in lines:
            lowered = line.casefold()
            if "rtx 3060" not in lowered or "ti" in lowered:
                raise RunError("remote GPU product differs from RTX 3060")
        torch_check = (
            "import torch; "
            "assert torch.cuda.is_available() and torch.cuda.device_count() == 2; "
            "assert all('3060' in torch.cuda.get_device_name(i) and "
            "'ti' not in torch.cuda.get_device_name(i).casefold() for i in range(2))"
        )
        check = subprocess.run(
            ["python3", "-c", torch_check],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if check.returncode != 0:
            raise RunError("PyTorch did not expose both RTX 3060 devices")
        return {"gpu_inventory": lines, "torch_cuda_devices": 2}

    def _free_gib(self) -> float:
        return shutil.disk_usage("/workspace").free / float(1024 ** 3)

    def _build_dataset(self) -> str:
        self.status("preparing")
        if self._free_gib() < INITIAL_FREE_GIB:
            raise RunError("less than 40 GiB is free before dataset construction")
        holdout_path = self.root / HOLDOUT_RELATIVE
        fetch = [
            "python3", str(self.source / ".github/scripts/fetch_public_holdout.py"),
            "--experiment-root", str(self.root),
            "--expected-revision", self.holdout_revision,
            "--expected-sha256", self.holdout_sha256,
        ]
        self._run("fetch_holdout", fetch, state="preparing")
        if not holdout_path.is_file():
            raise RunError("pinned holdout SQLite is missing")
        data_path = self.root / DATA_RELATIVE
        receipt_path = self.root / DATA_RECEIPT_RELATIVE
        build = [
            "python3", str(self.source / ".github/scripts/rebuild_frozen_dataset.py"),
            "--experiment-root", str(self.root),
            "--holdout-sqlite", str(holdout_path),
            "--output", str(data_path),
            "--receipt", str(receipt_path),
            "--builder-source-revision", self.source_revision,
            "--expected-holdout-revision", self.holdout_revision,
            "--expected-holdout-sha256", self.holdout_sha256,
            "--seed", "202611190001",
        ]
        self._run("build_frozen_dataset", build, state="preparing")
        sha_path = self.root / DATA_SHA_RELATIVE
        if not sha_path.is_file():
            raise RunError("frozen train dataset SHA receipt is missing")
        dataset_sha = sha_path.read_text(encoding="ascii").strip()
        if len(dataset_sha) != 64 or any(c not in "0123456789abcdef" for c in dataset_sha):
            raise RunError("frozen train dataset SHA is malformed")
        self._run(
            "validate_frozen_dataset",
            [
                str(self._runner("cnot103-validate-data")),
                str(data_path),
                dataset_sha,
            ],
            state="preparing",
        )
        self._run(
            "finalize_frozen_dataset",
            [
                "python3", str(self.source / ".github/scripts/rebuild_frozen_dataset.py"),
                "--finalize-dataset",
                "--experiment-root", str(self.root),
                "--output", str(data_path),
                "--receipt", str(receipt_path),
                "--expected-dataset-sha256", dataset_sha,
            ],
            state="preparing",
        )
        return dataset_sha

    def _build_native(self) -> None:
        self._ensure_cargo()
        self._run(
            "cargo_release_build",
            [
                "cargo", "build", "--release", "--features", "cuda",
                "--manifest-path", str(self.source / "native/Cargo.toml"),
                "--bin", "cnot103-train",
                "--bin", "cnot103-benchmark",
                "--bin", "cnot103-search",
                "--bin", "cnot103-coverage",
                "--bin", "cnot103-validate-data",
                "--bin", "cnot103-validate-manifest",
            ],
            state="preparing",
        )

    def _build_manifest(
        self,
        name: str,
        dataset_sha: str,
        attempts: int,
        target_attempts: int,
        resume: bool,
    ) -> pathlib.Path:
        manifest_path = self.root / "manifests" / f"{name}.json"
        command = [
            "python3", str(self.source / ".github/scripts/build_training_manifest.py"),
            "--profile", str(self.source / PROFILE_RELATIVE),
            "--dataset-path", str(self.root / DATA_RELATIVE),
            "--dataset-sha256", dataset_sha,
            "--checkpoint-dir", str(self.root / "checkpoints"),
            "--attempts", str(attempts),
            "--target-attempts", str(target_attempts),
            "--output", str(manifest_path),
        ]
        if resume:
            command.append("--resume")
        self._run(f"manifest_{name}", command, state="preparing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sidecar = json.loads(
                manifest_path.with_suffix(manifest_path.suffix + ".provenance.json")
                .read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            raise RunError("training manifest or provenance is malformed") from None
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if (
            manifest.get("schema") != "cnot103_training_run_v1"
            or manifest.get("training", {}).get("attempts") != attempts
            or manifest.get("training", {}).get("target_attempts") != target_attempts
            or manifest.get("training", {}).get("checkpoint_every_attempts") != CHECKPOINT_INTERVAL
            or manifest.get("model", {}).get("structural_blocks") != STRUCTURAL_BLOCKS
            or manifest.get("feature_policy", {}).get("schema_id") != "runtime_packet_layout_v25"
            or sidecar.get("model_input_slots") != MODEL_INPUTS
            or sidecar.get("invocation_attempts") != attempts
            or sidecar.get("target_attempts") != target_attempts
            or sidecar.get("manifest_sha256") != manifest_sha
        ):
            raise RunError("generated manifest violates the training contract")
        self._run(
            f"validate_manifest_{name}",
            [str(self._runner("cnot103-validate-manifest")), str(manifest_path)],
            state="preparing",
        )
        return manifest_path

    def _feature_coverage(self, dataset_sha: str) -> None:
        manifest = self._build_manifest(
            "coverage", dataset_sha, 1, TARGET_ATTEMPTS, False
        )
        for panel in ("train", "dev"):
            self._run(
                f"coverage_{panel}",
                [str(self._runner("cnot103-coverage")), str(manifest), panel],
                state="preparing",
            )

    def _benchmark(self, dataset_sha: str) -> tuple[float, float, dict[str, Any], float]:
        manifest = self._build_manifest(
            "speed_pilot", dataset_sha, PILOT_ATTEMPTS, PILOT_ATTEMPTS, False
        )
        started = time.monotonic()
        self.status("benchmarking")
        report = self._run(
            "speed_pilot",
            [str(self._runner("cnot103-benchmark")), str(manifest)],
            state="benchmarking",
            json_stdout=True,
        )
        elapsed = time.monotonic() - started
        update_seconds, setup_seconds = validate_benchmark(report, PILOT_ATTEMPTS)
        return update_seconds, setup_seconds, report, elapsed

    def _train(
        self,
        name: str,
        manifest: pathlib.Path,
        expected_start: int,
        expected_end: int,
        state: str,
    ) -> tuple[dict[str, Any], float]:
        started = time.monotonic()
        report = self._run(
            name,
            [str(self._runner("cnot103-train")), str(manifest)],
            state=state,
            json_stdout=True,
        )
        elapsed = time.monotonic() - started
        validate_training_result(report, expected_start, expected_end)
        return report, elapsed

    def _search_manifest(
        self,
        name: str,
        train_manifest: pathlib.Path,
        train_result: pathlib.Path,
        dev_indices: list[int],
    ) -> pathlib.Path:
        value = {
            "schema": "cnot103_dev_search_v1",
            "training_manifest": {
                "path": str(train_manifest),
                "sha256": hashlib.sha256(train_manifest.read_bytes()).hexdigest(),
            },
            "training_result": {
                "path": str(train_result),
                "sha256": hashlib.sha256(train_result.read_bytes()).hexdigest(),
            },
            "dev_indices": dev_indices,
            "search": BEAM_CONFIG,
            "head": "qhat",
        }
        path = self.root / "manifests" / f"{name}.json"
        _write_new_private_json(path, value)
        return path

    def _search(
        self,
        name: str,
        manifest: pathlib.Path,
        expected_indices: list[int],
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        started = time.monotonic()
        report = self._run(
            name,
            [str(self._runner("cnot103-search")), str(manifest)],
            state="baseline" if name.startswith("baseline") else "final_quality",
            json_stdout=True,
        )
        elapsed = time.monotonic() - started
        metrics = quality_metrics(report)
        if (
            report.get("search") != BEAM_CONFIG
            or report.get("head") != "qhat"
            or [row.get("dev_index") for row in report["rows"]] != expected_indices
        ):
            raise RunError("holdout search did not use the identical pinned budget and rows")
        return report, metrics, elapsed

    def _check_source(self) -> None:
        result = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() != self.source_revision:
            raise RunError("remote source revision differs from the workflow pin")

    def _private_failure(self, exc: BaseException) -> None:
        value = {
            "schema": "cnot103_full_training_failure_v1",
            "at_utc": _now(),
            "stage": self.current_state,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            _write_new_private_json(self.error_path, value)
        except Exception:
            pass

    def run(self) -> int:
        try:
            self.status("starting")
            self._check_source()
            hardware = self._hardware()
            dataset_sha = self._build_dataset()
            self._build_native()
            self._feature_coverage(dataset_sha)
            if self._free_gib() < CHECKPOINT_FREE_GIB:
                raise RunError("less than 24 GiB is free for immutable checkpoints")
            update_seconds, setup_seconds, benchmark_report, benchmark_wall = self._benchmark(dataset_sha)

            fresh_manifest = self._build_manifest(
                "fresh_step_1", dataset_sha, 1, TARGET_ATTEMPTS, False
            )
            fresh_report, fresh_wall = self._train(
                "fresh_step_1", fresh_manifest, 0, 1, "resuming"
            )
            resume_manifest = self._build_manifest(
                "resume_step_2", dataset_sha, 1, TARGET_ATTEMPTS, True
            )
            resume_report, resume_wall = self._train(
                "resume_step_2", resume_manifest, 1, 2, "resuming"
            )
            if self._free_gib() < CHECKPOINT_FREE_GIB:
                raise RunError("less than 24 GiB remains after resume proof")

            data_value = json.loads((self.root / DATA_RELATIVE).read_text(encoding="utf-8"))
            panels = data_value.get("panels") if isinstance(data_value, dict) else None
            dev = panels.get("dev") if isinstance(panels, dict) else None
            dev_indices = select_dev_indices(dev, per_width=8)
            baseline_manifest = self._search_manifest(
                "baseline_search", resume_manifest,
                self.root / "reports" / "resume_step_2.stdout.json",
                dev_indices,
            )
            _baseline_report, baseline_metrics, baseline_search_wall = self._search(
                "baseline_search",
                baseline_manifest,
                dev_indices,
            )
            self.status("baseline", "measured")

            elapsed_before_full = max(0.0, time.time() - self.started_at_epoch)
            remaining_updates = TARGET_ATTEMPTS - 2
            estimated_full_update_seconds = update_seconds / PILOT_ATTEMPTS * remaining_updates
            estimated_full_seconds = (
                elapsed_before_full
                + setup_seconds
                + estimated_full_update_seconds
                + CHECKPOINT_WRITE_RESERVE_SECONDS
                + baseline_search_wall
                + FORECAST_SAFETY_SECONDS
            )
            if estimated_full_seconds > MAX_RUNTIME_SECONDS:
                raise TimeLimit("measured full-model speed misses the 48-hour end-to-end gate")

            full_manifest = self._build_manifest(
                "full_resume", dataset_sha, remaining_updates, TARGET_ATTEMPTS, True
            )
            full_report, full_wall = self._train(
                "full_resume", full_manifest, 2, TARGET_ATTEMPTS, "training"
            )
            final_manifest = self._search_manifest(
                "final_search",
                full_manifest,
                self.root / "reports" / "full_resume.stdout.json",
                dev_indices,
            )
            _final_search_report, final_metrics, final_search_wall = self._search(
                "final_search",
                final_manifest,
                dev_indices,
            )
            self.quality_status = "passed"
            total_elapsed = max(0.0, time.time() - self.started_at_epoch)
            if total_elapsed > MAX_RUNTIME_SECONDS:
                raise TimeLimit("full run exceeded the end-to-end wall-time gate")
            final_report = {
                "schema": "cnot103_full_training_result_v1",
                "source_revision": self.source_revision,
                "holdout_revision": self.holdout_revision,
                "holdout_sha256": self.holdout_sha256,
                "dataset_sha256": dataset_sha,
                "started_at_epoch": self.started_at_epoch,
                "completed_at_utc": _now(),
                "total_elapsed_seconds": total_elapsed,
                "training": {
                    "target_attempts": TARGET_ATTEMPTS,
                    "batch_size": BATCH_SIZE,
                    "model_parameters": MODEL_PARAMETERS,
                    "structural_blocks": STRUCTURAL_BLOCKS,
                    "model_input_slots": MODEL_INPUTS,
                    "feature_schema": "runtime_packet_layout_v25",
                    "fresh_progress": fresh_report["end_progress"],
                    "resume_progress": resume_report["end_progress"],
                    "final_progress": full_report["end_progress"],
                    "checkpoint_files_expected": EXPECTED_CHECKPOINT_FILES,
                    "checkpoint_free_space_gate_gib": CHECKPOINT_FREE_GIB,
                    "checkpoint_interval_attempts": CHECKPOINT_INTERVAL,
                },
                "timing": {
                    "pilot_attempts": PILOT_ATTEMPTS,
                    "pilot_update_seconds": update_seconds,
                    "pilot_setup_seconds": setup_seconds,
                    "pilot_process_wall_seconds": benchmark_wall,
                    "fresh_step_wall_seconds": fresh_wall,
                    "resume_step_wall_seconds": resume_wall,
                    "baseline_search_wall_seconds": baseline_search_wall,
                    "full_training_wall_seconds": full_wall,
                    "final_search_wall_seconds": final_search_wall,
                    "forecasted_full_seconds_before_training": estimated_full_seconds,
                },
                "quality": {
                    "panel": "pinned dev subset stratified by width",
                    "dev_indices": dev_indices,
                    "beam_config": BEAM_CONFIG,
                    "head": "qhat",
                    "baseline": baseline_metrics,
                    "final": final_metrics,
                    "same_cases_and_budget": True,
                },
                "hardware": hardware,
                "budget": {
                    "spend_at_start": self.budget_baseline,
                    "max_total_usd": 20.0,
                    "billing_outcome": "bounded_by_controller",
                },
            }
            _write_new_private_json(self.private_report_path, final_report)
            self.status("completed", "passed")
            return 0
        except TimeLimit as exc:
            self._private_failure(exc)
            self.status("time_limit", self.quality_status)
            return 1
        except BaseException as exc:
            self._private_failure(exc)
            self.status("failed", "failed")
            return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--holdout-revision", required=True)
    parser.add_argument("--holdout-sha256", required=True)
    parser.add_argument("--started-at-epoch", required=True, type=float)
    parser.add_argument("--budget-baseline", required=True, type=float)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (
        len(args.source_revision) != 40
        or any(c not in "0123456789abcdef" for c in args.source_revision)
        or len(args.holdout_revision) != 40
        or any(c not in "0123456789abcdef" for c in args.holdout_revision)
        or len(args.holdout_sha256) != 64
        or any(c not in "0123456789abcdef" for c in args.holdout_sha256)
        or not math.isfinite(args.started_at_epoch)
        or not math.isfinite(args.budget_baseline)
        or args.started_at_epoch <= 0
        or args.budget_baseline < 0
    ):
        return 2
    root = pathlib.Path(args.root)
    if not str(root).startswith("/workspace/tqp_full_training_"):
        return 2
    supervisor = Supervisor(
        root=root,
        source_revision=args.source_revision,
        holdout_revision=args.holdout_revision,
        holdout_sha256=args.holdout_sha256,
        started_at_epoch=args.started_at_epoch,
        budget_baseline=args.budget_baseline,
    )
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
