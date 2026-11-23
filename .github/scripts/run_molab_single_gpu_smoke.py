#!/usr/bin/env python3
"""Run one full-model CUDA update, then resume it once in a fresh process.

This is a bounded Molab smoke only. It does not run a learning curve, search
quality evaluation, Vast control, or a multi-GPU comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

from build_training_manifest import (
    BATCH_SIZE,
    FEATURE_SCHEMA,
    MODEL_INPUT_COUNT,
    MODEL_ARCHITECTURE,
    _write_private_json,
)
from run_full_training import (
    MODEL_INPUTS,
    MODEL_PARAMETERS,
    STRUCTURAL_BLOCKS,
    RunError,
    validate_training_result,
)

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_FILE = re.compile(r"^step-([0-9]{12})-([0-9a-f]{64})\.qcp$")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_source_sha(repo_root: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    if result.returncode != 0 or _COMMIT.fullmatch(revision) is None:
        raise RunError("source revision could not be verified")
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise RunError("source checkout has modified tracked files")
    return revision



def _native_build_command(repo_root: pathlib.Path) -> list[str]:
    return [
        "cargo", "build", "--locked", "--release", "--features", "cuda",
        "--manifest-path", str(repo_root / "native" / "Cargo.toml"),
        "--bin", "cnot103-train",
    ]


def _write_private_text(path: pathlib.Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        output.write(value)


def _build_native_trainer(
    repo_root: pathlib.Path,
    source_sha: str,
    build_log_dir: pathlib.Path,
    source_revision: Callable[[pathlib.Path], str] = _current_source_sha,
) -> tuple[pathlib.Path, dict[str, Any]]:
    if source_revision(repo_root) != source_sha:
        raise RunError("source checkout changed before native trainer build")
    manifest_path = repo_root / "native" / "Cargo.toml"
    if not manifest_path.is_file():
        raise RunError("pinned native Cargo manifest is missing")
    if build_log_dir.exists():
        raise RunError("preserving existing native build logs")
    build_log_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    env = os.environ.copy()
    metadata_command = [
        "cargo", "metadata", "--locked", "--no-deps", "--format-version", "1",
        "--manifest-path", str(manifest_path),
    ]
    metadata_stdout_path = build_log_dir / "cargo-metadata.stdout.json"
    metadata_stderr_path = build_log_dir / "cargo-metadata.stderr.log"
    try:
        metadata_result = subprocess.run(
            metadata_command,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RunError("Cargo target directory could not be resolved") from None
    _write_private_text(metadata_stdout_path, metadata_result.stdout)
    _write_private_text(metadata_stderr_path, metadata_result.stderr)
    if metadata_result.returncode != 0:
        raise RunError("Cargo metadata failed; private build logs were preserved")
    try:
        metadata = json.loads(metadata_result.stdout)
        target_raw = metadata.get("target_directory") if isinstance(metadata, dict) else None
        if not isinstance(target_raw, str) or not target_raw:
            raise ValueError("target directory")
        target_dir = pathlib.Path(target_raw).resolve()
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        raise RunError("Cargo metadata target directory is invalid") from None

    versions: dict[str, str] = {}
    for name, command in (
        ("cargo", ["cargo", "--version"]),
        ("rustc", ["rustc", "--version"]),
    ):
        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise RunError("Rust toolchain version could not be recorded") from None
        _write_private_text(build_log_dir / f"{name}-version.stdout.log", result.stdout)
        _write_private_text(build_log_dir / f"{name}-version.stderr.log", result.stderr)
        if result.returncode != 0 or not result.stdout.strip():
            raise RunError("Rust toolchain version could not be recorded")
        versions[name] = result.stdout.strip()

    command = _native_build_command(repo_root)
    stdout_path = build_log_dir / "cargo-build.stdout.log"
    stderr_path = build_log_dir / "cargo-build.stderr.log"
    started = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        os.chmod(stdout_path, 0o600)
        os.chmod(stderr_path, 0o600)
        try:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
        except OSError:
            raise RunError("pinned native trainer build could not start") from None
        try:
            while process.poll() is None:
                elapsed = int(time.monotonic() - started)
                print(
                    f"molab_native_build=running; elapsed_seconds={elapsed}",
                    flush=True,
                )
                time.sleep(30)
            return_code = process.returncode
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if return_code != 0:
        raise RunError("pinned native trainer build failed; private logs were preserved")
    if source_revision(repo_root) != source_sha:
        raise RunError("source checkout changed during native trainer build")

    executable_name = "cnot103-train.exe" if os.name == "nt" else "cnot103-train"
    trainer_path = target_dir / "release" / executable_name
    if not trainer_path.is_file():
        raise RunError("Cargo did not produce the pinned native trainer")
    try:
        trainer_path.resolve().relative_to(target_dir)
    except ValueError:
        raise RunError("Cargo trainer escaped its target directory") from None
    trainer_sha256 = _sha256_file(trainer_path)
    relevant_environment = (
        "CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT", "CUDA_TOOLKIT_ROOT_DIR",
        "CUDACXX", "CMAKE_CUDA_COMPILER", "CNOT103_CUDA_ARCHITECTURES",
        "LIBTORCH", "LIBTORCH_USE_PYTORCH", "PYTHON", "CMAKE_PREFIX_PATH",
        "TORCH_CUDA_ARCH_LIST", "LD_LIBRARY_PATH",
    )
    build = {
        "source_sha": source_sha,
        "command": command,
        "target_directory": str(target_dir),
        "binary_sha256": trainer_sha256,
        "cargo_version": versions["cargo"],
        "rustc_version": versions["rustc"],
        "environment": {name: env.get(name) for name in relevant_environment if env.get(name)},
        "logs_directory": str(build_log_dir),
    }
    return trainer_path, build


def _build_manifest(
    profile_path: pathlib.Path,
    dataset_path: pathlib.Path,
    dataset_sha256: str,
    checkpoint_dir: pathlib.Path,
    resume: bool,
    output_path: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(pathlib.Path(__file__).with_name("build_training_manifest.py")),
        "--profile",
        str(profile_path),
        "--dataset-path",
        str(dataset_path),
        "--dataset-sha256",
        dataset_sha256,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--attempts",
        "1",
        "--target-attempts",
        "2",
        "--gpu-count",
        "1",
        "--output",
        str(output_path),
    ]
    if resume:
        command.append("--resume")
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RunError("pinned one-GPU manifest generation failed")
    try:
        manifest = json.loads(output_path.read_text(encoding="utf-8"))
        provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RunError("one-GPU manifest output is unreadable") from None
    if not isinstance(manifest, dict) or not isinstance(provenance, dict):
        raise RunError("one-GPU manifest output is malformed")
    return manifest, provenance


def _validate_manifest(
    manifest: object,
    provenance: object,
    *,
    profile_sha256: str,
    dataset_path: pathlib.Path,
    dataset_sha256: str,
    checkpoint_dir: pathlib.Path,
    resume: bool,
) -> str:
    if not isinstance(manifest, dict) or not isinstance(provenance, dict):
        raise RunError("one-GPU manifest is malformed")
    training = manifest.get("training")
    model = manifest.get("model")
    dataset = manifest.get("dataset")
    policy = manifest.get("feature_policy")
    if (
        manifest.get("schema") != "cnot103_training_run_v1"
        or manifest.get("resume") is not resume
        or manifest.get("checkpoint_dir") != str(checkpoint_dir)
        or not isinstance(training, dict)
        or training.get("attempts") != 1
        or training.get("target_attempts") != 2
        or training.get("cuda_devices") != [0]
        or training.get("batch_size") != BATCH_SIZE
        or not isinstance(model, dict)
        or model != MODEL_ARCHITECTURE
        or not isinstance(dataset, dict)
        or dataset.get("path") != str(dataset_path)
        or dataset.get("sha256") != dataset_sha256
        or not isinstance(policy, dict)
        or policy.get("schema_id") != FEATURE_SCHEMA
    ):
        raise RunError("one-GPU manifest differs from the full-model smoke contract")
    serialized = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    manifest_sha256 = hashlib.sha256(serialized).hexdigest()
    if (
        provenance.get("profile_sha256") != profile_sha256
        or provenance.get("dataset_sha256") != dataset_sha256
        or provenance.get("manifest_sha256") != manifest_sha256
        or provenance.get("model_input_slots") != MODEL_INPUT_COUNT
        or provenance.get("gpu_count") != 1
        or provenance.get("invocation_attempts") != 1
        or provenance.get("target_attempts") != 2
    ):
        raise RunError("one-GPU manifest provenance differs")
    return manifest_sha256


def _run_native(
    trainer_path: pathlib.Path,
    manifest_path: pathlib.Path,
    log_dir: pathlib.Path,
    step: int,
    repo_root: pathlib.Path,
) -> dict[str, Any]:
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    stdout_path = log_dir / f"step-{step}.stdout.json"
    stderr_path = log_dir / f"step-{step}.stderr.log"
    started = time.monotonic()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        os.chmod(stdout_path, 0o600)
        os.chmod(stderr_path, 0o600)
        try:
            process = subprocess.Popen(
                [str(trainer_path), str(manifest_path)],
                cwd=repo_root,
                env=os.environ.copy(),
                stdout=stdout,
                stderr=stderr,
            )
        except OSError:
            raise RunError("native trainer could not be started") from None
        try:
            while process.poll() is None:
                elapsed = int(time.monotonic() - started)
                print(
                    f"molab_smoke_step={step}; state=running; elapsed_seconds={elapsed}",
                    flush=True,
                )
                time.sleep(30)
            return_code = process.returncode
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if return_code != 0:
        raise RunError(f"native trainer step {step} failed; private logs are under {log_dir}")
    try:
        report = json.loads(stdout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RunError(f"native trainer step {step} returned invalid JSON") from None
    if not isinstance(report, dict):
        raise RunError(f"native trainer step {step} returned a malformed report")
    return report


def _verify_checkpoint(
    checkpoint_dir: pathlib.Path, report: dict[str, Any], expected_attempt: int
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RunError("trainer did not return a checkpoint manifest")
    filename = checkpoint.get("file")
    sha256 = checkpoint.get("sha256")
    progress = checkpoint.get("progress")
    if (
        not isinstance(filename, str)
        or not isinstance(sha256, str)
        or _SHA256.fullmatch(sha256) is None
        or not isinstance(progress, dict)
        or type(progress.get("attempted")) is not int
        or type(progress.get("accepted")) is not int
        or type(progress.get("rejected")) is not int
        or progress.get("attempted") != expected_attempt
        or progress.get("accepted") != expected_attempt
        or progress.get("rejected") != 0
    ):
        raise RunError("checkpoint progress or digest metadata is invalid")
    match = _CHECKPOINT_FILE.fullmatch(filename)
    if (
        match is None
        or int(match.group(1)) != expected_attempt
        or match.group(2) != sha256
    ):
        raise RunError("checkpoint filename is inconsistent with its receipt")
    checkpoint_path = checkpoint_dir / filename
    if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != sha256:
        raise RunError("checkpoint bytes do not match the trainer receipt")
    try:
        last = json.loads((checkpoint_dir / "last.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RunError("latest checkpoint pointer is missing or invalid") from None
    if last != checkpoint:
        raise RunError("latest checkpoint pointer differs from the trainer receipt")
    return {
        "file": filename,
        "sha256": sha256,
        "contract_sha256": checkpoint.get("contract_sha256"),
        "progress": {
            "attempted": expected_attempt,
            "accepted": expected_attempt,
            "rejected": 0,
        },
    }


def run_molab_single_gpu_smoke(
    *,
    profile_path: pathlib.Path,
    dataset_path: pathlib.Path,
    dataset_sha256: str,
    checkpoint_dir: pathlib.Path,
    source_sha: str,
    repo_root: pathlib.Path,
    source_revision: Callable[[pathlib.Path], str] = _current_source_sha,
    manifest_builder: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = _build_manifest,
    trainer_builder: Callable[..., tuple[pathlib.Path, dict[str, Any]]] = _build_native_trainer,
    native_runner: Callable[..., dict[str, Any]] = _run_native,
) -> pathlib.Path:
    if not isinstance(source_sha, str) or _COMMIT.fullmatch(source_sha) is None:
        raise RunError("expected source SHA is malformed")
    if source_revision(repo_root) != source_sha:
        raise RunError("checked-out source SHA differs from the requested pin")
    profile_path = profile_path.resolve()
    dataset_path = dataset_path.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    repo_root = repo_root.resolve()
    if not profile_path.is_file() or profile_path.stat().st_size > (4 << 20):
        raise RunError("pinned feature profile is missing or too large")
    if not dataset_path.is_file():
        raise RunError("pinned dataset is missing")
    if not isinstance(dataset_sha256, str) or _SHA256.fullmatch(dataset_sha256) is None:
        raise RunError("dataset SHA-256 is malformed")
    if _sha256_file(dataset_path) != dataset_sha256:
        raise RunError("dataset bytes differ from the requested SHA-256")
    if checkpoint_dir.exists():
        raise RunError("checkpoint directory must be new for the first one-GPU smoke")
    profile_sha256 = _sha256_file(profile_path)
    checkpoint_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".cnot103-molab-manifests-", dir=checkpoint_dir.parent
    ) as staging:
        staged = pathlib.Path(staging)
        built: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        for step, resume in ((1, False), (2, True)):
            output_path = staged / f"step-{step}.manifest.json"
            manifest, provenance = manifest_builder(
                profile_path,
                dataset_path,
                dataset_sha256,
                checkpoint_dir,
                resume,
                output_path,
            )
            manifest_sha256 = _validate_manifest(
                manifest,
                provenance,
                profile_sha256=profile_sha256,
                dataset_path=dataset_path,
                dataset_sha256=dataset_sha256,
                checkpoint_dir=checkpoint_dir,
                resume=resume,
            )
            built.append((manifest, provenance, manifest_sha256))

    build_log_dir = checkpoint_dir.parent / f".{checkpoint_dir.name}.native-build"
    trainer_path, trainer_build = trainer_builder(
        repo_root, source_sha, build_log_dir, source_revision
    )
    trainer_path = pathlib.Path(trainer_path).resolve()
    if not trainer_path.is_file():
        raise RunError("pinned source build did not produce a native trainer")
    trainer_sha256 = _sha256_file(trainer_path)
    if not isinstance(trainer_build, dict):
        raise RunError("native trainer build receipt is malformed")
    try:
        target_dir = pathlib.Path(trainer_build.get("target_directory", "")).resolve()
        expected_trainer_path = (
            target_dir / "release" /
            ("cnot103-train.exe" if os.name == "nt" else "cnot103-train")
        ).resolve()
    except (OSError, TypeError, ValueError):
        raise RunError("native trainer target directory is invalid") from None
    if (
        trainer_build.get("source_sha") != source_sha
        or trainer_build.get("command") != _native_build_command(repo_root)
        or trainer_build.get("binary_sha256") != trainer_sha256
        or trainer_build.get("logs_directory") != str(build_log_dir)
        or trainer_path != expected_trainer_path
        or not isinstance(trainer_build.get("cargo_version"), str)
        or not trainer_build.get("cargo_version")
        or not isinstance(trainer_build.get("rustc_version"), str)
        or not trainer_build.get("rustc_version")
    ):
        raise RunError("native trainer build receipt differs from the pinned source")
    if source_revision(repo_root) != source_sha:
        raise RunError("source checkout changed after native trainer build")

    checkpoint_dir.mkdir(mode=0o700, exist_ok=False)
    private_dir = checkpoint_dir / ".molab-single-gpu-smoke"
    private_dir.mkdir(mode=0o700)
    receipt_path = private_dir / "receipt.json"
    receipt: dict[str, Any] = {
        "schema": "cnot103_molab_single_gpu_smoke_v1",
        "status": "started",
        "scope": "smoke_only",
        "source_sha": source_sha,
        "dataset_sha256": dataset_sha256,
        "profile_sha256": profile_sha256,
        "trainer_sha256": trainer_sha256,
        "trainer_build": trainer_build,
        "model_parameters": MODEL_PARAMETERS,
        "feature_model_inputs": MODEL_INPUTS,
        "structural_blocks": STRUCTURAL_BLOCKS,
        "batch_size": BATCH_SIZE,
        "cuda_devices": [0],
        "strict_unseen": False,
        "steps": [],
    }
    _write_private_json(receipt_path, receipt)
    stage = "fresh_step"
    try:
        for step, (expected_start, expected_end, resume) in enumerate(
            ((0, 1, False), (1, 2, True)), start=1
        ):
            if source_revision(repo_root) != source_sha:
                raise RunError("source checkout changed during the one-GPU smoke")
            if _sha256_file(dataset_path) != dataset_sha256:
                raise RunError("dataset changed during the one-GPU smoke")
            if _sha256_file(profile_path) != profile_sha256:
                raise RunError("feature profile changed during the one-GPU smoke")
            if _sha256_file(trainer_path) != trainer_sha256:
                raise RunError("native trainer binary changed during the one-GPU smoke")
            manifest, provenance, manifest_sha256 = built[step - 1]
            manifest_path = private_dir / f"step-{step}.manifest.json"
            provenance_path = private_dir / f"step-{step}.manifest.provenance.json"
            _write_private_json(manifest_path, manifest)
            _write_private_json(provenance_path, provenance)
            stage = "fresh_step" if not resume else "resume_step"
            report = native_runner(
                trainer_path, manifest_path, private_dir / "logs", step, repo_root
            )
            validate_training_result(
                report,
                expected_start=expected_start,
                expected_end=expected_end,
                expected_cuda_devices=(0,),
            )
            checkpoint = _verify_checkpoint(checkpoint_dir, report, expected_end)
            report_path = private_dir / f"step-{step}.report.json"
            _write_private_json(report_path, report)
            receipt["steps"].append(
                {
                    "manifest_sha256": manifest_sha256,
                    "start_attempted": expected_start,
                    "end_attempted": expected_end,
                    "accepted": expected_end,
                    "rejected": 0,
                    "strict_unseen": True,
                    "checkpoint": checkpoint,
                }
            )
            receipt["strict_unseen"] = True
            receipt["status"] = "partial" if step == 1 else "complete"
            _write_private_json(receipt_path, receipt)
            print(
                f"molab_smoke_step={step}; state=verified; accepted={expected_end}; "
                "checkpoint_sha256=verified; strict_unseen=true",
                flush=True,
            )
    except BaseException:
        receipt["status"] = "partial" if receipt["steps"] else "failed"
        receipt["failure_stage"] = stage
        _write_private_json(receipt_path, receipt)
        raise
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=pathlib.Path)
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    try:
        receipt = run_molab_single_gpu_smoke(
            profile_path=args.profile,
            dataset_path=args.dataset,
            dataset_sha256=args.dataset_sha256,
            checkpoint_dir=args.checkpoint_dir,
            source_sha=args.source_sha,
            repo_root=args.repo_root,
        )
    except (OSError, ValueError, RunError):
        print("molab_single_gpu_smoke=failed")
        return 2
    print(
        f"molab_single_gpu_smoke=pass; accepted_updates=2; "
        f"cuda_devices=1; strict_unseen=true; receipt={receipt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
