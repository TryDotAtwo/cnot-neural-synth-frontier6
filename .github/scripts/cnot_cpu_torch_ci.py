#!/usr/bin/env python3
"""Redacted CI helpers for compiling and executing the CPU torch acceptance lanes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
from typing import Any

RUSTC_CODE = re.compile(r"^E[0-9]{4}$")
CHECK_NAMES = ("v13", "resume", "profile")


def _target_matches(target: dict[str, Any], target_name: str | None, artifact_kind: str) -> bool:
    if target_name is not None and target.get("name") != target_name:
        return False
    kind = target.get("kind")
    if not isinstance(kind, list):
        return False
    if artifact_kind == "lib_test":
        # Cargo's executable field is the reliable test-harness marker; metadata
        # has varied in whether the target.test flag is present on artifacts.
        return "lib" in kind
    if artifact_kind == "test":
        return target.get("test") is True
    if artifact_kind == "bin":
        return "bin" in kind
    raise ValueError("unknown artifact kind")


def _area_for_span(span: dict[str, Any]) -> str:
    filename = str(span.get("file_name", "")).replace("\\\\", "/").lower()
    line_start = span.get("line_start")
    if filename.endswith("/device_search/runtime.rs"):
        return "runtime_tests" if isinstance(line_start, int) and line_start >= 280 else "runtime_core"
    if filename.endswith("/device_search/frontier.rs"):
        return "frontier"
    if filename.endswith("/device_search/scoring.rs"):
        return "scoring"
    if filename.endswith("/device_search.rs"):
        return "device_search"
    if "/run_config/execution/" in filename:
        return "run_config_execution"
    if "/run_config/manifest/" in filename or filename.endswith("/run_config/manifest.rs"):
        return "run_config_manifest"
    if filename.endswith("/native/src/lib.rs") or filename == "src/lib.rs":
        return "crate_root"
    if filename.endswith("/native/build.rs") or filename == "build.rs":
        return "build_script"
    source_prefixes = ("/native/src/", "src/")
    for prefix in source_prefixes:
        if prefix in filename:
            relative = filename.split(prefix, 1)[1]
            top = relative.split("/", 1)[0]
            source_areas = {
                "device_search", "features", "model", "objective", "nn_training",
                "training_session", "training_data", "frozen_data", "run_config",
                "cuda", "torch_device", "bellman", "stage_profile", "device_state",
                "device_feature_plan", "checkpoint", "state", "alphabet",
            }
            if top in source_areas:
                return top
            if top.endswith(".rs"):
                root = top.removesuffix(".rs")
                if root in source_areas:
                    return f"{root}_root"
            return "native_source_other"
    if "/.cargo/registry/" in filename or "/cargo/registry/" in filename:
        return "dependency_source"
    return "unknown"


def _diagnostic_spans(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect source spans from the main diagnostic, children, and macro expansion."""
    found: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = [message]
    while pending:
        node = pending.pop(0)
        spans = node.get("spans")
        if isinstance(spans, list):
            for span in spans:
                if not isinstance(span, dict):
                    continue
                found.append(span)
                expansion = span.get("expansion")
                if isinstance(expansion, dict):
                    expanded_span = expansion.get("span")
                    if isinstance(expanded_span, dict):
                        found.append(expanded_span)
        children = node.get("children")
        if isinstance(children, list):
            pending.extend(child for child in children if isinstance(child, dict))
    return found


def _diagnostic_area(message: dict[str, Any], code: str) -> str:
    spans = _diagnostic_spans(message)
    ordered = [span for span in spans if span.get("is_primary") is True]
    ordered.extend(span for span in spans if span.get("is_primary") is not True)
    for span in ordered:
        area = _area_for_span(span)
        if area != "unknown":
            return area
    # E0425 is a stable Rust error class. Keep the missing identifier private.
    message_text = str(message.get("message", "")).lower()
    if code == "E0425" or message_text.startswith("cannot find "):
        return "unresolved_name"
    return "rustc_without_source_span"


def _cargo_failure_area(cargo_json: str) -> str:
    text = cargo_json.lower()
    if "linking with" in text and "failed" in text:
        return "linker"
    if "undefined reference" in text or "undefined symbol" in text:
        return "linker_symbol"
    if "no space left on device" in text or "disk quota exceeded" in text:
        return "disk_full"
    if "signal 9" in text or "process didn't exit successfully" in text and "killed" in text:
        return "process_killed"
    if "could not find bin" in text or "no bin target" in text or "no test target" in text:
        return "target_not_found"
    if "error:" in text:
        return "cargo_error_without_rustc_code"
    return "cargo_failed_unknown"


def compile_evidence(
    cargo_json: str,
    cargo_exit: int,
    artifact_kind: str,
    target_name: str | None = None,
) -> dict[str, str | None]:
    executables: list[str] = []
    errors: list[tuple[str | None, str]] = []
    for line in cargo_json.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if item.get("reason") == "compiler-artifact":
            target = item.get("target")
            executable = item.get("executable")
            if (
                isinstance(target, dict)
                and isinstance(executable, str)
                and executable
                and _target_matches(target, target_name, artifact_kind)
            ):
                executables.append(executable)
        elif item.get("reason") == "compiler-message":
            message = item.get("message")
            if not isinstance(message, dict) or message.get("level") != "error":
                continue
            code = message.get("code")
            code_value = code.get("code") if isinstance(code, dict) else None
            if code_value is not None and not (
                isinstance(code_value, str) and RUSTC_CODE.fullmatch(code_value)
            ):
                code_value = None
            if code_value is not None or message.get("level") == "error":
                area = _diagnostic_area(message, str(code_value or ""))
                errors.append((code_value, area))

    unique_executables = sorted(set(executables))
    if cargo_exit == 0 and len(unique_executables) == 1:
        return {
            "status": "compiled",
            "rustc_code": "none",
            "executable": unique_executables[0],
        }
    if errors:
        return {
            "status": f"compile_failed_{errors[0][1]}",
            "rustc_code": errors[0][0] or "none",
            "executable": None,
        }
    if cargo_exit == 0:
        status = "compile_failed_artifact_missing" if not unique_executables else "compile_failed_artifact_ambiguous"
    else:
        status = f"compile_failed_{_cargo_failure_area(cargo_json)}"
    return {
        "status": status,
        "rustc_code": "none",
        "executable": None,
    }


def gate_receipt(outcomes: dict[str, str]) -> dict[str, Any]:
    expected = {
        "torch": "success",
        "v13_compile": "compiled",
        "v13": "passed",
        "resume_compile": "compiled",
        "resume": "passed",
        "profile_compile": "compiled",
        "profile": "passed",
    }
    checks = {name: outcomes.get(name, "not_run") for name in expected}
    return {
        "status": "passed" if all(checks[name] == expected[name] for name in expected) else "failed",
        "outcomes": checks,
    }


def run_named_tests(executable: str, test_names: list[str]) -> dict[str, str]:
    path = pathlib.Path(executable)
    if not path.is_file():
        return {"status": "failed", "code": "binary_missing"}
    try:
        listing = subprocess.run(
            [str(path), "--list"], capture_output=True, text=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "code": "test_list_timeout"}
    if listing.returncode != 0:
        return {"status": "failed", "code": "test_list_failed"}
    available = {
        line.rsplit(": ", 1)[0]
        for line in listing.stdout.splitlines()
        if line.endswith(": test")
    }
    if any(name not in available for name in test_names):
        return {"status": "failed", "code": "expected_test_missing"}
    for name in test_names:
        try:
            result = subprocess.run(
                [str(path), "--exact", name],
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "failed", "code": "test_timeout"}
        if result.returncode != 0 or f"test {name} ... ok" not in result.stdout:
            return {"status": "failed", "code": "test_failed"}
    return {"status": "passed", "code": "none"}


def _emit(values: dict[str, str], output: str | None) -> None:
    rendered = "\n".join(f"{key}={value}" for key, value in values.items())
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


def command_compile(args: argparse.Namespace) -> int:
    cargo_text = pathlib.Path(args.cargo_json).read_text(encoding="utf-8", errors="replace")
    evidence = compile_evidence(
        cargo_text, args.cargo_exit, args.artifact_kind, args.target_name
    )
    binary_file = pathlib.Path(args.binary_file)
    if evidence["status"] == "compiled":
        binary_file.write_text(str(evidence["executable"]), encoding="utf-8")
    else:
        binary_file.unlink(missing_ok=True)
    _emit(
        {"status": str(evidence["status"]), "rustc_code": str(evidence["rustc_code"])},
        args.output,
    )
    return 0


def command_run_tests(args: argparse.Namespace) -> int:
    try:
        executable = pathlib.Path(args.binary_file).read_text(encoding="utf-8").strip()
    except OSError:
        result = {"status": "failed", "code": "binary_missing"}
    else:
        result = run_named_tests(executable, args.tests)
    _emit(result, args.output)
    return 0


def command_gate(args: argparse.Namespace) -> int:
    receipt = gate_receipt(
        {
            "torch": args.torch,
            "v13_compile": args.v13_compile,
            "v13": args.v13,
            "resume_compile": args.resume_compile,
            "resume": args.resume,
            "profile_compile": args.profile_compile,
            "profile": args.profile,
        }
    )
    _emit({"status": receipt["status"]}, args.output)
    return 0 if receipt["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--cargo-json", required=True)
    compile_parser.add_argument("--cargo-exit", required=True, type=int)
    compile_parser.add_argument("--artifact-kind", choices=("lib_test", "test", "bin"), required=True)
    compile_parser.add_argument("--target-name")
    compile_parser.add_argument("--binary-file", required=True)
    compile_parser.add_argument("--output", default=os.environ.get("GITHUB_OUTPUT"))
    compile_parser.set_defaults(run=command_compile)

    tests_parser = subparsers.add_parser("run-tests")
    tests_parser.add_argument("--binary-file", required=True)
    tests_parser.add_argument("--tests", nargs="+", required=True)
    tests_parser.add_argument("--output", default=os.environ.get("GITHUB_OUTPUT"))
    tests_parser.set_defaults(run=command_run_tests)

    gate_parser = subparsers.add_parser("gate")
    for name in (
        "torch",
        "v13",
        "resume",
        "profile",
        "v13-compile",
        "resume-compile",
        "profile-compile",
    ):
        gate_parser.add_argument("--" + name, required=True)
    gate_parser.add_argument("--output", default=os.environ.get("GITHUB_OUTPUT"))
    gate_parser.set_defaults(run=command_gate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
