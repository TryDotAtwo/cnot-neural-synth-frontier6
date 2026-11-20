#!/usr/bin/env python3
"""Build a pinned 2x3060 full-model run manifest from an explicit feature profile.

The profile is source-controlled and passed in by path. This tool never invents
feature contexts from matrix width.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

WIDTHS = (8, 10, 13, 25, 26, 34, 45, 50, 58, 72, 90, 108, 122, 144)
MODEL_INPUT_COUNT = 102
TARGET_ATTEMPTS = 250_000
INVOCATION_ATTEMPTS = TARGET_ATTEMPTS
BATCH_SIZE = 4
CHECKPOINT_EVERY = 2_500
PROFILE_SCHEMA = "cnot103_feature_context_profile_v2"
SOURCE_REPOSITORY = "TryDotAtwo/cnot-neural-synth-frontier6"
RUN_SCHEMA = "cnot103_training_run_v1"
FEATURE_SCHEMA = "runtime_packet_layout_v25"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_SOURCE_BYTES = 2 * 1024 * 1024

_CONTEXT_KEYS = {
    "max_n",
    "labeled_cuts",
    "restricted_intersections",
    "rank_increment_probes",
    "alphabet",
    "selected_residual_gates",
    "block_parity_contexts",
    "max_parity_patterns",
    "max_parity_pairs",
    "matrix_boundaries",
    "max_boundary_ports",
    "max_boundary_equations",
    "paid_basis_contexts",
    "paid_shear_config",
    "template_response",
    "bilinear_contexts",
    "shared_relation_contexts",
    "frequency_contexts",
    "positive_packet_contexts",
    "i09_contexts",
    "i09_limits",
    "i10_config",
}


def _all_to_all_alphabet(n: int) -> dict[str, Any]:
    gates = [
        {"control": control, "target": target}
        for control in range(n)
        for target in range(n)
        if control != target
    ]
    return {
        "dimension": n,
        "id": f"all_to_all_unit_cnot:{n}",
        "gates": gates,
        "costs": [1] * len(gates),
    }


def _json_exact(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int equality shortcut."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _json_exact(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _validate_objective_compatible_alphabet(alphabet: object) -> None:
    if not isinstance(alphabet, dict):
        raise ValueError("feature context alphabet")
    dimension = alphabet.get("dimension")
    if type(dimension) is not int or dimension not in WIDTHS:
        raise ValueError("feature context alphabet dimension is not an objective width")
    if not _json_exact(alphabet, _all_to_all_alphabet(dimension)):
        raise ValueError("feature context alphabet must exactly match an objective alphabet")


def fetch_source_file(revision: str, source_path: str) -> bytes:
    url = (
        f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/"
        f"{quote(revision, safe='')}/{quote(source_path, safe='/')}"
    )
    request = Request(
        url,
        headers={"User-Agent": "TheQuantumProject-FeatureProfilePin/1.0"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https" or final_url.hostname != "raw.githubusercontent.com":
                raise ValueError("profile source redirect is invalid")
            payload = response.read(MAX_SOURCE_BYTES + 1)
    except ValueError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError):
        raise ValueError("profile source is unavailable") from None
    if not payload or len(payload) > MAX_SOURCE_BYTES:
        raise ValueError("profile source size is invalid")
    return payload


def validate_profile_source(profile: object, source_fetcher=None) -> None:
    validate_profile(profile)
    source = profile["source"]
    fetcher = source_fetcher or fetch_source_file
    try:
        payload = fetcher(source["revision"], source["path"])
    except ValueError:
        raise
    except Exception:
        raise ValueError("profile source is unavailable") from None
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_SOURCE_BYTES:
        raise ValueError("profile source size is invalid")
    if hashlib.sha256(payload).hexdigest() != source["sha256"]:
        raise ValueError("profile source checksum differs")


def validate_profile(profile: object) -> dict[str, Any]:
    if not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError("profile schema")
    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or _PROFILE_ID.fullmatch(profile_id) is None:
        raise ValueError("profile identity")
    source = profile.get("source")
    if not isinstance(source, dict):
        raise ValueError("profile source")
    source_repository = source.get("repository")
    if source_repository != SOURCE_REPOSITORY:
        raise ValueError("profile source repository")
    revision = source.get("revision")
    source_path = source.get("path")
    source_sha256 = source.get("sha256")
    if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
        raise ValueError("profile source revision")
    if not isinstance(source_sha256, str) or _SHA256.fullmatch(source_sha256) is None:
        raise ValueError("profile source sha256")
    if (
        not isinstance(source_path, str)
        or not source_path
        or source_path.startswith("/")
        or "\\" in source_path
        or any(part in {"", ".", ".."} for part in source_path.split("/"))
    ):
        raise ValueError("profile source path")
    contexts = profile.get("feature_contexts")
    if not isinstance(contexts, dict) or not _CONTEXT_KEYS.issubset(contexts):
        raise ValueError("feature context fields")
    if contexts.get("max_n") != max(WIDTHS):
        raise ValueError("feature context maximum width")
    _validate_objective_compatible_alphabet(contexts.get("alphabet"))
    additional = contexts.get("selected_residual_additional_contexts", [])
    if not isinstance(additional, list):
        raise ValueError("additional selected-residual contexts")
    for context in additional:
        if not isinstance(context, dict) or set(context) != {"alphabet", "gates"}:
            raise ValueError("additional selected-residual context fields")
        if not isinstance(context["gates"], list):
            raise ValueError("additional selected-residual gates")
        _validate_objective_compatible_alphabet(context["alphabet"])
    return contexts


def context_widths(contexts: dict[str, Any]) -> dict[str, list[int]]:
    """Summarize explicitly configured widths; absent widths remain missing-token cases."""
    result: dict[str, list[int]] = {}
    for key in (
        "labeled_cuts",
        "restricted_intersections",
        "rank_increment_probes",
        "block_parity_contexts",
        "matrix_boundaries",
        "bilinear_contexts",
        "shared_relation_contexts",
        "frequency_contexts",
        "positive_packet_contexts",
        "i09_contexts",
    ):
        rows = contexts.get(key)
        if isinstance(rows, list):
            result[key] = sorted(
                {row["n"] for row in rows if isinstance(row, dict) and type(row.get("n")) is int}
            )
    paid_basis = contexts.get("paid_basis_contexts")
    if isinstance(paid_basis, list):
        result["paid_basis_contexts"] = sorted(
            {
                row["dimension"]
                for row in paid_basis
                if isinstance(row, dict) and type(row.get("dimension")) is int
            }
        )
    alphabet = contexts.get("alphabet")
    if isinstance(alphabet, dict) and type(alphabet.get("dimension")) is int:
        result["selected_residual_widths"] = [alphabet["dimension"]]
    additional = contexts.get("selected_residual_additional_contexts")
    if isinstance(additional, list):
        widths = set(result.get("selected_residual_widths", []))
        for row in additional:
            if isinstance(row, dict) and isinstance(row.get("alphabet"), dict):
                dimension = row["alphabet"].get("dimension")
                if type(dimension) is int:
                    widths.add(dimension)
        result["selected_residual_widths"] = sorted(widths)
    i10 = contexts.get("i10_config")
    if isinstance(i10, dict) and type(i10.get("max_n")) is int:
        result["i10_max_n"] = [i10["max_n"]]
    return result


def _validate_run_inputs(
    dataset_path: object,
    dataset_sha256: object,
    checkpoint_dir: object,
    resume: object,
) -> None:
    if not isinstance(dataset_path, str) or not dataset_path or "\x00" in dataset_path:
        raise ValueError("dataset path")
    if not isinstance(dataset_sha256, str) or _SHA256.fullmatch(dataset_sha256) is None:
        raise ValueError("dataset SHA-256")
    if not isinstance(checkpoint_dir, str) or not checkpoint_dir or "\x00" in checkpoint_dir:
        raise ValueError("checkpoint directory")
    if type(resume) is not bool:
        raise ValueError("resume flag")


def build_manifest(
    profile: dict[str, Any],
    profile_bytes: bytes,
    dataset_path: str,
    dataset_sha256: str,
    checkpoint_dir: str,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contexts = validate_profile(profile)
    _validate_run_inputs(dataset_path, dataset_sha256, checkpoint_dir, resume)
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "dataset": {"path": dataset_path, "sha256": dataset_sha256},
        "checkpoint_dir": checkpoint_dir,
        "resume": resume,
        "training": {
            "seed": 202611200001,
            "cuda_device": 0,
            "cuda_devices": [0, 1],
            "attempts": INVOCATION_ATTEMPTS,
            "target_attempts": TARGET_ATTEMPTS,
            "batch_size": BATCH_SIZE,
            "checkpoint_every_attempts": CHECKPOINT_EVERY,
        },
        "feature_policy": {
            "schema_id": FEATURE_SCHEMA,
            "solver_assisted": True,
            "evaluations": None,
            "max_bytes": None,
        },
        "model": {
            "wire_dim": 744,
            "pair_dim": 186,
            "structural_blocks": 8,
            "attention_heads": 8,
            "feedforward_dim": 1488,
            "fusion_blocks": 2,
            "basis_proposal_hidden_dim": 256,
        },
        "optimizer": {
            "learning_rate": 0.0001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 0.00000001,
            "weight_decay": 0.01,
        },
        "objective": {
            "bellman_weight": 1.0,
            "supervision_weight": 1.0,
            "qhat_weight": 0.1,
            "teacher_weight": 0.0,
            "neighbor_chunk_size": 8,
            "sampled_neighbor_count": 4,
            "adaptive_neighbor_chunk": True,
            "share_candidate_wire_projections": True,
            "share_training_wire_projections_min_width": 8,
            "cache_root_f08": True,
            "sequential_batch_gradients": True,
            "prevalidated_device_closure_audit": False,
            "teacher_temperature": 4.0,
        },
        "alphabets": [_all_to_all_alphabet(n) for n in WIDTHS],
        "feature_contexts": contexts,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    provenance = {
        "schema": "cnot103_training_manifest_provenance_v1",
        "profile_id": profile["profile_id"],
        "profile_source_repository": profile["source"]["repository"],
        "profile_source_revision": profile["source"]["revision"],
        "profile_source_path": profile["source"]["path"],
        "profile_source_sha256": profile["source"]["sha256"],
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "context_widths": context_widths(contexts),
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_input_slots": MODEL_INPUT_COUNT,
        "target_attempts": TARGET_ATTEMPTS,
    }
    return manifest, provenance


def _write_private_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    temporary = pathlib.Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _self_test_profile() -> dict[str, Any]:
    return {
        "schema": PROFILE_SCHEMA,
        "profile_id": "self-test",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": "a" * 40,
            "path": "native/tests/support/feature_contexts.rs",
            "sha256": "c" * 64,
        },
        "feature_contexts": {
            "max_n": 144,
            "labeled_cuts": [{"n": 3, "a": [0]}],
            "restricted_intersections": [
                {"n": 3, "left": [0], "right": [1], "coordinates": [0]}
            ],
            "rank_increment_probes": [
                {"n": 3, "span_rows": [0], "coordinates": [0], "candidate_support": []}
            ],
            "alphabet": _all_to_all_alphabet(8),
            "selected_residual_gates": [{"control": 0, "target": 1}],
            "block_parity_contexts": [{"n": 3, "sources": [0], "targets": [1]}],
            "max_parity_patterns": 8,
            "max_parity_pairs": 8,
            "matrix_boundaries": [
                {
                    "n": 3,
                    "inputs": [0],
                    "outputs": [2],
                    "protected_inputs": [],
                    "zero_outputs": [],
                    "weights": [],
                }
            ],
            "max_boundary_ports": 8,
            "max_boundary_equations": 1024,
            "paid_basis_contexts": [
                {"dimension": 3, "source_wires": [0], "target_wires": [1]}
            ],
            "paid_shear_config": {"local_width": 2, "sweeps": 1},
            "template_response": {
                "choices": [
                    {"kind": "add", "control": 0, "target": 1},
                    {"kind": "swap", "control": 1, "target": 2},
                ],
                "points": [
                    [
                        {"numerator": 1, "denominator": 2},
                        {"numerator": 1, "denominator": 2},
                    ]
                ],
                "budgets": {
                    "max_width": 3,
                    "max_assignments": 16,
                    "max_evaluations": 128,
                    "max_bytes": 65536,
                },
            },
            "bilinear_contexts": [
                {
                    "n": 3,
                    "rows": [0],
                    "columns": [1],
                    "hidden_width": 1,
                    "u_rows": [[1]],
                    "v_rows": [[1]],
                    "q": 1,
                    "a0_rows": [[0]],
                    "d0_rows": [[0]],
                }
            ],
            "shared_relation_contexts": [
                {"n": 3, "rows": [0], "coordinates": [1], "covectors": [[1]]}
            ],
            "frequency_contexts": [
                {"n": 3, "sources": [0], "internal": [1], "targets": [2], "central_rows": [[1]]}
            ],
            "positive_packet_contexts": [
                {"n": 3, "gates": [[0, 1]], "factors": [{"scope": [0], "allowed": ["0", "1"]}]}
            ],
            "i09_contexts": [
                {
                    "n": 3,
                    "horizon": 1,
                    "cut": 0,
                    "prefix_bits": [],
                    "allowed_gates": [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
                }
            ],
            "i09_limits": {"max_variables": 1024, "max_equations": 4096, "max_bytes": 1048576},
            "i10_config": {
                "max_n": 3,
                "max_degree": 1,
                "max_layer_states": 4096,
                "max_transitions": 65536,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            if any((args.profile, args.dataset_path, args.dataset_sha256, args.checkpoint_dir, args.resume)):
                raise ValueError("self-test arguments")
            profile = _self_test_profile()
            profile_bytes = json.dumps(profile, sort_keys=True).encode("utf-8")
            dataset_path = "/workspace/self-test/frozen_train_10k.json"
            dataset_sha = "b" * 64
            checkpoint_dir = "/workspace/self-test/checkpoints"
            resume = False
        else:
            if not all((args.profile, args.dataset_path, args.dataset_sha256, args.checkpoint_dir)):
                raise ValueError("required arguments")
            profile_path = pathlib.Path(args.profile)
            profile_bytes = profile_path.read_bytes()
            if not profile_bytes or len(profile_bytes) > (4 << 20):
                raise ValueError("profile size")
            profile = json.loads(profile_bytes)
            validate_profile(profile)
            try:
                validate_profile_source(profile)
            except ValueError:
                print("manifest_build=source_unverified")
                return 2
            dataset_path = args.dataset_path
            dataset_sha = args.dataset_sha256
            checkpoint_dir = args.checkpoint_dir
            resume = args.resume

        manifest, provenance = build_manifest(
            profile,
            profile_bytes,
            dataset_path,
            dataset_sha,
            checkpoint_dir,
            resume,
        )
        output_path = pathlib.Path(args.output)
        provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
        _write_private_json(output_path, manifest)
        _write_private_json(provenance_path, provenance)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        print("manifest_build=invalid_profile")
        return 2

    print("manifest_build=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
