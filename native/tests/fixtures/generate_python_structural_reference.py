#!/usr/bin/env python3
"""Regenerate the pinned Python-v17 structural and subspace parity fixtures.

Run from the repository root with NumPy available:
    python native/tests/fixtures/generate_python_structural_reference.py
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import numpy as np
import pathlib
import tempfile
import sys
import zipfile
from collections import deque
from dataclasses import dataclass
from random import Random
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "packages/v13_recovery_candidate_v17_20261001.zip"
OUTPUT = pathlib.Path(__file__).resolve().with_name("python_structural_33.json")
OUTPUT_SUBSPACE = pathlib.Path(__file__).resolve().with_name("python_subspace_views_6.json")
OUTPUT_BASIS = pathlib.Path(__file__).resolve().with_name("python_basis_gradient_3.json")
ARCHIVE_SHA256 = "9947df2dc27080b38eb34e7551be39ce0536f2409aa733c5f201c317ed1a4866"
STRUCTURAL_PATH = "research_v3/features/structural.py"
SUBSPACE_PATH = "research_v3/features/subspace_views.py"
GF2_PATH = "research_v3/gf2.py"
PACKET_PATH = "research_v3/features/packet.py"
BASIS_GRADIENT_PATH = "research_v3/features/basis_gradient.py"
BASIS_MACROS_PATH = "research_v3/search/basis_macros.py"
SCHEMA_PATH = "research_v3/configs/feature_schema.json"
SUBSPACE_SHA256 = "78c06d9e542016f4cb0875f7a1a9807e8a638de6eebeddc7aadff57cd3084a99"
GF2_SHA256 = "223d7bc72029457e8119946c6cd320c95063559ff63a61d1c67a9e7293bc21f2"
PACKET_SHA256 = "b4e8b22065a7ec45a2de8c2b4d2a5e75f527abf4e131a7b9409951d1fa7341f1"
BASIS_GRADIENT_SHA256 = "d450f15ed32454bff2dd069c66ccf3cb6395667f8828e1a563d522006bbc96e2"
BASIS_MACROS_SHA256 = "ed34348949806453dd1c70daf957ee071c1cd445de787b70dc5346dd38840b53"

@dataclass(frozen=True)
class OracleState:
    n: int
    packed_rows: tuple[int, ...]


class CaptureRegistry:
    def __init__(self):
        self.producers = {}

    def register(self, feature_spec, producer, **_kwargs):
        self.producers[feature_spec.id] = producer


WIDTH4_PATHS = [
    ("identity4", []),
    ("triangular_chain4", [[0, 1], [1, 2], [2, 3]]),
    ("reciprocal_chain4", [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]),
    ("dense_cycle4", [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]),
    ("seeded_walk4", [
        [3, 2], [1, 2], [0, 2], [0, 2], [3, 1], [2, 3], [0, 1], [0, 3],
        [1, 2], [0, 3], [1, 0], [1, 2], [2, 0], [3, 2], [3, 0], [1, 0],
        [2, 0], [1, 2], [3, 1], [2, 0], [3, 2], [1, 2], [2, 0], [3, 0],
        [1, 3], [2, 0], [3, 1], [0, 2], [0, 1], [3, 2], [0, 2],
    ]),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def apply_path(n: int, path: list[list[int]]) -> tuple[int, ...]:
    rows = [1 << wire for wire in range(n)]
    for control, target in path:
        if control == target or not (0 <= control < n and 0 <= target < n):
            raise ValueError(f"invalid CNOT ({control}, {target}) at width {n}")
        rows[target] ^= rows[control]
    return tuple(rows)


def gl3_states() -> list[tuple[tuple[int, ...], list[list[int]]]]:
    identity = (1, 2, 4)
    paths = {identity: []}
    queue = deque([identity])
    gates = [(control, target) for control in range(3) for target in range(3) if control != target]
    while queue:
        rows = queue.popleft()
        for control, target in gates:
            next_rows = list(rows)
            next_rows[target] ^= next_rows[control]
            state = tuple(next_rows)
            if state not in paths:
                paths[state] = paths[rows] + [[control, target]]
                queue.append(state)
    if len(paths) != 168:
        raise RuntimeError(f"GL(3,2) enumeration produced {len(paths)} matrices, expected 168")
    return sorted(paths.items())


def load_oracle():
    archive_bytes = ARCHIVE.read_bytes()
    if sha256(archive_bytes) != ARCHIVE_SHA256:
        raise RuntimeError("pinned Python-v17 source archive SHA-256 mismatch")
    with zipfile.ZipFile(ARCHIVE) as archive:
        schema_bytes = archive.read(SCHEMA_PATH)
        source_bytes = archive.read(STRUCTURAL_PATH)
        subspace_bytes = archive.read(SUBSPACE_PATH)
        gf2_bytes = archive.read(GF2_PATH)
        packet_bytes = archive.read(PACKET_PATH)
        basis_gradient_bytes = archive.read(BASIS_GRADIENT_PATH)
        basis_macros_bytes = archive.read(BASIS_MACROS_PATH)
    schema = json.loads(schema_bytes)
    if sha256(schema_bytes) != "3b4063047ccc9d337e74d92de90108653ac5cf9da166733eba2f4c429c48c0b8":
        raise RuntimeError("pinned feature schema manifest SHA-256 mismatch")
    if sha256(source_bytes) != "fe1c479ceef39c6b8a64f078a6166993d31ef833c43f7ad78b5ed481f4821049":
        raise RuntimeError("pinned structural producer SHA-256 mismatch")
    if sha256(subspace_bytes) != SUBSPACE_SHA256:
        raise RuntimeError("pinned subspace-view producer SHA-256 mismatch")
    if sha256(gf2_bytes) != GF2_SHA256:
        raise RuntimeError("pinned GF(2) helper SHA-256 mismatch")
    if sha256(packet_bytes) != PACKET_SHA256:
        raise RuntimeError("pinned Python feature packet SHA-256 mismatch")
    if sha256(basis_gradient_bytes) != BASIS_GRADIENT_SHA256:
        raise RuntimeError("pinned Python basis-gradient producer SHA-256 mismatch")
    if sha256(basis_macros_bytes) != BASIS_MACROS_SHA256:
        raise RuntimeError("pinned Python basis-macro helper SHA-256 mismatch")
    implemented = [entry for entry in schema["entries"] if entry["implemented"]]
    expected = [f"A{i:02}" for i in range(1, 21)]
    expected += [f"B{i:02}" for i in range(1, 13)] + ["F01"]
    if [entry["id"] for entry in implemented] != expected:
        raise RuntimeError("the pinned schema no longer declares the expected 33 source-backed IDs")

    with tempfile.TemporaryDirectory(prefix="tqp-python-v17-") as temp:
        source_path = pathlib.Path(temp) / "structural.py"
        source_path.write_bytes(source_bytes)
        spec = importlib.util.spec_from_file_location("tqp_python_v17_structural", source_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(temp)
        sys.path.insert(0, temp)
        try:
            subspace_module = importlib.import_module("research_v3.features.subspace_views")
            subspace_capture = CaptureRegistry()
            subspace_specs = subspace_module.register_subspace_views(subspace_capture)
            basis_module = importlib.import_module("research_v3.features.basis_gradient")
            basis_capture = CaptureRegistry()
            basis_specs = basis_module.register_basis_gradient(basis_capture)
        finally:
            sys.path.remove(temp)
        expected_subspace = ["C01", "D03", "H01", "H02", "H03", "H04"]
        if list(subspace_specs) != expected_subspace:
            raise RuntimeError("pinned subspace module declared unexpected feature IDs")
        return (
            schema, implemented, module, subspace_specs, subspace_capture.producers,
            basis_specs, basis_capture.producers, sha256(schema_bytes), sha256(source_bytes),
        )


def basis_gradient_batches(specs, producers):
    feature_ids = list(specs)
    if feature_ids != ["E01", "E04", "E05"]:
        raise RuntimeError(f"pinned basis-gradient module declared unexpected IDs: {feature_ids}")
    batches = []
    target_count = 32

    for n in (4, 5, 6):
        accepted = []
        seen = set()
        candidate_count = 0
        unsupported_count = 0

        def consider(name, path):
            nonlocal candidate_count, unsupported_count
            rows = apply_path(n, path)
            if rows in seen:
                return
            seen.add(rows)
            candidate_count += 1
            state = OracleState(n, rows)
            results = [producers[feature_id](state, None) for feature_id in feature_ids]
            unsupported = [isinstance(result, str) for result in results]
            if any(unsupported):
                if not all(unsupported) or any(result != "unsupported" for result in results):
                    raise RuntimeError("basis-gradient providers disagree on applicability")
                unsupported_count += 1
                return
            accepted.append({
                "name": name,
                "n": n,
                "packed_rows": list(rows),
                "cnot_path": path,
                "oracle_state": state,
            })

        if n == 4:
            for original_name, path in WIDTH4_PATHS:
                consider(f"basis_n4_{original_name}", path)

        rng = Random(20261103 + n)
        gates = [
            (control, target)
            for control in range(n)
            for target in range(n)
            if control != target
        ]
        for attempt in range(20000):
            if len(accepted) >= target_count:
                break
            length = rng.randint(2 * n, 5 * n)
            path = [list(rng.choice(gates)) for _ in range(length)]
            consider(f"basis_n{n}_seeded_{attempt:04}", path)
        if len(accepted) != target_count:
            raise RuntimeError(
                f"only generated {len(accepted)} applicable width-{n} states; expected {target_count}"
            )

        states = []
        for row in accepted:
            states.append({key: value for key, value in row.items() if key != "oracle_state"})
        feature_rows = []
        for feature_id in feature_ids:
            arrays = [
                producers[feature_id](row["oracle_state"], None)[0]
                for row in accepted
            ]
            first_shape = list(arrays[0].shape)
            if any(list(array.shape) != first_shape for array in arrays):
                raise RuntimeError(f"{feature_id} changes shape within width-{n} batch")
            stacked = np.stack(arrays, axis=0).astype("float32", copy=False)
            values = [float(value) for value in stacked.reshape(-1)]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"{feature_id} contains a non-finite value")
            spec = specs[feature_id]
            feature_rows.append({
                "id": feature_id,
                "role": spec.role,
                "channels": spec.channels,
                "shape": first_shape,
                "values": values,
            })
        batches.append({
            "n": n,
            "candidate_count": candidate_count,
            "unsupported_candidate_count": unsupported_count,
            "state_count": len(states),
            "states": states,
            "features": feature_rows,
        })
    return batches


def compact_json(value, level=0, key=None):
    if key in {"values", "shape", "packed_rows", "cnot_path"}:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if isinstance(value, dict):
        if not value:
            return "{}"
        pad, child = "  " * level, "  " * (level + 1)
        lines = [
            child + json.dumps(name, ensure_ascii=False) + ": " + compact_json(item, level + 1, name)
            for name, item in value.items()
        ]
        return "{\n" + ",\n".join(lines) + "\n" + pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        pad, child = "  " * level, "  " * (level + 1)
        return "[\n" + ",\n".join(child + compact_json(item, level + 1) for item in value) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def main():
    (schema, entries, oracle, subspace_specs, subspace_producers, basis_specs, basis_producers,
     schema_sha, source_sha) = load_oracle()
    batches = []
    subspace_batches = []
    inputs = [
        (3, [(f"gl3_{index:03}", rows, path) for index, (rows, path) in enumerate(gl3_states())]),
        (4, sorted(
            ((apply_path(4, path), path, name) for name, path in WIDTH4_PATHS),
            key=lambda row: row[0],
        )),
    ]
    for n, raw_states in inputs:
        states = []
        for index, item in enumerate(raw_states):
            if n == 3:
                name, rows, path = item
            else:
                rows, path, _original_name = item
                name = f"gl4_{index:03}"
            states.append({
                "name": name,
                "n": n,
                "packed_rows": list(rows),
                "cnot_path": path,
            })
        if len({tuple(row["packed_rows"]) for row in states}) != len(states):
            raise RuntimeError(f"duplicate width-{n} matrix in fixture inputs")

        per_state = [
            oracle.compute(SimpleNamespace(n=n, packed_rows=state["packed_rows"]))
            for state in states
        ]
        feature_rows = []
        for entry in entries:
            feature_id = entry["id"]
            arrays = [result[feature_id] for result in per_state]
            first_shape = list(arrays[0].shape)
            if any(list(array.shape) != first_shape for array in arrays):
                raise RuntimeError(f"{feature_id} changes shape within width-{n} batch")
            stacked = np.stack(arrays, axis=0).astype("float32", copy=False)
            values = [float(value) for value in stacked.reshape(-1)]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"{feature_id} contains a non-finite value")
            feature_rows.append({
                "id": feature_id,
                "role": entry["current_role"],
                "channels": entry["channels"],
                "shape": first_shape,
                "values": values,
            })
        subspace_feature_rows = []
        for feature_id, feature_spec in subspace_specs.items():
            arrays = [
                subspace_producers[feature_id](
                    SimpleNamespace(n=n, packed_rows=state["packed_rows"]), None
                )[0]
                for state in states
            ]
            first_shape = list(arrays[0].shape)
            if any(list(array.shape) != first_shape for array in arrays):
                raise RuntimeError(f"{feature_id} changes shape within width-{n} batch")
            stacked = np.stack(arrays, axis=0).astype("float32", copy=False)
            values = [float(value) for value in stacked.reshape(-1)]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"{feature_id} contains a non-finite value")
            subspace_feature_rows.append({
                "id": feature_id,
                "role": feature_spec.role,
                "channels": feature_spec.channels,
                "shape": first_shape,
                "values": values,
            })
        batches.append({
            "n": n,
            "state_count": len(states),
            "states": states,
            "features": feature_rows,
        })
        subspace_batches.append({
            "n": n,
            "state_count": len(states),
            "states": states,
            "features": subspace_feature_rows,
        })

    fixture = {
        "schema": "python_structural_reference_v2",
        "source_archive_sha256": ARCHIVE_SHA256,
        "feature_schema_source_sha256": schema["source_sha256"],
        "feature_schema_manifest_sha256": schema_sha,
        "python_structural_sha256": source_sha,
        "feature_ids": [entry["id"] for entry in entries],
        "batches": batches,
    }
    subspace_fixture = {
        "schema": "python_subspace_reference_v1",
        "source_archive_sha256": ARCHIVE_SHA256,
        "subspace_views_sha256": SUBSPACE_SHA256,
        "gf2_helper_sha256": GF2_SHA256,
        "feature_packet_sha256": PACKET_SHA256,
        "feature_ids": list(subspace_specs),
        "batches": subspace_batches,
    }
    basis_fixture = {
        "schema": "python_basis_gradient_reference_v1",
        "source_archive_sha256": ARCHIVE_SHA256,
        "basis_gradient_sha256": BASIS_GRADIENT_SHA256,
        "basis_macros_sha256": BASIS_MACROS_SHA256,
        "gf2_helper_sha256": GF2_SHA256,
        "feature_packet_sha256": PACKET_SHA256,
        "feature_ids": list(basis_specs),
        "batches": basis_gradient_batches(basis_specs, basis_producers),
    }
    OUTPUT.write_text(compact_json(fixture) + "\n", encoding="utf-8")
    OUTPUT_SUBSPACE.write_text(compact_json(subspace_fixture) + "\n", encoding="utf-8")
    OUTPUT_BASIS.write_text(compact_json(basis_fixture) + "\n", encoding="utf-8")
    matrix_count = sum(batch["state_count"] for batch in batches)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({matrix_count} matrices, {len(entries)} structural features)")
    basis_state_count = sum(batch["state_count"] for batch in basis_fixture["batches"])
    print(
        f"wrote {OUTPUT_SUBSPACE.relative_to(ROOT)} "
        f"({matrix_count} matrices, {len(subspace_specs)} subspace features)"
    )
    print(
        f"wrote {OUTPUT_BASIS.relative_to(ROOT)} "
        f"({basis_state_count} matrices, {len(basis_specs)} basis-gradient features)"
    )


if __name__ == "__main__":
    main()
