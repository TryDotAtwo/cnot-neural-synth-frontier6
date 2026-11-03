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
import pathlib
import tempfile
import sys
import zipfile
from collections import deque
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "packages/v13_recovery_candidate_v17_20261001.zip"
OUTPUT = pathlib.Path(__file__).resolve().with_name("python_structural_33.json")
OUTPUT_SUBSPACE = pathlib.Path(__file__).resolve().with_name("python_subspace_views_6.json")
ARCHIVE_SHA256 = "9947df2dc27080b38eb34e7551be39ce0536f2409aa733c5f201c317ed1a4866"
STRUCTURAL_PATH = "research_v3/features/structural.py"
SUBSPACE_PATH = "research_v3/features/subspace_views.py"
GF2_PATH = "research_v3/gf2.py"
PACKET_PATH = "research_v3/features/packet.py"
SCHEMA_PATH = "research_v3/configs/feature_schema.json"
SUBSPACE_SHA256 = "78c06d9e542016f4cb0875f7a1a9807e8a638de6eebeddc7aadff57cd3084a99"
GF2_SHA256 = "223d7bc72029457e8119946c6cd320c95063559ff63a61d1c67a9e7293bc21f2"
PACKET_SHA256 = "b4e8b22065a7ec45a2de8c2b4d2a5e75f527abf4e131a7b9409951d1fa7341f1"
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
            class CaptureRegistry:
                def __init__(self):
                    self.producers = {}
                def register(self, feature_spec, producer):
                    self.producers[feature_spec.id] = producer
            capture = CaptureRegistry()
            subspace_specs = subspace_module.register_subspace_views(capture)
        finally:
            sys.path.remove(temp)
        expected_subspace = ["C01", "D03", "H01", "H02", "H03", "H04"]
        if list(subspace_specs) != expected_subspace:
            raise RuntimeError("pinned subspace module declared unexpected feature IDs")
        return (
            schema, implemented, module, subspace_specs, capture.producers,
            sha256(schema_bytes), sha256(source_bytes),
        )


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
    schema, entries, oracle, subspace_specs, subspace_producers, schema_sha, source_sha = load_oracle()
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
            stacked = __import__("numpy").stack(arrays, axis=0).astype("float32", copy=False)
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
            stacked = __import__("numpy").stack(arrays, axis=0).astype("float32", copy=False)
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
    OUTPUT.write_text(compact_json(fixture) + "\n", encoding="utf-8")
    OUTPUT_SUBSPACE.write_text(compact_json(subspace_fixture) + "\n", encoding="utf-8")
    matrix_count = sum(batch["state_count"] for batch in batches)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({matrix_count} matrices, {len(entries)} structural features)")
    print(f"wrote {OUTPUT_SUBSPACE.relative_to(ROOT)} ({matrix_count} matrices, {len(subspace_specs)} subspace features)")


if __name__ == "__main__":
    main()
