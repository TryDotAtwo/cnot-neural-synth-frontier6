from __future__ import annotations

"""Regenerate the pinned Python-v17 diffusion feature oracle."""

from collections import deque
from pathlib import Path
from random import Random
import importlib
import sys
import tempfile
import zipfile
from types import SimpleNamespace

import numpy as np
import scipy

from generate_python_structural_reference import (
    ARCHIVE,
    ARCHIVE_SHA256,
    OracleState,
    apply_path,
    compact_json,
    sha256,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().with_name("python_diffusion_10.json")
DIFFUSION_PATH = "research_v3/features/diffusion.py"
PROBES_PATH = "research_v3/diffusion/probes.py"
CONFIG = {
    "taus": (0.0, 0.25, 1.0, 4.0),
    "floor": 1e-30,
    "max_n": 16,
    "max_pair_states": 10_000,
    "mc_samples": 128,
    "mc_tau": 1.0,
}
MC_SEED = 91


class CaptureRegistry:
    def __init__(self):
        self.specs = {}
        self.producers = {}

    def register(self, spec, producer, *_args, **_kwargs):
        self.specs[spec.id] = spec
        self.producers[spec.id] = producer


def exhaustive_cnot_states(n):
    identity = tuple(1 << wire for wire in range(n))
    queue = deque([(identity, [])])
    seen = {identity}
    result = []
    gates = [(control, target) for control in range(n) for target in range(n) if control != target]
    while queue:
        rows, path = queue.popleft()
        result.append({
            "name": f"diffusion_gl{n}_{len(result):03}",
            "n": n,
            "packed_rows": list(rows),
            "cnot_path": path,
        })
        if n == 2 and len(result) == 6:
            break
        if n == 3 and len(result) == 168:
            break
        for control, target in gates:
            next_rows = list(rows)
            next_rows[target] ^= next_rows[control]
            frozen = tuple(next_rows)
            if frozen not in seen:
                seen.add(frozen)
                queue.append((frozen, path + [[control, target]]))
    expected = 6 if n == 2 else 168
    if len(result) != expected:
        raise RuntimeError(f"BFS generated {len(result)} states for GL({n},2), expected {expected}")
    return result


def sampled_cnot_states(n, count=12):
    result = [{
        "name": f"diffusion_n{n}_identity",
        "n": n,
        "packed_rows": [1 << wire for wire in range(n)],
        "cnot_path": [],
    }]
    seen = {tuple(result[0]["packed_rows"])}
    gates = [(control, target) for control in range(n) for target in range(n) if control != target]
    rng = Random(20261103 + n)
    for attempt in range(20_000):
        if len(result) >= count:
            break
        length = rng.randint(2 * n, 5 * n)
        path = [[*rng.choice(gates)] for _ in range(length)]
        rows = apply_path(n, path)
        if rows in seen:
            continue
        seen.add(rows)
        result.append({
            "name": f"diffusion_n{n}_seeded_{attempt:04}",
            "n": n,
            "packed_rows": list(rows),
            "cnot_path": path,
        })
    if len(result) != count:
        raise RuntimeError(f"generated only {len(result)} unique width-{n} states")
    return result


def make_feature_rows(feature_ids, states, specs, producers, seed):
    policy = SimpleNamespace(seed=seed)
    rows = []
    for feature_id in feature_ids:
        spec = specs[feature_id]
        arrays = []
        for fixture_state in states:
            state = OracleState(fixture_state["n"], tuple(fixture_state["packed_rows"]))
            output = producers[feature_id](state, policy)
            if output is None or isinstance(output, str):
                raise RuntimeError(f"{feature_id} unavailable for {fixture_state['name']}: {output}")
            array = output[0]
            if not np.isfinite(array).all():
                raise RuntimeError(f"{feature_id} has non-finite source values")
            arrays.append(array.astype(np.float32, copy=False))
        shape = list(arrays[0].shape)
        if any(list(array.shape) != shape for array in arrays):
            raise RuntimeError(f"{feature_id} shape changes within width {states[0]['n']}")
        values = np.stack(arrays, axis=0).reshape(-1)
        rows.append({
            "id": feature_id,
            "role": spec.role,
            "channels": spec.channels,
            "shape": shape,
            "values": [float(value) for value in values],
        })
    return rows


def main():
    archive_bytes = ARCHIVE.read_bytes()
    if sha256(archive_bytes) != ARCHIVE_SHA256:
        raise RuntimeError("pinned source archive SHA-256 mismatch")

    with zipfile.ZipFile(ARCHIVE) as archive:
        diffusion_sha = sha256(archive.read(DIFFUSION_PATH))
        probes_sha = sha256(archive.read(PROBES_PATH))
        with tempfile.TemporaryDirectory(prefix="tqp-diffusion-oracle-") as temp:
            archive.extractall(temp)
            sys.path.insert(0, temp)
            try:
                module = importlib.import_module("research_v3.features.diffusion")
                capture = CaptureRegistry()
                module.register_diffusion(capture, **CONFIG)
                feature_ids = list(capture.specs)
                expected_ids = [f"G{index:02}" for index in range(1, 11)]
                if feature_ids != expected_ids:
                    raise RuntimeError(f"unexpected diffusion IDs: {feature_ids}")

                batches = []
                for n in range(2, 7):
                    fixture_states = (
                        exhaustive_cnot_states(n)
                        if n in (2, 3)
                        else sampled_cnot_states(n)
                    )
                    exact_ids = feature_ids[:6]
                    mc_ids = feature_ids[6:]
                    exact_features = make_feature_rows(
                        exact_ids, fixture_states, capture.specs, capture.producers, 0
                    )
                    monte_carlo_features = make_feature_rows(
                        mc_ids, fixture_states, capture.specs, capture.producers, MC_SEED
                    )
                    batches.append({
                        "n": n,
                        "state_count": len(fixture_states),
                        "states": fixture_states,
                        "exact_features": exact_features,
                        "monte_carlo_features": monte_carlo_features,
                    })

                fixture = {
                    "schema": "python_diffusion_reference_v1",
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "diffusion_module_sha256": diffusion_sha,
                    "probes_module_sha256": probes_sha,
                    "numpy_version": np.__version__,
                    "scipy_version": scipy.__version__,
                    "feature_ids": feature_ids,
                    "config": {
                        "taus": list(CONFIG["taus"]),
                        "floor": CONFIG["floor"],
                        "max_n": CONFIG["max_n"],
                        "max_pair_states": CONFIG["max_pair_states"],
                        "mc_samples": CONFIG["mc_samples"],
                        "mc_tau": CONFIG["mc_tau"],
                    },
                    "mc_seed": MC_SEED,
                    "batches": batches,
                }
                OUTPUT.write_text(compact_json(fixture) + "\n", encoding="utf-8")
            finally:
                sys.path.remove(temp)

    print(
        f"wrote {OUTPUT.relative_to(ROOT)}: 6 exhaustive GL(2,2), "
        f"168 exhaustive GL(3,2), and 12 seeded states at each width 4-6"
    )


if __name__ == "__main__":
    main()
