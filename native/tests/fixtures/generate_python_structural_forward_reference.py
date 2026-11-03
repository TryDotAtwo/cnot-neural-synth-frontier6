#!/usr/bin/env python3
"""Generate a small full-forward oracle for all active Python-v17 features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "packages/v13_recovery_candidate_v17_20261001.zip"
OUTPUT = Path(__file__).resolve().with_name("python_structural_33_forward_small.json")
ARCHIVE_SHA256 = "9947df2dc27080b38eb34e7551be39ce0536f2409aa733c5f201c317ed1a4866"
SCHEMA_SHA256 = "3b4063047ccc9d337e74d92de90108653ac5cf9da166733eba2f4c429c48c0b8"
WEIGHT_SCHEME = "sorted_state_dict_mod257_v1"
CONFIG = {
    "d": 8,
    "pair_d": 4,
    "blocks": 1,
    "heads": 2,
    "ffn": 16,
    "fusion_blocks": 1,
    "basis_proposal_hidden_dim": 128,
}
SCHEMA_ID = "python-forward-parity-structural-33-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_weights(state_dict):
    with torch.no_grad():
        for tensor_index, name in enumerate(sorted(state_dict)):
            target = state_dict[name]
            count = target.numel()
            values = np.fromiter(
                (
                    ((tensor_index * 131 + flat_index * 17 + 43) % 257 - 128)
                    / 512.0
                    for flat_index in range(count)
                ),
                dtype=np.float32,
                count=count,
            ).reshape(tuple(target.shape))
            target.copy_(torch.from_numpy(values))


def main() -> None:
    archive_bytes = ARCHIVE.read_bytes()
    if sha256(archive_bytes) != ARCHIVE_SHA256:
        raise RuntimeError("pinned Python-v17 source archive SHA-256 mismatch")

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="tqp-python-v17-forward-") as directory:
        with zipfile.ZipFile(ARCHIVE) as archive:
            schema_bytes = archive.read("research_v3/configs/feature_schema.json")
            if sha256(schema_bytes) != SCHEMA_SHA256:
                raise RuntimeError("pinned source feature-schema hash mismatch")
            schema = json.loads(schema_bytes)
            archive.extractall(directory)

        sys.path.insert(0, directory)
        try:
            from research_v3.contracts import AlphabetSpec, MatrixBatch, MatrixState
            from research_v3.features.registry import FeaturePolicy, FeatureRegistry
            from research_v3.model.field import FusionDistanceField

            ids = [row["id"] for row in schema["entries"] if row["implemented"]]
            expected_ids = [f"A{i:02}" for i in range(1, 21)]
            expected_ids += [f"B{i:02}" for i in range(1, 13)] + ["F01"]
            if ids != expected_ids or len(ids) != 33:
                raise RuntimeError("Python-v17 active feature set changed")

            alphabet3 = AlphabetSpec.all_to_all(3)
            alphabet4 = AlphabetSpec.all_to_all(4)
            states = (
                MatrixState(3, (1, 2, 4), alphabet3.id),
                MatrixState(3, (1, 3, 7), alphabet3.id),
                MatrixState(4, (1, 2, 4, 8), alphabet4.id),
                MatrixState(4, (1, 3, 7, 15), alphabet4.id),
            )
            policy = FeaturePolicy(tuple(ids), schema_id=SCHEMA_ID)
            registry = FeatureRegistry()
            packet = registry.build(states, policy)
            specs = tuple(registry.specs[feature_id] for feature_id in ids)

            for row, spec in zip(
                (entry for entry in schema["entries"] if entry["implemented"]),
                specs,
                strict=True,
            ):
                if spec.role != row["current_role"] or spec.channels != row["channels"]:
                    raise RuntimeError(f"Python model schema drift for {spec.id}")

            model = FusionDistanceField(
                specs,
                d=CONFIG["d"],
                pair_d=CONFIG["pair_d"],
                blocks=CONFIG["blocks"],
                heads=CONFIG["heads"],
                ffn=CONFIG["ffn"],
                fusion_blocks=CONFIG["fusion_blocks"],
            )
            state_dict = model.state_dict()
            deterministic_weights(state_dict)
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            batch = MatrixBatch.from_states(states, device="cpu")
            with torch.no_grad():
                output = model(batch, packet)

            features = []
            for feature_id in ids:
                feature = packet.features[feature_id]
                values = np.asarray(feature.values, dtype=np.float32)
                if not np.isfinite(values).all():
                    raise RuntimeError(f"{feature_id} source values are non-finite")
                features.append({
                    "id": feature_id,
                    "role": feature.role,
                    "channels": int(values.shape[-1]),
                    "shape": list(values.shape),
                    "values": [float(value) for value in values.reshape(-1)],
                })

            actions = [
                [batch_index, control, target]
                for batch_index, state in enumerate(states)
                for control in range(state.n)
                for target in range(state.n)
                if control != target
            ]
            fixture = {
                "schema": "python_reference_forward_structural_33_v1",
                "reference_archive_sha256": ARCHIVE_SHA256,
                "feature_schema_manifest_sha256": SCHEMA_SHA256,
                "feature_ids": ids,
                "feature_values": features,
                "weight_scheme": WEIGHT_SCHEME,
                "config": CONFIG,
                "states": [
                    {"n": state.n, "packed_rows": list(state.packed_rows)}
                    for state in states
                ],
                "state_dict_shapes": {
                    name: list(tensor.shape)
                    for name, tensor in sorted(state_dict.items())
                },
                "expected": {
                    "allowed_actions": actions,
                    "value": output.value.detach().cpu().reshape(-1).tolist(),
                    "proposal": [
                        float(output.proposal[b, c, t])
                        for b, c, t in actions
                    ],
                    "qhat": [
                        float(output.qhat[b, c, t])
                        for b, c, t in actions
                    ],
                },
                "reference_runtime": {
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                },
            }
        finally:
            sys.path.remove(directory)

    OUTPUT.write_text(
        json.dumps(fixture, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {OUTPUT.relative_to(ROOT)} ({len(ids)} features, {len(states)} mixed-width states)"
    )
    print(f"sha256={sha256(OUTPUT.read_bytes())}")


if __name__ == "__main__":
    main()
