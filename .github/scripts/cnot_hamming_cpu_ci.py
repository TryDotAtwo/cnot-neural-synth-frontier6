#!/usr/bin/env python3
"""Run and validate the sealed CPU Hamming acceptance fixture without exposing data."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_full_training as training  # noqa: E402

REVISION = re.compile(r"^[0-9a-f]{40}$")
RUNTIME_CONTRACT = "native_device_search_v13"


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _panel_inputs(dataset_path: pathlib.Path) -> tuple[str, list[str], list[int], dict[str, dict[str, Any]]]:
    sidecar = dataset_path.with_suffix(".sha256")
    dataset_sha = sidecar.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_sha) or _digest(dataset_path) != dataset_sha:
        raise ValueError("fixture dataset digest mismatch")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = dataset.get("panels", {}).get("dev")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("synthetic dev panel is malformed")
    record = records[0]
    case_id = record.get("lineage_id")
    state = record.get("state")
    if not isinstance(case_id, str) or not isinstance(state, dict):
        raise ValueError("synthetic case identity is malformed")
    dimension = state.get("n")
    alphabet_id = state.get("alphabet_id")
    packed_rows = state.get("packed_rows")
    if (
        type(dimension) is not int or dimension < 2
        or not isinstance(alphabet_id, str) or not alphabet_id
        or not isinstance(packed_rows, list) or len(packed_rows) != dimension
    ):
        raise ValueError("synthetic case matrix is malformed")
    words_per_row = (dimension + 63) // 64
    flattened: list[int] = []
    for row in packed_rows:
        if type(row) is not int or row < 0 or row.bit_length() > dimension:
            raise ValueError("synthetic case packed rows are invalid")
        flattened.extend((row >> (64 * w)) & 0xFFFFFFFFFFFFFFFF for w in range(words_per_row))
    indices = [0]
    ids = [case_id]
    roots = {
        case_id: {
            "panel_index": 0,
            "dimension": dimension,
            "alphabet_id": alphabet_id,
            "start_packed_rows": flattened,
        }
    }
    return dataset_sha, ids, indices, roots


def _manifest(
    *,
    dataset_path: pathlib.Path,
    dataset_sha: str,
    source_revision: str,
    panel: str,
    ids: list[str],
    indices: list[int],
    roots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    digest = training._panel_digest(panel, indices, ids)
    return {
        "schema": "cnot103_hamming_search_v1",
        "source_revision": source_revision,
        "dataset": {"path": str(dataset_path.resolve()), "sha256": dataset_sha},
        "panel": panel,
        "panel_indices": indices,
        "panel_ids": ids,
        "panel_roots": [
            {
                "panel_index": indices[position],
                "case_id": case_id,
                "dimension": roots[case_id]["dimension"],
                "alphabet_id": roots[case_id]["alphabet_id"],
                "start_packed_rows": roots[case_id]["start_packed_rows"],
            }
            for position, case_id in enumerate(ids)
        ],
        "panel_digest": digest,
        "runtime_contract": RUNTIME_CONTRACT,
        "search": training.BEAM_CONFIG,
    }


def _invoke(binary: pathlib.Path, manifest_path: pathlib.Path) -> tuple[int, str]:
    result = subprocess.run(
        [str(binary), str(manifest_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return result.returncode, result.stdout


def _expect_rejected(
    binary: pathlib.Path,
    folder: pathlib.Path,
    name: str,
    manifest: dict[str, Any],
) -> bool:
    path = folder / f"{name}.json"
    path.write_bytes(_canonical_bytes(manifest))
    status, _ = _invoke(binary, path)
    return status != 0


def run(args: argparse.Namespace) -> None:
    dataset_path = pathlib.Path(args.dataset).resolve()
    binary_pointer = pathlib.Path(args.binary_file)
    binary = pathlib.Path(binary_pointer.read_text(encoding="utf-8").strip())
    source_revision = args.source_revision
    if not binary.is_file() or not REVISION.fullmatch(source_revision):
        raise ValueError("binary or source pin is unavailable")
    dataset_sha, ids, indices, roots = _panel_inputs(dataset_path)
    base = _manifest(
        dataset_path=dataset_path,
        dataset_sha=dataset_sha,
        source_revision=source_revision,
        panel="dev",
        ids=ids,
        indices=indices,
        roots=roots,
    )
    folder = pathlib.Path(args.output_dir).resolve() if args.output_dir else dataset_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    valid_path = folder / "hamming-valid.json"
    valid_path.write_bytes(_canonical_bytes(base))
    return_code, stdout = _invoke(binary, valid_path)
    if return_code != 0:
        raise RuntimeError("valid CPU Hamming manifest rejected")
    report = json.loads(stdout)
    expected_pins = {
        "search_manifest_sha256": _digest(valid_path),
        "dataset_sha256": dataset_sha,
        "source_revision": source_revision,
        "runtime_contract": RUNTIME_CONTRACT,
        "training_manifest_sha256": None,
        "training_result_sha256": None,
        "checkpoint_sha256": None,
    }
    metrics = training._panel_search_metrics(
        report,
        expected_case_ids=ids,
        expected_panel_indices=indices,
        expected_case_roots=roots,
        expected_panel="dev",
        expected_panel_digest=training._panel_digest("dev", indices, ids),
        expected_search=training.BEAM_CONFIG,
        expected_receipt_pins=expected_pins,
        expected_head="hamming",
    )
    if metrics["solved_count"] != 1 or metrics["cases"][ids[0]]["physical_cost"] != 1:
        raise ValueError("synthetic Hamming solve or replay failed")

    negative = folder / "negative"
    negative.mkdir(exist_ok=True)
    bad_pin = copy.deepcopy(base)
    bad_pin["dataset"]["sha256"] = "0" * 64
    pin_rejected = _expect_rejected(binary, negative, "bad-pin", bad_pin)

    bad_index = copy.deepcopy(base)
    bad_index["panel_indices"] = [1]
    bad_index["panel_digest"] = training._panel_digest("dev", [1], ids)
    bad_index["panel_roots"][0]["panel_index"] = 1
    index_rejected = _expect_rejected(binary, negative, "bad-index", bad_index)

    bad_root = copy.deepcopy(base)
    bad_root["panel_roots"][0]["start_packed_rows"][0] ^= 1
    root_rejected = _expect_rejected(binary, negative, "bad-root", bad_root)
    if not all((pin_rejected, index_rejected, root_rejected)):
        raise ValueError("one or more malformed Hamming pins were accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--binary-file", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
        output = "cpu_hamming=failed code=receipt_or_runtime"
        if args.output:
            with open(args.output, "a", encoding="utf-8") as stream:
                stream.write("status=failed" + chr(10))
                stream.write(output + chr(10))
        print("status=failed")
        print(output)
        return 1
    output = "cpu_hamming=passed bad_pin=rejected bad_index=rejected bad_root=rejected"
    if args.output:
        with open(args.output, "a", encoding="utf-8") as stream:
            stream.write("status=passed" + chr(10))
            stream.write(output + chr(10))
    print("status=passed")
    print(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
