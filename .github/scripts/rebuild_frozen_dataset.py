#!/usr/bin/env python3
"""Rebuild a sealed 10k train panel beside the public shared holdout.

All generated records, receipts, and input databases belong on the remote
experiment disk. The command prints only an allowlisted status value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import sys
import time
from collections import Counter


WIDTHS = (8, 10, 13, 25, 26, 34, 45, 50, 58, 72, 90, 108, 122, 144)
TRAIN_COUNT = 10_000
DEV_COUNT = 1_792
TEST_COUNT = 1_792
SCHEMA = "own_frozen_panels_v1"
SYMMETRIES = "identity,inverse,transpose,inverse_transpose and wire reversal; not full orbit"
MASK64 = (1 << 64) - 1
GOLDEN64 = 0x9E3779B97F4A7C15


class BuildError(RuntimeError):
    pass


class SplitMix64:
    """Small explicit RNG whose sequence does not depend on Python versions."""

    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + GOLDEN64) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) & MASK64

    def below(self, limit: int) -> int:
        if limit <= 0:
            raise BuildError("invalid random range")
        return self.next_u64() % limit


def identity(n: int) -> list[int]:
    return [1 << row for row in range(n)]


def apply_gate(rows: list[int], control: int, target: int) -> None:
    if not (0 <= control < len(rows) and 0 <= target < len(rows)) or control == target:
        raise BuildError("generated gate is outside the CNOT alphabet")
    rows[target] ^= rows[control]


def random_gate(rng: SplitMix64, n: int) -> tuple[int, int]:
    control = rng.below(n)
    target = rng.below(n - 1)
    if target >= control:
        target += 1
    return control, target


def walk_component(rng: SplitMix64, n: int, rows: list[int], length: int) -> list[tuple[int, int]]:
    gates = []
    for _ in range(length):
        gate = random_gate(rng, n)
        apply_gate(rows, *gate)
        gates.append(gate)
    return gates


def reverse_wire_order(rows: list[int], n: int) -> list[int]:
    result = [0] * n
    for row_index, row in enumerate(rows):
        reversed_row = 0
        while row:
            bit = row & -row
            column = bit.bit_length() - 1
            reversed_row |= 1 << (n - 1 - column)
            row ^= bit
        result[n - 1 - row_index] = reversed_row
    return result


def transpose(rows: list[int], n: int) -> list[int]:
    result = [0] * n
    for row_index, row in enumerate(rows):
        while row:
            bit = row & -row
            column = bit.bit_length() - 1
            result[column] |= 1 << row_index
            row ^= bit
    return result


def inverse(rows: list[int], n: int) -> list[int]:
    left = list(rows)
    right = identity(n)
    for column in range(n):
        bit = 1 << column
        pivot = next((row for row in range(column, n) if left[row] & bit), None)
        if pivot is None:
            raise BuildError("holdout contains a singular matrix")
        if pivot != column:
            left[column], left[pivot] = left[pivot], left[column]
            right[column], right[pivot] = right[pivot], right[column]
        for row in range(n):
            if row != column and left[row] & bit:
                left[row] ^= left[column]
                right[row] ^= right[column]
    return right


def finite_views(state: dict) -> list[tuple[int, str, tuple[int, ...]]]:
    n = state["n"]
    alphabet_id = state["alphabet_id"]
    rows = state["packed_rows"]
    inv = inverse(rows, n)
    trans = transpose(rows, n)
    bases = (rows, inv, trans, transpose(inv, n))
    result = set()
    for matrix in bases:
        result.add((n, alphabet_id, tuple(matrix)))
        result.add((n, alphabet_id, tuple(reverse_wire_order(matrix, n))))
    return sorted(result)


def state_wire(n: int, rows: list[int]) -> dict:
    return {
        "n": n,
        "packed_rows": list(rows),
        "alphabet_id": f"all_to_all_unit_cnot:{n}",
    }


def normalize_state(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise BuildError("holdout state has an invalid shape")
    n = raw.get("n")
    rows = raw.get("packed_rows")
    alphabet_id = raw.get("alphabet_id")
    if isinstance(n, bool) or not isinstance(n, int) or n not in WIDTHS:
        raise BuildError("holdout state has an unsupported width")
    if alphabet_id != f"all_to_all_unit_cnot:{n}":
        raise BuildError("holdout state uses an unexpected alphabet")
    if not isinstance(rows, list) or len(rows) != n:
        raise BuildError("holdout state has an invalid row count")
    if any(isinstance(row, bool) or not isinstance(row, int) or row < 0 or row >= (1 << n) for row in rows):
        raise BuildError("holdout state has an invalid packed row")
    return state_wire(n, rows)


def normalize_record(raw: object, split: str) -> dict:
    if not isinstance(raw, dict) or raw.get("split") != split:
        raise BuildError("holdout record has a split mismatch")
    state = normalize_state(raw.get("state"))
    n = state["n"]
    lineage_id = raw.get("lineage_id")
    source = raw.get("source")
    seed = raw.get("seed")
    words = raw.get("verified_words")
    if not isinstance(lineage_id, str) or not lineage_id.strip():
        raise BuildError("holdout record has no lineage")
    if not isinstance(source, str) or not source.strip():
        raise BuildError("holdout record has no source")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MASK64:
        raise BuildError("holdout record has an invalid seed")
    if not isinstance(words, list) or not words:
        raise BuildError("holdout record has no verified reduction word")

    normalized_words = []
    end = state_wire(n, identity(n))
    for word in words:
        if not isinstance(word, dict):
            raise BuildError("holdout word certificate has an invalid shape")
        start_state = normalize_state(word.get("start"))
        end_state = normalize_state(word.get("end"))
        gates = word.get("gates")
        cost = word.get("physical_cost")
        producer = word.get("producer")
        word_lineage = word.get("lineage_id")
        if start_state != state or end_state != end:
            raise BuildError("holdout certificate endpoints do not match")
        if not isinstance(gates, list) or isinstance(cost, bool) or not isinstance(cost, int) or cost != len(gates):
            raise BuildError("holdout certificate cost does not match its gates")
        if not isinstance(producer, str) or not producer.strip() or not isinstance(word_lineage, str) or not word_lineage.strip():
            raise BuildError("holdout certificate provenance is incomplete")
        replay = list(state["packed_rows"])
        normalized_gates = []
        for gate in gates:
            if not isinstance(gate, dict):
                raise BuildError("holdout gate has an invalid shape")
            control, target = gate.get("control"), gate.get("target")
            if isinstance(control, bool) or not isinstance(control, int) or isinstance(target, bool) or not isinstance(target, int):
                raise BuildError("holdout gate has invalid indices")
            apply_gate(replay, control, target)
            normalized_gates.append({"control": control, "target": target})
        if replay != end["packed_rows"]:
            raise BuildError("holdout reduction word does not replay to identity")
        normalized_words.append({
            "start": start_state,
            "end": end_state,
            "gates": normalized_gates,
            "physical_cost": cost,
            "producer": producer,
            "lineage_id": word_lineage,
        })

    exact_distance = raw.get("exact_distance")
    exact_evidence = raw.get("exact_evidence")
    if exact_distance is not None and (isinstance(exact_distance, bool) or not isinstance(exact_distance, int) or exact_distance < 0):
        raise BuildError("holdout exact-distance field is invalid")
    if exact_evidence is not None and not isinstance(exact_evidence, str):
        raise BuildError("holdout exact-evidence field is invalid")
    return {
        "state": state,
        "verified_words": normalized_words,
        "lineage_id": lineage_id,
        "seed": seed,
        "source": source,
        "split": split,
        "exact_distance": exact_distance,
        "exact_evidence": exact_evidence,
    }


def read_holdout(path: pathlib.Path) -> tuple[list[dict], list[dict]]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        database = sqlite3.connect(uri, uri=True)
        tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "roots" not in tables:
            raise BuildError("holdout database has no roots table")
        columns = {row[1] for row in database.execute("PRAGMA table_info(roots)")}
        if not {"split", "epoch", "position", "record_json"}.issubset(columns):
            raise BuildError("holdout roots table has an unsupported schema")
        rows = database.execute(
            "SELECT split, record_json FROM roots WHERE split IN ('dev','test') ORDER BY split, epoch, position"
        ).fetchall()
    except sqlite3.Error:
        raise BuildError("holdout database could not be read") from None
    finally:
        if "database" in locals():
            database.close()

    panels = {"dev": [], "test": []}
    for split, raw in rows:
        if split not in panels:
            raise BuildError("holdout contains an unsupported split")
        try:
            record = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            raise BuildError("holdout record JSON is invalid") from None
        panels[split].append(normalize_record(record, split))
    if len(panels["dev"]) != DEV_COUNT or len(panels["test"]) != TEST_COUNT:
        raise BuildError("holdout does not contain the pinned 1792+1792 panels")
    return panels["dev"], panels["test"]


def generate_record(position: int, seed_base: int, attempt: int = 0) -> dict:
    width_index = position % len(WIDTHS)
    n = WIDTHS[width_index]
    family = ("walk", "shear", "composition")[position % 3]
    seed = (seed_base + position * 1024 + attempt) & MASK64
    rng = SplitMix64(seed ^ (width_index * GOLDEN64))
    rows = identity(n)
    if family == "walk":
        length = n * (1 + seed % 3)
        gates = walk_component(rng, n, rows, length)
    elif family == "shear":
        wires = list(range(n))
        for i in range(n - 1, 0, -1):
            j = rng.below(i + 1)
            wires[i], wires[j] = wires[j], wires[i]
        sources = wires[: n // 2]
        targets = wires[n // 2 :]
        gates = []
        for target in targets:
            for control in sources:
                if rng.below(100) < 35:
                    gate = (control, target)
                    apply_gate(rows, *gate)
                    gates.append(gate)
        if not gates:
            gate = (sources[0], targets[0])
            apply_gate(rows, *gate)
            gates.append(gate)
    else:
        components = 2 + rng.below(2)
        gates = []
        for component in range(components):
            length = n * (1 + rng.below(3))
            gates.extend(walk_component(rng, n, rows, length))

    if rows == identity(n):
        return generate_record(position, seed_base, attempt + 1)
    state = state_wire(n, rows)
    end = state_wire(n, identity(n))
    lineage = f"rebuild_20261119:{family}:{seed}:{n}"
    reduction = [
        {"control": control, "target": target}
        for control, target in reversed(gates)
    ]
    certificate = {
        "start": state,
        "end": end,
        "gates": reduction,
        "physical_cost": len(reduction),
        "producer": f"rebuild_20261119_{family}",
        "lineage_id": lineage,
    }
    replay = list(rows)
    for gate in reduction:
        apply_gate(replay, gate["control"], gate["target"])
    if replay != end["packed_rows"]:
        raise BuildError("generated reduction certificate failed replay")
    return {
        "state": state,
        "verified_words": [certificate],
        "lineage_id": lineage,
        "seed": seed,
        "source": f"rebuild_20261119_{family}",
        "split": "train",
        "exact_distance": None,
        "exact_evidence": None,
    }


def key_wire(key: tuple[int, str, tuple[int, ...]]) -> list:
    n, alphabet_id, rows = key
    return [n, alphabet_id, list(rows)]


def build_body(dev: list[dict], test: list[dict], seed_base: int) -> tuple[dict, dict]:
    dev_views: set[tuple[int, str, tuple[int, ...]]] = set()
    test_views: set[tuple[int, str, tuple[int, ...]]] = set()
    heldout_states = set()
    heldout_units = set()
    heldout_lineages = set()
    for split, panel, panel_views in (("dev", dev, dev_views), ("test", test, test_views)):
        for record in panel:
            state = record["state"]
            signature = (state["n"], state["alphabet_id"], tuple(state["packed_rows"]))
            if signature in heldout_states:
                raise BuildError("dev/test repeats a heldout root")
            heldout_states.add(signature)
            unit = (record["source"], record["seed"])
            if unit in heldout_units or record["lineage_id"] in heldout_lineages:
                raise BuildError("dev/test source or lineage identity repeats")
            heldout_units.add(unit)
            heldout_lineages.add(record["lineage_id"])
            panel_views.update(finite_views(state))
    if dev_views & test_views:
        raise BuildError("dev/test finite-symmetry views overlap")
    forbidden = dev_views | test_views

    if len(forbidden) < DEV_COUNT + TEST_COUNT:
        raise BuildError("heldout finite-symmetry view set is incomplete")

    train = []
    seen = set(heldout_states)
    train_units = set(heldout_units)
    train_lineages = set(heldout_lineages)
    widths = Counter()
    families = Counter()
    for position in range(TRAIN_COUNT):
        for attempt in range(1_000):
            record = generate_record(position, seed_base, attempt)
            state = record["state"]
            key = (state["n"], state["alphabet_id"], tuple(state["packed_rows"]))
            if key in seen or any(view in forbidden for view in finite_views(state)):
                continue
            unit = (record["source"], record["seed"])
            if unit in train_units or record["lineage_id"] in train_lineages:
                continue
            seen.add(key)
            train_units.add(unit)
            train_lineages.add(record["lineage_id"])
            widths[state["n"]] += 1
            families[record["source"]] += 1
            train.append(record)
            break
        else:
            raise BuildError("could not generate a unique train root within the retry bound")

    body = {
        "schema": SCHEMA,
        "provenance": {
            "builder": "rebuild_frozen_dataset.py",
            "builder_schema": "deterministic_walk_shear_composition_v1",
            "seed_base": seed_base,
            "heldout_source": "TryDotAtwo/quantum-generator",
            "heldout_source_revision": None,
            "heldout_sqlite_sha256": None,
        },
        "symmetries": SYMMETRIES,
        "forbidden_keys": [key_wire(key) for key in sorted(forbidden)],
        "historical_keys": [],
        "rejected_train_lineages": [],
        "panels": {"train": train, "dev": dev, "test": test},
    }
    stats = {
        "train_count": len(train),
        "dev_count": len(dev),
        "test_count": len(test),
        "forbidden_key_count": len(forbidden),
        "train_width_counts": dict(sorted(widths.items())),
        "train_family_counts": dict(sorted(families.items())),
    }
    return body, stats


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_under(path: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise BuildError("output path is outside the run directory") from None
    return resolved


def write_json_new(path: pathlib.Path, value: dict) -> bytes:
    if path.exists():
        raise BuildError("refusing to overwrite an existing artifact")
    raw = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise BuildError("preserving an existing partial artifact")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise BuildError("artifact destination appeared during the write")
    os.rename(temporary, path)
    return raw


def self_test() -> None:
    for position, expected_width in enumerate(WIDTHS):
        record = generate_record(position, 202611190001)
        normalized = normalize_record(record, "train")
        assert normalized["state"]["n"] == expected_width
        assert normalized["verified_words"][0]["physical_cost"] > 0
        assert 1 <= len(finite_views(normalized["state"])) <= 8

    # Rust stores each row as one decimal BigUint in the JSON wire format.
    # Exercise that contract above 64 bits and at the largest supported width.
    for n in (72, 144):
        rows = identity(n)
        gate = (n - 1, 0)
        apply_gate(rows, *gate)
        state = state_wire(n, rows)
        end = state_wire(n, identity(n))
        lineage = f"selftest:{n}"
        raw = {
            "state": state,
            "verified_words": [{
                "start": state,
                "end": end,
                "gates": [{"control": gate[0], "target": gate[1]}],
                "physical_cost": 1,
                "producer": "dataset-builder-selftest",
                "lineage_id": lineage,
            }],
            "lineage_id": lineage,
            "seed": n,
            "source": "dataset-builder-selftest",
            "split": "train",
            "exact_distance": None,
            "exact_evidence": None,
        }
        roundtrip = json.loads(json.dumps(raw, separators=(",", ":")))
        assert normalize_record(roundtrip, "train")["state"] == state
        assert inverse(rows, n) == rows
        assert transpose(transpose(rows, n), n) == rows
        assert len(finite_views(state)) <= 8
    state = state_wire(8, identity(8))
    assert inverse(state["packed_rows"], 8) == identity(8)
    assert transpose(state["packed_rows"], 8) == identity(8)
    assert reverse_wire_order(state["packed_rows"], 8) == identity(8)


def run(args: argparse.Namespace) -> None:
    root = pathlib.Path(args.experiment_root).resolve()
    if not root.is_dir():
        raise BuildError("run directory does not exist")
    holdout_path = pathlib.Path(args.holdout_sqlite).resolve()
    output_path = require_under(pathlib.Path(args.output), root)
    receipt_path = require_under(pathlib.Path(args.receipt), root)
    if output_path == receipt_path or output_path.suffix != ".json":
        raise BuildError("output file names are invalid")
    if receipt_path.exists() or output_path.with_suffix(".sha256").exists():
        raise BuildError("preserving an existing dataset or receipt")

    holdout_sha = sha256_file(holdout_path)
    dev, test = read_holdout(holdout_path)
    body, stats = build_body(dev, test, args.seed)
    body["provenance"]["heldout_source_revision"] = args.source_revision
    body["provenance"]["heldout_sqlite_sha256"] = holdout_sha
    raw = write_json_new(output_path, body)
    dataset_sha = hashlib.sha256(raw).hexdigest()
    sidecar = output_path.with_suffix(".sha256")
    with sidecar.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(dataset_sha + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "schema": "cnot103_dataset_rebuild_receipt_v1",
        "complete": True,
        "created_unix_seconds": int(time.time()),
        "dataset_sha256": dataset_sha,
        "dataset_bytes": len(raw),
        "holdout_sqlite_sha256": holdout_sha,
        "holdout_source_revision": args.source_revision,
        "seed_base": args.seed,
        **stats,
    }
    write_json_new(receipt_path, receipt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--experiment-root")
    parser.add_argument("--holdout-sqlite")
    parser.add_argument("--output")
    parser.add_argument("--receipt")
    parser.add_argument("--source-revision")
    parser.add_argument("--seed", type=int, default=202611190001)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("dataset_builder_selftest=pass")
            return 0
        if not all((args.experiment_root, args.holdout_sqlite, args.output, args.receipt, args.source_revision)):
            raise BuildError("remote rebuild arguments are incomplete")
        if not 0 <= args.seed <= MASK64:
            raise BuildError("seed must fit an unsigned 64-bit integer")
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
            raise BuildError("holdout source revision must be a lowercase commit SHA")
        run(args)
        print("dataset_build=accepted")
        return 0
    except Exception as error:
        log_path = pathlib.Path(args.experiment_root, "dataset-builder.private.log") if args.experiment_root else None
        if log_path:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"{type(error).__name__}: {error}\n")
            except OSError:
                pass
        print("dataset_build=failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

