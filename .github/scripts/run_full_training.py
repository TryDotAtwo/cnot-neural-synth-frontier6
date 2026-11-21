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
import secrets
import time
import traceback
from datetime import datetime, timezone
from typing import Any

from build_training_manifest import context_widths as manifest_context_widths

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
FORECAST_CLEANUP_RESERVE_SECONDS = 10 * 60
FORECAST_AUTHORIZATION_MAX_AGE_SECONDS = 6 * 60
FORECAST_APPROVAL_WAIT_SECONDS = 20 * 60
FORECAST_APPROVAL_POLL_SECONDS = 5
FORECAST_SAFETY_SECONDS = 15 * 60
CHECKPOINT_WRITE_RESERVE_SECONDS = 30 * 60
PROFILE_RELATIVE = ".github/training-profiles/n10-test-support-v1.json"
PROFILE_ID = "n10-test-support-v1"
DATA_RELATIVE = "data/frozen_train_10k.json"
DATA_SHA_RELATIVE = "data/frozen_train_10k.sha256"
DATA_RECEIPT_RELATIVE = "data/build.receipt.json"
HOLDOUT_RELATIVE = "input/full_shared_native_holdout_20261001.sqlite"
# Compact v13 geometry: capacities are 1, 4, 16, then five layers of 36;
# E=201 admits all eight planned parent frontiers without a budget stop.
BEAM_CONFIG = {
    "beam_width": 36,
    "shortlist": 4,
    "max_depth": 8,
    "max_expanded": 201,
    "inference_batch": 32,
}
BASELINE_MIN_SOLVED = 14
BASELINE_MIN_WIDTHS = 7
QUALITY_MIN_ABSOLUTE_SOLVED_RATE_GAIN = 0.05



class RunError(RuntimeError):
    pass


class TimeLimit(RunError):
    pass



def estimate_remaining_work_seconds(
    *,
    setup_seconds: float,
    update_seconds: float,
    remaining_updates: int,
    checkpoint_reserve_seconds: float,
    baseline_search_seconds: float,
    forecast_safety_seconds: float,
) -> float:
    """Estimate work still required after the measured bounded pilot."""
    values = (
        setup_seconds,
        update_seconds,
        checkpoint_reserve_seconds,
        baseline_search_seconds,
        forecast_safety_seconds,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < 0
        for value in values
    ):
        raise RunError("runtime forecast contains invalid timing inputs")
    if type(remaining_updates) is not int or remaining_updates < 0:
        raise RunError("runtime forecast has an invalid remaining update count")
    if update_seconds <= 0:
        raise RunError("runtime forecast requires a positive update measurement")
    return (
        float(setup_seconds)
        + float(update_seconds) * remaining_updates
        + float(checkpoint_reserve_seconds)
        + float(baseline_search_seconds)
        + float(forecast_safety_seconds)
    )


def estimate_full_run_seconds(
    *,
    elapsed_before_full: float,
    setup_seconds: float,
    update_seconds: float,
    remaining_updates: int,
    checkpoint_reserve_seconds: float,
    baseline_search_seconds: float,
    forecast_safety_seconds: float,
) -> float:
    """Report elapsed campaign time plus remaining measured work."""
    elapsed = elapsed_before_full
    if (
        isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed) or elapsed < 0
    ):
        raise RunError("runtime forecast has invalid prior elapsed time")
    return float(elapsed) + estimate_remaining_work_seconds(
        setup_seconds=setup_seconds,
        update_seconds=update_seconds,
        remaining_updates=remaining_updates,
        checkpoint_reserve_seconds=checkpoint_reserve_seconds,
        baseline_search_seconds=baseline_search_seconds,
        forecast_safety_seconds=forecast_safety_seconds,
    )


def forecast_time_fits(
    started_at_epoch: object,
    remaining_work_seconds: object,
    now_epoch: object,
    cleanup_reserve_seconds: object = FORECAST_CLEANUP_RESERVE_SECONDS,
) -> bool:
    values = (started_at_epoch, remaining_work_seconds, now_epoch, cleanup_reserve_seconds)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        return False
    started, remaining, now, cleanup = map(float, values)
    if started <= 0 or remaining < 0 or now < started or cleanup < 0:
        return False
    return now - started + remaining + cleanup <= MAX_RUNTIME_SECONDS


def forecast_approval_matches(
    value: object,
    token: str,
    forecast_total_seconds: float,
    remaining_work_seconds: float,
    now_epoch: float,
) -> bool:
    if not isinstance(value, dict):
        return False
    values = (forecast_total_seconds, remaining_work_seconds, now_epoch)
    if (
        not isinstance(token, str)
        or len(token) != 32
        or any(character not in "0123456789abcdef" for character in token)
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in values
        )
        or forecast_total_seconds <= 0
        or remaining_work_seconds <= 0
        or now_epoch <= 0
    ):
        return False
    observed_total = value.get("forecast_total_seconds")
    observed_remaining = value.get("remaining_work_seconds")
    authorized_at = value.get("authorized_at_epoch")
    return (
        value.get("schema") == "cnot103_full_training_forecast_approval_v1"
        and value.get("authorized") is True
        and value.get("forecast_token") == token
        and not isinstance(observed_total, bool)
        and isinstance(observed_total, (int, float))
        and math.isfinite(observed_total)
        and float(observed_total) == float(forecast_total_seconds)
        and not isinstance(observed_remaining, bool)
        and isinstance(observed_remaining, (int, float))
        and math.isfinite(observed_remaining)
        and float(observed_remaining) == float(remaining_work_seconds)
        and not isinstance(authorized_at, bool)
        and isinstance(authorized_at, (int, float))
        and math.isfinite(authorized_at)
        and 0 < float(authorized_at) <= float(now_epoch)
        and float(now_epoch) - float(authorized_at) <= FORECAST_AUTHORIZATION_MAX_AGE_SECONDS
    )


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


def _panel_digest(panel: str, indices: list[int], case_ids: list[str]) -> str:
    canonical = json.dumps(
        {"panel": panel, "indices": indices, "ids": case_ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def select_panel_cohorts(
    dataset_value: object, per_width: int = 8
) -> dict[str, dict[str, Any]]:
    if not isinstance(dataset_value, dict) or not isinstance(dataset_value.get("panels"), dict):
        raise RunError("sealed dataset panels are malformed")
    result: dict[str, dict[str, Any]] = {}
    for panel in ("dev", "test"):
        records = dataset_value["panels"].get(panel)
        indices = select_dev_indices(records, per_width=per_width)
        case_ids: list[str] = []
        for index in indices:
            record = records[index]
            case_id = record.get("lineage_id") if isinstance(record, dict) else None
            if not isinstance(case_id, str) or not case_id.strip():
                raise RunError(f"{panel} cohort has a missing case identity")
            case_ids.append(case_id)
        if len(case_ids) != len(set(case_ids)):
            raise RunError(f"{panel} cohort contains duplicate case identities")
        result[panel] = {
            "indices": indices,
            "case_ids": case_ids,
            "digest": _panel_digest(panel, indices, case_ids),
            "case_count": len(indices),
        }
    return result


def _identity_packed_rows(dimension: int) -> list[int]:
    words_per_row = (dimension + 63) // 64
    rows = [0] * (dimension * words_per_row)
    for row in range(dimension):
        rows[row * words_per_row + row // 64] |= 1 << (row % 64)
    return rows


def _panel_search_metrics(
    report: object,
    *,
    expected_case_ids: list[str],
    expected_panel: str,
    expected_panel_digest: str,
    expected_search: dict[str, Any],
    expected_head: str = "qhat",
) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema") != "cnot103_panel_search_result_v2":
        raise RunError("panel search report schema is invalid")
    if report.get("panel") != expected_panel:
        raise RunError("panel search evaluated the wrong frozen split")
    if report.get("panel_digest") != expected_panel_digest:
        raise RunError("panel search digest differs from its frozen cohort")
    if report.get("panel_ids") != expected_case_ids:
        raise RunError("panel search case identities differ from the frozen cohort")
    if not expected_case_ids:
        raise RunError("frozen panel is empty")
    if len(expected_case_ids) != len(set(expected_case_ids)):
        raise RunError("frozen panel contains duplicate case identities")
    expected_backend = "cpu_hamming" if expected_head == "hamming" else "cuda_model"
    if (
        expected_head not in {"qhat", "proposal", "hamming"}
        or report.get("search") != expected_search
        or report.get("head") != expected_head
        or report.get("execution_backend") != expected_backend
    ):
        raise RunError("panel search budget, scorer, or execution backend differs")
    expected_feature_execution = (
        "not_used_by_hamming" if expected_head == "hamming" else "used_for_model_scoring"
    )
    if (
        report.get("feature_model_inputs") != MODEL_INPUTS
        or report.get("feature_execution") != expected_feature_execution
    ):
        raise RunError("panel search model-input slots or feature execution differs")
    audit = report.get("training_exposure_audit")
    if not isinstance(audit, dict) or audit.get("strict_unseen") is not True:
        raise RunError("panel search exposure audit failed")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_case_ids):
        raise RunError("panel search row count differs from the frozen cohort")
    cases: dict[str, dict[str, Any]] = {}
    statuses = {"solved", "depth_limit", "resource_limit", "exhausted"}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in statuses:
            raise RunError("panel search row is malformed")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in expected_case_ids or case_id in cases:
            raise RunError("panel search case identity is missing, duplicated, or unexpected")
        dimension = row.get("dimension")
        if type(dimension) is not int or dimension < 1:
            raise RunError("panel search case dimension is invalid")
        words_per_row = (dimension + 63) // 64
        start = row.get("start_packed_rows")
        if (
            not isinstance(start, list)
            or len(start) != dimension * words_per_row
            or any(type(word) is not int or word < 0 or word > 0xFFFFFFFFFFFFFFFF for word in start)
        ):
            raise RunError("panel search original case state is malformed")
        width = row.get("width", dimension)
        if type(width) is not int or width != dimension:
            raise RunError("panel search case width differs from its state")
        wall_seconds = row.get("wall_seconds")
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds)
            or wall_seconds < 0
        ):
            raise RunError("panel search per-case timing is invalid")
        word = row.get("word")
        cost: int | None = None
        if row["status"] == "solved":
            if not isinstance(word, dict) or word.get("replayed") is not True:
                raise RunError("a solved panel word lacks native replay confirmation")
            gates = word.get("control_target")
            cost = word.get("physical_cost")
            if (
                not isinstance(gates, list)
                or type(cost) is not int
                or cost != len(gates)
                or cost < 0
            ):
                raise RunError("solved panel word physical cost differs from its gates")
            replay = start.copy()
            for gate in gates:
                if (
                    not isinstance(gate, list) or len(gate) != 2
                    or type(gate[0]) is not int or type(gate[1]) is not int
                    or gate[0] < 0 or gate[1] < 0
                    or gate[0] >= dimension or gate[1] >= dimension
                    or gate[0] == gate[1]
                ):
                    raise RunError("solved panel word contains an invalid physical gate")
                control, target = gate
                for word_index in range(words_per_row):
                    replay[target * words_per_row + word_index] ^= replay[
                        control * words_per_row + word_index
                    ]
            identity = _identity_packed_rows(dimension)
            if replay != identity or row.get("end_packed_rows") != identity:
                raise RunError("panel word does not replay to identity in original orientation")
        elif word is not None or row.get("end_packed_rows") is not None:
            raise RunError("unsolved panel row unexpectedly contains a word or endpoint")
        cases[case_id] = {
            "dimension": dimension,
            "start_packed_rows": start,
            "status": row["status"],
            "physical_cost": cost,
            "wall_seconds": float(wall_seconds),
        }
    if set(cases) != set(expected_case_ids):
        raise RunError("panel search omitted one or more frozen cases")
    solved_ids = [case_id for case_id in expected_case_ids if cases[case_id]["status"] == "solved"]
    costs = [cases[case_id]["physical_cost"] for case_id in solved_ids]
    widths: dict[int, dict[str, Any]] = {}
    for case_id in expected_case_ids:
        dimension = cases[case_id]["dimension"]
        row = widths.setdefault(
            dimension,
            {"case_count": 0, "solved_count": 0, "wall_seconds": []},
        )
        row["case_count"] += 1
        row["wall_seconds"].append(cases[case_id]["wall_seconds"])
        if cases[case_id]["status"] == "solved":
            row["solved_count"] += 1
    return {
        "case_count": len(expected_case_ids),
        "solved_count": len(solved_ids),
        "solved_rate": len(solved_ids) / len(expected_case_ids),
        "solved_case_ids": solved_ids,
        "physical_costs": costs,
        "total_physical_cost": sum(costs),
        "median_physical_cost": statistics.median(costs) if costs else None,
        "search_wall_seconds": sum(
            cases[case_id]["wall_seconds"] for case_id in expected_case_ids
        ),
        "cases": cases,
        "by_width": {
            str(width): {
                "case_count": row["case_count"],
                "solved_count": row["solved_count"],
                "solved_rate": row["solved_count"] / row["case_count"],
                "search_wall_seconds": sum(row["wall_seconds"]),
                "median_case_wall_seconds": statistics.median(row["wall_seconds"]),
            }
            for width, row in sorted(widths.items())
        },
    }


def evaluate_baseline_gate(
    report: object,
    *,
    expected_case_ids: list[str],
    expected_panel_digest: str,
    expected_search: dict[str, Any],
    minimum_solved: int = BASELINE_MIN_SOLVED,
    minimum_widths: int = BASELINE_MIN_WIDTHS,
) -> dict[str, Any]:
    """Admit use of a fixed Hamming baseline before any locked-test search."""
    try:
        metrics = _panel_search_metrics(
            report,
            expected_case_ids=expected_case_ids,
            expected_panel="dev",
            expected_panel_digest=expected_panel_digest,
            expected_search=expected_search,
            expected_head="hamming",
        )
    except RunError as exc:
        return {"status": "invalid", "reason": str(exc)}
    solved_widths = sum(
        1 for summary in metrics["by_width"].values()
        if summary["solved_count"] > 0
    )
    if metrics["solved_count"] == 0:
        status = "inconclusive"
        reason = "Hamming baseline solved no frozen dev cases"
    elif (
        metrics["solved_count"] < minimum_solved
        or solved_widths < minimum_widths
    ):
        status = "failed"
        reason = "Hamming baseline is below the frozen solve or width-coverage threshold"
    else:
        status = "passed"
        reason = "Hamming baseline meets the frozen dev admission threshold"
    return {
        "status": status,
        "reason": reason,
        "panel": "dev",
        "case_count": metrics["case_count"],
        "solved_count": metrics["solved_count"],
        "solved_width_count": solved_widths,
        "minimum_solved": minimum_solved,
        "minimum_widths": minimum_widths,
        "panel_digest": expected_panel_digest,
        "metrics": metrics,
    }


def train_after_baseline_gate(gate: object, full_train):
    """Fail closed so full training cannot run without a passed dev baseline."""
    if not isinstance(gate, dict) or gate.get("status") != "passed":
        raise RunError("full training refused because the frozen dev baseline gate did not pass")
    return full_train()


def compare_quality(
    baseline_report: object,
    candidate_report: object,
    *,
    expected_case_ids: list[str],
    expected_panel: str,
    expected_panel_digest: str,
    expected_search: dict[str, Any],
    baseline_head: str = "qhat",
    candidate_head: str = "qhat",
) -> dict[str, Any]:
    """Compare paired searches on one frozen panel and identical beam budget."""
    try:
        baseline = _panel_search_metrics(
            baseline_report,
            expected_case_ids=expected_case_ids,
            expected_panel=expected_panel,
            expected_panel_digest=expected_panel_digest,
            expected_search=expected_search,
            expected_head=baseline_head,
        )
        candidate = _panel_search_metrics(
            candidate_report,
            expected_case_ids=expected_case_ids,
            expected_panel=expected_panel,
            expected_panel_digest=expected_panel_digest,
            expected_search=expected_search,
            expected_head=candidate_head,
        )
        for case_id in expected_case_ids:
            before = baseline["cases"][case_id]
            after = candidate["cases"][case_id]
            if (
                before["dimension"] != after["dimension"]
                or before["start_packed_rows"] != after["start_packed_rows"]
            ):
                raise RunError("paired quality case differs in original orientation")
    except RunError as exc:
        return {"status": "invalid", "reason": str(exc)}

    baseline_solved = set(baseline["solved_case_ids"])
    candidate_solved = set(candidate["solved_case_ids"])
    required_solved_gain = max(
        1,
        math.ceil(QUALITY_MIN_ABSOLUTE_SOLVED_RATE_GAIN * len(expected_case_ids)),
    )
    solved_gain = candidate["solved_count"] - baseline["solved_count"]
    if not baseline_solved:
        status = "inconclusive"
        reason = "baseline solved no locked test cases"
        paired_baseline_cost = 0
        paired_candidate_cost = 0
    elif not baseline_solved.issubset(candidate_solved):
        status = "failed"
        reason = "candidate lost one or more baseline solutions"
        paired_baseline_cost = sum(
            baseline["cases"][case_id]["physical_cost"] for case_id in baseline_solved
        )
        paired_candidate_cost = sum(
            candidate["cases"][case_id]["physical_cost"]
            for case_id in baseline_solved & candidate_solved
        )
    else:
        paired_baseline_cost = sum(
            baseline["cases"][case_id]["physical_cost"] for case_id in baseline_solved
        )
        paired_candidate_cost = sum(
            candidate["cases"][case_id]["physical_cost"] for case_id in baseline_solved
        )
        if paired_candidate_cost > paired_baseline_cost:
            status = "failed"
            reason = "candidate increased paired physical cost"
        elif solved_gain < required_solved_gain:
            status = "failed"
            reason = "candidate did not reach the frozen practical solved-rate gain"
        else:
            status = "passed"
            reason = "candidate preserved baseline solves, reduced paired cost, and reached the frozen solved-rate gain"
    widths = sorted(
        set(baseline["by_width"]) | set(candidate["by_width"]),
        key=int,
    )
    return {
        "status": status,
        "reason": reason,
        "baseline_head": baseline_head,
        "candidate_head": candidate_head,
        "panel": expected_panel,
        "case_count": len(expected_case_ids),
        "baseline_solved_count": baseline["solved_count"],
        "candidate_solved_count": candidate["solved_count"],
        "solved_gain": solved_gain,
        "required_solved_gain": required_solved_gain,
        "minimum_absolute_solved_rate_gain": QUALITY_MIN_ABSOLUTE_SOLVED_RATE_GAIN,
        "baseline_solved_rate": baseline["solved_rate"],
        "candidate_solved_rate": candidate["solved_rate"],
        "paired_baseline_physical_cost": paired_baseline_cost,
        "paired_candidate_physical_cost": paired_candidate_cost,
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "by_width": [
            {
                "width": int(width),
                "baseline_case_count": baseline["by_width"].get(width, {}).get("case_count", 0),
                "baseline_solved_count": baseline["by_width"].get(width, {}).get("solved_count", 0),
                "candidate_case_count": candidate["by_width"].get(width, {}).get("case_count", 0),
                "candidate_solved_count": candidate["by_width"].get(width, {}).get("solved_count", 0),
                "solved_delta": (
                    candidate["by_width"].get(width, {}).get("solved_count", 0)
                    - baseline["by_width"].get(width, {}).get("solved_count", 0)
                ),
            }
            for width in widths
        ],
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


def _normalise_profile_context_widths(value: object) -> dict[str, list[int]]:
    if not isinstance(value, dict):
        raise RunError("feature coverage profile contexts are malformed")
    normalized: dict[str, list[int]] = {}
    for name, widths in value.items():
        if (
            not isinstance(name, str) or not name
            or not isinstance(widths, list)
            or any(type(width) is not int or width < 1 for width in widths)
            or len(set(widths)) != len(widths)
        ):
            raise RunError("feature coverage profile context widths are malformed")
        normalized[name] = sorted(widths)
    return dict(sorted(normalized.items()))


def _availability_by_width(
    width_row: dict[str, Any],
    profile_context_widths: dict[str, list[int]],
) -> dict[str, Any]:
    width = width_row.get("width")
    rows = width_row.get("rows")
    features = width_row.get("features")
    if (
        type(width) is not int or width < 1
        or type(rows) is not int or rows < 1
        or not isinstance(features, list) or len(features) != MODEL_INPUTS
    ):
        raise RunError("feature availability width record is malformed")
    totals = {
        "computed": 0,
        "inapplicable": 0,
        "unsupported": 0,
        "resource_limit": 0,
    }
    computed_features = 0
    ids: set[str] = set()
    for feature in features:
        if not isinstance(feature, dict):
            raise RunError("feature availability entry is malformed")
        feature_id = feature.get("id")
        if not isinstance(feature_id, str) or not feature_id or feature_id in ids:
            raise RunError("feature availability identity is malformed")
        ids.add(feature_id)
        counts = {}
        for status in ("not_requested", "computed", "inapplicable", "unsupported", "resource_limit"):
            count = feature.get(status)
            if type(count) is not int or count < 0:
                raise RunError("feature availability count is malformed")
            counts[status] = count
        if counts["not_requested"] != 0 or sum(counts.values()) != rows:
            raise RunError("feature availability does not account for every input slot")
        computed_features += int(counts["computed"] > 0)
        for status in totals:
            totals[status] += counts[status]
    return {
        "width": width,
        "rows": rows,
        "model_input_slots": MODEL_INPUTS,
        "profile_context_keys": sorted(
            key for key, configured_widths in profile_context_widths.items()
            if width in configured_widths
        ),
        "features_with_any_computed_rows": computed_features,
        "computed_values": totals["computed"],
        "inapplicable_values": totals["inapplicable"],
        "unsupported_values": totals["unsupported"],
        "resource_limit_values": totals["resource_limit"],
    }


def validate_feature_coverage(
    report: object,
    *,
    expected_manifest_sha256: str,
    expected_dataset_sha256: str,
    expected_panel: str,
    expected_rows_by_width: dict[int, int],
    expected_feature_count: int,
    profile_id: str,
    profile_sha256: str,
    profile_context_widths: dict[str, list[int]],
) -> dict[str, Any]:
    """Validate every v25 input status and compare availability with profile widths."""
    if expected_feature_count != MODEL_INPUTS:
        raise RunError("coverage must account for all 102 model input slots")
    if profile_id != PROFILE_ID:
        raise RunError("coverage profile identity differs")
    if not isinstance(report, dict) or report.get("schema") != "cnot103_feature_coverage_v1":
        raise RunError("feature coverage report schema is invalid")
    if (
        report.get("input_schema") != "runtime_packet_layout_v25"
        or not isinstance(report.get("policy_id"), str)
        or not report.get("policy_id")
        or report.get("manifest_sha256") != expected_manifest_sha256
        or report.get("dataset_sha256") != expected_dataset_sha256
        or report.get("panel") != expected_panel
        or report.get("batch_size") != 1
    ):
        raise RunError("feature coverage source, schema, or split pin differs")
    if (
        not isinstance(profile_sha256, str) or len(profile_sha256) != 64
        or any(char not in "0123456789abcdef" for char in profile_sha256)
    ):
        raise RunError("feature coverage profile hash is malformed")
    context_widths = _normalise_profile_context_widths(profile_context_widths)
    if (
        expected_panel not in {"train", "dev"}
        or not isinstance(expected_rows_by_width, dict)
        or not expected_rows_by_width
        or any(
            type(width) is not int or width < 1
            or type(count) is not int or count < 1
            for width, count in expected_rows_by_width.items()
        )
    ):
        raise RunError("feature coverage expected width inventory is malformed")

    raw_widths = report.get("widths")
    if not isinstance(raw_widths, list):
        raise RunError("feature coverage widths are malformed")
    normalized_widths: list[dict[str, Any]] = []
    seen_widths: set[int] = set()
    expected_feature_ids: list[str] | None = None
    status_names = (
        "not_requested", "computed", "inapplicable", "unsupported", "resource_limit"
    )
    for width_row in raw_widths:
        if not isinstance(width_row, dict):
            raise RunError("feature coverage width row is malformed")
        width = width_row.get("width")
        rows = width_row.get("rows")
        rejected = width_row.get("rejected_rows")
        features = width_row.get("features")
        if (
            type(width) is not int or width in seen_widths
            or width not in expected_rows_by_width
            or type(rows) is not int or rows != expected_rows_by_width[width]
            or type(rejected) is not int or rejected != 0
            or not isinstance(features, list) or len(features) != expected_feature_count
        ):
            raise RunError("feature coverage width or rejected-row accounting differs")
        seen_widths.add(width)
        normalized_features: list[dict[str, Any]] = []
        feature_ids: list[str] = []
        for feature in features:
            if not isinstance(feature, dict):
                raise RunError("feature coverage entry is malformed")
            feature_id = feature.get("id")
            if not isinstance(feature_id, str) or not feature_id:
                raise RunError("feature coverage id is malformed")
            feature_ids.append(feature_id)
            counts: dict[str, int] = {}
            for status_name in status_names:
                count = feature.get(status_name)
                if type(count) is not int or count < 0:
                    raise RunError("feature coverage status count is malformed")
                counts[status_name] = count
            if (
                counts["not_requested"] != 0
                or sum(counts.values()) != rows
            ):
                raise RunError("feature coverage does not account for every requested row")
            normalized_features.append({
                "id": feature_id,
                "role": feature.get("role"),
                "scope": feature.get("scope"),
                "provider": feature.get("provider"),
                **counts,
            })
        if len(set(feature_ids)) != expected_feature_count:
            raise RunError("feature coverage selected specs are duplicated")
        if expected_feature_ids is None:
            expected_feature_ids = feature_ids
        elif feature_ids != expected_feature_ids:
            raise RunError("feature coverage selected specs changed across widths")
        normalized_widths.append({
            "width": width,
            "rows": rows,
            "rejected_rows": rejected,
            "features": normalized_features,
        })
    if seen_widths != set(expected_rows_by_width):
        raise RunError("feature coverage omitted a dataset width")
    normalized_widths.sort(key=lambda row: row["width"])
    return {
        "panel": expected_panel,
        "policy_id": report["policy_id"],
        "manifest_sha256": expected_manifest_sha256,
        "dataset_sha256": expected_dataset_sha256,
        "profile_id": profile_id,
        "profile_sha256": profile_sha256,
        "profile_context_widths": context_widths,
        "model_input_slots": MODEL_INPUTS,
        "selected_feature_count": expected_feature_count,
        "widths": normalized_widths,
        "availability_by_width": [
            _availability_by_width(row, context_widths)
            for row in normalized_widths
        ],
    }


def validate_feature_coverage_preflight(
    summary: object,
    *,
    expected_dataset_sha256: str,
    expected_profile_id: str,
    expected_profile_sha256: str,
    expected_profile_context_widths: dict[str, list[int]],
) -> None:
    """Gate expensive training on pinned, complete private availability evidence."""
    if not isinstance(summary, dict) or summary.get("schema") != "cnot103_feature_coverage_summary_v1":
        raise RunError("feature coverage preflight summary schema is invalid")
    expected_contexts = _normalise_profile_context_widths(expected_profile_context_widths)
    for field, expected in (
        ("dataset_sha256", expected_dataset_sha256),
        ("profile_id", expected_profile_id),
        ("profile_sha256", expected_profile_sha256),
        ("profile_context_widths", expected_contexts),
    ):
        if summary.get(field) != expected:
            raise RunError("feature coverage preflight pin differs")
    manifest_sha256 = summary.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in manifest_sha256)
        or summary.get("model_input_slots") != MODEL_INPUTS
    ):
        raise RunError("feature coverage preflight manifest or input-slot count differs")
    panels = summary.get("panels")
    if not isinstance(panels, dict) or set(panels) != {"train", "dev"}:
        raise RunError("feature coverage preflight must include train and dev")
    policy_ids: set[str] = set()
    for panel_name in ("train", "dev"):
        panel = panels[panel_name]
        if (
            not isinstance(panel, dict)
            or panel.get("panel") != panel_name
            or panel.get("dataset_sha256") != expected_dataset_sha256
            or panel.get("profile_id") != expected_profile_id
            or panel.get("profile_sha256") != expected_profile_sha256
            or panel.get("profile_context_widths") != expected_contexts
            or panel.get("model_input_slots") != MODEL_INPUTS
            or panel.get("manifest_sha256") != manifest_sha256
        ):
            raise RunError("feature coverage preflight panel pin differs")
        policy_id = panel.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id:
            raise RunError("feature coverage preflight policy is missing")
        policy_ids.add(policy_id)
        widths = panel.get("widths")
        availability = panel.get("availability_by_width")
        if not isinstance(widths, list) or not widths or not isinstance(availability, list):
            raise RunError("feature coverage preflight availability is missing")
        if len(widths) != len(availability):
            raise RunError("feature coverage preflight widths and availability differ")
        seen: set[int] = set()
        for width_row, actual in zip(widths, availability):
            if not isinstance(width_row, dict):
                raise RunError("feature coverage preflight width is malformed")
            width = width_row.get("width")
            if type(width) is not int or width in seen:
                raise RunError("feature coverage preflight width is duplicated")
            seen.add(width)
            expected = _availability_by_width(width_row, expected_contexts)
            if actual != expected:
                raise RunError("feature availability differs from the accepted profile comparison")
    if len(policy_ids) != 1:
        raise RunError("train and dev feature coverage policies differ")

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
        bootstrap_deadline_epoch: float,
        billing_baseline_epoch: float | None = None,
    ):
        self.root = root
        self.source = root / "source"
        self.source_revision = source_revision
        self.holdout_revision = holdout_revision
        self.holdout_sha256 = holdout_sha256
        self.started_at_epoch = started_at_epoch
        self.budget_baseline = budget_baseline
        self.billing_baseline_epoch = (
            started_at_epoch if billing_baseline_epoch is None else billing_baseline_epoch
        )
        self.bootstrap_deadline_epoch = bootstrap_deadline_epoch
        self.full_run_authorized = False
        self.forecast_total_seconds: float | None = None
        self.remaining_work_seconds: float | None = None
        self.forecast_token: str | None = None
        self.status_path = root / "status.private.json"
        self.events_path = root / "events.private.jsonl"
        self.error_path = root / "error.private.json"
        self.private_report_path = root / "final-result.private.json"
        self.feature_coverage_summary: dict[str, Any] | None = None
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
        allowed_quality = {"not_ready", "measured", "passed", "failed", "inconclusive", "invalid"}
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
            "billing_baseline_epoch": self.billing_baseline_epoch,
        }
        if self.remaining_work_seconds is not None:
            value["remaining_work_seconds"] = self.remaining_work_seconds
        if self.forecast_total_seconds is not None and self.forecast_token is not None:
            value["forecast_total_seconds"] = self.forecast_total_seconds
            value["forecast_token"] = self.forecast_token
        _atomic_private_json(self.status_path, value)
        event = {"at_utc": _now(), "state": state, "quality_status": self.quality_status}
        fd = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as output:
            output.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def check_time(self) -> None:
        now = time.time()
        if now >= self.deadline:
            raise TimeLimit("48-hour wall-time guard reached")
        if not self.full_run_authorized and now >= self.bootstrap_deadline_epoch:
            raise TimeLimit("bounded bootstrap pilot deadline reached")

    def _await_forecast_approval(self) -> None:
        if (
            self.forecast_total_seconds is None
            or self.forecast_token is None
            or self.remaining_work_seconds is None
        ):
            raise RunError("controller forecast is missing")
        deadline = min(
            self.bootstrap_deadline_epoch,
            time.time() + FORECAST_APPROVAL_WAIT_SECONDS,
        )
        approval_path = self.root / "forecast-approval.private.json"
        while time.time() < deadline:
            self.check_time()
            try:
                value = json.loads(approval_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = None
            now_epoch = time.time()
            if forecast_approval_matches(
                value,
                self.forecast_token,
                self.forecast_total_seconds,
                self.remaining_work_seconds,
                now_epoch,
            ):
                if not forecast_time_fits(
                    self.started_at_epoch, self.remaining_work_seconds, now_epoch
                ):
                    raise TimeLimit("remaining work and stop reserve no longer fit the 48-hour gate")
                self.full_run_authorized = True
                return
            time.sleep(FORECAST_APPROVAL_POLL_SECONDS)
        raise TimeLimit("controller did not approve the forecast within the pilot window")


    def _active_deadline(self) -> float:
        if self.full_run_authorized:
            return self.deadline
        return min(self.deadline, self.bootstrap_deadline_epoch)

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
        active_deadline = self._active_deadline()
        remaining_seconds = active_deadline - time.time()
        if remaining_seconds < 1:
            self.check_time()
            raise TimeLimit("runtime gate expired before stage launch")
        timeout = int(remaining_seconds)
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
                raise TimeLimit(f"{name} exceeded its active runtime gate") from None
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
        self.check_time()
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
                timeout=min(180, max(1, int(self._active_deadline() - time.time()))),
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
                timeout=min(1200, max(1, int(self._active_deadline() - time.time()))),
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
        """Build candidate bytes and provenance before compiling Rust tools."""
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
        if not data_path.is_file() or not sha_path.is_file() or not receipt_path.is_file():
            raise RunError("candidate dataset or its provenance receipt is missing")
        dataset_sha = sha_path.read_text(encoding="ascii").strip()
        if len(dataset_sha) != 64 or any(c not in "0123456789abcdef" for c in dataset_sha):
            raise RunError("frozen train dataset SHA receipt is malformed")
        hasher = hashlib.sha256()
        with data_path.open("rb") as candidate:
            for chunk in iter(lambda: candidate.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != dataset_sha:
            raise RunError("candidate dataset bytes differ from the SHA receipt")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RunError("candidate dataset provenance receipt is invalid") from None
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != "cnot103_dataset_rebuild_receipt_v1"
            or receipt.get("complete") is not False
            or receipt.get("validation_status") != "pending"
            or receipt.get("dataset_sha256") != dataset_sha
            or receipt.get("dataset_bytes") != data_path.stat().st_size
            or receipt.get("holdout_sqlite_sha256") != self.holdout_sha256
            or receipt.get("heldout_source_revision") != self.holdout_revision
            or receipt.get("builder_source_revision") != self.source_revision
            or not isinstance(receipt.get("heldout_source_repository"), str)
            or not receipt["heldout_source_repository"].strip()
            or not isinstance(receipt.get("heldout_source_filename"), str)
            or not receipt["heldout_source_filename"].strip()
        ):
            raise RunError("candidate dataset provenance does not match its pinned inputs")
        return dataset_sha

    def _validate_and_seal_dataset(self, dataset_sha: str) -> None:
        """Use the freshly built canonical Rust loader, then seal its receipt."""
        data_path = self.root / DATA_RELATIVE
        receipt_path = self.root / DATA_RECEIPT_RELATIVE
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
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RunError("sealed dataset receipt is unreadable") from None
        if (
            not isinstance(receipt, dict)
            or receipt.get("complete") is not True
            or receipt.get("validation_status") != "rust_loader_accepted"
            or receipt.get("dataset_sha256") != dataset_sha
            or receipt.get("dataset_bytes") != data_path.stat().st_size
        ):
            raise RunError("Rust-accepted dataset receipt was not sealed")
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
            profile_bytes = (self.source / PROFILE_RELATIVE).read_bytes()
            profile = json.loads(profile_bytes)
        except (OSError, json.JSONDecodeError):
            raise RunError("training manifest, profile, or provenance is malformed") from None
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        profile_sha = hashlib.sha256(profile_bytes).hexdigest()
        profile_source = profile.get("source") if isinstance(profile, dict) else None
        if (
            not isinstance(profile_source, dict)
            or profile.get("profile_id") != PROFILE_ID
            or sidecar.get("profile_id") != PROFILE_ID
            or sidecar.get("profile_sha256") != profile_sha
            or sidecar.get("profile_source_repository") != profile_source.get("repository")
            or sidecar.get("profile_source_revision") != profile_source.get("revision")
            or sidecar.get("profile_source_path") != profile_source.get("path")
            or sidecar.get("profile_source_sha256") != profile_source.get("sha256")
            or sidecar.get("dataset_sha256") != dataset_sha
            or manifest.get("feature_contexts") != profile.get("feature_contexts")
        ):
            raise RunError("training manifest profile or dataset provenance differs")
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

    def _feature_coverage(self, dataset_sha: str) -> dict[str, Any]:
        manifest_path = self._build_manifest(
            "coverage", dataset_sha, 1, TARGET_ATTEMPTS, False
        )
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            manifest = json.loads(manifest_bytes)
            provenance = json.loads(
                manifest_path.with_suffix(manifest_path.suffix + ".provenance.json")
                .read_text(encoding="utf-8")
            )
            profile_path = self.source / PROFILE_RELATIVE
            profile_bytes = profile_path.read_bytes()
            profile = json.loads(profile_bytes)
            dataset = json.loads(
                (self.root / DATA_RELATIVE).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            raise RunError("coverage manifest, profile, or dataset is malformed") from None
        actual_profile_sha = hashlib.sha256(profile_bytes).hexdigest()
        if (
            profile.get("profile_id") != PROFILE_ID
            or provenance.get("profile_id") != PROFILE_ID
            or provenance.get("profile_sha256") != actual_profile_sha
            or provenance.get("dataset_sha256") != dataset_sha
            or manifest.get("feature_policy", {}).get("schema_id")
                != "runtime_packet_layout_v25"
            or manifest.get("dataset", {}).get("sha256") != dataset_sha
        ):
            raise RunError("coverage profile or dataset pin differs")
        try:
            expected_context_widths = manifest_context_widths(profile["feature_contexts"])
        except (KeyError, TypeError, ValueError):
            raise RunError("accepted feature profile contexts are malformed") from None
        if (
            provenance.get("context_widths") != expected_context_widths
            or manifest.get("feature_contexts") != profile.get("feature_contexts")
        ):
            raise RunError("coverage context widths differ from the accepted profile")
        panels = dataset.get("panels") if isinstance(dataset, dict) else None
        if not isinstance(panels, dict):
            raise RunError("coverage dataset panels are malformed")

        def width_counts(panel: str) -> dict[int, int]:
            records = panels.get(panel)
            if not isinstance(records, list) or not records:
                raise RunError("coverage dataset panel is empty or malformed")
            counts: dict[int, int] = {}
            for record in records:
                state = record.get("state") if isinstance(record, dict) else None
                width = state.get("n") if isinstance(state, dict) else None
                if type(width) is not int or width < 1:
                    raise RunError("coverage dataset record width is malformed")
                counts[width] = counts.get(width, 0) + 1
            return counts

        expected_rows_by_panel = {
            panel: width_counts(panel) for panel in ("train", "dev")
        }
        panel_summaries: dict[str, dict[str, Any]] = {}
        for panel, expected_rows in expected_rows_by_panel.items():
            report = self._run(
                f"coverage_{panel}",
                [str(self._runner("cnot103-coverage")), str(manifest_path), panel],
                state="preparing",
                json_stdout=True,
            )
            panel_summaries[panel] = validate_feature_coverage(
                report,
                expected_manifest_sha256=manifest_sha,
                expected_dataset_sha256=dataset_sha,
                expected_panel=panel,
                expected_rows_by_width=expected_rows,
                expected_feature_count=MODEL_INPUTS,
                profile_id=profile["profile_id"],
                profile_sha256=actual_profile_sha,
                profile_context_widths=expected_context_widths,
            )
        summary = {
            "schema": "cnot103_feature_coverage_summary_v1",
            "manifest_sha256": manifest_sha,
            "dataset_sha256": dataset_sha,
            "profile_id": PROFILE_ID,
            "profile_sha256": actual_profile_sha,
            "profile_context_widths": expected_context_widths,
            "model_input_slots": MODEL_INPUTS,
            "policy_id": panel_summaries["train"]["policy_id"],
            "panels": panel_summaries,
        }
        # This profile-pinned summary is a hard gate before benchmark, pilot,
        # resume proof, baseline search, or the 250,000-update training.
        validate_feature_coverage_preflight(
            summary,
            expected_dataset_sha256=dataset_sha,
            expected_profile_id=PROFILE_ID,
            expected_profile_sha256=actual_profile_sha,
            expected_profile_context_widths=expected_context_widths,
        )
        _write_new_private_json(
            self.root / "reports" / "feature_coverage.summary.json",
            summary,
        )
        return summary

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
        panel: str,
        cohort: dict[str, Any],
        *,
        head: str = "qhat",
    ) -> pathlib.Path:
        if panel not in {"dev", "test"} or head not in {"qhat", "hamming"}:
            raise RunError("search manifest panel or scorer is invalid")
        indices = cohort.get("indices")
        case_ids = cohort.get("case_ids")
        panel_digest = cohort.get("digest")
        if (
            not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or not isinstance(case_ids, list) or len(case_ids) != len(indices)
            or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))
            or panel_digest != _panel_digest(panel, indices, case_ids)
        ):
            raise RunError("search manifest cohort is malformed")
        value = {
            "schema": "cnot103_panel_search_v2",
            "training_manifest": {
                "path": str(train_manifest),
                "sha256": hashlib.sha256(train_manifest.read_bytes()).hexdigest(),
            },
            "training_result": {
                "path": str(train_result),
                "sha256": hashlib.sha256(train_result.read_bytes()).hexdigest(),
            },
            "panel": panel,
            "panel_indices": indices,
            "panel_ids": case_ids,
            "panel_digest": panel_digest,
            "search": BEAM_CONFIG,
            "head": head,
        }
        path = self.root / "manifests" / f"{name}.json"
        _write_new_private_json(path, value)
        return path

    def _search(
        self,
        name: str,
        manifest: pathlib.Path,
        cohort: dict[str, Any],
        panel: str,
        *,
        expected_head: str = "qhat",
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        started = time.monotonic()
        report = self._run(
            name,
            [str(self._runner("cnot103-search")), str(manifest)],
            state="baseline" if name.startswith("baseline") else "final_quality",
            json_stdout=True,
        )
        elapsed = time.monotonic() - started
        metrics = _panel_search_metrics(
            report,
            expected_case_ids=cohort["case_ids"],
            expected_panel=panel,
            expected_panel_digest=cohort["digest"],
            expected_search=BEAM_CONFIG,
            expected_head=expected_head,
        )
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
            self._validate_and_seal_dataset(dataset_sha)
            data_value = json.loads((self.root / DATA_RELATIVE).read_text(encoding="utf-8"))
            panel_cohorts = select_panel_cohorts(data_value, per_width=8)
            _write_new_private_json(
                self.root / "reports" / "panel-cohorts.private.json",
                {
                    "schema": "cnot103_locked_panel_cohorts_v1",
                    "dataset_sha256": dataset_sha,
                    "panels": panel_cohorts,
                },
            )
            self.feature_coverage_summary = self._feature_coverage(dataset_sha)
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

            dev_cohort = panel_cohorts["dev"]
            test_cohort = panel_cohorts["test"]
            baseline_plan = {
                "schema": "cnot103_hamming_baseline_plan_v1",
                "source_revision": self.source_revision,
                "holdout_revision": self.holdout_revision,
                "holdout_sha256": self.holdout_sha256,
                "dataset_sha256": dataset_sha,
                "profile_id": PROFILE_ID,
                "profile_sha256": hashlib.sha256(
                    (self.source / PROFILE_RELATIVE).read_bytes()
                ).hexdigest(),
                "feature_schema": "runtime_packet_layout_v25",
                "model": {
                    "parameters": MODEL_PARAMETERS,
                    "structural_blocks": STRUCTURAL_BLOCKS,
                    "input_slots": MODEL_INPUTS,
                    "seed": json.loads(resume_manifest.read_text(encoding="utf-8"))[
                        "training"
                    ]["seed"],
                },
                "checkpoint_inputs": {
                    "training_manifest_sha256": hashlib.sha256(
                        resume_manifest.read_bytes()
                    ).hexdigest(),
                    "training_result_sha256": hashlib.sha256(
                        (self.root / "reports" / "resume_step_2.stdout.json").read_bytes()
                    ).hexdigest(),
                    "receipt": json.loads(
                        (self.root / "reports" / "resume_step_2.stdout.json").read_text(
                            encoding="utf-8"
                        )
                    ).get("checkpoint"),
                },
                "dev_panel": dev_cohort,
                "locked_test_panel": test_cohort,
                "search": BEAM_CONFIG,
                "runtime_contract": {
                    "id": "native_device_search_v13",
                    "frontier_charge": "charge_complete_live_parent_frontier_before_goal_scan",
                    "terminal_scan": "full_action_alphabet_one_cnot_before_shortlist",
                    "final_layer_scan": False,
                    "orientation": "row_target_xor_row_control",
                    "alphabet": "all_to_all_unit_cnot",
                    "tie_break": "ascending_score_then_lexicographic_complete_word",
                    "replay": "exact_gf2_rows_to_identity",
                    "state_dtype": "packed_u64",
                    "neural_score_dtype": "float32",
                    "hamming_score": "integer_popcount_then_float32",
                },
                "head": "hamming",
                "execution_backend": "cpu_hamming",
                "quality_acceptance": {
                    "minimum_absolute_solved_rate_gain": QUALITY_MIN_ABSOLUTE_SOLVED_RATE_GAIN,
                    "preserve_all_hamming_solutions": True,
                    "paired_physical_cost_may_not_increase": True,
                },
                "criterion": {
                    "minimum_solved_dev_cases": BASELINE_MIN_SOLVED,
                    "minimum_solved_dev_widths": BASELINE_MIN_WIDTHS,
                    "zero_solved_is": "inconclusive",
                    "locked_test_is_never_used_to_tune": True,
                },
            }
            _write_new_private_json(
                self.root / "reports" / "baseline-plan.private.json",
                baseline_plan,
            )
            baseline_dev_manifest = self._search_manifest(
                "baseline_dev_search", resume_manifest,
                self.root / "reports" / "resume_step_2.stdout.json",
                "dev", dev_cohort, head="hamming",
            )
            baseline_dev_report, baseline_dev_metrics, baseline_dev_wall = self._search(
                "baseline_dev_search", baseline_dev_manifest, dev_cohort, "dev",
                expected_head="hamming",
            )
            baseline_gate = evaluate_baseline_gate(
                baseline_dev_report,
                expected_case_ids=dev_cohort["case_ids"],
                expected_panel_digest=dev_cohort["digest"],
                expected_search=BEAM_CONFIG,
            )
            _write_new_private_json(
                self.root / "reports" / "baseline-dev-gate.private.json",
                {
                    "schema": "cnot103_hamming_baseline_gate_receipt_v1",
                    "plan_sha256": hashlib.sha256(
                        (self.root / "reports" / "baseline-plan.private.json").read_bytes()
                    ).hexdigest(),
                    "result": baseline_gate,
                    "search_wall_seconds": baseline_dev_wall,
                },
            )
            self.status("baseline", baseline_gate["status"])
            if baseline_gate["status"] != "passed":
                self.quality_status = baseline_gate["status"]
                raise RunError(
                    "frozen dev Hamming baseline failed the predeclared admission threshold"
                )

            baseline_test_report = None
            baseline_test_metrics = None
            baseline_test_wall = 0.0
            baseline_search_wall = baseline_dev_wall

            # The two-update network is a timing diagnostic only. It never
            # selects a checkpoint or contributes a quality claim.
            smoke_qhat_manifest = self._search_manifest(
                "two_update_dev_search", resume_manifest,
                self.root / "reports" / "resume_step_2.stdout.json",
                "dev", dev_cohort, head="qhat",
            )
            smoke_qhat_report, smoke_qhat_metrics, smoke_qhat_dev_wall = self._search(
                "two_update_dev_search", smoke_qhat_manifest, dev_cohort, "dev",
                expected_head="qhat",
            )

            forecast_measured_at = time.time()
            elapsed_before_full = max(0.0, forecast_measured_at - self.started_at_epoch)
            remaining_updates = TARGET_ATTEMPTS - 2
            panel_scale = len(test_cohort["case_ids"]) / max(1, len(dev_cohort["case_ids"]))
            estimated_locked_hamming_seconds = baseline_dev_wall * panel_scale
            estimated_candidate_dev_seconds = smoke_qhat_dev_wall
            estimated_candidate_test_seconds = smoke_qhat_dev_wall * panel_scale
            search_eval_reserve_seconds = (
                baseline_dev_wall
                + estimated_locked_hamming_seconds
                + estimated_candidate_dev_seconds
                + estimated_candidate_test_seconds
            )
            remaining_work_seconds = estimate_remaining_work_seconds(
                setup_seconds=setup_seconds,
                update_seconds=update_seconds / PILOT_ATTEMPTS,
                remaining_updates=remaining_updates,
                checkpoint_reserve_seconds=CHECKPOINT_WRITE_RESERVE_SECONDS,
                baseline_search_seconds=search_eval_reserve_seconds,
                forecast_safety_seconds=FORECAST_SAFETY_SECONDS,
            )
            estimated_full_seconds = elapsed_before_full + remaining_work_seconds
            self.remaining_work_seconds = float(remaining_work_seconds)
            self.forecast_total_seconds = float(estimated_full_seconds)
            self.forecast_token = secrets.token_hex(16)
            runtime_gate = forecast_time_fits(
                self.started_at_epoch,
                self.remaining_work_seconds,
                time.time(),
            )
            _write_new_private_json(
                self.root / "reports" / "bounded-pilot-decision.private.json",
                {
                    "schema": "cnot103_bounded_pilot_decision_v1",
                    "source_revision": self.source_revision,
                    "holdout_revision": self.holdout_revision,
                    "holdout_sha256": self.holdout_sha256,
                    "dataset_sha256": dataset_sha,
                    "hardware": hardware,
                    "model": {
                        "parameters": MODEL_PARAMETERS,
                        "structural_blocks": STRUCTURAL_BLOCKS,
                        "input_slots": MODEL_INPUTS,
                        "batch_size": BATCH_SIZE,
                        "target_attempts": TARGET_ATTEMPTS,
                    },
                    "pilot": {
                        "attempts": PILOT_ATTEMPTS,
                        "fresh_end_progress": fresh_report["end_progress"],
                        "resume_end_progress": resume_report["end_progress"],
                        "setup_seconds": setup_seconds,
                        "update_seconds_per_attempt": update_seconds / PILOT_ATTEMPTS,
                        "benchmark_wall_seconds": benchmark_wall,
                        "baseline_dev": baseline_dev_metrics,
                        "two_update_qhat_dev_diagnostic": smoke_qhat_metrics,
                    },
                    "search_eval_reserve": {
                        "measured_hamming_dev_seconds": baseline_dev_wall,
                        "measured_two_update_qhat_dev_seconds": smoke_qhat_dev_wall,
                        "estimated_locked_hamming_seconds": estimated_locked_hamming_seconds,
                        "estimated_final_qhat_dev_seconds": estimated_candidate_dev_seconds,
                        "estimated_final_qhat_test_seconds": estimated_candidate_test_seconds,
                        "locked_test_not_run_before_dev_checkpoint_selection": True,
                    },
                    "forecast": {
                        "measured_at_epoch": forecast_measured_at,
                        "elapsed_before_full_seconds": elapsed_before_full,
                        "remaining_updates": remaining_updates,
                        "remaining_work_seconds": remaining_work_seconds,
                        "campaign_total_seconds": estimated_full_seconds,
                        "cleanup_reserve_seconds": FORECAST_CLEANUP_RESERVE_SECONDS,
                        "runtime_gate": "within_limit" if runtime_gate else "exceeds_limit",
                        "controller_budget_gate": "pending",
                    },
                },
            )
            self.status("baseline", "measured")
            if not runtime_gate:
                self.status("time_limit", "measured")
                raise TimeLimit("measured remaining work plus cleanup misses the 48-hour end-to-end gate")
            self._await_forecast_approval()

            full_manifest = self._build_manifest(
                "full_resume", dataset_sha, remaining_updates, TARGET_ATTEMPTS, True
            )
            full_report, full_wall = train_after_baseline_gate(
                baseline_gate,
                lambda: self._train(
                    "full_resume", full_manifest, 2, TARGET_ATTEMPTS, "training"
                ),
            )
            final_dev_manifest = self._search_manifest(
                "final_dev_search",
                full_manifest,
                self.root / "reports" / "full_resume.stdout.json",
                "dev", dev_cohort,
            )
            final_dev_report, final_dev_metrics, final_dev_wall = self._search(
                "final_dev_search", final_dev_manifest, dev_cohort, "dev"
            )
            dev_quality_comparison = compare_quality(
                baseline_dev_report,
                final_dev_report,
                expected_case_ids=dev_cohort["case_ids"],
                expected_panel="dev",
                expected_panel_digest=dev_cohort["digest"],
                expected_search=BEAM_CONFIG,
                baseline_head="hamming",
                candidate_head="qhat",
            )
            _write_new_private_json(
                self.root / "reports" / "dev-checkpoint-selection.private.json",
                {
                    "schema": "cnot103_dev_checkpoint_selection_v1",
                    "cohort_digest": dev_cohort["digest"],
                    "search": BEAM_CONFIG,
                    "comparison": dev_quality_comparison,
                },
            )

            final_test_report = None
            final_test_metrics = None
            final_test_wall = 0.0
            if dev_quality_comparison["status"] == "passed":
                # The selected full-training checkpoint has passed the frozen
                # dev criterion. Only now open the locked test panel.
                baseline_test_manifest = self._search_manifest(
                    "baseline_test_search", resume_manifest,
                    self.root / "reports" / "resume_step_2.stdout.json",
                    "test", test_cohort, head="hamming",
                )
                baseline_test_report, baseline_test_metrics, baseline_test_wall = self._search(
                    "baseline_test_search", baseline_test_manifest, test_cohort, "test",
                    expected_head="hamming",
                )
                baseline_search_wall = baseline_dev_wall + baseline_test_wall
                final_test_manifest = self._search_manifest(
                    "final_test_search",
                    full_manifest,
                    self.root / "reports" / "full_resume.stdout.json",
                    "test", test_cohort,
                )
                final_test_report, final_test_metrics, final_test_wall = self._search(
                    "final_test_search", final_test_manifest, test_cohort, "test"
                )
                quality_comparison = compare_quality(
                    baseline_test_report,
                    final_test_report,
                    expected_case_ids=test_cohort["case_ids"],
                    expected_panel="test",
                    expected_panel_digest=test_cohort["digest"],
                    expected_search=BEAM_CONFIG,
                    baseline_head="hamming",
                    candidate_head="qhat",
                )
                self.quality_status = quality_comparison["status"]
            else:
                quality_comparison = {
                    "status": "not_run",
                    "reason": "locked test withheld because the frozen dev acceptance threshold failed",
                    "dev_status": dev_quality_comparison["status"],
                }
                self.quality_status = dev_quality_comparison["status"]
            total_elapsed = max(0.0, time.time() - self.started_at_epoch)
            if total_elapsed > MAX_RUNTIME_SECONDS:
                raise TimeLimit("full run exceeded the end-to-end wall-time gate")
            final_report = {
                "schema": "cnot103_full_training_result_v1",
                "source_revision": self.source_revision,
                "holdout_revision": self.holdout_revision,
                "holdout_sha256": self.holdout_sha256,
                "dataset_sha256": dataset_sha,
                "feature_coverage": self.feature_coverage_summary,
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
                    "baseline_dev_search_wall_seconds": baseline_dev_wall,
                    "baseline_test_search_wall_seconds": baseline_test_wall,
                    "final_dev_search_wall_seconds": final_dev_wall,
                    "final_test_search_wall_seconds": final_test_wall,
                    "forecasted_full_seconds_before_training": estimated_full_seconds,
                },
                "quality": {
                    "cohort_receipt": "reports/panel-cohorts.private.json",
                    "dev_cohort": dev_cohort,
                    "locked_test_cohort": test_cohort,
                    "beam_config": BEAM_CONFIG,
                    "head": "qhat",
                    "baseline_head": "hamming",
                    "acceptance_minimum_solved_rate_gain": QUALITY_MIN_ABSOLUTE_SOLVED_RATE_GAIN,
                    "baseline_dev": baseline_dev_metrics,
                    "dev_paired_comparison": dev_quality_comparison,
                    "locked_test_evaluated": final_test_report is not None,
                    "final_dev_diagnostic": final_dev_metrics,
                    "baseline_locked_test": baseline_test_metrics,
                    "candidate_locked_test": final_test_metrics,
                    "paired_locked_test_comparison": quality_comparison,
                    "same_locked_cases_and_budget": True,
                },
                "hardware": hardware,
                "budget": {
                    "spend_at_start": self.budget_baseline,
                    "max_total_usd": 20.0,
                    "billing_outcome": "bounded_by_controller",
                },
            }
            _write_new_private_json(self.private_report_path, final_report)
            if self.quality_status == "passed":
                self.status("completed", "passed")
                return 0
            self.status("failed", self.quality_status)
            return 1
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
    parser.add_argument("--billing-baseline-epoch", required=True, type=float)
    parser.add_argument("--bootstrap-deadline-epoch", required=True, type=float)
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
        or not math.isfinite(args.billing_baseline_epoch)
        or not math.isfinite(args.bootstrap_deadline_epoch)
        or args.started_at_epoch <= 0
        or args.budget_baseline < 0
        or args.billing_baseline_epoch < args.started_at_epoch
        or args.billing_baseline_epoch > args.bootstrap_deadline_epoch
        or args.bootstrap_deadline_epoch <= args.started_at_epoch
        or args.bootstrap_deadline_epoch > args.started_at_epoch + MAX_RUNTIME_SECONDS
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
        bootstrap_deadline_epoch=args.bootstrap_deadline_epoch,
        billing_baseline_epoch=args.billing_baseline_epoch,
    )
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
