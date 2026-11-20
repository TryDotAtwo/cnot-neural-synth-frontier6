# Full-feature CNOT training run implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Run and verify 250,000 updates of the original CNOT network on one 2×RTX 3060 instance in no more than 48 hours and $20 total Vast spend.

**Architecture:** A detached supervisor performs training on the existing Vast disk. A chained GitHub Actions workflow starts it, monitors it in sequential windows within the six-hour per-job limit, and stops the exact target. Detailed artifacts stay on Vast; Actions emits only allowlisted operational states.

**Tech Stack:** Rust/CUDA native trainer, Python manifest and remote controller, GitHub Actions, Vast API through Actions.

**Spec:** `docs/superpowers/specs/2026-11-20-full-feature-training-design.md`

## Global Constraints

- Preserve the original 8-block, 136,421,807-parameter model and all 102 model inputs.
- Use `runtime_packet_layout_v25`; unavailable features have zero numeric payload and an explicit unavailable status.
- Train exactly 250,000 attempted updates at batch size 4 over 10,000 records.
- Keep dataset, logs, checkpoints, measurements, and quality results on the existing Vast disk.
- Do not use local Vast CLI, SSH, SCP, desktop, browser automation, CUA, or local datasets/results.
- Do not delete existing Vast data or unique checkpoint snapshots.
- Keep total Vast charges under $20 and measured end-to-end time at or below 48 hours.
- Keep public Actions logs, summaries, and artifacts redacted to allowlisted operational state.

## Review Focus

- A malformed or stale dataset/profile pin must fail before paid compute.
- A target in loading/starting/stopping must not be treated as stopped.
- A target with insufficient projected budget or checkpoint free space must not start training.
- A rerun must detect an active supervisor and resume only from a checkpoint whose contract matches.
- Private progress and quality numbers must never appear in Actions logs or summaries.

---

### Task 1: Bound immutable checkpoint storage

**Files:**
- Modify: `.github/scripts/build_training_manifest.py`
- Test: `.github/scripts/test_build_training_manifest.py`
- Workflow: `.github/workflows/check-full-training-manifest.yml`

- [x] Add a manifest behavior test bounding the full-run snapshot count at ten.
- [x] Observe it fail at 100 snapshots with a 2,500-update interval.
- [x] Set the manifest interval to 25,000 updates and verify the branch workflow passes.

### Task 2: Build the private remote supervisor

**Files:**
- Create: `.github/scripts/vast_run_full_training.py`
- Test: `.github/scripts/test_vast_run_full_training.py`
- Create: `.github/scripts/run_full_training_on_vast.sh`

- [ ] Create a dedicated run directory on Vast and reject collisions unless the stored run identity matches exactly.
- [ ] Build and validate the training manifest on Vast; check dataset SHA, source/profile pins, 102 inputs, v25 schema, 2-GPU target, and checkpoint free-space reserve.
- [ ] Run train/dev coverage and a bounded throughput measurement, saving raw output privately.
- [ ] Run two fresh-process updates to prove resume, then write an equal-budget baseline search over a pinned dev index set.
- [ ] Resume the same checkpoint contract to exactly 250,000 attempts with 25,000-attempt checkpoint cadence.
- [ ] Run the same pinned equal-budget search on the final checkpoint and store the paired comparison privately.
- [ ] Write a small status file containing only enumerated phase/state values; keep full progress, metrics, receipts, and logs private.

### Task 3: Add the target and spend controller

**Files:**
- Create: `.github/scripts/vast_full_training_controller.py`
- Test: `.github/scripts/test_vast_full_training_controller.py`

- [ ] Require exactly the configured target label and two verified RTX 3060 GPUs.
- [ ] Require observed stopped/exited before start and check projected spend for training plus stopped-storage and transfer reserve before any paid transition.
- [ ] Start or monitor only this target. Attach a temporary SSH key for each operation and verify its removal.
- [ ] Treat the budget deadline, disk gate, state transitions, supervisor identity, and incomplete run as explicit closed-failure states.
- [ ] Stop and verify the exact target on terminal completion or failure, without deleting its disk or artifacts.

### Task 4: Orchestrate under GitHub Actions limits

**Files:**
- Create: `.github/workflows/vast-full-feature-2x3060-training.yml`
- Create: `.github/paid-runs/trigger-full-training-2026-11-20.txt`

- [ ] Add one start job and sequential monitor jobs each below the GitHub-hosted six-hour limit.
- [ ] Add an `always()` finalizer that stops and verifies the exact target on success, failure, timeout, or cancellation.
- [ ] Keep permissions minimal, disable overlapping runs, and publish only redacted allowlisted operational status.
- [ ] Trigger the workflow only after the rebuilt dataset passes pinned validation and all source checks pass.

### Task 5: Run scientific and operational acceptance

- [ ] Rebuild and validate the 10,000-record dataset from the pinned holdout source.
- [ ] Run the end-to-end training workflow on the same 2×RTX 3060 target.
- [ ] Verify full architecture, 102-input/v25 contract, coverage, fresh-process resume, and 250,000 attempts.
- [ ] Compare baseline and final search on the identical pinned holdout roots and identical beam budget.
- [ ] Record preparation/audit/checkpoint/training/evaluation elapsed time and total Vast charges.
- [ ] Verify the exact target is stopped and all private artifacts remain on its disk.
