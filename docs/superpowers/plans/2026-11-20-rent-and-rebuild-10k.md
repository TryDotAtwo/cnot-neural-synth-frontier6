# 2x3060 Dataset Rebuild Implementation Plan

> **For agentic workers:** Execute tasks inline under the approved GitHub Actions boundary; do not use a local data or compute path.

**Goal:** Rebuild and validate a new 10k training dataset on one 2x RTX 3060 host without exceeding the authorized $20 Vast cap.

**Architecture:** A pure decision layer selects one strict 2x3060 offer using current project charges and a 51-hour projection. A paid Actions workflow is armed only by a dedicated marker commit, creates one stable-label target, runs the existing remote dataset builder, then stops and verifies the instance while retaining its disk.

**Tech Stack:** Python 3, Vast CLI/API, GitHub Actions, existing Rust dataset validator.

**Spec:** docs/superpowers/specs/2026-11-20-rent-and-rebuild-10k-design.md

## Global Constraints
- Vast control and SSH run only on GitHub-hosted runners.
- Total project spend plus the 51-hour forecast, seven days of stopped storage, and a $0.25 status-only transfer reserve must stay at or below $20.
- Public outputs contain only time, query result, and allowlisted status.
- Data, hashes, receipts, and private results stay on the retained Vast disk.
- Cleanup stops only the exact stable-label instance and never destroys it.

## Review Focus
- A 10k train panel at batch 4 uses 250,000 attempted updates for exactly 100 sampler epochs. The CLI treats `attempts` as one invocation's limit; `target_attempts` is an absolute cap applied against restored checkpoint progress so resumes cannot add another 100 epochs.
- Missing or malformed price, reliability, GPU count, or invoice data fails closed.
- A duplicate or unrelated 2x3060 target prevents creation.
- A create timeout never causes a second create request.
- Failed dataset build still stops the newly created instance.
- SSH key removal failure is surfaced and the instance disk remains intact.

## Interfaces
- vast_rent_rebuild_dataset.py imports existing vast_probe_existing_2x3060.py API/key helpers and invokes vast_rebuild_remote_dataset.main() after setting the exact target label.
- The paid workflow installs vastai, supplies VAST_API_KEY only to the controller, and runs cleanup in an if: always() step.
- The source-check workflow runs only Python unit tests and syntax checks; it has no Vast secret or paid side effects.

## Tasks

### Task 1: Pure budget and offer selection
1. Add test_vast_rent_rebuild_dataset.py first.
2. Add failing cases: test_offer_filter_requires_exact_verified_2x3060, test_budget_gate_includes_50h_and_transfer_reserve, test_invalid_costs_fail_closed, and test_label_resolution_rejects_duplicates.
3. Run the new unittest workflow and confirm the expected import/behavior failures.
4. Implement pure selection functions in .github/scripts/vast_rent_rebuild_dataset.py.
5. Run the same unit tests; all named cases must pass.

### Task 2: Remote rebuild controller
1. Add a bounded controller that inventories targets/volumes, reads project charges, queries eligible offers, creates once, and resolves by the stable label.
2. Bootstrap Cargo on the target only when missing, then run the existing vast_rebuild_remote_dataset.main() with VAST_TARGET_LABEL set.
3. In finally, remove any temporary key and stop/poll only the exact target label; never destroy the instance.
4. Extend the unit suite with create-timeout/no-duplicate and cleanup state cases.
5. Run the full Python controller suite through GitHub Actions.

### Task 3: Redacted source and paid workflows
1. Add .github/workflows/vast-rent-rebuild-dataset-check.yml for syntax and unit checks, triggered by controller/test/workflow source changes and never given a Vast secret.
2. Add .github/workflows/vast-rent-rebuild-dataset.yml, triggered only by .github/paid-runs/2026-11-20-rebuild-10k.json or explicit workflow dispatch; set a 175-minute job timeout and a concurrency guard.
3. Add the inert marker file only after Task 1 and Task 2 checks pass.
4. Run the workflow. Acceptance requires dataset_build=accepted, target_stop=verified, and temporary_ssh_key=detached; no SHA or private artifacts appear in logs.
5. Read the run/job status and stop after the dataset phase. Preserve the stopped instance for the separately measured benchmark/quality phase.

## Verification
- python3 -m unittest .github/scripts/test_vast_rent_rebuild_dataset.py passes on a GitHub-hosted Linux runner.
- Controller syntax parses.
- The paid Action has no raw API/log/artifact outputs and its cleanup step verifies stopped state.
- The dataset validator accepts the rebuilt file on Vast; no full training or quality claim is made in this task.
