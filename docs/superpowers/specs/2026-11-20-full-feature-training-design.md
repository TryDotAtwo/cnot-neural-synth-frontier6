# Full-feature CNOT training run design

## Intent

Produce a reproducible run of the original 8-block, 136,421,807-parameter network for 100 epochs over 10,000 records on one 2×RTX 3060 instance. Keep all 102 feature inputs, the v25 unavailable-feature status path, pinned holdout data, checkpoint resume, equal-budget holdout search, the $20 total Vast cap, and the 48-hour end-to-end limit.

## Current boundaries

- The training manifest and Rust validator are present. The source-pinned profile currently has explicit n=10 contexts only; other contexts remain unavailable and must use v25.
- Existing checkpoints are immutable files. Keep every snapshot on Vast; do not prune old checkpoints.
- The target is 64 GB. At about 1.64 GB per full checkpoint, a 2,500-update interval would create about 100 snapshots (roughly 164 GB). The manifest now uses 25,000 updates per checkpoint, about 10 full-run snapshots, and a pre-run free-space gate.
- Vast status and control, remote execution, and private experiment files remain behind GitHub Actions. Windows and public Actions output remain source-only and redacted.

## Approaches considered

1. **One foreground GitHub-hosted job.** Rejected because GitHub-hosted jobs are limited to six hours, shorter than the 48-hour target.
2. **Local SSH watcher.** Rejected by the project boundary.
3. **Detached remote supervisor with sequential GitHub Actions monitors.** Selected. The supervisor runs on the existing Vast disk, keeps detailed logs and quality reports there, resumes only from a matching checkpoint, and writes a small status file. Sequential monitor jobs use fresh temporary SSH keys, report only allowlisted state, and a finalizer stops and verifies the exact target.

## Run contract

- One exact labelled target with two RTX 3060 GPUs; no second training target.
- 250,000 attempted updates at batch size 4.
- Full model and input profile are validated before paid training; manifest must select all 102 inputs with `runtime_packet_layout_v25`.
- Audit actual feature statuses on train and dev. Do not infer contexts from width; unavailable providers remain v25 status tokens with zeroed numeric payloads.
- Perform a small fresh-process resume check before the full segment. Use a private baseline and final beam-search manifest with the same pinned dev roots and identical search budget.
- Keep data, checkpoints, logs, measurements, and quality reports on the Vast disk. Public Actions output may contain only check time and allowlisted operational state.
- Gate projected total Vast charges before start and at each restart. Preserve the disk and stop the exact target after completion or failure.
- Count dataset preparation, profile/data audits, builds, resume check, checkpoint writes, training, and final quality evaluation in the measured end-to-end time.

## Completion evidence

The goal is complete only when the private Vast receipt proves 250,000 attempted updates, the full model/102-input/v25 contract, successful fresh-process resume, pinned holdout comparison at equal beam budget, measured total elapsed time at or below 48 hours, total Vast charges below $20, and verified stopped status for the exact target.
