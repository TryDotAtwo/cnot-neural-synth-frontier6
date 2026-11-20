# Budget-gated 2x3060 dataset rebuild

## Intent
Rebuild a fresh, reproducible 10,000-row training panel on a newly rented 2x RTX 3060 host so the next benchmark can use a pinned dataset SHA. The Windows workspace remains source-only; generated data and receipts stay on the Vast container disk.

## Goals
- Rent at most one verified, reliable 2x RTX 3060 target only when current project charges plus a 51-hour projection (48h training plus up to 3h rebuild), seven days of stopped storage, and a $0.25 reserve for redacted status traffic fit the existing $20 cap. Dataset, checkpoint, log, and quality-result files stay on Vast and are not exported.
- Rebuild and validate the frozen dataset at the exact source revision.
- Emit only redacted status values in public Actions logs.
- Stop the target on every exit path and preserve its container disk for follow-up benchmark and quality work.

## Non-goals
- No model training, quality evaluation, benchmark, dataset upload, volume deletion, or instance destruction in this stage.
- No automatic rental from ordinary source pushes.

## Constraints
- All Vast API and SSH operations run from GitHub-hosted Actions.
- The Vast API key and temporary SSH private key never enter repository files or logs.
- No raw API output, dataset content, SHA-256, training output, checkpoint, or private quality result is published to GitHub.
- Selection requires one exact target label, 2x RTX 3060, 12 GB per GPU, verified=true, reliability >= 0.95, a direct SSH port, and enough free workspace.
- Failed or ambiguous preflight stops before creation; uncertain creation is resolved by the stable label without retrying create.
- Cleanup stops and verifies the selected instance, removes the ephemeral SSH key, and keeps the instance/disk.

## Approaches
1. Explicit marker commit starts a one-shot Actions run — chosen. It isolates paid side effects from code pushes, preserves the current Vast-only data boundary, and makes the $20 gate part of the run.
2. Rent automatically whenever the dataset workflow sees no target — rejected because routine source changes could incur charges.
3. Store the regenerated dataset in a public Actions artifact or repository — rejected because it violates the private data boundary.

## Implementation decisions
Use a focused controller script with pure, unit-tested offer, label, and budget decisions. Reuse the existing remote dataset rebuild validator; bootstrap Cargo on the remote target if needed. A separate paid workflow reacts only to the committed run marker, reports allowlisted states, and always invokes a label-scoped stop/verification cleanup.

## Acceptance
A run is accepted only when the dataset builder and canonical validator both report success, the temporary SSH key is detached, and the exact target is verified stopped. The follow-up benchmark and holdout-quality evaluation are a separate phase using this retained disk and dataset SHA.
