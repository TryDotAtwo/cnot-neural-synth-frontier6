# Agent instructions

## Local and remote data boundary

- Keep the Windows workspace limited to source code and small text configuration. Do not download or retain datasets, training records, checkpoints, model weights, experiment results, run logs, build archives, or generated payloads on Windows.
- Keep the existing Vast disk as the source for remote data. Do not delete, destroy, or clean it during transfer or recovery work.
- Route Vast status and control through GitHub Actions. Do not use local Vast CLI, SSH, SCP, browser automation, desktop control, or CUA for this project.
- Use GitHub-hosted runners for remote operations. Never print API keys, SSH credentials, remote addresses, raw API responses, raw logs, datasets, checkpoints, model weights, or private quality results to a public repository, Actions log, summary, or artifact.
- Public Actions summaries and artifacts may contain only a redacted operational report: check time, query outcome, and a small allowlisted status value. Keep raw and private experiment outputs on the existing Vast disk unless Ivan explicitly authorizes a private destination.
- Do not start a stopped instance for a smoke check. Do not rent another instance. Keep all authorized Vast compute within the active $20 total budget; stop at that ceiling. Do not start paid compute unless the current task requires a real training run.
- Keep source changes isolated from unrelated local modifications. Never stage local archives, logs, checkpoints, datasets, or generated outputs.