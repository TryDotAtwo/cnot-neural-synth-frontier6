# CNOT neural synthesis — Rust/CUDA refactor

This repository contains the native refactor on branch refactor/rust-cuda-103.
The runtime is Rust with LibTorch and project-owned CUDA/C++ kernels. Build,
GPU validation, and training work for this branch run on Vast.

No Kelipaev code, words, labels, or derived training data are included. Source
archives and the pinned feature/model manifests are retained for provenance.
Old Molab bootstrap and replay scripts are archived under history/molab-bootstrap;
they are not part of the runtime or build path.

The model currently has 102 registered model-input features. H07 is implemented
as a post-forward prediction-spread diagnostic, not as a model input. The
archived source contract leaves H07's probe-generation and availability policy
unspecified; no source-parity claim is made for it or for the candidate
providers whose pinned sources are marked unimplemented.

CUDA acceptance now covers a real 102-input Bellman training update, checkpoint
save, restore in a fresh process, and the next update on RTX 3060. A matching
H07 diagnostic test runs actual CUDA model forwards over an equivalent inverse
view and a cost-preserving wire relabeling. These are small synthetic fixtures,
not production training or a synthesis-quality result.

Production-scale training, held-out leakage audit, the 1000x10000 campaign,
quality acceptance, and exact source parity for the unimplemented providers
remain open.

## Narrow Python forward-parity fixture

The test fixture native/tests/fixtures/python_a01_forward_small.json records a
small numerical comparison against the Python model from the pinned
packages/v13_recovery_candidate_v17_20261001.zip archive
(SHA-256 9947df2dc27080b38eb34e7551be39ce0536f2409aa733c5f201c317ed1a4866).
It covers the A01-only feature profile, two three-wire matrices (identity and a
nonidentity state), and the one-block model configuration stored in the fixture.
The same 10,887 state-dict parameters are loaded into the Rust model before the
forward comparison.

The check compares the A01 feature values, value prediction, allowed-action
mask, proposal logits, and qhat outputs. Floating outputs use a relative-plus-
absolute tolerance of 2e-5 times (1 + absolute reference value); the identity
value is checked as exact zero. Run it with
cargo test --locked --manifest-path native/Cargo.toml --features torch --test python_model_forward_parity.

This is a narrow CPU forward-parity check for one source-backed feature and one
model profile. It does not establish parity for all 102 model inputs, the 103-ID
catalog, CUDA training, production data, or the complete training workflow.

## Python structural-provider parity fixture

The fixture at native/tests/fixtures/python_structural_33.json is generated from
the pinned Python feature schema and structural producer. It covers all 33 feature
IDs marked implemented by that Python producer (A01-A20, B01-B12, and F01) over
all 168 matrices in GL(3,2), plus five width-4 matrices constructed from recorded
CNOT paths. The test replays each path, verifies matrix uniqueness, and compares
IDs, roles, channel counts, shapes, availability, computed status, and every
numeric value against the pinned Python fixture.

Run with: cargo test --locked --manifest-path native/Cargo.toml --no-default-features --test python_structural_parity

This establishes parity only for those 33 source-backed structural providers. It
does not establish parity for the remaining 70 features, model forward/training,
CUDA behavior, production data, or synthesis quality.
