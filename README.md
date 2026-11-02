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
