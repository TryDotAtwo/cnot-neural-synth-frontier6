# Native 103-feature port

The canonical packet layout contains 103 feature IDs. The model consumes 102
inputs; H07 is a post-forward diagnostic only. Producer coverage is deliberately
split by input contract:

- 41 feature IDs have built-in structural producers: A01-A20, B01-B12, F01,
  G05, C01, D03, and H01-H05.
- FeatureRegistry::with_all_model_input_providers(contexts) registers all 102
  model-input features using explicit validated context records. The runtime
  coverage gate accepts computed values and correctly masked inapplicable inputs,
  but rejects unsupported, resource-limited, or unrequested model inputs. H07
  remains outside the input tensor and is evaluated after the model forward pass.
- FeatureRegistry::with_default_context_free_providers(max_n) adds 23 more
  providers (C05, E01/E04/E05, F02-F10, G01-G04/G06-G10, and H06), so 64 IDs
  are selectable with the default operational profile. It leaves 31 physical
  context-bound IDs, six template-configured E IDs, horizon-configured I10, and
  H07 unregistered: full-103 selection still fails closed for those 39 IDs.
- C02, C03, and C04 can be registered for explicit labeled wire cuts. They
  require an `augmentation_only` symmetry policy and are bound to the ordered
  cut set by the producer version. A matrix width with no registered cut is
  reported as unsupported.
- C05 can be registered for the deterministic partial-shear elimination block
  shared with the paid-basis adapter. It reports that block's rank only when a
  nonzero partial shear is applicable.
- C06 and C07 can be registered for explicit restricted row- and column-space
  intersection probes. C09 can be registered for explicit restricted-rank
  increment probes that include a candidate parity vector. These providers
  also require `augmentation_only`, and their context is bound to the producer
  version.
- C10 can be registered for selected physical CNOTs from an explicit alphabet;
  it reports exact residual rank after that action, not distance.
- C08, I01-I04, and I11 can be registered for an explicit physical linear
  boundary. The provider reproduces the source's ordered port/equation tokens,
  incidence, protected-input projection, weighted zero-phase eligibility, and
  rank descriptors. I01/I02 retain dynamic group shapes. Port/equation budgets
  are checked before producing values, and `FeaturePolicy.max_bytes` can cap
  the emitted payload.
- D01, D02, and D04-D08 can be registered for explicit disjoint physical
  source/target blocks. Pattern and pair budgets are enforced before features
  are emitted; absent sparse bins keep their exact count semantics.
- F02 can be registered as a solver-assisted stage-35 pair view. It emits
  `connectivity + zero_diagonal_count` only after exact pair-scope recognition
  and independent word replay; outside that scope it is unavailable. The
  value alone carries no theorem witness or lower-bound claim.
- F03-F05 port the stage-26 projected shared-parity certificate. F03 exposes
  the deletion/target/obstruction dimensions, F04 preserves labelled projected
  coset and multiplicity relations, and F05 reports the checked lower bound
  beside its difference from the elementary bound. A separate low-pivot
  pairwise-coset verifier checks the constructed certificate before values are
  admitted. The result is a lower bound, never an exact distance. Width and
  checker-pair budgets are explicit. Pinned constructor and checker SHA-256:
  `6d29d12d2eb369e120533e1961f4b984429643a08f93cc4db99b148a9088bc05` and
  `4dfda79863fe7a7b57785e023471cc5d0e8dac21cc3313271cacf4bf97db08ce`.
  The source metadata also pins proof-note hash
  `0daadfb3738478b6ee95c11aad4d67177a1c17613eba5e3c91dc041151676fea`; that
  Markdown note is absent from the checked-in local source snapshot.
- F06 registers the source stage-35 pair recognizer as a six-channel group
  feature. It emits the active 2x2 frame bits, transformed external-column
  counts, and physical incidence roles for the two active wires and remaining
  sources. Exact values pass independent re-recognition and word replay;
  outside the theorem scope the feature is unavailable, and width/alphabet
  limits remain explicit. It requires `augmentation_only` symmetry and the
  solver-assisted policy.
- F07 can be registered as the three-channel guarded stage-37 result:
  exact zero-excess, certified failure of only the changed-row lower-bound
  length, or unsupported. Exact results are independently re-recognized and
  their CNOT word is replay-verified; a negative result does not claim that
  longer words do not exist. Width is explicitly capped and the solver-assisted
  policy is required.
- F08 ports the guarded stage-38 one-excess recognizer. It contracts temporal
  row histories with arbitrary-precision integer counts, admits exact labels
  only after reconstructing and replay-verifying a word, and preserves budget
  exhaustion as `resource_limit` rather than a negative label.
- F09 and F10 can be explicitly registered with
  `register_general_compiler_bounds` as solver-assisted global providers. F09
  emits `[verified upper bound, certified lower bound, gap, bounds meet]`.
  Its upper bound is a deterministic full-matrix Gauss-Jordan CNOT word, replayed
  against the endpoint; its lower bound is the maximum of changed-row count,
  changed-column count, and `rank(M xor I)`. F10 exposes proof flags and
  explicitly reports that no exhaustive optimality search was performed. A
  shortest-word result is proved only when the certified bounds meet. These are
  versioned T2 contracts: the archived schema marks F09/F10 unimplemented and
  has no legacy producers, so this does not claim source parity.
- The separate stage-35 recognizer is an exact-label oracle for invertible
  matrices differing from identity in at most two rows, under the full
  all-to-all unit CNOT alphabet. It emits a replayable word and accepts the
  label only when its cost matches the theorem's global lower bound. The proof
  source is the archived
  `research_v3/artifacts/frontier61_source_v1/stage35_geodesic_vertex/35_pair_kernel_to_global_geodesic_ru.md`,
  pinned by SHA-256
  `1b8ef924af6c68af51dd67e0be98148ad7f8e8c272d379ad44943b7baf3fade5`.
  This oracle is exposed independently from F06 registration.
- E02, E03, and E06-E09 register together for an explicit ordered
  source-template choice set, rational probe points, and exact resource budgets.
  They expose the all-half expectation, configured-point expectations, pairwise
  mixed differences with physical-wire incidence, coordinate-rounding cost,
  finite-template optimum, and multistart diagnostics. Values are exact rational
  calculations rounded to float32; checked `u128` overflow is reported as a
  resource limit. Their scope is only that fixed template and partial-shear
  block, not a full-alphabet distance or optimum claim.
- E01, E04, and E05 can be explicitly registered for the source-defined
  zero-template partial-shear cost, ordered-pair gradient, and gradient
  statistics. They require `augmentation_only` because the source block fixes
  the first half of the physical wires. A direct source-parity fixture pins the
  Python-v17 archive plus the basis-gradient and basis-macro module hashes, then
  compares values, roles, channels, shapes, and status on 32 CNOT-constructed
  matrices at each width 4, 5, and 6. This establishes parity for those sampled
  states only; it does not establish parity for other E-family providers.
- G01-G04 can be explicitly registered as source-relative CNOT column-diffusion
  profiles. Rust uses uniformization over the exact finite class chain and
  exposes the source's `floating_semigroup` status without claiming a formal
  numerical error certificate. Four time samples are fixed by the canonical
  feature layout; their values, floor, and width budget are bound into the
  producer version. These providers are opt-in.
- G06 can be explicitly registered as the source-relative rank-2 pair-diffusion
  profile. It uses the exact ordered-pair transition counts, class-size
  stationary masses, source floor/zero/underflow channels, and an explicit
  pair-state budget. Diagonal pairs are zero-filled as in the source. It is
  opt-in and reports singleton inputs as unsupported.
- G07-G10 can be explicitly registered as shared-word Monte Carlo diffusion
  moments. They reuse each sampled ordered-CNOT word across all physical
  columns and report empirical covariance/uncertainty channels. The seed path
  matches NumPy 2.2.6 `default_rng` for the source's u64 seeds; these providers
  require `augmentation_only` because finite seeded samples are not exactly
  relabel invariant. The sample count, time multiplier, width budget, and RNG
  compatibility identity are bound into their producer version. Use
  `FeaturePolicy::complete_103_augmentation_only` when requesting the full
  manifest with these providers.
  A pinned Python-source parity fixture checks all G01-G10 on all six
  GL(2,2) and 168 GL(3,2) states, plus 12 seeded CNOT-generated states at
  each width 4-6. It pins the source archive and producer-module hashes,
  NumPy 2.2.6, SciPy 1.15.3, and the diffusion grid; this establishes parity
  for the Rust CPU providers at that configuration, not the CUDA
  implementation's separate RNG stream.
- I10 can be explicitly registered as an exact leading-term response over the
  full ordered CNOT alphabet. It is solver-assisted and bounded by configured
  maximum width, degree, layer-state count, and transition count. It returns a
  leading positive degree and its exact multiplicity projection only after all
  requested layers complete. Exact coefficients use checked `u128` counters;
  counter overflow is reported as a resource limit. A horizon miss, alphabet
  mismatch, or exhausted budget is unavailable; no-hit is never reported as an
  exact zero. `evaluate_i10` also exposes the complete bounded positive series,
  scope, transition count, and status for direct auditing.
- I05 can be registered for explicit labelled `U`, `V`, `A0`, and `D0` port
  matrices and an ordered physical row/column block `C`. The Rust provider ports
  the stage-55 GF(2) reduction and emits only residual rank and free hidden
  width. Its producer version binds every port, physical index, budget, and the
  primary-source SHA-256
  `30c90c396e8f5f1d43921f09a0d07ca723afa6e38a9c0ebed4f68b5b2aab5353`.
  It requires `augmentation_only`; the values describe a matrix-product fiber,
  not CNOT distance. This is a CPU Rust provider; CUDA integration remains open.
- I06 ports the pinned stage-56 labelled shared-relation context. It retains the
  full relation kernel of the selected physical rows and each independent,
  ordered covector readout; rank-only summaries are not substituted. The
  producer version binds the ordered rows, coordinates, covectors, budgets, and
  source SHA-256
  `b14e85497632860665754a4014c34a4bcf149a9f29a1af6cd3710f401559c7ef`.
  It requires `augmentation_only` and is currently a CPU Rust provider; GPU
  feature-kernel integration remains open.
- I07 ports the pinned stage-52 labelled interface sector. It emits the exact
  central Fourier rank `kappa` and representation-size exponent
  `internal_width * kappa`, never the exponential dimension itself. It checks
  the physical block form, all-to-all alphabet, width/frequency budgets, and
  `augmentation_only` symmetry. States outside the explicit interface are
  `inapplicable` and masked; alphabet/policy/context mismatches remain
  unsupported and exhausted budgets remain resource-limited. Source SHA-256:
  `44fa3c82aa7b5c59542cb4101c40b8f437fca1e7baf47f66a3c395627c49c465`.
  Rust source-parity fixtures cover all 16 labelled 2x2 sectors under a
  noncanonical wire partition, a 144-wire source case, a frequency row crossing
  the 64-bit boundary, and fail-closed interface/policy/budget cases. This is a
  CPU Rust provider; CUDA integration remains open.
- I08 ports the pinned stage-59 affine optional-CNOT packet. It preserves the
  declared gate order and local selector-factor scopes, emits exact cut width,
  live-selector count, and their sum, and rejects nonlinear endpoint maps
  instead of linearizing them. The source SHA-256 is
  `63824b056fa1983a10b3d535532c2975c7c3a1c96cfb2a90267ead2a1b168987`.
  It requires `augmentation_only`; the current implementation is a CPU Rust
  provider. CUDA feature-kernel integration remains open.
- G05 is an exact four-bin histogram of the ordered pair of physical columns;
  its channel order is `00, 10, 01, 11` and includes diagonal pairs.
- H05 emits four aligned raw values for every ordered physical wire pair:
  `M[i,j]`, `Mᵀ[i,j]`, `M⁻¹[i,j]`, and `M⁻ᵀ[i,j]`. The model's H05 pair
  projection turns these into learned pair tokens. This producer is T2 because
  it computes a GF(2) inverse. This is a concrete input representation; it does
  not claim exact relabel invariance or that all four transformed tasks have
  equal cost for every CNOT alphabet.
- H06 can be explicitly registered as the exact 15-channel difference of
  A15/A16/D03 descriptors across `Mᵀ`, `M⁻¹`, and `M⁻ᵀ` relative to `M`.
  Ordered wire pairs transform equivariantly with physical relabelings; the
  producer binds exact inverse and matching conventions.
- D09-D14 can be registered for explicit physical source/target blocks. They
  use the verified paid-basis candidate and replay its reduced CNOT word over
  the full matrix before emitting endpoint features. D14 keeps the endpoint
  digest in packet provenance, outside the numeric tensor. The archived
  feature schema marks these IDs unimplemented and contains no legacy
  producers; this is a versioned T2 operational definition, not a claim of
  legacy-source parity or global optimality.
- The pinned reference archive's `research_v3/configs/feature_schema.json`
  marks 33 of 103 IDs as implemented by its base registry. Its separate opt-in
  Python modules also provide C01, D03, E01/E04/E05, G01-G10, H01-H04, and I10.
  Together, the archive has executable producers for 53 IDs; the other 50 have
  no executable Python producer there. Rust implementations for those missing
  IDs are versioned T2 contracts, not legacy-source parity. This distinction
  does not establish a complete production registry. H07's pinned entry is
  a group/1 placeholder with producer version unimplemented; the local packet
  layout preserves that placeholder. H07 is a post-forward diagnostic/ensemble,
  not a model input. The native runtime now executes the actual model on
  caller-built canonical/view/relabel packets, runs the value head for each,
  and reports population variance with probe provenance. The exact source
  probe-generation and availability policy
  remain unspecified, so this does not claim source parity or complete-103
  model-input coverage.
  I09 now ports the bounded quadratic compiler equations and fixed-prefix
  alpha/beta/rank response from the separately pinned Stage-60 source in the
  A15 source-only candidate, SHA-256
  `fcdaec83bff7c111261dfc46c2040ddbbc349dbbc11a712a31af945dca3c6339`.
  It requires `augmentation_only` and emits a source-hashed CPU Rust feature;
  it does not run a solver or port Stage 60's separate weighted exact
  contraction. The A15 candidate declares `runtime_acceptance:false`,
  `gpu_executed:false`, and `full103_width_group_parity:false`, so these bounded
  fixtures do not establish wide-source parity or complete-103 acceptance.

Native `runtime_packet_layout_v24` keeps the stable 103-ID manifest while
resolving role/channel shape from each research-v3 provider. Built-in shapes
are fixed; reserved IDs use a non-executable placeholder until a provider
declares its actual contract. `selected_specs` rejects those IDs, so
placeholders cannot enter the model.
`code/cnot103/layout.py` is a separate legacy packet contract (`v23`) with fixed
role/channel shapes. Conversion to that interface succeeds only when every
provider shape matches it; the two layout IDs are not interchangeable.

The native feature tests currently exercise I05's hand-derived and 72-case
stage-55 source-golden fixtures, paired-rank edge case, and fail-closed
width/symmetry cases; I06's stage-56 full-kernel/readout fixture, independent
GF(2) enumeration across every invertible 3x3 matrix, labelled-port distinction,
resource limits, and packed-word boundary; aligned H05 multi-view pair inputs,
H06 descriptor differences, wire relabeling, and packed-word boundaries; H07's
post-forward population variance, digest binding, order stability, and rejection
of missing probes/nonfinite predictions; exact
F02's guarded connectivity/diagonal value and fail-closed scope; F03-F05's
source-golden outputs for every GL(3,2) matrix, pinned to the Python constructor
and independent checker, plus the complete lower-bound comparison; F06's
six-channel stage-35 values, group incidence,
source metadata, and fail-closed scope; F07's exact/negative/unsupported source
projection; F08's guarded one-excess projection; F09/F10's bound values and
proof flags, solver and evaluation-budget separation, all 168 invertible 3x3
compiler replays, row-swap reconstruction, and gate-limit failure; exact GL(3,2)
distance partition and replayed certificates, width and solver gates, and
producer-version parity;
counts, G01-G04 source-relative column diffusion, G05 ordered column-pair
histograms, I10 exact full-alphabet response (source degree/coefficient fixture,
wire relabeling, horizon miss, solver-policy gate and resource cap), G06
source-relative pair diffusion (SciPy profile values and exact
transition closure across all GL(3,2) matrices), G07-G10 fixed-seed NumPy goldens plus direct G01-G10 Python-source parity on all GL(2,2)/GL(3,2) states and seeded width-4-to-6 states, E02/E03/E06-E09 fixed-template exact response and
E01/E04/E05 partial-shear
gradients and statistics, C02-C05 cut and partial-shear ranks, C06/C07
restricted-subspace intersections, C09 candidate-rank increment,
C10 selected residual ranks, C08/I01-I04/I11 explicit boundary features,
D01/D02/D04-D08 physical-block parity features,
incidence, pattern budgets, symmetry policy, alphabet mismatch, width mismatch,
invalid contexts, relabeling, transactional provider registration, I09's
direct one-gate compiler replay, all four GL(2,2) pinned Stage-60 alpha/beta/
rank/mixed-rank fixtures, physical incidence, and fail-closed limits, the
bounded stage-35 exact-label oracle, I07 stage-52 Fourier ranks over the source's 16
labelled sectors and wide packed inputs, and I08 stage-59 source-golden cut/live
values, selector IDs beyond 64 bits, physical incidence, affine rejection, and
budget handling. Native
training primitives also include a CPython-compatible sampler, checked progress
counters, and an immutable hash-checked checkpoint container. With the `torch`
feature, `nn_training` applies guarded scalar-loss updates using AdamW and saves
model weights, named optimizer moments, sampler, progress, objective bytes, and
caller-managed RNG bytes in the checkpoint. Its fresh-process integration test
compares a resumed autograd training run with an uninterrupted run. This is a
generic training runtime test, not the CNOT Bellman objective or full 103-feature
training acceptance. Torch-gated tests pin Rust parameter names and shapes to
the Python reference state dict generated from
`v13_recovery_candidate_v17_20261001.zip` (SHA-256
`9947df2dc27080b38eb34e7551be39ce0536f2409aa733c5f201c317ed1a4866`). The Rust
model now uses the reference module paths, fused attention projection layouts,
and PyTorch-compatible linear initialization bounds. The source model's 94
state-dict names/shapes were independently regenerated and matched to the
fixture. The complete native suite passed on Windows/MSVC with CPU LibTorch
2.9.0: 73 tests across Rust, model, Bellman objective, and resume/checkpoint
targets. This run used an external D:-drive Cargo target cache and a temporary
local Cargo patch that disables `zip`'s unused zstd feature; neither changes the
project dependency manifest or lockfile. The fresh-process test proves generic
trainer/autograd resume equality, while the Bellman objective test exercises a
synthetic feature fixture. These checks do not establish production 103-feature
training or a fresh-process run of the complete CNOT objective. External sealed
data admission now requires a stable SHA-256 fingerprint, included in the
objective/checkpoint contract and checked by the runner. CUDA compilation and
runtime were previously validated with CUDA 12.5/MSVC and a CMake CTest
CPU-oracle executable on a local RTX 3070 Laptop GPU. On 2026-11-02, the same
CUDA CNOT-successor target compiled with NVCC 13.2.86 for SM86 and its CMake
CTest passed on Vast.ai with two RTX 3060 GPUs. The test compares all ordered
CNOT successors bit-for-bit at dimensions 2, 3, 7, 64, and 65 and checks launch
rejection for undersized output buffers and invalid packed row widths. On
2026-11-02, the full Cargo suite passed on Vast.ai with
`cuda,tch/download-libtorch` and `CNOT103_REQUIRE_CUDA=1`: 149 tests passed,
0 failed, and 1 child-process
test was intentionally ignored across 39 test targets. The CUDA-feature build
compiled and linked the Rust FFI against the project CUDA library. GPU smoke
coverage includes the Rust FFI successor output and a real CUDA
`FusionDistanceField` forward pass followed by one AdamW update on device 0.
`CNOT103_REQUIRE_CUDA=1` makes those smoke tests fail instead of silently
skipping if CUDA is unavailable. The first Vast.ai CUDA run exposed a link issue: the Rust executable omitted
`libtorch_cuda.so` under the default `--as-needed` linker behavior. The initial
workaround used `LD_PRELOAD`; the loader fix described below removes that
requirement. The model
fixture uses a synthetic I11
producer and a small model; it is not production 103-feature training. No
throughput claim was measured, and I08 CUDA integration remains open.
Source-parity for the unimplemented feature contracts, production-scale
training, held-out integrity, and end-to-end search acceptance remain open.

On 2026-11-02, the current native tree passed
`cargo test --no-default-features` on Vast.ai: 137 tests passed, 0 failed,
across 35 targets, including the F03-F05 exhaustive source-parity fixture for
all 168 GL(3,2) states.
After the I09 port was added, the local Windows/MSVC
`cargo test --locked --no-default-features` run passed 142 tests. The later I09
GF(2) rank-workspace change was rechecked by all five I09 tests. These are
CPU-only local checks; this I09 revision still needs Vast Linux/CUDA acceptance.
On the managed Windows runner, `TEMP` and `TMP` were pointed to `native/target/test-temp`
so child-process checkpoint tests stayed inside the writable worktree. The
installed PyTorch 2.8 wheel cannot build `torch-sys` 0.22: it lacks
`torch/csrc/autograd/engine.h`, while `tch 0.22.0` requires LibTorch 2.9.0. The
Torch-gated tests for this source revision were instead run on Vast.ai with the
official LibTorch 2.9.0 CUDA 12.8 package; see the full-suite result above. The
73-test LibTorch result is from an earlier source revision. The source-bound
E-family overflow regression and matrix-boundary reference tests ran in the
current no-default-features suite; I05 includes 72 source-golden cases and I06
checks exact relation/readout behavior. Those CPU provider tests do not imply
GPU feature-producer coverage.

For Windows Cargo builds, initialize the MSVC environment and use the Ninja
generator when Visual Studio has no CUDA integration toolset. Set `CUDA_PATH`
and `CMAKE_CUDA_COMPILER`; `CNOT103_CUDA_ARCHITECTURES` optionally pins the
compiled GPU targets. `build.rs` adds the platform CUDA runtime library path so
the Rust FFI link step resolves `cudart`.


## Frozen dataset reader and checkpoint identity

`FrozenDataset::load_pinned_with_catalogs(path, trusted_sha256, catalogs)` reads a
`own_frozen_panels_v1` artifact and requires both its `.sha256` sidecar and the
caller-supplied trusted digest to match the exact bytes. It replays reduction
certificates, checks labels and train/dev/test source-seed separation, recomputes
the declared finite symmetry views, enforces dev/test and historical/held-out
separation, and rejects train roots or one-CNOT successors in the forbidden set.
The sidecar alone is only an integrity check; production callers should pin the
expected digest in a separately versioned experiment manifest.

Exact labels that are not proved by equality of source bounds require a
proof-checked catalog. load_exact_catalogs_pinned(path, trusted_sha256) reads
an exact_distance_catalog_bundle_v1 artifact with a provenance field and one
catalog per matrix width. Entries carry packed matrix rows, an exact distance,
and a replayable reduction word under the unit-cost all-to-all CNOT alphabet.
The loader requires both the adjacent .sha256 sidecar and the separate trusted
pin, then verifies catalog closure, replayed costs, and the global Lipschitz
condition before returning catalogs for FrozenDataset::load_pinned_with_catalogs.
A sidecar by itself is not a trusted pin.

Before constructing `FullGraphObjectiveRunner`, call
`objective.bind_frozen_dataset(&dataset)`. Checkpoint contract v3 binds the
frozen artifact digest and the exact ordered train-panel digest immediately;
runner construction rechecks that its records match the frozen panel. A resume
with a different artifact or record sequence is rejected. This is a library path;
a complete production training CLI and full source/103-feature acceptance are
still open.

## GPU fresh-process checkpoint acceptance

The CUDA loader was corrected on 2026-11-02: torch_device explicitly opens
libtorch_cuda.so with global symbol visibility before checking the selected
device, so Rust CUDA tests no longer need LD_PRELOAD. On this Vast.ai
2xRTX3060 instance, both CUDA smoke targets passed with LD_PRELOAD unset.
The fresh-process checkpoint test also passed on cuda:0 and cuda:1: it applies
updates, saves at the split, restores in a child process, applies the remaining
updates, and matches the uninterrupted run's model value, optimizer count,
sampler, progress, objective bytes, and caller RNG bytes. To run it on either
GPU, set CNOT103_REQUIRE_CUDA=1 and
CNOT103_RESUME_DEVICE=cuda:0 or cuda:1 before invoking
cargo test --features cuda --test tch_training_resume
fresh_process_resume_matches_uninterrupted. The full current CUDA suite also
passed on cuda:0 with LD_PRELOAD unset: 156 passed, 0 failed, 2 ignored across
41 test targets. The second-GPU resume run passed separately. This exercises
the generic trainer and checkpoint path; it is not a full 103-feature CNOT
objective run.

The 41-feature CUDA smoke and fresh-process resume tests remain as a regression
profile. A second profile now executes all 102 registered model-input features
through the small Bellman objective, saves after one update, restores in a child
process, applies the next update, and matches the uninterrupted model, optimizer,
sampler, objective, progress, and RNG state. The 102-input resume test passed on
cuda:0 and cuda:1. These are synthetic three-wire fixtures, not production
training or quality acceptance.

H07 now has a Torch-gated evaluator that runs the model value head on
caller-built canonical, equivalent-view, and relabel packets, then records the
population variance and probe provenance. Its CUDA regression executes an
inverse view and an all-to-all unit-cost wire swap. The exact source probe
schedule and availability contract remain unspecified; H07 remains outside the
102 model inputs.


## Current Vast acceptance — 2026-11-03

At the current branch tip, the full CUDA Cargo suite passed after adding the
102-input checkpoint profile and real H07 forwards. After the H07 producer
identity was versioned, the H07 unit tests and complete CUDA resume target were
rerun successfully on cuda:0. The 102-input fresh-process resume also passed
separately on cuda:1.

The 102-input synthetic GL(4,2) CLI run also completed one CUDA update, saved
a checkpoint, resumed it in a fresh process, and completed the next update.
This exercises the bounded launcher, pinned exact fixture artifacts, complete
feature profile, exposure audit, and checkpoint contract together.

These acceptances use small synthetic inputs. They do not establish exact parity
for the archived-unimplemented source features, production data coverage,
held-out quality, or completion of the 1000x10000 campaign.


## Bounded CUDA training command

Build and run a bounded invocation on a CUDA host:

```sh
cargo run --locked --manifest-path native/Cargo.toml --features cuda --bin cnot103-train -- run.json
```

The strict JSON manifest schema is `cnot103_training_run_v1`. Its required
top-level keys are `dataset`, `exact_catalogs`, `checkpoint_dir`, `resume`,
`training`, `feature_policy`, `model`, `optimizer`, `objective`,
`alphabets`, and `feature_contexts`, plus the schema string. Unknown keys
and omitted context fields are errors. Relative artifact and checkpoint paths
are resolved from the manifest file. Dataset and exact-catalog artifacts each
need an adjacent `.sha256` sidecar and a caller-supplied lowercase SHA-256 pin.

`training` explicitly sets the seed, CUDA device index, attempted updates,
batch size, and checkpoint interval. `feature_policy` explicitly sets the
feature schema identity, solver-assisted setting, evaluation limit, and byte
limit; use JSON `null` to state that a limit is unbounded. The launcher always
selects the complete 102 model inputs and keeps the full exposure audit enabled.
It does not offer a reduced-input or compact-audit mode. `model`,
`optimizer`, and `objective` expose their dimensions and numeric parameters;
none are silently defaulted by the launcher.

`feature_contexts` must supply the validated cuts, intersections, rank probes,
alphabet, residual gates, parity blocks and budgets, boundaries and budgets,
paid-basis blocks and shear config, template choices/probe points/budgets,
bilinear matrices, shared-relation frames, Fourier partitions, optional-gate
packets, I09 horizons/limits, and I10 horizon/budgets. Constructors validate
every context before a model is created. Exact-catalog input currently supports
all-to-all unit CNOT alphabets only and requires one catalog for every configured
alphabet. Any context or feature that is unavailable for a training example
fails the full-input objective; it is not replaced with a fabricated value.

A fresh invocation refuses to use a nonempty checkpoint directory. A resumed
invocation requires `last.json` and lets the checkpoint contract reject changed
data, feature contexts, or objective settings. Each invocation writes a final
checkpoint and prints JSON with input hashes, split sizes, update counts,
exposure audit, and checkpoint manifest. The repository contains no production
frozen training panel or matching exact-catalog bundle, so the launcher is
infrastructure until those pinned artifacts and their real physical feature
contexts are supplied; it does not make a training or quality claim by itself.
