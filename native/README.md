# Native 103-feature port

The canonical packet layout contains 103 feature IDs. Current producer coverage
is deliberately split by input contract:

- 40 feature IDs have built-in structural producers: A01-A20, B01-B12, F01-F02,
  C01, D03, and H01-H04.
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
- D01, D02, and D04-D08 can be registered for explicit disjoint physical
  source/target blocks. Pattern and pair budgets are enforced before features
  are emitted; absent sparse bins keep their exact count semantics.
- F02 computes `n - support_components + zero_diagonal_count` exactly. This is
  the descriptor used by the stage-35 lower-bound branch; the feature value
  alone carries no theorem witness or lower-bound claim.
- E01, E04, and E05 can be explicitly registered for the source-defined
  zero-template partial-shear cost, ordered-pair gradient, and gradient
  statistics. They require `augmentation_only` because the source block fixes
  the first half of the physical wires.
- 58 IDs can produce features under their stated input contracts; the remaining
  45 IDs have no native producer yet. Their packet rows remain
  unavailable; the native code does not claim complete-103 coverage.

The native feature tests currently exercise exact F02 connectivity/diagonal
counts, E01/E04/E05 partial-shear gradients and statistics, C02-C05 cut and partial-shear
ranks, C06/C07 restricted-subspace intersections, C09 candidate-rank increment,
C10 selected residual ranks, D01/D02/D04-D08 physical-block parity features,
incidence, pattern budgets, symmetry policy, alphabet mismatch, width mismatch,
invalid contexts, relabeling, and transactional provider registration. These
CPU tests do not establish Torch, CUDA, model checkpoint migration,
training-loop, or end-to-end search acceptance.
