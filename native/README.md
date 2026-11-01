# Native 103-feature port

The canonical packet layout contains 103 feature IDs. Current producer coverage
is deliberately split by input contract:

- 39 feature IDs have built-in structural producers: A01-A20, B01-B12, F01,
  C01, D03, and H01-H04.
- C02, C03, and C04 can be registered for explicit labeled wire cuts. They
  require an `augmentation_only` symmetry policy and are bound to the ordered
  cut set by the producer version. A matrix width with no registered cut is
  reported as unsupported.
- The remaining 61 IDs have no native producer yet. Their packet rows remain
  unavailable; the native code does not claim complete-103 coverage.

The native feature tests currently exercise C02-C04 ranks, cut incidence,
symmetry policy, width mismatch, invalid cuts, relabeling, and transactional
provider registration. These CPU tests do not establish Torch, CUDA, model
checkpoint migration, training-loop, or end-to-end search acceptance.
