# Adaptive candidate chunks

`objective.adaptive_neighbor_chunk` is an optional training-manifest field. It
defaults to `false`, preserving the existing fixed `neighbor_chunk_size` and
checkpoint contract.

When explicitly enabled, `neighbor_chunk_size` is the maximum candidate chunk.
For each matrix width `n`, the effective size is the lesser of that maximum and
`feature_policy.max_bytes / (1024*n*n + 256*n)`. A null `max_bytes` means no
feature-workspace cap. The run fails before training if the cap cannot fit one
state or the estimate overflows.

```json
{
  "feature_policy": {
    "schema_id": "runtime_packet_layout_v25",
    "solver_assisted": true,
    "evaluations": null,
    "max_bytes": 268435456
  },
  "objective": {
    "neighbor_chunk_size": 32,
    "adaptive_neighbor_chunk": true
  }
}
```

These are excerpts; a complete run manifest contains the other required
fields. With this budget the effective chunk is 32 at width 72 and 12 at width
144. This is a feature-packet estimate, not a bound on total GPU memory: model
activations, optimizer state, allocations, and other GPU users still count.
Measure the chosen maximum on the target hardware; a configuration that fits
the feature limit can still run out of GPU memory.

The limit applies to nonterminal candidate scoring. For an identity state with
`qhat_weight > 0`, the existing terminal Q-hat path still evaluates all
primitive successors in one packet. If that packet exceeds the feature-workspace
budget, the run fails closed; adaptive chunks do not split that terminal packet.

Adaptive chunks require the missing-token schema, all model inputs, and
unlimited provider evaluations. Finite evaluation budgets can make feature
availability depend on packet size. Legacy schemas retain their fixed chunk
layout. The adaptive schedule has a distinct objective fingerprint and cannot
silently resume a fixed-chunk checkpoint. Changing chunk size can alter floating
point execution order, so assess training and search quality separately.
