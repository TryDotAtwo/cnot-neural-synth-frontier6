# CNOT model input contract

This table distinguishes the canonical feature-ID registry from tensors consumed by the trained network.

| Layer | Count / identifier | Contract |
|---|---:|---|
| Feature registry | 103 IDs, `cnot_feature_ids_a01_i11_v1` | Canonical registry includes H07. |
| Network feature inputs | 102, `runtime_packet_layout_v25` | `all_model_input_specs()` is the canonical 103-ID list with H07 removed. The trainer, exporter, native scorer, and consumer handoff pin this count. |
| Feature value | One registered feature entry per input slot | A value of zero remains a computed value when its status is `Computed`; storage zeros do not establish availability. |
| Missing-status service token | 3 status indicators per model input | `Inapplicable`, `Unsupported`, and `ResourceLimit` are encoded in the v25 learned missing-status projection. They are service tokens, not additional feature slots. |
| Not requested | ABI code 0 | A selected model input may not be `NotRequested`; admission rejects it. |
| H07 | One registry ID, outside the 102 network inputs | H07 is a read-only post-forward diagnostic and is rejected as a model input. |

The stable device status codes are `NotRequested=0`, `Computed=1`, `Inapplicable=2`, `Unsupported=3`, and `ResourceLimit=4`. The v25 input contract keeps the computed-zero value distinct from each missing reason.

The current compatibility decision is to retain the 102-input v25 checkpoint contract. Adding an independent 103rd neural input requires a new feature layout, model/checkpoint compatibility version, and separately validated producer/scorer mapping. H07 is not silently reused for that purpose.

The CPU-torch regression checks the 103-to-102 registry selection, canonical ordering, status ABI, and v25 schema. CUDA packet regressions separately verify missing-reason embeddings, computed zero, and H07 rejection. The producer's training manifest pins v25 and 102; the consumer resolver pins the same schema, architecture, and input count.
