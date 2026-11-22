# CNOT model input contract: v25

## Three counts that must stay separate

The canonical feature registry contains 103 IDs. The model policy selects 102 of them. H07 remains a registry ID for compatibility and diagnostics, but it is not a neural input.

| Registry IDs | Canonical registry positions (1-based) | Neural input slots (1-based) | Contract |
| --- | ---: | ---: | --- |
| A01–A20 | 1–20 | 1–20 | 20 selected feature IDs |
| B01–B12 | 21–32 | 21–32 | 12 selected feature IDs |
| C01–C10 | 33–42 | 33–42 | 10 selected feature IDs |
| D01–D14 | 43–56 | 43–56 | 14 selected feature IDs |
| E01–E09 | 57–65 | 57–65 | 9 selected feature IDs |
| F01–F10 | 66–75 | 66–75 | 10 selected feature IDs |
| G01–G10 | 76–85 | 76–85 | 10 selected feature IDs |
| H01–H06 | 86–91 | 86–91 | 6 selected feature IDs |
| H07 | 92 | — | Post-forward diagnostic; excluded from model policy and scorer input |
| I01–I11 | 93–103 | 92–102 | 11 selected feature IDs |

The ID manifest is `cnot_feature_ids_a01_i11_v1`. The feature registry stores each selected ID's role, channel count, scope, producer version, dependencies, exactness, cost tier, wire roles, and solver-assisted flag. Roles and channels are defined in `native/src/schema/spec/roles/{abc,def,ghi}.rs`; a contextual provider may replace a reserved shape only by declaring its actual metadata and producer. The static structural module currently lists 41 built-in IDs; contextual providers require their explicit physical contexts, so that count does not describe per-width train or dev coverage.

The full training contract pins 8 structural blocks, 2 fusion blocks, 136,421,807 parameters, 102 input slots, and schema `runtime_packet_layout_v25`. The runtime matrix and valid-node mask travel in the packet as structural state data; they are not additional feature IDs.

## Missing-value tokens are service metadata

| Packet state | Availability | Stable device status | Model treatment |
| --- | ---: | ---: | --- |
| Computed value, including numeric zero | 1 | `Computed = 1` | Project the value and include its role-shaped contribution |
| Outside the feature's mathematical domain | 0 | `Inapplicable = 2` | Mask the value and add the per-feature inapplicable token |
| Provider/context is unavailable | 0 | `Unsupported = 3` | Mask the value and add the per-feature unsupported token |
| Computation stopped by an explicit resource cap | 0 | `ResourceLimit = 4` | Mask the value and add the per-feature resource-limit token |
| Feature was not selected | not admissible for a selected input | `NotRequested = 0` | Reject for a selected v25 model input |

In v25, unavailable rows are zeroed before their feature projection, while a learned context projection receives three reason bits for each of the 102 selected IDs. That token is an internal model context input; it is not a 103rd feature ID or a replacement for H07. Group `token_valid` masks and incidence arrays describe the shape of group-valued features and also do not add feature slots. A computed numeric zero remains distinguishable because its availability and status are `1 / Computed`.

## Where the contract is checked

- `native/src/schema.rs` declares 103 registry IDs, 102 model inputs, the H07 diagnostic ID, and v25.
- `native/src/features/registry/builtins.rs` excludes H07 from registered model-input providers. `all_model_input_specs()` filters exactly H07.
- `native/src/model/field/construct.rs` builds the model from the registry and policy. `native/src/model/field/packet.rs` checks the canonical 103-ID packet layout and exact selected IDs/provider specs.
- `.github/scripts/build_training_manifest.py` and `.github/scripts/run_full_training.py` pin the training manifest and checkpoint metadata to 102 slots and v25. Native train and search use the shared model/manifest loader.
- Producer-to-consumer handoff receipts pin source, training manifest/result, dataset, checkpoint, search manifest, test-panel IDs and digest. The MultiGPUBeamSearch bridge validates the same 8-block / 136,421,807-parameter / 102-slot / v25 contract before starting paired search.

The CPU-torch regression `feature_schema::model_input_contract_has_102_slots_and_h07_stays_post_forward` checks the ID counts, H07 exclusion, policy order, v24/v25 IDs, and status ABI. The companion `model_layout::v25_missing_status_token_distinguishes_computed_zero_from_unsupported` exercises a real CPU model forward with a synthetic pair feature. CUDA packet tests cover the device status path; compilation without a GPU does not establish physical CUDA/NCCL runtime acceptance.

Adding H07 or another independent feature as a model input changes the model/checkpoint ABI. It requires a new schema and compatible weights, not a silent slot insertion.
