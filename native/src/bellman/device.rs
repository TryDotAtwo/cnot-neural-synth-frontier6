//! Complete primitive Bellman backup for one already-admitted training record.
//! Alphabet metadata is uploaded once. The detached selection and its gradient
//! recomputation run in the same call with the exact winning feature row.
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeaturePacket, FieldOutput, FusionDistanceField};
use crate::CnotAlphabet;
use tch::{Device, Kind, Reduction, Tensor};
mod layout;
mod sampling;
pub(crate) use layout::RecordLayout;

/// Static record classification is checked against the resident matrix. A terminal
/// has no Bellman branch; only Q-hat may request its primitive targets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootBranch {
    Nonterminal,
    Terminal { qhat: bool },
}

/// An optimizer must require the device validity vote before committing an update.
pub struct DevicePrimitiveBackup {
    pub parent: FieldOutput,
    pub loss: Tensor,
    pub primitive_totals: Tensor,
    /// Alphabet ordinals corresponding to totals when the approximate mode is
    /// enabled. None means all primitive actions in declared order.
    pub primitive_indices: Option<Tensor>,
    pub selected_index: Option<Tensor>,
    pub valid: Tensor,
    /// OR of source group-axis presence on the parent and frozen successor.
    /// A false slot means its parameters skip AdamW, including decay and moments.
    pub group_activity: Tensor,
    pub lower_bound: Tensor,
}

#[derive(Clone, Copy)]
pub(crate) struct SampleRequest {
    pub count: usize,
    pub seed: u64,
}

pub struct DevicePrimitiveBellman {
    alphabet: CnotAlphabet,
    index: usize,
    controls: Tensor,
    targets: Tensor,
    costs: Tensor,
    action_indices: Tensor,
}

struct Winner {
    score: Tensor,
    index: Tensor,
    packet: DeviceFeaturePacket,
    group_activity: Tensor,
}

impl DevicePrimitiveBellman {
    pub fn new(alphabet: &CnotAlphabet, index: usize) -> Result<Self, String> {
        crate::torch_device::require_cuda_device(index)?;
        let device = Device::Cuda(index);
        let n = i64::from(alphabet.dimension());
        n.checked_mul(n)
            .ok_or("primitive action matrix extent overflow")?;
        let controls: Vec<_> = alphabet
            .gates()
            .iter()
            .map(|g| i64::from(g.control))
            .collect();
        let targets: Vec<_> = alphabet
            .gates()
            .iter()
            .map(|g| i64::from(g.target))
            .collect();
        let costs: Vec<_> = (0..controls.len())
            .map(|i| f64::from(alphabet.cost_at(i).expect("validated gate cost")))
            .collect();
        let controls = Tensor::from_slice(&controls).to_device(device);
        let targets = Tensor::from_slice(&targets).to_device(device);
        let action_indices = &controls * n + &targets;
        Ok(Self {
            alphabet: alphabet.clone(),
            index,
            controls,
            targets,
            costs: Tensor::from_slice(&costs).to_device(device),
            action_indices,
        })
    }

    /// Flat control-major model positions for the declared alphabet, in its own order.
    pub fn action_indices(&self) -> &Tensor {
        &self.action_indices
    }

    fn children(
        &self,
        root: &DeviceStateBatch,
        start: i64,
        count: i64,
    ) -> Result<DeviceStateBatch, String> {
        let packed = root.tensor();
        let n = i64::from(root.dimension());
        let words = packed.size()[2];
        count
            .checked_mul(n)
            .and_then(|x| x.checked_mul(words))
            .and_then(|x| x.checked_mul(8))
            .ok_or("primitive successor extent overflow")?;
        let targets = self.targets.narrow(0, start, count);
        let rows = packed.select(0, 0);
        let controls = rows.index_select(0, &self.controls.narrow(0, start, count));
        let updated = rows.index_select(0, &targets).bitwise_xor_tensor(&controls);
        let target_rows = targets.view([count, 1, 1]).expand([count, 1, words], true);
        let packed =
            packed
                .expand([count, n, words], true)
                .scatter(1, &target_rows, &updated.unsqueeze(1));
        DeviceStateBatch::from_packed_tensor_for_alphabet(packed, &self.alphabet, self.index)
    }

    fn selected_children(
        &self,
        root: &DeviceStateBatch,
        ordinals: &Tensor,
    ) -> Result<DeviceStateBatch, String> {
        let packed = root.tensor();
        let n = i64::from(root.dimension());
        let words = packed.size()[2];
        let count = ordinals.size()[0];
        count.checked_mul(n).and_then(|x| x.checked_mul(words))
            .and_then(|x| x.checked_mul(8)).ok_or("primitive successor extent overflow")?;
        let targets = self.targets.index_select(0, ordinals);
        let rows = packed.select(0, 0);
        let controls = rows.index_select(0, &self.controls.index_select(0, ordinals));
        let updated = rows.index_select(0, &targets).bitwise_xor_tensor(&controls);
        let target_rows = targets.view([count, 1, 1]).expand([count, 1, words], true);
        let packed = packed.expand([count, n, words], true)
            .scatter(1, &target_rows, &updated.unsqueeze(1));
        DeviceStateBatch::from_packed_tensor_for_alphabet(packed, &self.alphabet, self.index)
    }

    pub fn backup(
        &self,
        model: &FusionDistanceField,
        plan: &DeviceFeaturePlan,
        root: &DeviceStateBatch,
        branch: RootBranch,
        chunk_size: usize,
        strict_coverage: bool,
    ) -> Result<DevicePrimitiveBackup, String> {
        self.backup_with_layout(
            model,
            plan,
            root,
            branch,
            chunk_size,
            strict_coverage,
            None,
            false,
            false,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backup_with_layout(
        &self,
        model: &FusionDistanceField,
        plan: &DeviceFeaturePlan,
        root: &DeviceStateBatch,
        branch: RootBranch,
        chunk_size: usize,
        strict_coverage: bool,
        layout: Option<&RecordLayout>,
        share_candidate_wire_projections: bool,
        share_training_wire_projections: bool,
        sample: Option<SampleRequest>,
        cached_root_f08: Option<&DeviceFeatureEntry>,
    ) -> Result<DevicePrimitiveBackup, String> {
        let device = Device::Cuda(self.index);
        if root.batch_size() != 1
            || root.dimension() != self.alphabet.dimension()
            || root.alphabet_id() != Some(self.alphabet.id())
            || root.tensor().device() != device
            || model.device() != device
        {
            return Err(
                "primitive Bellman requires one root bound to the same alphabet and CUDA device"
                    .into(),
            );
        }
        let (all_count, full_width) = self.schedule(branch, chunk_size)?;
        let sampling = sample.is_some_and(|request| all_count > request.count);
        if sampling && (layout.is_some() || !model.uses_missing_tokens()) {
            return Err("sampled Bellman requires the missing-token layout".into());
        }
        let count = if sampling { sample.unwrap().count } else { all_count };
        let width = if sampling { chunk_size } else { full_width };
        let largest_packet = count.min(width).max(1);
        if strict_coverage && !plan.fits_workspace(root.dimension(), largest_packet)? {
            if branch == (RootBranch::Terminal { qhat: true }) {
                return Err(format!(
                    "terminal Q-hat scores all {largest_packet} successors in one packet at width {}, exceeding the feature workspace budget; raise max_bytes or set qhat_weight=0 to disable the Q-hat loss",
                    root.dimension()
                ));
            }
            return Err(format!(
                "candidate chunk of {largest_packet} states at width {} exceeds the feature workspace budget and would make mandatory A01 unavailable; reduce neighbor_chunk_size or raise max_bytes",
                root.dimension()
            ));
        }
        let nonterminal = branch == RootBranch::Nonterminal;
        let parent_packet = crate::stage_profile::measure("parent_packet", || {
            plan.build_with_root_f08(root, cached_root_f08)
        })?;
        let parent_packet = if let Some(layout) = layout {
            parent_packet.source_layout(&layout.parent)?
        } else {
            parent_packet
        };
        let n = i64::from(root.dimension());
        let residual_matrix = parent_packet
            .matrix()
            .ne_tensor(&Tensor::eye(n, (Kind::Float, device)));
        // rank(M-I) cannot exceed either nonzero row or column count, so their
        // maximum is exactly max(changed_rows, changed_columns, rank(M-I)).
        let lower_bound = residual_matrix
            .any_dim(2, false)
            .sum(Kind::Int64)
            .maximum(&residual_matrix.any_dim(1, false).sum(Kind::Int64));
        let identity = lower_bound.eq(0);
        let mut valid = if nonterminal {
            identity.logical_not()
        } else {
            identity
        };
        if strict_coverage {
            valid = valid.logical_and(&parent_packet.complete_coverage().map_err(|e| e.0)?);
        }

        // Retain the parent graph so that the prediction used for selection is
        // exactly the one receiving the sampled Q-hat gradient.
        let early_parent = if sampling {
            Some(model.training_forward_device(&parent_packet, share_training_wire_projections).map_err(|e| e.0)?)
        } else {
            None
        };
        let primitive_indices = if let Some(request) = sample {
            if let Some(parent) = &early_parent {
                let scores = parent.qhat.view([-1]).index_select(0, &self.action_indices);
                valid = valid.logical_and(&scores.isfinite().all());
                valid = valid.logical_and(&parent.action_mask.view([-1])
                    .index_select(0, &self.action_indices).all());
                Some(sampling::sampled_indices(&scores, request.count, request.seed)?)
            } else {
                Some(Tensor::arange(all_count as i64, (Kind::Int64, device)))
            }
        } else {
            None
        };

        let (totals, winner) = tch::no_grad(|| -> Result<_, String> {
            let mut totals = Vec::new();
            let mut winner: Option<Winner> = None;
            for (chunk, start) in (0..count).step_by(width).enumerate() {
                if crate::stage_profile::skip_candidate_chunk(chunk) {
                    continue;
                }
                let probe_start = crate::stage_profile::candidate_chunk_start();
                let size = (count - start).min(width) as i64;
                let ordinals = primitive_indices.as_ref().map(|indices| indices.narrow(0, start as i64, size));
                let packed = if sampling {
                    self.selected_children(root, ordinals.as_ref().expect("sampled indices"))?
                } else {
                    self.children(root, start as i64, size)?
                };
                let packet = crate::stage_profile::measure("candidate_packet", || plan.build(&packed))?;
                if strict_coverage {
                    valid = valid.logical_and(&packet.complete_coverage().map_err(|e| e.0)?);
                }
                let shaped;
                let scoring_packet = if let Some(layout) = layout {
                    shaped =
                        packet.source_layout(&layout.signatures[layout.chunk_signatures[chunk]])?;
                    &shaped
                } else {
                    &packet
                };
                let values = model
                    .candidate_value_device(scoring_packet, share_candidate_wire_projections)
                    .map_err(|e| e.0)?;
                let costs = if sampling {
                    self.costs.index_select(0, ordinals.as_ref().expect("sampled indices"))
                } else {
                    self.costs.narrow(0, start as i64, size)
                };
                let scores = values.view([size]).to_kind(Kind::Double) + costs;
                valid = valid.logical_and(&scores.isfinite().all());
                if nonterminal {
                    let (score, local) = scores.min_dim(0, false);
                    let index = if sampling {
                        primitive_indices.as_ref().unwrap()
                            .index_select(0, &(&local + start as i64).view([1])).view([])
                    } else {
                        &local + start as i64
                    };
                    let activity = packet.group_activity();
                    winner = Some(match winner {
                        None => Winner {
                            score,
                            index,
                            packet: packet.freeze_winner(&local, None),
                            group_activity: activity,
                        },
                        Some(old) => {
                            let replace = score.lt_tensor(&old.score);
                            Winner {
                                score: score.where_self(&replace, &old.score),
                                index: index.where_self(&replace, &old.index),
                                packet: packet.freeze_winner(&local, Some((&old.packet, &replace))),
                                group_activity: activity.where_self(&replace, &old.group_activity),
                            }
                        }
                    });
                }
                totals.push(scores);
                crate::stage_profile::candidate_chunk_complete(probe_start)?;
            }
            crate::stage_profile::finish_candidate_probe()?;
            let totals = if totals.is_empty() {
                Tensor::empty([0], (Kind::Double, device))
            } else {
                Tensor::cat(&totals, 0)
            };
            Ok((totals, winner))
        })?;
        let mut group_activity = parent_packet.group_activity();
        let shared_parent = if early_parent.is_some() {
            early_parent
        } else {
            model.uses_missing_tokens().then(|| model.training_forward_device(&parent_packet, share_training_wire_projections))
                .transpose().map_err(|e| e.0)?
        };
        let value = match &shared_parent {
            Some(parent) => parent.value.view([]),
            None => model
                .candidate_value_device(&parent_packet, share_training_wire_projections)
                .map_err(|e| e.0)?
                .view([]),
        };
        let (residual, selected_index) = match winner {
            None => (value, None),
            Some(winner) => {
                group_activity = group_activity.logical_or(&winner.group_activity);
                let next = if let Some(layout) = layout {
                    // Only the global winner is differentiated. When immutable
                    // chunk widths differ, evaluate that same row in each layout
                    // and gather its source-shaped value using its device index.
                    let values = layout
                        .signatures
                        .iter()
                        .map(|sig| {
                            let packet = winner.packet.source_layout(sig)?;
                            Ok(model.candidate_value_device(&packet, share_training_wire_projections).map_err(|e| e.0)?.view([]))
                        })
                        .collect::<Result<Vec<Tensor>, String>>()?;
                    if values.len() == 1 {
                        values.into_iter().next().expect("one signature")
                    } else {
                        let chunk = winner.index.floor_divide_scalar(width as i64).view([1]);
                        let signature = layout.device_signatures.index_select(0, &chunk);
                        Tensor::stack(&values, 0)
                            .index_select(0, &signature)
                            .view([])
                    }
                } else {
                    model
                        .candidate_value_device(&winner.packet, share_training_wire_projections)
                        .map_err(|e| e.0)?
                        .view([])
                };
                let cost = self
                    .costs
                    .index_select(0, &winner.index.view([1]))
                    .view([])
                    .to_kind(value.kind());
                (value - cost - next, Some(winner.index))
            }
        };
        let loss = residual.smooth_l1_loss(&Tensor::zeros_like(&residual), Reduction::Mean, 1.0);
        // Older schemas retain the reference FP32 gradient accumulation order.
        let parent = match shared_parent {
            Some(parent) => parent,
            None => model.training_forward_device(&parent_packet, share_training_wire_projections).map_err(|e| e.0)?,
        };
        valid = valid.logical_and(&loss.isfinite());
        Ok(DevicePrimitiveBackup {
            parent,
            loss,
            primitive_totals: totals,
            primitive_indices,
            selected_index,
            valid,
            group_activity,
            lower_bound,
        })
    }
}

#[cfg(test)]
mod projection_tests {
    use super::*;
    use crate::model::ModelConfig;
    use crate::{FeatureId, FeaturePolicy, FeatureRegistry, MatrixState};
    use tch::nn::VarStore;

    #[test]
    fn factored_candidate_values_retain_the_winner_and_gradient_branch() {
        crate::torch_device::require_cuda_device(0).unwrap();
        tch::manual_seed(4121);
        let a = CnotAlphabet::all_to_all_unit(5).unwrap();
        let r = FeatureRegistry::new();
        let mut p = FeaturePolicy::all_model_inputs(crate::schema::MISSING_TOKEN_LAYOUT_ID);
        p.enabled_ids = (1..=20).map(|i| FeatureId::new(b'A', i).unwrap()).collect();
        let plan = DeviceFeaturePlan::context_free(&r, &p, 0).unwrap();
        let vs = VarStore::new(Device::Cuda(0));
        let config = ModelConfig {
            wire_dim: 16,
            pair_dim: 5,
            structural_blocks: 2,
            attention_heads: 4,
            feedforward_dim: 25,
            fusion_blocks: 1,
            basis_proposal_hidden_dim: 8,
        };
        let model =
            FusionDistanceField::new_for_policy(&vs.root(), &r, &p, vs.device(), config).unwrap();
        let actions = DevicePrimitiveBellman::new(&a, 0).unwrap();
        for gates in [[1, 5, 8, 11], [6, 0, 15, 10]] {
            let mut root = MatrixState::identity(5, a.id()).unwrap();
            for g in gates {
                root.apply_cnot(a.gates()[g]).unwrap();
            }
            let packed: Vec<_> = root.packed_rows().iter().map(|&x| x as i64).collect();
            let gpu = DeviceStateBatch::from_packed_tensor_for_alphabet(
                Tensor::from_slice(&packed)
                    .view([1, 5, 1])
                    .to_device(vs.device()),
                &a,
                0,
            )
            .unwrap();
            for chunk in [3, 8] {
                let run = |factor| {
                    actions
                        .backup_with_layout(
                            &model,
                            &plan,
                            &gpu,
                            RootBranch::Nonterminal,
                            chunk,
                            false,
                            None,
                            false,
                            factor,
                            None,
                            None,
                        )
                        .unwrap()
                };
                let reference = run(false);
                let measured = run(true);
                assert_eq!(measured.valid.int64_value(&[]), 1);
                assert!(reference.primitive_totals.allclose(
                    &measured.primitive_totals,
                    1e-5,
                    2e-6,
                    false
                ));
                assert_eq!(
                    reference.selected_index.unwrap().int64_value(&[]),
                    measured.selected_index.unwrap().int64_value(&[])
                );
                assert!(measured.loss.requires_grad());
                assert!(reference.loss.equal(&measured.loss));
            }
        }
    }

    #[test]
    fn sampled_backup_scores_real_subset_and_full_cap_matches_complete_backup() {
        crate::torch_device::require_cuda_device(0).unwrap();
        tch::manual_seed(731);
        let a = CnotAlphabet::all_to_all_unit(5).unwrap();
        let registry = FeatureRegistry::new();
        let mut policy = FeaturePolicy::all_model_inputs(crate::schema::MISSING_TOKEN_LAYOUT_ID);
        policy.enabled_ids = (1..=20).map(|i| FeatureId::new(b'A', i).unwrap()).collect();
        let plan = DeviceFeaturePlan::context_free(&registry, &policy, 0).unwrap();
        let vs = VarStore::new(Device::Cuda(0));
        let model = FusionDistanceField::new_for_policy(&vs.root(), &registry, &policy, vs.device(), ModelConfig {
            wire_dim: 16, pair_dim: 5, structural_blocks: 2, attention_heads: 4,
            feedforward_dim: 25, fusion_blocks: 1, basis_proposal_hidden_dim: 8,
        }).unwrap();
        let actions = DevicePrimitiveBellman::new(&a, 0).unwrap();
        let mut root = MatrixState::identity(5, a.id()).unwrap();
        for index in [1, 5, 8, 11] { root.apply_cnot(a.gates()[index]).unwrap(); }
        let packed: Vec<_> = root.packed_rows().iter().map(|&x| x as i64).collect();
        let gpu = DeviceStateBatch::from_packed_tensor_for_alphabet(
            Tensor::from_slice(&packed).view([1, 5, 1]).to_device(vs.device()), &a, 0,
        ).unwrap();
        let run = |branch, sample| actions.backup_with_layout(
            &model, &plan, &gpu, branch, 4, false, None, false, false, sample, None,
        ).unwrap();
        let full = run(RootBranch::Nonterminal, None);
        let wide = run(RootBranch::Nonterminal, Some(SampleRequest { count: 20, seed: 123 }));
        assert!(wide.primitive_totals.equal(&full.primitive_totals));
        assert!(wide.loss.equal(&full.loss));
        assert_eq!(wide.selected_index.unwrap().int64_value(&[]), full.selected_index.as_ref().unwrap().int64_value(&[]));
        let partial = run(RootBranch::Nonterminal, Some(SampleRequest { count: 6, seed: 123 }));
        assert_eq!(partial.valid.int64_value(&[]), 1);
        let indices = partial.primitive_indices.unwrap();
        assert_eq!(indices.size(), [6]);
        assert!(partial.primitive_totals.allclose(&full.primitive_totals.index_select(0, &indices), 2e-5, 2e-5, false));
        let minimum = partial.primitive_totals.argmin(0, false).view([1]);
        assert_eq!(partial.selected_index.unwrap().int64_value(&[]), indices.index_select(0, &minimum).int64_value(&[0]));
        assert!(partial.loss.requires_grad());
    }

    #[test]
    fn factored_training_backup_preserves_loss_winner_and_gradients() {
        crate::torch_device::require_cuda_device(0).unwrap();
        tch::manual_seed(9624);
        let alphabet = CnotAlphabet::all_to_all_unit(5).unwrap();
        let registry = FeatureRegistry::new();
        let mut policy = FeaturePolicy::all_model_inputs(crate::schema::MISSING_TOKEN_LAYOUT_ID);
        policy.enabled_ids = (1..=20).map(|i| FeatureId::new(b'A', i).unwrap()).collect();
        let plan = DeviceFeaturePlan::context_free(&registry, &policy, 0).unwrap();
        let store = VarStore::new(Device::Cuda(0));
        let model = FusionDistanceField::new_for_policy(&store.root(), &registry, &policy, store.device(), ModelConfig {
            wire_dim: 16, pair_dim: 5, structural_blocks: 2, attention_heads: 4,
            feedforward_dim: 25, fusion_blocks: 1, basis_proposal_hidden_dim: 8,
        }).unwrap();
        let actions = DevicePrimitiveBellman::new(&alphabet, 0).unwrap();
        let mut root = MatrixState::identity(5, alphabet.id()).unwrap();
        for index in [1, 5, 8, 11] { root.apply_cnot(alphabet.gates()[index]).unwrap(); }
        let packed: Vec<_> = root.packed_rows().iter().map(|&x| x as i64).collect();
        let gpu = DeviceStateBatch::from_packed_tensor_for_alphabet(
            Tensor::from_slice(&packed).view([1, 5, 1]).to_device(store.device()), &alphabet, 0,
        ).unwrap();
        let parameters = store.trainable_variables();
        for sample in [None, Some(SampleRequest { count: 6, seed: 123 })] {
            let run = |factor| actions.backup_with_layout(
                &model, &plan, &gpu, RootBranch::Nonterminal, 4, false,
                None, false, factor, sample, None,
            ).unwrap();
            let scalar = |backup: &DevicePrimitiveBackup| {
                let masked_qhat = backup.parent.qhat.masked_fill(&backup.parent.action_mask.logical_not(), 0.0);
                let masked_logits = backup.parent.proposal_logits.masked_fill(&backup.parent.action_mask.logical_not(), 0.0);
                &backup.loss + masked_qhat.mean(Kind::Float) + masked_logits.mean(Kind::Float)
            };
            let reference = run(false);
            let expected_loss = scalar(&reference);
            expected_loss.backward();
            let gradients: Vec<_> = parameters.iter().map(|v| {
                let g = v.grad();
                g.defined().then(|| g.detach() * 1.0)
            }).collect();
            for parameter in &parameters {
                let mut grad = parameter.grad();
                if grad.defined() { let _ = grad.zero_(); }
            }
            let factored = run(true);
            let actual_loss = scalar(&factored);
            actual_loss.backward();
            assert_eq!(factored.valid.int64_value(&[]), 1);
            assert!(factored.primitive_totals.allclose(&reference.primitive_totals, 2e-5, 2e-5, false));
            assert_eq!(factored.selected_index.as_ref().unwrap().int64_value(&[]),
                       reference.selected_index.as_ref().unwrap().int64_value(&[]));
            assert!(actual_loss.allclose(&expected_loss, 2e-4, 2e-5, false));
            let mut compared = 0;
            for (parameter, expected) in parameters.iter().zip(&gradients) {
                let actual = parameter.grad();
                match expected {
                    Some(expected) => {
                        assert!(actual.defined());
                        assert!(actual.allclose(expected, 1e-3, 2e-4, false), "parameter gradient changed");
                        compared += 1;
                    }
                    None => assert!(!actual.defined()),
                }
            }
            assert!(compared > 10);
            for parameter in &parameters {
                let mut grad = parameter.grad();
                if grad.defined() { let _ = grad.zero_(); }
            }
        }
    }
}
