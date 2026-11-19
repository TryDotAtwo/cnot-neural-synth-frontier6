use super::preflight::admitted_upper_bound;
use crate::bellman::device::{DevicePrimitiveBellman, RootBranch, SampleRequest};
use crate::device_feature_plan::DeviceFeaturePlan;
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, FusionDistanceField};
use crate::nn_training::DeviceUpdateLoss;
use crate::objective::{
    losses::first_gate_labels, FullGraphObjective, FullGraphObjectiveConfig, ObjectiveDiagnostics,
    StateAdmission,
};
use crate::training_data::TrainingRecord;
use std::collections::{BTreeMap, HashMap};
use tch::{Device, Kind, Reduction, Tensor};

struct Group {
    roots: DeviceStateBatch,
    actions: DevicePrimitiveBellman,
    neighbor_chunk_size: usize,
    targets: Tensor,
    teacher_indices: Tensor,
    teacher_cost_deltas: Tensor,
    root_f08: Option<RootF08>,
}

struct RootF08 {
    spec: crate::FeatureSpec,
    values: Tensor,
    status: Tensor,
}

impl RootF08 {
    fn prepare(plan: &DeviceFeaturePlan, roots: &DeviceStateBatch) -> Result<Self, String> {
        let id = crate::FeatureId::new(b'F', 8).expect("F08 is in the stable schema");
        let mut values = Vec::new();
        let mut statuses = Vec::new();
        let mut spec = None;
        let mut start = 0;
        while start < roots.batch_size() {
            let mut count = (roots.batch_size() - start).min(8);
            while count > 1 && !plan.fits_workspace(roots.dimension(), count as usize)? {
                count /= 2;
            }
            let packet = plan.build(&roots.rows(start, count)?)?;
            let f08 = packet.feature(id).ok_or("root F08 cache requires F08")?;
            if let Some(previous) = &spec {
                if previous != f08.spec() { return Err("root F08 cache contract changed".into()); }
            } else {
                spec = Some(f08.spec().clone());
            }
            values.push(f08.values().shallow_clone());
            statuses.push(f08.status().shallow_clone());
            start += count;
        }
        Ok(Self {
            spec: spec.ok_or("root F08 cache has no rows")?,
            values: Tensor::cat(&values, 0),
            status: Tensor::cat(&statuses, 0),
        })
    }

    fn row(&self, row: i64) -> DeviceFeatureEntry {
        DeviceFeatureEntry::new(self.spec.clone(), self.values.narrow(0, row, 1),
                                self.status.narrow(0, row, 1), None, None)
    }
}

struct Record {
    group: usize,
    row: i64,
    branch: RootBranch,
    exact: bool,
    teacher_start: i64,
    teacher_len: i64,
}

pub(in crate::objective) struct DeviceTrainingData {
    groups: Vec<Group>,
    layouts: Option<Vec<crate::bellman::device::RecordLayout>>,
    records: Vec<Record>,
    plans: HashMap<u32, DeviceFeaturePlan>,
    config: FullGraphObjectiveConfig,
}

pub(in crate::objective) struct BatchAudit {
    pub(in crate::objective) records: Vec<usize>,
    pub(in crate::objective) selected: Tensor,
    pub(in crate::objective) sampled_candidates: Option<Vec<Tensor>>,
    pub(in crate::objective) diagnostics: ObjectiveDiagnostics,
}

impl DeviceTrainingData {
    /// Return the resident plans when a discard-only benchmark evaluates the
    /// just-trained model. This consumes the training data and its GPU buffers.
    pub(in crate::objective) fn into_plans(self) -> HashMap<u32, DeviceFeaturePlan> {
        self.plans
    }

    /// Validate sealed records, certificates and the entire forbidden-neighbor
    /// boundary before any forward. Upload packed roots and ragged teacher labels
    /// once per alphabet. No successor matrix is cached on the host for scoring.
    pub(in crate::objective) fn prepare(
        objective: &FullGraphObjective,
        records: &[TrainingRecord],
        admission: Option<&dyn StateAdmission>,
        plans: HashMap<u32, DeviceFeaturePlan>,
        index: usize,
    ) -> Result<Self, String> {
        let device = Device::Cuda(index);
        let mut partitions: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (i, r) in records.iter().enumerate() {
            partitions.entry(r.state.alphabet_id()).or_default().push(i);
        }
        let mut metadata: Vec<Option<Record>> = (0..records.len()).map(|_| None).collect();
        let mut groups = Vec::new();
        for (id, indices) in partitions {
            let alphabet = objective
                .alphabets
                .get(id)
                .ok_or("unregistered training alphabet")?;
            let n = alphabet.dimension();
            if !plans.contains_key(&n) {
                return Err(format!("no resident feature plan for training width {n}"));
            }
            let mut packed = Vec::new();
            let mut targets = Vec::new();
            let mut teacher_indices = Vec::new();
            let mut deltas = Vec::new();
            for (row, i) in indices.iter().copied().enumerate() {
                let record = &records[i];
                let upper = admitted_upper_bound(
                    alphabet,
                    objective.exact_catalogs.get(alphabet.id()),
                    &objective.forbidden,
                    record,
                    admission,
                )?;
                let teacher_start = teacher_indices.len() as i64;
                if objective.config.teacher_weight > 0.0 {
                    let (indices, relative) = first_gate_labels(record, alphabet, n as usize)?;
                    teacher_indices.extend(indices);
                    deltas.extend(relative);
                }
                metadata[i] = Some(Record {
                    group: groups.len(),
                    row: i64::try_from(row).map_err(|_| "training row overflow")?,
                    branch: if record.state.is_identity() {
                        RootBranch::Terminal {
                            qhat: objective.config.qhat_weight > 0.0,
                        }
                    } else {
                        RootBranch::Nonterminal
                    },
                    exact: record.exact_distance.is_some(),
                    teacher_start,
                    teacher_len: teacher_indices.len() as i64 - teacher_start,
                });
                targets.extend([upper as f64, record.exact_distance.unwrap_or(0) as f64]);
                packed.extend(record.state.packed_rows().iter().map(|&w| w as i64));
            }
            let count = indices.len() as i64;
            let roots = DeviceStateBatch::from_packed_tensor_for_alphabet(
                Tensor::from_slice(&packed)
                    .view([count, i64::from(n), i64::from(n.div_ceil(64))])
                    .to_device(device),
                alphabet,
                index,
            )?;
            groups.push(Group {
                neighbor_chunk_size: objective
                    .config
                    .effective_neighbor_chunk_size(objective.feature_policy(), n as usize)?,
                roots,
                actions: DevicePrimitiveBellman::new(alphabet, index)?,
                targets: Tensor::from_slice(&targets)
                    .view([count, 2])
                    .to_device(device),
                teacher_indices: Tensor::from_slice(&teacher_indices).to_device(device),
                teacher_cost_deltas: Tensor::from_slice(&deltas).to_device(device),
                root_f08: None,
            });
        }
        // Complete the admission checks for every group before evaluating any
        // immutable root features.
        if objective.config.cache_root_f08 {
            for group in &mut groups {
                let width = group.roots.dimension();
                group.root_f08 = Some(RootF08::prepare(&plans[&width], &group.roots)?);
            }
        }
        let records: Vec<_> = metadata
            .into_iter()
            .map(|r| r.expect("every sealed row belongs to one alphabet"))
            .collect();
        // V25 consumes padded groups with device masks. Legacy checkpoints use
        // exact per-chunk extents to reproduce their FP32 update order.
        let layouts =
            if objective.feature_policy().schema_id == crate::schema::MISSING_TOKEN_LAYOUT_ID {
                None
            } else {
                Some(
                    records
                        .iter()
                        .map(|record| {
                            let group = &groups[record.group];
                            let root = group.roots.row(record.row)?;
                            group.actions.prepare_layout(
                                &root,
                                &plans[&root.dimension()],
                                record.branch,
                                group.neighbor_chunk_size,
                            )
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                )
            };
        Ok(Self {
            groups,
            records,
            layouts,
            plans,
            config: objective.config.clone(),
        })
    }

    pub(in crate::objective) fn compute(
        &self,
        model: &FusionDistanceField,
        indices: &[usize],
        attempt: u64,
    ) -> Result<(DeviceUpdateLoss, BatchAudit), String> {
        self.compute_at_position(model, indices, attempt, 0)
    }

    /// The offset preserves each member's sampling seed when the logical
    /// optimizer batch is evaluated as separate one-record microbatches.
    pub(in crate::objective) fn compute_at_position(
        &self,
        model: &FusionDistanceField,
        indices: &[usize],
        attempt: u64,
        position_offset: usize,
    ) -> Result<(DeviceUpdateLoss, BatchAudit), String> {
        if indices.is_empty() || indices.iter().any(|&i| i >= self.records.len()) {
            return Err("invalid device training sample indices".into());
        }
        let mut losses = Vec::new();
        let mut validity = Vec::new();
        let mut group_activity = Vec::new();
        let mut selected = Vec::new();
        let mut sampled_candidates = self.config.sampled_neighbor_count.map(|_| Vec::new());
        let mut diagnostics = ObjectiveDiagnostics {
            examples: indices.len(),
            ..Default::default()
        };
        for (position, &i) in indices.iter().enumerate() {
            let logical_position = position.checked_add(position_offset)
                .ok_or("training batch position overflow")?;
            let record = &self.records[i];
            let group = &self.groups[record.group];
            let root = group.roots.row(record.row)?;
            let root_f08 = group.root_f08.as_ref().map(|cache| cache.row(record.row));
            let backup = group.actions.backup_with_layout(
                model,
                &self.plans[&root.dimension()],
                &root,
                record.branch,
                group.neighbor_chunk_size,
                self.config.require_all_model_inputs,
                self.layouts.as_ref().map(|layouts| &layouts[i]),
                self.config.share_candidate_wire_projections,
                self.config.share_training_wire_projections_min_width
                    .is_some_and(|min_width| root.dimension() >= min_width),
                self.config.sampled_neighbor_count.map(|count| SampleRequest {
                    count,
                    seed: sampling_seed(attempt, i, logical_position),
                }),
                root_f08.as_ref(),
            )?;
            let output = backup.parent;
            let value = output.value.view([]);
            let target = group.targets.select(0, record.row).to_kind(value.kind());
            let upper = target.select(0, 0);
            let interval = (backup.lower_bound.to_kind(value.kind()) - &value)
                .relu()
                .square()
                + (&value - upper).relu().square();
            let exact = if record.exact {
                diagnostics.exact_examples += 1;
                value.smooth_l1_loss(&target.select(0, 1), Reduction::Mean, 1.0)
            } else {
                &value * 0.0
            };
            let mut valid = backup.valid;
            let flat_mask = output.action_mask.view([-1]);
            let qhat = if self.config.qhat_weight > 0.0 && backup.primitive_totals.numel() > 0 {
                let actions = match &backup.primitive_indices {
                    Some(indices) => group.actions.action_indices().index_select(0, indices),
                    None => group.actions.action_indices().shallow_clone(),
                };
                valid = valid.logical_and(&flat_mask.index_select(0, &actions).all());
                output
                    .qhat
                    .view([-1])
                    .index_select(0, &actions)
                    .smooth_l1_loss(
                        &backup.primitive_totals.to_kind(output.qhat.kind()),
                        Reduction::Mean,
                        1.0,
                    )
            } else {
                output
                    .qhat
                    .masked_fill(&Tensor::ones_like(&output.action_mask), 0.0)
                    .sum(output.qhat.kind())
                    * 0.0
            };
            let teacher = if record.teacher_len > 0 {
                diagnostics.teacher_examples += 1;
                let indices =
                    group
                        .teacher_indices
                        .narrow(0, record.teacher_start, record.teacher_len);
                valid = valid.logical_and(&flat_mask.index_select(0, &indices).all());
                let kind = output.proposal_logits.kind();
                let costs =
                    group
                        .teacher_cost_deltas
                        .narrow(0, record.teacher_start, record.teacher_len);
                let targets = (-costs / self.config.teacher_temperature)
                    .to_kind(kind)
                    .softmax(0, kind);
                let log_probs = output
                    .proposal_logits
                    .masked_fill(&output.action_mask.logical_not(), f64::NEG_INFINITY)
                    .view([-1])
                    .log_softmax(0, kind)
                    .index_select(0, &indices);
                -(targets * log_probs).sum(kind)
            } else {
                output
                    .proposal_logits
                    .masked_fill(&output.action_mask.logical_not(), 0.0)
                    .sum(output.proposal_logits.kind())
                    * 0.0
            };
            let term = backup.loss * self.config.bellman_weight
                + (interval + exact) * self.config.supervision_weight
                + qhat * self.config.qhat_weight
                + teacher * self.config.teacher_weight;
            if record.branch == RootBranch::Nonterminal {
                diagnostics.primitive_candidates = diagnostics
                    .primitive_candidates
                    .checked_add(backup.primitive_totals.numel())
                    .ok_or("primitive candidate counter overflow")?;
            }
            let winner = if !term.requires_grad() {
                Tensor::full([], -2i64, (Kind::Int64, model.device()))
            } else {
                backup
                    .selected_index
                    .unwrap_or_else(|| Tensor::full([], -1i64, (Kind::Int64, model.device())))
            };
            selected.push(winner);
            if let Some(candidates) = &mut sampled_candidates {
                candidates.push(backup.primitive_indices.ok_or("missing sampled action audit")?);
            }
            group_activity.push(backup.group_activity);
            validity.push(valid);
            losses.push(term);
        }
        let loss = Tensor::stack(&losses, 0).mean(Kind::Float);
        let valid = Tensor::stack(&validity, 0)
            .all()
            .logical_and(&loss.isfinite());
        let groups = Tensor::stack(&group_activity, 0).any_dim(0, false);
        Ok((
            DeviceUpdateLoss {
                loss,
                valid,
                group_activity: groups,
            },
            BatchAudit {
                records: indices.to_vec(),
                selected: Tensor::stack(&selected, 0),
                sampled_candidates,
                diagnostics,
            },
        ))
    }
}

/// A stable stateless schedule: checkpointed attempt number and record index
/// determine exploration, including across resume and interval rollback.
fn sampling_seed(attempt: u64, record: usize, position: usize) -> u64 {
    let mut value = attempt ^ (record as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (position as u64).wrapping_mul(0xd1b54a32d192ed03);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}
