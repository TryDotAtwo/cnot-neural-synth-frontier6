use super::contexts::BlockParityPlanInputs;
use crate::cuda::BlockHistogramArgs;
use crate::device_context::DeviceBlockParityContext;
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::{FeaturePolicy, FeatureSpec};
use tch::{Device, Kind, Tensor};

/// Setup-time capacities. Runtime token counts and ordering stay on the device.
pub(super) struct BlockHistogramPlan {
    patterns: i64,
    pairs: i64,
}

fn empty(shape: &[i64], kind: Kind, device: Device) -> Result<Tensor, String> {
    let elements = shape
        .iter()
        .try_fold(1i64, |product, &size| {
            if size < 0 {
                None
            } else {
                product.checked_mul(size)
            }
        })
        .ok_or("block-histogram tensor extent overflow")?;
    elements
        .checked_mul(match kind {
            Kind::Int64 => 8,
            Kind::Uint8 => 1,
            _ => 4,
        })
        .ok_or("block-histogram tensor byte extent overflow")?;
    Tensor::f_empty(shape, (kind, device)).map_err(|e| e.to_string())
}

impl BlockHistogramPlan {
    pub(super) fn new(
        dimension: u32,
        input: BlockParityPlanInputs<'_>,
        policy: &FeaturePolicy,
    ) -> Result<Self, String> {
        let max_targets = input
            .contexts
            .iter()
            .filter(|context| context.dimension() == dimension)
            .map(|context| context.targets().len())
            .max()
            .ok_or("no physical block matches the histogram plan width")?;
        let selected = |ordinals: &[u8]| {
            policy
                .enabled_ids
                .iter()
                .any(|id| id.family() == 'D' && ordinals.contains(&id.ordinal()))
        };
        let patterns = input.max_patterns.min(
            max_targets
                .checked_add(1)
                .ok_or("block pattern capacity overflow")?,
        );
        let pairs = if selected(&[6]) {
            input.max_pairs.min(
                max_targets
                    .checked_mul(max_targets.saturating_sub(1))
                    .ok_or("block pair capacity overflow")?
                    / 2,
            )
        } else {
            0
        };
        Ok(Self {
            patterns: i64::try_from(patterns).map_err(|_| "block pattern capacity overflow")?,
            pairs: i64::try_from(pairs).map_err(|_| "block pair capacity overflow")?,
        })
    }

    pub(super) fn append_entries(
        &self,
        states: &DeviceStateBatch,
        context: &DeviceBlockParityContext,
        specs: &[FeatureSpec],
        computed: &Tensor,
        entries: &mut Vec<DeviceFeatureEntry>,
    ) -> Result<(), String> {
        let b = states.batch_size();
        let c = context.context_count;
        let n = i64::from(states.dimension());
        let device = states.tensor().device();
        let items = b
            .checked_mul(c)
            .ok_or("block histogram launch extent overflow")?;
        if items > i64::from(i32::MAX) {
            return Err("block histogram launch exceeds CUDA grid capacity".into());
        }
        let p = self.patterns;
        let q = if super::selected(specs, 'D', &[6]) {
            self.pairs
        } else {
            0
        };
        let emit_patterns = super::selected(specs, 'D', &[1, 2, 4]);
        let frequency = empty(&[b, c, n + 1], Kind::Int, device)?;
        let stats = empty(&[b, c, 4], Kind::Int64, device)?;
        let sorted = empty(
            &[b, c, if emit_patterns || q > 0 { p } else { 0 }],
            Kind::Int,
            device,
        )?;
        let emitted = if emit_patterns { p } else { 0 };
        let pattern_values = empty(&[b, c, emitted, 3], Kind::Float, device)?;
        let pattern_valid = empty(&[b, c, emitted], Kind::Uint8, device)?;
        let pattern_incidence = empty(&[b, c, emitted, n, 3], Kind::Float, device)?;
        let pair_values = empty(&[b, c, q, 1], Kind::Float, device)?;
        let pair_valid = empty(&[b, c, q], Kind::Uint8, device)?;
        let pair_incidence = empty(&[b, c, q, n, 3], Kind::Float, device)?;
        let args = BlockHistogramArgs {
            states: states.tensor().data_ptr().cast(),
            source_masks: context.source_masks.data_ptr().cast(),
            target_masks: context.target_masks.data_ptr().cast(),
            frequencies: frequency.data_ptr().cast(),
            statistics: stats.data_ptr().cast(),
            ordered_ids: sorted.data_ptr().cast(),
            pattern_values: pattern_values.data_ptr().cast(),
            pattern_valid: pattern_valid.data_ptr().cast(),
            pattern_incidence: pattern_incidence.data_ptr().cast(),
            pair_values: pair_values.data_ptr().cast(),
            pair_valid: pair_valid.data_ptr().cast(),
            pair_incidence: pair_incidence.data_ptr().cast(),
            stream: crate::cuda::current_stream(context.device_index).map_err(str::to_owned)?,
            pair_capacity: q as u64,
            frequency_elements: frequency.numel() as u64,
            statistic_elements: stats.numel() as u64,
            pattern_groups: sorted.numel() as u64,
            pattern_incidence_elements: pattern_incidence.numel() as u64,
            pair_groups: pair_valid.numel() as u64,
            pair_incidence_elements: pair_incidence.numel() as u64,
            batch: u32::try_from(b).map_err(|_| "histogram batch exceeds CUDA range")?,
            contexts: u32::try_from(c).map_err(|_| "histogram contexts exceed CUDA range")?,
            dimension: states.dimension(),
            words_per_row: states.dimension().div_ceil(64),
            pattern_capacity: u32::try_from(p)
                .map_err(|_| "histogram patterns exceed CUDA range")?,
            emit_patterns: u32::from(emit_patterns),
        };
        // All buffers are contiguous, on the current PyTorch stream, and kept
        // live until every kernel and downstream tensor operation is queued.
        unsafe {
            crate::cuda::launch_block_histograms(&args)?;
        }
        let available = stats.select(2, 0).le(p).all_dim(1, false);
        let pair_available = stats
            .select(2, 3)
            .le(q)
            .all_dim(1, false)
            .logical_and(&available);
        let limited = computed.full_like(DeviceFeatureStatus::ResourceLimit as i64);
        let rank = if super::selected(specs, 'D', &[7]) {
            Some(states.projected_ranks(&context.target_masks, &context.source_masks, false)?)
        } else {
            None
        };
        for spec in specs.iter().filter(|spec| spec.id.family() == 'D') {
            let (values, valid, incidence, active) = match spec.id.ordinal() {
                ordinal @ (1 | 2 | 4) => {
                    let channel = match ordinal {
                        1 => 0,
                        2 => 1,
                        _ => 2,
                    };
                    (
                        pattern_values.select(3, channel).reshape([b, c * p, 1]),
                        pattern_valid.view([b, c * p]).to_kind(Kind::Bool),
                        pattern_incidence.view([b, c * p, n, 3]),
                        &available,
                    )
                }
                6 => (
                    pair_values.view([b, c * q, 1]),
                    pair_valid.view([b, c * q]).to_kind(Kind::Bool),
                    pair_incidence.view([b, c * q, n, 3]),
                    &pair_available,
                ),
                7 => {
                    let nonzero = stats.select(2, 0) - 1;
                    let total = stats.select(2, 1);
                    let summary = Tensor::stack(
                        &[
                            nonzero.shallow_clone(),
                            total.shallow_clone(),
                            &total - &nonzero,
                            stats.select(2, 2),
                        ],
                        2,
                    )
                    .to_kind(Kind::Float);
                    (
                        Tensor::cat(&[rank.as_ref().expect("selected D07 rank"), &summary], 2),
                        Tensor::ones([b, c], (Kind::Bool, device)),
                        context.incidence.expand([b, c, n, 3], true),
                        &available,
                    )
                }
                _ => continue,
            };
            // A single over-budget context makes the entire sample unavailable.
            // Fixed-capacity padding stays zero, including its physical roles.
            let mask = valid.logical_and(&active.unsqueeze(1));
            entries.push(DeviceFeatureEntry::new(
                spec.clone(),
                values.where_self(&mask.unsqueeze(-1), &Tensor::zeros_like(&values)),
                computed.where_self(active, &limited),
                Some(mask.shallow_clone()),
                Some(incidence.where_self(
                    &mask.unsqueeze(-1).unsqueeze(-1),
                    &Tensor::zeros_like(&incidence),
                )),
            ));
        }
        Ok(())
    }
}
