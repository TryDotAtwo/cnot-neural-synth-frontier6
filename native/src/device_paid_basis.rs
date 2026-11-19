//! D09-D14 from resident exact local compilers and ordered physical words.
//! Setup inspects metadata only. No state-dependent decisions, counts, bases,
//! gates or endpoint values cross the device boundary during evaluation.
use crate::cuda::{PaidBasisAppendArgs, PaidBasisArgs};
use crate::device_paid_shear::{allocate, DevicePaidShearBatch, DevicePaidShearPlan, PaidAtlasSet};
use crate::device_state::DeviceStateBatch;
use crate::{PaidBasisBlockContext, PaidShearConfig};
use tch::{Device, Kind, Tensor};

pub struct DevicePaidBasisBatch {
    groups: Tensor,
    pairs: Tensor,
    incidence: Tensor,
    token_valid: Tensor,
    status: Tensor,
    word: Tensor,
    lengths: Tensor,
    endpoint: Tensor,
}
impl DevicePaidBasisBatch {
    pub fn feature(&self, ordinal: u8) -> Option<Tensor> {
        let (offset, width) = match ordinal {
            9 => (0, 1),
            10 => (1, 1),
            11 => (2, 2),
            12 => (4, 2),
            13 => (6, 12),
            14 => return Some(self.pairs.shallow_clone()),
            _ => return None,
        };
        Some(self.groups.narrow(2, offset, width))
    }
    pub fn incidence(&self) -> &Tensor {
        &self.incidence
    }
    pub fn token_valid(&self) -> &Tensor {
        &self.token_valid
    }
    pub fn status(&self) -> &Tensor {
        &self.status
    }
    /// Complete reduced physical word [B,Q,2]; slots after lengths are zero.
    pub fn word(&self) -> &Tensor {
        &self.word
    }
    pub fn lengths(&self) -> &Tensor {
        &self.lengths
    }
    /// Independently composed original shears, verified against the full word.
    pub fn endpoint(&self) -> &Tensor {
        &self.endpoint
    }
}
pub struct DevicePaidBasisPlan {
    dimension: u32,
    device_index: usize,
    contexts: Vec<DevicePaidShearPlan>,
    groups: i64,
    capacity: i64,
    unavailable: Option<u8>,
}
impl DevicePaidBasisPlan {
    /// Binds every configured context (including inactive widths) to the
    /// canonical registration. Byte policy uses the source's exact estimate.
    pub fn new(
        dimension: u32,
        device_index: usize,
        contexts: &[PaidBasisBlockContext],
        config: PaidShearConfig,
        max_bytes: Option<usize>,
    ) -> Result<Self, String> {
        crate::register_paid_basis_features(
            &mut crate::FeatureRegistry::new(),
            contexts.to_vec(),
            config,
        )
        .map_err(|e| e.0)?;
        crate::torch_device::require_cuda_device(device_index)?;
        if dimension == 0 || dimension > (1 << 24) {
            return Err("paid-basis width exceeds device state contract".into());
        }
        let matching: Vec<_> = contexts
            .iter()
            .filter(|c| c.dimension() == dimension)
            .collect();
        let unavailable = if matching.is_empty() {
            Some(3)
        } else {
            match crate::paid_basis_features::workspace_estimate(dimension as usize, matching.len())
            {
                Some(bytes) if max_bytes.is_none_or(|limit| bytes <= limit) => None,
                _ => Some(4),
            }
        };
        let mut prepared = Vec::new();
        let mut capacity = 0i64;
        if unavailable.is_none() {
            let atlases = PaidAtlasSet::new(config.local_width, device_index)?;
            for context in &matching {
                let cells = context
                    .source_wires()
                    .len()
                    .checked_mul(context.target_wires().len())
                    .and_then(|n| i64::try_from(n).ok())
                    .ok_or("paid-basis word capacity overflow")?;
                capacity = capacity
                    .checked_add(cells)
                    .ok_or("paid-basis combined word capacity overflow")?;
                prepared.push(DevicePaidShearPlan::with_atlases(
                    (*context).clone(),
                    config,
                    device_index,
                    &atlases,
                )?);
            }
        }
        Ok(Self {
            dimension,
            device_index,
            contexts: prepared,
            groups: matching.len().max(1) as i64,
            capacity: capacity.max(1),
            unavailable,
        })
    }
    pub fn evaluate(&self, states: &DeviceStateBatch) -> Result<DevicePaidBasisBatch, String> {
        let device = Device::Cuda(self.device_index);
        if states.dimension() != self.dimension || states.tensor().device() != device {
            return Err("paid-basis states and plan widths/devices differ".into());
        }
        let alphabet = states
            .alphabet_id()
            .ok_or("paid-basis requires an explicit state alphabet identity")?;
        // Alphabet rejection precedes the source workspace budget.
        let unavailable = if alphabet != format!("all_to_all_unit_cnot:{}", self.dimension) {
            Some(3)
        } else {
            self.unavailable
        };
        let b = states.batch_size();
        let n = i64::from(self.dimension);
        let g = self.groups;
        let q = self.capacity;
        let zero = unavailable.is_some();
        let output = DevicePaidBasisBatch {
            groups: allocate(&[b, g, 18], Kind::Float, device, zero)?,
            pairs: allocate(&[b, n, n, 4], Kind::Float, device, zero)?,
            incidence: allocate(&[b, g, n, 2], Kind::Float, device, zero)?,
            token_valid: allocate(&[b, g], Kind::Uint8, device, zero)?,
            status: allocate(&[b], Kind::Uint8, device, false)?,
            word: allocate(&[b, q, 2], Kind::Int, device, zero)?,
            lengths: allocate(&[b], Kind::Int64, device, zero)?,
            endpoint: allocate(&states.tensor().size(), Kind::Int64, device, zero)?,
        };
        if let Some(status) = unavailable {
            return Ok(DevicePaidBasisBatch {
                status: output.status.full_like(i64::from(status)),
                ..output
            });
        }
        let labels = allocate(&[b, q], Kind::Int64, device, false)?;
        let counts = allocate(&[b, n, n, 3], Kind::Int64, device, false)?;
        let replay = allocate(&states.tensor().size(), Kind::Int64, device, false)?;
        let verified = allocate(&[b], Kind::Uint8, device, false)?;
        let overflow = allocate(&[b], Kind::Int, device, false)?;
        let args = PaidBasisArgs {
            word: output.word.data_ptr().cast(),
            labels: labels.data_ptr().cast(),
            lengths: output.lengths.data_ptr().cast(),
            endpoint: output.endpoint.data_ptr().cast(),
            replay: replay.data_ptr().cast(),
            counts: counts.data_ptr().cast(),
            groups: output.groups.data_ptr().cast(),
            incidence: output.incidence.data_ptr().cast(),
            valid: output.token_valid.data_ptr().cast(),
            pairs: output.pairs.data_ptr().cast(),
            status: output.status.data_ptr().cast(),
            verified: verified.data_ptr().cast(),
            overflow: overflow.data_ptr().cast(),
            stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
            capacity: q as u64,
            state_words: states.tensor().numel() as u64,
            gate_elements: output.word.numel() as u64,
            label_words: labels.numel() as u64,
            count_elements: counts.numel() as u64,
            group_elements: output.groups.numel() as u64,
            incidence_elements: output.incidence.numel() as u64,
            token_elements: output.token_valid.numel() as u64,
            pair_elements: output.pairs.numel() as u64,
            batch_elements: b as u64,
            batch: u32::try_from(b).map_err(|_| "paid-basis batch exceeds CUDA range")?,
            dimension: self.dimension,
            words_per_row: self.dimension.div_ceil(64),
            context_count: g as u32,
        };
        // Keep original neighboring words alive, including empty ones. Every
        // descriptor below is static shape metadata plus resident pointers.
        let mut candidates: Vec<DevicePaidShearBatch> = Vec::with_capacity(self.contexts.len());
        unsafe {
            crate::cuda::paid_basis_init(&args)?;
        }
        for (index, plan) in self.contexts.iter().enumerate() {
            let current = plan.compile(states)?;
            let previous = candidates.last();
            let (sources, targets) = plan.physical_wires();
            let append = PaidBasisAppendArgs {
                output: args,
                states: states.tensor().data_ptr().cast(),
                sources: sources.data_ptr().cast(),
                targets: targets.data_ptr().cast(),
                word: current.word().data_ptr().cast(),
                lengths: current.lengths().data_ptr().cast(),
                statistics: current.statistics().data_ptr().cast(),
                status: current.status().data_ptr().cast(),
                previous_word: previous.map_or(std::ptr::null(), |p| p.word().data_ptr().cast()),
                previous_lengths: previous
                    .map_or(std::ptr::null(), |p| p.lengths().data_ptr().cast()),
                capacity: current.word().size()[1] as u64,
                gate_elements: current.word().numel() as u64,
                stat_elements: current.statistics().numel() as u64,
                source_elements: sources.numel() as u64,
                target_elements: targets.numel() as u64,
                previous_capacity: previous.map_or(0, |p| p.word().size()[1] as u64),
                previous_gate_elements: previous.map_or(0, |p| p.word().numel() as u64),
                source_count: sources.numel() as u32,
                target_count: targets.numel() as u32,
                context_index: index as u32,
            };
            unsafe {
                crate::cuda::paid_basis_append(&append)?;
            }
            candidates.push(current);
        }
        unsafe {
            crate::cuda::paid_basis_finish(&args)?;
        }
        Ok(output)
    }
}
