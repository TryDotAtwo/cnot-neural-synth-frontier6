//! Full source-order paid-shear compilation and independent endpoint replay.
//! Only explicit physical-context metadata is prepared on host. All block,
//! basis, local selection, words, costs and accepted moves remain on CUDA.
use crate::cuda::PaidShearArgs;
use crate::device_paid_basis_atlas::DevicePaidBasisAtlas;
use crate::device_state::DeviceStateBatch;
use crate::{BasisSide, PaidBasisBlockContext, PaidShearConfig};
use tch::{Device, Kind, Tensor};

pub struct DevicePaidShearBatch {
    transformed_block: Tensor,
    source_basis: Tensor,
    target_basis: Tensor,
    preparation: Tensor,
    restoration: Tensor,
    core: Tensor,
    word: Tensor,
    cancellation_events: Tensor,
    accepted_moves: Tensor,
    statistics: Tensor,
    lengths: Tensor,
    endpoint: Tensor,
    status: Tensor,
}
impl DevicePaidShearBatch {
    pub fn transformed_block(&self) -> &Tensor {
        &self.transformed_block
    }
    pub fn source_basis(&self) -> &Tensor {
        &self.source_basis
    }
    pub fn target_basis(&self) -> &Tensor {
        &self.target_basis
    }
    pub fn preparation(&self) -> &Tensor {
        &self.preparation
    }
    pub fn restoration(&self) -> &Tensor {
        &self.restoration
    }
    pub fn core(&self) -> &Tensor {
        &self.core
    }
    pub fn word(&self) -> &Tensor {
        &self.word
    }
    /// One physical gate for each canceled cross-movement pair, in traversal
    /// order. statistics[:,14] is the count; unused slots are zero.
    pub fn cancellation_events(&self) -> &Tensor {
        &self.cancellation_events
    }
    /// [B,P,11]: sweep,side,width,indices[4],before,after,prep_start,prep_end.
    pub fn accepted_moves(&self) -> &Tensor {
        &self.accepted_moves
    }
    /// [B,16]: direct,paid,prep,source_prep,target_prep,core,restoration,tested,
    /// moves,source_moves,target_moves,source_changed,target_changed,swaps,
    /// cancellation pairs,unshortened. Counts are exact int64.
    pub fn statistics(&self) -> &Tensor {
        &self.statistics
    }
    pub fn lengths(&self) -> &Tensor {
        &self.lengths
    }
    /// Packed full shear target, independently compared with the reduced word.
    pub fn endpoint(&self) -> &Tensor {
        &self.endpoint
    }
    /// 1 verified;3 unsupported alphabet/invalid certificate with zero payloads.
    pub fn status(&self) -> &Tensor {
        &self.status
    }
}
pub(crate) struct PaidAtlasSet {
    tables: [Option<DevicePaidBasisAtlas>; 6],
    width: usize,
    device_index: usize,
}
impl PaidAtlasSet {
    pub(crate) fn new(width: usize, device_index: usize) -> Result<Self, String> {
        if !(2..=4).contains(&width) {
            return Err("paid-shear local width must be2..4".into());
        }
        let mut tables = std::array::from_fn(|_| None);
        for side in 0..2 {
            for w in 2..=width {
                tables[side * 3 + w - 2] = Some(DevicePaidBasisAtlas::new(
                    w,
                    if side == 0 {
                        BasisSide::Source
                    } else {
                        BasisSide::Target
                    },
                    device_index,
                )?);
            }
        }
        Ok(Self {
            tables,
            width,
            device_index,
        })
    }
    fn shallow_clone(&self) -> Self {
        Self {
            tables: std::array::from_fn(|i| {
                self.tables[i]
                    .as_ref()
                    .map(DevicePaidBasisAtlas::shallow_clone)
            }),
            width: self.width,
            device_index: self.device_index,
        }
    }
}
pub struct DevicePaidShearPlan {
    context: PaidBasisBlockContext,
    config: PaidShearConfig,
    device_index: usize,
    sources: Tensor,
    targets: Tensor,
    atlases: PaidAtlasSet,
}
pub(crate) fn allocate(
    shape: &[i64],
    kind: Kind,
    device: Device,
    zero: bool,
) -> Result<Tensor, String> {
    shape
        .iter()
        .try_fold(
            1i64,
            |count, &v| if v < 0 { None } else { count.checked_mul(v) },
        )
        .and_then(|count| {
            count.checked_mul(match kind {
                Kind::Int64 => 8,
                Kind::Uint8 => 1,
                _ => 4,
            })
        })
        .ok_or("paid-shear tensor byte extent overflow")?;
    if zero {
        Tensor::f_zeros(shape, (kind, device))
    } else {
        Tensor::f_empty(shape, (kind, device))
    }
    .map_err(|e| e.to_string())
}
impl DevicePaidShearPlan {
    pub fn new(
        context: PaidBasisBlockContext,
        config: PaidShearConfig,
        device_index: usize,
    ) -> Result<Self, String> {
        Self::validate(&context, config)?;
        let atlases = PaidAtlasSet::new(config.local_width, device_index)?;
        Self::with_atlases(context, config, device_index, &atlases)
    }
    fn validate(context: &PaidBasisBlockContext, config: PaidShearConfig) -> Result<(), String> {
        if context.dimension() > (1 << 24) {
            return Err("paid-shear width exceeds device state contract".into());
        }
        crate::register_paid_basis_features(
            &mut crate::FeatureRegistry::new(),
            vec![context.clone()],
            config,
        )
        .map_err(|e| e.0)
    }
    pub(crate) fn with_atlases(
        context: PaidBasisBlockContext,
        config: PaidShearConfig,
        device_index: usize,
        atlases: &PaidAtlasSet,
    ) -> Result<Self, String> {
        Self::validate(&context, config)?;
        if atlases.width != config.local_width || atlases.device_index != device_index {
            return Err("paid-shear atlas metadata mismatch".into());
        }
        crate::torch_device::require_cuda_device(device_index)?;
        let upload = |indices: &[usize]| -> Result<Tensor, String> {
            let values = indices
                .iter()
                .map(|&x| i32::try_from(x).map_err(|_| "paid-shear wire exceeds device range"))
                .collect::<Result<Vec<_>, _>>()?;
            Tensor::f_from_slice(&values)
                .and_then(|t| t.f_to_device(Device::Cuda(device_index)))
                .map_err(|e| e.to_string())
        };
        Ok(Self {
            sources: upload(context.source_wires())?,
            targets: upload(context.target_wires())?,
            context,
            config,
            device_index,
            atlases: atlases.shallow_clone(),
        })
    }
    pub fn context(&self) -> &PaidBasisBlockContext {
        &self.context
    }
    pub fn config(&self) -> PaidShearConfig {
        self.config
    }
    pub(crate) fn physical_wires(&self) -> (&Tensor, &Tensor) {
        (&self.sources, &self.targets)
    }
    pub fn compile(&self, states: &DeviceStateBatch) -> Result<DevicePaidShearBatch, String> {
        let n = self.context.dimension();
        let device = Device::Cuda(self.device_index);
        if states.dimension() != n || states.tensor().device() != device {
            return Err("paid-shear states and plan widths/devices differ".into());
        }
        let alphabet = states
            .alphabet_id()
            .ok_or("paid shear requires an explicit state alphabet identity")?;
        let unavailable = alphabet != format!("all_to_all_unit_cnot:{n}");
        let b = states.batch_size();
        let batch = u32::try_from(b).map_err(|_| "paid-shear batch exceeds CUDA range")?;
        let s = self.context.source_wires().len() as i64;
        let t = self.context.target_wires().len() as i64;
        let c = s.checked_mul(t).ok_or("paid-shear block extent overflow")?;
        let p = (c / 2).max(1);
        let output = DevicePaidShearBatch {
            transformed_block: allocate(&[b, t, s], Kind::Uint8, device, unavailable)?,
            source_basis: allocate(&[b, s, s], Kind::Uint8, device, unavailable)?,
            target_basis: allocate(&[b, t, t], Kind::Uint8, device, unavailable)?,
            preparation: allocate(&[b, p, 2], Kind::Int, device, unavailable)?,
            restoration: allocate(&[b, p, 2], Kind::Int, device, unavailable)?,
            core: allocate(&[b, c, 2], Kind::Int, device, unavailable)?,
            word: allocate(&[b, c, 2], Kind::Int, device, unavailable)?,
            cancellation_events: allocate(&[b, p, 2], Kind::Int, device, unavailable)?,
            accepted_moves: allocate(&[b, p, 11], Kind::Int64, device, unavailable)?,
            statistics: allocate(&[b, 16], Kind::Int64, device, unavailable)?,
            lengths: allocate(&[b], Kind::Int64, device, unavailable)?,
            endpoint: allocate(&states.tensor().size(), Kind::Int64, device, unavailable)?,
            status: allocate(&[b], Kind::Uint8, device, false)?,
        };
        if unavailable {
            return Ok(DevicePaidShearBatch {
                status: output.status.full_like(3),
                ..output
            });
        }
        let scratch = allocate(&[b, 15 * s.max(t) + p + c], Kind::Int64, device, false)?;
        let replay = allocate(&states.tensor().size(), Kind::Int64, device, false)?;
        let verified = allocate(&[b], Kind::Uint8, device, false)?;
        let args = PaidShearArgs {
            states: states.tensor().data_ptr().cast(),
            sources: self.sources.data_ptr().cast(),
            targets: self.targets.data_ptr().cast(),
            atlases: std::array::from_fn(|i| {
                self.atlases.tables[i]
                    .as_ref()
                    .map_or(std::ptr::null(), |a| a.storage().data_ptr().cast())
            }),
            block: output.transformed_block.data_ptr().cast(),
            source_basis: output.source_basis.data_ptr().cast(),
            target_basis: output.target_basis.data_ptr().cast(),
            preparation: output.preparation.data_ptr().cast(),
            restoration: output.restoration.data_ptr().cast(),
            core: output.core.data_ptr().cast(),
            word: output.word.data_ptr().cast(),
            cancellations: output.cancellation_events.data_ptr().cast(),
            moves: output.accepted_moves.data_ptr().cast(),
            statistics: output.statistics.data_ptr().cast(),
            lengths: output.lengths.data_ptr().cast(),
            endpoint: output.endpoint.data_ptr().cast(),
            replay: replay.data_ptr().cast(),
            scratch: scratch.data_ptr().cast(),
            status: output.status.data_ptr().cast(),
            verified: verified.data_ptr().cast(),
            stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
            state_words: states.tensor().numel() as u64,
            block_elements: output.transformed_block.numel() as u64,
            source_basis_elements: output.source_basis.numel() as u64,
            target_basis_elements: output.target_basis.numel() as u64,
            preparation_elements: output.preparation.numel() as u64,
            core_elements: output.core.numel() as u64,
            move_elements: output.accepted_moves.numel() as u64,
            stat_elements: output.statistics.numel() as u64,
            scratch_words: scratch.numel() as u64,
            batch_elements: b as u64,
            source_elements: s as u64,
            target_elements: t as u64,
            atlas_words: std::array::from_fn(|i| {
                self.atlases.tables[i]
                    .as_ref()
                    .map_or(0, |a| a.storage().numel() as u64)
            }),
            batch,
            dimension: n,
            words_per_row: n.div_ceil(64),
            source_count: s as u32,
            target_count: t as u32,
            local_width: self.config.local_width as u32,
            sweeps: self.config.sweeps as u32,
        };
        unsafe {
            crate::cuda::launch_paid_shear(&args)?;
        }
        Ok(output)
    }
}
