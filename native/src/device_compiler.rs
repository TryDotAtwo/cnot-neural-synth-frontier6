//! Source-order CNOT synthesis, independent replay and certified feature bounds.
//! All matrix/word calculations stay on CUDA; host checks inspect metadata only.
use crate::cuda::GeneralCompilerArgs;
use crate::device_state::DeviceStateBatch;
use crate::general_compiler::GeneralCompilerLimits;
use crate::model::DeviceFeatureStatus;
use tch::{Device, Kind, Tensor};

#[derive(Debug)]
pub struct DeviceGeneralWordBatch {
    gates: Tensor,
    lengths: Tensor,
    status: Tensor,
    lower_bounds: Tensor,
    f09: Tensor,
    f10: Tensor,
}
impl DeviceGeneralWordBatch {
    pub fn gates(&self) -> &Tensor {
        &self.gates
    }
    pub fn lengths(&self) -> &Tensor {
        &self.lengths
    }
    pub fn status(&self) -> &Tensor {
        &self.status
    }
    pub fn lower_bounds(&self) -> &Tensor {
        &self.lower_bounds
    }
    pub fn f09(&self) -> &Tensor {
        &self.f09
    }
    pub fn f10(&self) -> &Tensor {
        &self.f10
    }
    pub fn producer(&self) -> &'static str {
        "general_gauss_jordan_v1"
    }
}

pub struct DeviceGeneralCompilerPlan {
    dimension: u32,
    device_index: usize,
    limits: GeneralCompilerLimits,
    capacity: i64,
    alphabet_id: String,
}

fn allocate(shape: &[i64], kind: Kind, device: Device, zero: bool) -> Result<Tensor, String> {
    let count = shape
        .iter()
        .try_fold(
            1i64,
            |count, &n| if n < 0 { None } else { count.checked_mul(n) },
        )
        .ok_or("device compiler tensor extent overflow")?;
    count
        .checked_mul(match kind {
            Kind::Int64 => 8,
            Kind::Uint8 => 1,
            _ => 4,
        })
        .ok_or("device compiler tensor byte extent overflow")?;
    if zero {
        Tensor::f_zeros(shape, (kind, device))
    } else {
        Tensor::f_empty(shape, (kind, device))
    }
    .map_err(|e| e.to_string())
}

impl DeviceGeneralCompilerPlan {
    /// Compile immutable widths, limits and alphabet identity, without states.
    pub fn new(
        dimension: u32,
        device_index: usize,
        limits: GeneralCompilerLimits,
    ) -> Result<Self, String> {
        crate::torch_device::require_cuda_device(device_index)?;
        if !(2..=1 << 24).contains(&dimension) {
            return Err("device compiler width exceeds kernel capacity".into());
        }
        let capacity = if dimension > limits.max_dimension {
            0
        } else {
            // At most n-1 swaps and n*(n-1) non-swap eliminations.
            limits
                .max_gates
                .min((u64::from(dimension) + 3) * (u64::from(dimension) - 1))
        };
        Ok(Self {
            dimension,
            device_index,
            limits,
            capacity: i64::try_from(capacity)
                .map_err(|_| "device compiler capacity exceeds tensor range")?,
            alphabet_id: format!("all_to_all_unit_cnot:{dimension}"),
        })
    }
    pub fn limits(&self) -> GeneralCompilerLimits {
        self.limits
    }

    /// Produces synthesis-order words and F09/F10 only after full device replay.
    /// Status uses DeviceFeatureStatus. Failed samples have zero words/lengths
    /// and payloads. Counts and numerical statuses are never read by the host.
    /// This route also enforces F09's exact-f32 cost limit (2^24 gates); larger
    /// words are ResourceLimit even when a custom compiler budget permits them.
    pub fn compile(&self, states: &DeviceStateBatch) -> Result<DeviceGeneralWordBatch, String> {
        if states.dimension() != self.dimension
            || states.tensor().device() != Device::Cuda(self.device_index)
        {
            return Err("device compiler and states have different width/device".into());
        }
        let alphabet = states
            .alphabet_id()
            .ok_or("device compiler requires an explicit state alphabet identity")?;
        let unavailable = if alphabet != self.alphabet_id {
            Some(DeviceFeatureStatus::Unsupported)
        } else if self.dimension > self.limits.max_dimension {
            Some(DeviceFeatureStatus::ResourceLimit)
        } else {
            None
        };
        let b = states.batch_size();
        let device = states.tensor().device();
        let capacity = if unavailable.is_some() {
            0
        } else {
            self.capacity
        };
        let output = DeviceGeneralWordBatch {
            gates: allocate(&[b, capacity, 2], Kind::Int, device, unavailable.is_some())?,
            lengths: allocate(&[b], Kind::Int64, device, unavailable.is_some())?,
            status: allocate(&[b], Kind::Uint8, device, false)?,
            lower_bounds: allocate(&[b], Kind::Int64, device, unavailable.is_some())?,
            f09: allocate(&[b, 4], Kind::Float, device, unavailable.is_some())?,
            f10: allocate(&[b, 5], Kind::Float, device, unavailable.is_some())?,
        };
        if let Some(code) = unavailable {
            return Ok(DeviceGeneralWordBatch {
                status: output.status.full_like(code as i64),
                ..output
            });
        }
        let scratch = allocate(&states.tensor().size(), Kind::Int64, device, false)?;
        let verified = allocate(&[b], Kind::Uint8, device, false)?;
        let args = GeneralCompilerArgs {
            states: states.tensor().data_ptr().cast(),
            scratch: scratch.data_ptr().cast(),
            gates: output.gates.data_ptr().cast(),
            lengths: output.lengths.data_ptr().cast(),
            status: output.status.data_ptr().cast(),
            verified: verified.data_ptr().cast(),
            lower_bounds: output.lower_bounds.data_ptr().cast(),
            f09: output.f09.data_ptr().cast(),
            f10: output.f10.data_ptr().cast(),
            stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
            word_capacity: capacity as u64,
            workspace_words: scratch.numel() as u64,
            gate_elements: output.gates.numel() as u64,
            batch: u32::try_from(b).map_err(|_| "device compiler batch exceeds CUDA range")?,
            dimension: self.dimension,
            words_per_row: self.dimension.div_ceil(64),
        };
        // Allocations and consumers use the current Torch stream. Dropping
        // scratch after enqueue is safe under its stream-ordered allocator.
        unsafe {
            crate::cuda::launch_general_compiler(&args)?;
        }
        Ok(output)
    }
}

/// Independent device replay of fixed-capacity words; returns [batch] bool.
/// Invalid indices, self-gates, negative/oversized lengths and wrong endpoints
/// produce false. This verifier does not establish optimality.
pub fn verify_device_cnot_words(
    states: &DeviceStateBatch,
    gates: &Tensor,
    lengths: &Tensor,
) -> Result<Tensor, String> {
    let b = states.batch_size();
    let n = states.dimension();
    let device = states.tensor().device();
    let Device::Cuda(index) = device else {
        return Err("CNOT replay requires CUDA".into());
    };
    if states.alphabet_id() != Some(format!("all_to_all_unit_cnot:{n}").as_str()) {
        return Err("CNOT replay requires an explicit all-to-all unit alphabet".into());
    }
    let shape = gates.size();
    if shape.len() != 3
        || shape[0] != b
        || shape[2] != 2
        || gates.kind() != Kind::Int
        || !gates.is_contiguous()
        || gates.device() != device
        || lengths.size() != [b]
        || lengths.kind() != Kind::Int64
        || !lengths.is_contiguous()
        || lengths.device() != device
    {
        return Err("CNOT replay words/counts have invalid shapes, types, layout or device".into());
    }
    let scratch = allocate(&states.tensor().size(), Kind::Int64, device, false)?;
    let verified = allocate(&[b], Kind::Uint8, device, false)?;
    unsafe {
        crate::cuda::verify_cnot_words(
            states.tensor().data_ptr().cast(),
            gates.data_ptr().cast(),
            lengths.data_ptr().cast(),
            scratch.data_ptr().cast(),
            verified.data_ptr().cast(),
            u32::try_from(b).map_err(|_| "replay batch exceeds CUDA range")?,
            n,
            n.div_ceil(64),
            shape[1] as u64,
            scratch.numel() as u64,
            gates.numel() as u64,
            crate::cuda::current_stream(index).map_err(str::to_owned)?,
        )?;
    }
    Ok(verified.to_kind(Kind::Bool))
}
