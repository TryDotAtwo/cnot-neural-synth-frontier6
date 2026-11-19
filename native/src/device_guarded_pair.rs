//! Complete Stage35 encoder, independent GPU certificate/word verification,
//! and source-compatible F02/F06. Host inspects static metadata only.
use crate::cuda::GuardedPairArgs;
use crate::device_state::DeviceStateBatch;
use tch::{Device, Kind, Tensor};
pub struct DeviceGuardedPairBatch {
    gates: Tensor,
    lengths: Tensor,
    certificate: Tensor,
    status: Tensor,
    f02: Tensor,
    f06: Tensor,
    valid: Tensor,
    incidence: Tensor,
}
impl DeviceGuardedPairBatch {
    /// [B,n+1,2] encoder from identity to input; reverse for source M->I word.
    pub fn encoder_gates(&self) -> &Tensor {
        &self.gates
    }
    pub fn lengths(&self) -> &Tensor {
        &self.lengths
    }
    /// [B,16] slots documented in Cnot103GuardedPairArgs, absent indices=n.
    pub fn certificate(&self) -> &Tensor {
        &self.certificate
    }
    pub fn status(&self) -> &Tensor {
        &self.status
    }
    pub fn f02(&self) -> &Tensor {
        &self.f02
    }
    pub fn f06(&self) -> &Tensor {
        &self.f06
    }
    pub fn token_valid(&self) -> &Tensor {
        &self.valid
    }
    pub fn incidence(&self) -> &Tensor {
        &self.incidence
    }
}
pub struct DeviceGuardedPairPlan {
    dimension: u32,
    device_index: usize,
    max_n: u32,
}
fn allocate(shape: &[i64], kind: Kind, device: Device, zero: bool) -> Result<Tensor, String> {
    shape
        .iter()
        .try_fold(
            1i64,
            |count, &n| if n < 0 { None } else { count.checked_mul(n) },
        )
        .and_then(|count| {
            count.checked_mul(match kind {
                Kind::Int64 => 8,
                Kind::Uint8 => 1,
                _ => 4,
            })
        })
        .ok_or("guarded-pair tensor extent overflow")?;
    if zero {
        Tensor::f_zeros(shape, (kind, device))
    } else {
        Tensor::f_empty(shape, (kind, device))
    }
    .map_err(|e| e.to_string())
}
impl DeviceGuardedPairPlan {
    pub fn new(dimension: u32, device_index: usize, max_n: u32) -> Result<Self, String> {
        crate::torch_device::require_cuda_device(device_index)?;
        if !(2..=1 << 24).contains(&dimension) || max_n < 2 {
            return Err("guarded-pair width exceeds device/configuration contract".into());
        }
        Ok(Self {
            dimension,
            device_index,
            max_n,
        })
    }
    pub fn evaluate(&self, states: &DeviceStateBatch) -> Result<DeviceGuardedPairBatch, String> {
        let device = Device::Cuda(self.device_index);
        if states.dimension() != self.dimension || states.tensor().device() != device {
            return Err("guarded-pair states and plan differ in width/device".into());
        }
        let label = states
            .alphabet_id()
            .ok_or("guarded-pair requires an explicit state alphabet identity")?;
        // Source guard order: configured width before complete-alphabet check.
        let unavailable = if self.dimension > self.max_n {
            Some(4)
        } else if label != format!("all_to_all_unit_cnot:{}", self.dimension) {
            Some(3)
        } else {
            None
        };
        let b = states.batch_size();
        let n = i64::from(self.dimension);
        let zero = unavailable.is_some();
        let output = DeviceGuardedPairBatch {
            gates: allocate(&[b, n + 1, 2], Kind::Int, device, zero)?,
            lengths: allocate(&[b], Kind::Int64, device, zero)?,
            certificate: allocate(&[b, 16], Kind::Int64, device, zero)?,
            status: allocate(&[b], Kind::Uint8, device, false)?,
            f02: allocate(&[b, 1], Kind::Float, device, zero)?,
            f06: allocate(&[b, 1, 6], Kind::Float, device, zero)?,
            valid: allocate(&[b, 1], Kind::Uint8, device, zero)?,
            incidence: allocate(&[b, 1, n, 3], Kind::Float, device, zero)?,
        };
        if let Some(code) = unavailable {
            return Ok(DeviceGuardedPairBatch {
                status: output.status.full_like(code),
                ..output
            });
        }
        let scratch = allocate(&states.tensor().size(), Kind::Int64, device, false)?;
        let graph = allocate(&[b, 2 * n], Kind::Int64, device, false)?;
        let verified = allocate(&[b], Kind::Uint8, device, false)?;
        let args = GuardedPairArgs {
            states: states.tensor().data_ptr().cast(),
            scratch: scratch.data_ptr().cast(),
            graph: graph.data_ptr().cast(),
            gates: output.gates.data_ptr().cast(),
            lengths: output.lengths.data_ptr().cast(),
            certificate: output.certificate.data_ptr().cast(),
            status: output.status.data_ptr().cast(),
            verified: verified.data_ptr().cast(),
            f02: output.f02.data_ptr().cast(),
            f06: output.f06.data_ptr().cast(),
            valid: output.valid.data_ptr().cast(),
            incidence: output.incidence.data_ptr().cast(),
            stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
            word_capacity: (n + 1) as u64,
            state_words: states.tensor().numel() as u64,
            scratch_words: scratch.numel() as u64,
            graph_words: graph.numel() as u64,
            gate_elements: output.gates.numel() as u64,
            certificate_words: output.certificate.numel() as u64,
            f06_elements: output.f06.numel() as u64,
            incidence_elements: output.incidence.numel() as u64,
            batch_elements: b as u64,
            batch: u32::try_from(b).map_err(|_| "guarded-pair batch exceeds CUDA range")?,
            dimension: self.dimension,
            words_per_row: self.dimension.div_ceil(64),
        };
        unsafe {
            crate::cuda::launch_guarded_pair(&args)?;
        }
        Ok(output)
    }
}
