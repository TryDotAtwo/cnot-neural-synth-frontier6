//! Resident Stage37 exact and negative certificates, full word verification,
//! and F07. The negative class excludes only the changed-row bound length.
use crate::cuda::ZeroExcessArgs;
use crate::device_state::DeviceStateBatch;
use tch::{Device, Kind, Tensor};
pub struct DeviceZeroExcessBatch {
    headers: Tensor,
    parents: Tensor,
    changed: Tensor,
    missing: Tensor,
    order: Tensor,
    gates: Tensor,
    lengths: Tensor,
    status: Tensor,
    feature: Tensor,
}
impl DeviceZeroExcessBatch {
    /// [B,3]: class1Exact/2NoZeroExcess, changed count, missing count.
    pub fn headers(&self) -> &Tensor {
        &self.headers
    }
    /// [B,n,2]: physical control and mode0Initial/1Final/2Absent per target.
    pub fn parents(&self) -> &Tensor {
        &self.parents
    }
    /// Sorted physical indices, padded with n for computed samples.
    pub fn changed_rows(&self) -> &Tensor {
        &self.changed
    }
    pub fn missing_rows(&self) -> &Tensor {
        &self.missing
    }
    /// Exact source FIFO update order, padded with n; all n for negative rows.
    pub fn update_order(&self) -> &Tensor {
        &self.order
    }
    /// Encoder I->M is the reverse of the source M->I word; unused gates=0.
    pub fn encoder_gates(&self) -> &Tensor {
        &self.gates
    }
    pub fn lengths(&self) -> &Tensor {
        &self.lengths
    }
    pub fn status(&self) -> &Tensor {
        &self.status
    }
    pub fn feature(&self) -> &Tensor {
        &self.feature
    }
}
pub struct DeviceZeroExcessPlan {
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
        .ok_or("zero-excess tensor extent overflow")?;
    if zero {
        Tensor::f_zeros(shape, (kind, device))
    } else {
        Tensor::f_empty(shape, (kind, device))
    }
    .map_err(|e| e.to_string())
}
impl DeviceZeroExcessPlan {
    pub fn new(dimension: u32, device_index: usize, max_n: u32) -> Result<Self, String> {
        crate::torch_device::require_cuda_device(device_index)?;
        if !(2..=1 << 24).contains(&dimension) || max_n < 2 {
            return Err("zero-excess width exceeds device/configuration contract".into());
        }
        Ok(Self {
            dimension,
            device_index,
            max_n,
        })
    }
    pub fn evaluate(&self, states: &DeviceStateBatch) -> Result<DeviceZeroExcessBatch, String> {
        let device = Device::Cuda(self.device_index);
        if states.dimension() != self.dimension || states.tensor().device() != device {
            return Err("zero-excess states and plan differ in width/device".into());
        }
        let label = states
            .alphabet_id()
            .ok_or("zero-excess requires an explicit state alphabet identity")?;
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
        let output = DeviceZeroExcessBatch {
            headers: allocate(&[b, 3], Kind::Int64, device, zero)?,
            parents: allocate(&[b, n, 2], Kind::Int, device, zero)?,
            changed: allocate(&[b, n], Kind::Int, device, zero)?,
            missing: allocate(&[b, n], Kind::Int, device, zero)?,
            order: allocate(&[b, n], Kind::Int, device, zero)?,
            gates: allocate(&[b, n, 2], Kind::Int, device, zero)?,
            lengths: allocate(&[b], Kind::Int64, device, zero)?,
            status: allocate(&[b], Kind::Uint8, device, false)?,
            feature: allocate(&[b, 3], Kind::Float, device, zero)?,
        };
        if let Some(code) = unavailable {
            return Ok(DeviceZeroExcessBatch {
                status: output.status.full_like(code),
                ..output
            });
        }
        let scratch = allocate(&states.tensor().size(), Kind::Int64, device, false)?;
        let graph = allocate(&[b, 3 * n], Kind::Int, device, false)?;
        let verified = allocate(&[b], Kind::Uint8, device, false)?;
        let args = ZeroExcessArgs {
            states: states.tensor().data_ptr().cast(),
            scratch: scratch.data_ptr().cast(),
            graph: graph.data_ptr().cast(),
            parents: output.parents.data_ptr().cast(),
            changed_rows: output.changed.data_ptr().cast(),
            missing_rows: output.missing.data_ptr().cast(),
            update_order: output.order.data_ptr().cast(),
            gates: output.gates.data_ptr().cast(),
            headers: output.headers.data_ptr().cast(),
            lengths: output.lengths.data_ptr().cast(),
            verified: verified.data_ptr().cast(),
            status: output.status.data_ptr().cast(),
            feature: output.feature.data_ptr().cast(),
            stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
            state_words: states.tensor().numel() as u64,
            scratch_words: scratch.numel() as u64,
            graph_elements: graph.numel() as u64,
            parent_elements: output.parents.numel() as u64,
            changed_elements: output.changed.numel() as u64,
            missing_elements: output.missing.numel() as u64,
            order_elements: output.order.numel() as u64,
            gate_elements: output.gates.numel() as u64,
            header_elements: output.headers.numel() as u64,
            feature_elements: output.feature.numel() as u64,
            batch_elements: b as u64,
            batch: u32::try_from(b).map_err(|_| "zero-excess batch exceeds CUDA range")?,
            dimension: self.dimension,
            words_per_row: self.dimension.div_ceil(64),
        };
        unsafe {
            crate::cuda::launch_zero_excess(&args)?;
        }
        Ok(output)
    }
}
