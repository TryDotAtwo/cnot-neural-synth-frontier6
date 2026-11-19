//! Exact local paid-basis primitive. Atlas BFS, histogram, stable argmin and
//! reconstructed operations stay on CUDA. This alone does not admit D09-D14.
use crate::cuda::{PaidAtlasArgs, PaidAtlasPrepareArgs};
use crate::BasisSide;
use tch::{Device, Kind, Tensor};

pub struct DeviceExactBasisBatch {
    operations: Tensor,
    lengths: Tensor,
    direct_cost: Tensor,
    paid_cost: Tensor,
    transformed: Tensor,
    status: Tensor,
}
impl DeviceExactBasisBatch {
    pub fn operations(&self) -> &Tensor {
        &self.operations
    }
    pub fn lengths(&self) -> &Tensor {
        &self.lengths
    }
    pub fn direct_cost(&self) -> &Tensor {
        &self.direct_cost
    }
    pub fn paid_cost(&self) -> &Tensor {
        &self.paid_cost
    }
    pub fn transformed(&self) -> &Tensor {
        &self.transformed
    }
    /// 1=exact,3=invalid binary vector; invalid samples have zero payloads.
    pub fn status(&self) -> &Tensor {
        &self.status
    }
}
pub struct DevicePaidBasisAtlas {
    width: usize,
    side: BasisSide,
    device_index: usize,
    storage: Tensor,
}
fn allocate(shape: &[i64], kind: Kind, device: Device) -> Result<Tensor, String> {
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
        .ok_or("paid-basis tensor byte extent overflow")?;
    Tensor::f_empty(shape, (kind, device)).map_err(|e| e.to_string())
}
impl DevicePaidBasisAtlas {
    pub(crate) fn storage(&self) -> &Tensor {
        &self.storage
    }
    pub(crate) fn shallow_clone(&self) -> Self {
        Self {
            width: self.width,
            side: self.side,
            device_index: self.device_index,
            storage: self.storage.shallow_clone(),
        }
    }
    pub fn new(width: usize, side: BasisSide, device_index: usize) -> Result<Self, String> {
        if !(1..=4).contains(&width) {
            return Err("exact paid-basis atlas supports width1..4".into());
        }
        crate::torch_device::require_cuda_device(device_index)?;
        let words = (1i64 << (width * width)) * (6 + (1i64 << width)) + 1;
        let storage = allocate(&[words], Kind::Int, Device::Cuda(device_index))?;
        let args = PaidAtlasPrepareArgs {
            storage: storage.data_ptr().cast(),
            stream: crate::cuda::current_stream(device_index).map_err(str::to_owned)?,
            storage_words: words as u64,
            width: width as u32,
            side: if side == BasisSide::Source { 0 } else { 1 },
        };
        unsafe {
            crate::cuda::prepare_paid_atlas(&args)?;
        }
        Ok(Self {
            width,
            side,
            device_index,
            storage,
        })
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn side(&self) -> BasisSide {
        self.side
    }
    /// Independent device copy of the scalar, with no host numerical query.
    /// A mutable tensor view must not expose the internal traversal count.
    pub fn state_count(&self) -> Tensor {
        self.storage.narrow(0, self.storage.size()[0] - 1, 1).copy()
    }
    pub fn optimize(&self, vectors: &Tensor) -> Result<DeviceExactBasisBatch, String> {
        let shape = vectors.size();
        if shape.len() != 3
            || shape[0] <= 0
            || shape[1] <= 0
            || shape[2] != self.width as i64
            || vectors.kind() != Kind::Uint8
            || vectors.device() != Device::Cuda(self.device_index)
            || !vectors.is_contiguous()
        {
            return Err(
                "paid-basis vectors require contiguous CUDA uint8 [batch,positive_rows,width]"
                    .into(),
            );
        }
        let batch = u32::try_from(shape[0]).map_err(|_| "paid-basis batch exceeds CUDA range")?;
        let capacity = ((self.width + 3) * (self.width - 1)).max(1) as i64;
        let device = vectors.device();
        let output = DeviceExactBasisBatch {
            operations: allocate(&[shape[0], capacity, 2], Kind::Int, device)?,
            lengths: allocate(&[shape[0]], Kind::Int64, device)?,
            direct_cost: allocate(&[shape[0]], Kind::Int64, device)?,
            paid_cost: allocate(&[shape[0]], Kind::Int64, device)?,
            transformed: allocate(&shape, Kind::Uint8, device)?,
            status: allocate(&[shape[0]], Kind::Uint8, device)?,
        };
        let args = PaidAtlasArgs {
            storage: self.storage.data_ptr().cast(),
            vectors: vectors.data_ptr().cast(),
            transformed: output.transformed.data_ptr().cast(),
            operations: output.operations.data_ptr().cast(),
            lengths: output.lengths.data_ptr().cast(),
            direct_cost: output.direct_cost.data_ptr().cast(),
            paid_cost: output.paid_cost.data_ptr().cast(),
            status: output.status.data_ptr().cast(),
            stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
            storage_words: self.storage.numel() as u64,
            vector_elements: vectors.numel() as u64,
            gate_elements: output.operations.numel() as u64,
            batch_elements: shape[0] as u64,
            word_capacity: capacity as u64,
            rows: shape[1] as u64,
            batch,
            width: self.width as u32,
            side: if self.side == BasisSide::Source { 0 } else { 1 },
        };
        unsafe {
            crate::cuda::optimize_paid_atlas(&args)?;
        }
        Ok(output)
    }
}
