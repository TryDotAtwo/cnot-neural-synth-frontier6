use crate::cuda::Stage26Args;
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::FeatureSpec;
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct Stage26PlanInputs {
    pub max_n: u32,
    pub max_checker_pairs: usize,
}
impl Default for Stage26PlanInputs {
    fn default() -> Self {
        Self {
            max_n: crate::stage26::DEFAULT_MAX_N,
            max_checker_pairs: crate::stage26::DEFAULT_MAX_CHECKER_PAIRS,
        }
    }
}
pub(super) struct Stage26Plan {
    dimension: u32,
    device_index: usize,
    limited: bool,
}
fn allocate(shape: &[i64], kind: Kind, device: Device, zeros: bool) -> Result<Tensor, String> {
    let count = shape
        .iter()
        .try_fold(
            1i64,
            |count, &n| {
                if n < 0 {
                    None
                } else {
                    count.checked_mul(n)
                }
            },
        )
        .ok_or("Stage26 tensor extent overflow")?;
    count
        .checked_mul(match kind {
            Kind::Int64 => 8,
            Kind::Uint8 => 1,
            _ => 4,
        })
        .ok_or("Stage26 tensor byte extent overflow")?;
    if zeros {
        Tensor::f_zeros(shape, (kind, device))
    } else {
        Tensor::f_empty(shape, (kind, device))
    }
    .map_err(|e| e.to_string())
}
impl Stage26Plan {
    pub(super) fn new(
        dimension: u32,
        device_index: usize,
        input: Stage26PlanInputs,
    ) -> Result<Self, String> {
        if input.max_n == 0 || input.max_checker_pairs == 0 {
            return Err("Stage26 requires positive width/checker-pair budgets".into());
        }
        let pool = 2 * u64::from(dimension);
        let checker_pairs = pool
            .checked_mul(pool.saturating_sub(1))
            .ok_or("Stage26 checker-pair extent overflow")?
            / 2;
        Ok(Self {
            dimension,
            device_index,
            limited: dimension > input.max_n || checker_pairs > input.max_checker_pairs as u64,
        })
    }
    pub(super) fn append_entries(
        &self,
        states: &DeviceStateBatch,
        specs: &[FeatureSpec],
        entries: &mut Vec<DeviceFeatureEntry>,
    ) -> Result<(), String> {
        let device = Device::Cuda(self.device_index);
        if states.dimension() != self.dimension || states.tensor().device() != device {
            return Err("Stage26 plan and states have different widths/devices".into());
        }
        let alphabet = states
            .alphabet_id()
            .ok_or("Stage26 requires an explicit state alphabet identity")?;
        let unavailable = if alphabet != format!("all_to_all_unit_cnot:{}", self.dimension) {
            Some(DeviceFeatureStatus::Unsupported)
        } else if self.limited {
            Some(DeviceFeatureStatus::ResourceLimit)
        } else {
            None
        };
        let n = i64::from(self.dimension);
        let w = i64::from(self.dimension.div_ceil(64));
        let b = states.batch_size();
        let f03 = allocate(&[b, 5], Kind::Float, device, unavailable.is_some())?;
        let f04 = allocate(&[b, n, n, 3], Kind::Float, device, unavailable.is_some())?;
        let f05 = allocate(&[b, 2], Kind::Float, device, unavailable.is_some())?;
        let mut status = allocate(&[b], Kind::Uint8, device, false)?;
        if let Some(code) = unavailable {
            status = status.full_like(code as i64);
        } else {
            // The fixed layout matches Cnot103Stage26Args and is checked again
            // by its C ABI. Only shapes/static metadata are examined on host.
            let certificate = allocate(&[b, 6 + n + (4 * n + 2) * w], Kind::Int64, device, false)?;
            let scratch = allocate(&[b, (7 * n + 4) * w + n + 1], Kind::Int64, device, false)?;
            let verified = allocate(&[b], Kind::Uint8, device, false)?;
            let args = Stage26Args {
                states: states.tensor().data_ptr().cast(),
                certificate: certificate.data_ptr().cast(),
                scratch: scratch.data_ptr().cast(),
                verified: verified.data_ptr().cast(),
                status: status.data_ptr().cast(),
                f03: f03.data_ptr().cast(),
                f04: f04.data_ptr().cast(),
                f05: f05.data_ptr().cast(),
                stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
                certificate_words: certificate.numel() as u64,
                workspace_words: scratch.numel() as u64,
                pair_elements: f04.numel() as u64,
                batch: u32::try_from(b).map_err(|_| "Stage26 batch exceeds CUDA range")?,
                dimension: self.dimension,
                words_per_row: self.dimension.div_ceil(64),
            };
            // Torch's current-stream allocator keeps enqueued scratch accesses
            // valid after handles drop. No values or proof flags reach the CPU.
            unsafe {
                crate::cuda::launch_stage26(&args)?;
            }
        }
        for spec in specs
            .iter()
            .filter(|s| s.id.family() == 'F' && matches!(s.id.ordinal(), 3..=5))
        {
            let value = match spec.id.ordinal() {
                3 => &f03,
                4 => &f04,
                _ => &f05,
            };
            entries.push(DeviceFeatureEntry::new(
                spec.clone(),
                value.shallow_clone(),
                status.shallow_clone(),
                None,
                None,
            ));
        }
        Ok(())
    }
}
