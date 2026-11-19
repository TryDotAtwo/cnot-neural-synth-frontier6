//! Exact all-word finite-horizon I10 dynamic programming on CUDA. Host work
//! checks metadata and allocates tables; no graph counts or state readbacks.
use crate::cuda::{EndpointArgs, EndpointPrepareArgs};
use crate::device_state::DeviceStateBatch;
use crate::model::DeviceFeatureStatus;
use crate::I10Config;
use tch::{Device, Kind, Tensor};

pub struct DeviceEndpointResponse {
    values: Tensor,
    status: Tensor,
    oracle_status: Tensor,
    leading_degrees: Tensor,
    leading_counts: Tensor,
    terminal_counts: Tensor,
    evaluations: Tensor,
}
impl DeviceEndpointResponse {
    pub fn values(&self) -> &Tensor {
        &self.values
    }
    pub fn status(&self) -> &Tensor {
        &self.status
    }
    /// DeviceFeatureStatus::Computed means the entire oracle horizon completed,
    /// even if the target is absent and the feature status is Unsupported.
    pub fn oracle_status(&self) -> &Tensor {
        &self.oracle_status
    }
    pub fn leading_degrees(&self) -> &Tensor {
        &self.leading_degrees
    }
    /// [batch,2] int64 bit patterns for unsigned low/high u128 limbs.
    pub fn leading_counts(&self) -> &Tensor {
        &self.leading_counts
    }
    /// Exact coefficient at max_degree on accepted runs, with the same limbs.
    pub fn terminal_counts(&self) -> &Tensor {
        &self.terminal_counts
    }
    /// Exact number of transitions on completed runs. Failed runs return zero;
    /// their source HashMap overflow traversal is unordered and is not claimed.
    pub fn evaluations(&self) -> &Tensor {
        &self.evaluations
    }
}
struct Prepared {
    storage: Tensor,
    status: Tensor,
    capacity: u64,
    buckets: u64,
    history_buckets: u64,
}
pub struct DeviceEndpointResponsePlan {
    dimension: u32,
    device_index: usize,
    config: I10Config,
    prepared: Option<Prepared>,
}
fn allocate(shape: &[i64], kind: Kind, device: Device, zero: bool) -> Result<Tensor, String> {
    shape
        .iter()
        .try_fold(
            1i64,
            |size, &v| if v < 0 { None } else { size.checked_mul(v) },
        )
        .and_then(|size| {
            size.checked_mul(match kind {
                Kind::Int64 => 8,
                Kind::Uint8 => 1,
                _ => 4,
            })
        })
        .ok_or("I10 tensor byte extent overflow")?;
    if zero {
        Tensor::f_zeros(shape, (kind, device))
    } else {
        Tensor::f_empty(shape, (kind, device))
    }
    .map_err(|e| e.to_string())
}
impl DeviceEndpointResponsePlan {
    pub fn new(dimension: u32, device_index: usize, config: I10Config) -> Result<Self, String> {
        crate::register_tiny_i10(&mut crate::FeatureRegistry::new(), config).map_err(|e| e.0)?;
        crate::torch_device::require_cuda_device(device_index)?;
        if dimension == 0 || dimension > (1 << 24) {
            return Err("I10 state width is outside the device matrix contract".into());
        }
        let prepared = if dimension < 2 || dimension > config.max_n {
            None
        } else {
            // The source ignores layer/transition caps at degree0. At positive
            // degrees an accepted layer cannot exceed the total transition
            // budget or the number of binary n*n matrices. No numerical DP is
            // performed to derive these static allocation bounds.
            let mut capacity = if config.max_degree == 0 {
                1
            } else {
                (config.max_layer_states as u64)
                    .min(config.max_transitions as u64)
                    .max(1)
            };
            let bits = u64::from(dimension) * u64::from(dimension);
            if bits < 64 {
                capacity = capacity.min(1u64 << bits);
            }
            let extents = (|| {
                let buckets = capacity
                    .checked_mul(2)?
                    .checked_add(1)?
                    .checked_next_power_of_two()?;
                let history_buckets = capacity
                    .checked_mul(4)?
                    .checked_add(1)?
                    .checked_next_power_of_two()?;
                let words = u64::from(dimension).checked_mul(u64::from(dimension.div_ceil(64)))?;
                let count = capacity
                    .checked_mul(words)?
                    .checked_mul(4)?
                    .checked_add(capacity.checked_mul(10)?)?
                    .checked_add(buckets.checked_mul(2)?)?
                    .checked_add(history_buckets)?
                    .checked_add(words.checked_mul(128)?)?
                    .checked_add(387)?;
                Some((buckets, history_buckets, i64::try_from(count).ok()?))
            })()
            .ok_or("I10 exact table allocation extent overflow")?;
            let device = Device::Cuda(device_index);
            let p = Prepared {
                storage: allocate(&[extents.2], Kind::Int64, device, false)?,
                status: allocate(&[1], Kind::Uint8, device, false)?,
                capacity,
                buckets: extents.0,
                history_buckets: extents.1,
            };
            let args = EndpointPrepareArgs {
                storage: p.storage.data_ptr().cast(),
                status: p.status.data_ptr().cast(),
                stream: crate::cuda::current_stream(device_index).map_err(str::to_owned)?,
                storage_words: p.storage.numel() as u64,
                capacity,
                buckets: p.buckets,
                history_buckets: p.history_buckets,
                max_degree: config.max_degree as u64,
                max_layer_states: config.max_layer_states as u64,
                max_transitions: config.max_transitions as u64,
                dimension,
                words_per_row: dimension.div_ceil(64),
            };
            unsafe {
                crate::cuda::prepare_endpoint_response(&args)?;
            }
            Some(p)
        };
        Ok(Self {
            dimension,
            device_index,
            config,
            prepared,
        })
    }
    pub fn config(&self) -> I10Config {
        self.config
    }
    pub fn evaluate(&self, states: &DeviceStateBatch) -> Result<DeviceEndpointResponse, String> {
        let device = Device::Cuda(self.device_index);
        if states.dimension() != self.dimension || states.tensor().device() != device {
            return Err("I10 plan and states have different widths/devices".into());
        }
        let alphabet = states
            .alphabet_id()
            .ok_or("I10 requires an explicit state alphabet identity")?;
        let unavailable = self.prepared.is_none()
            || alphabet != format!("all_to_all_unit_cnot:{}", self.dimension);
        let b = states.batch_size();
        let output = DeviceEndpointResponse {
            values: allocate(&[b, 4], Kind::Float, device, unavailable)?,
            status: allocate(&[b], Kind::Uint8, device, false)?,
            oracle_status: allocate(&[b], Kind::Uint8, device, false)?,
            leading_degrees: allocate(&[b], Kind::Int64, device, unavailable)?,
            leading_counts: allocate(&[b, 2], Kind::Int64, device, unavailable)?,
            terminal_counts: allocate(&[b, 2], Kind::Int64, device, unavailable)?,
            evaluations: allocate(&[b], Kind::Int64, device, unavailable)?,
        };
        if unavailable {
            return Ok(DeviceEndpointResponse {
                status: output
                    .status
                    .full_like(DeviceFeatureStatus::Unsupported as i64),
                oracle_status: output
                    .oracle_status
                    .full_like(DeviceFeatureStatus::Unsupported as i64),
                ..output
            });
        }
        let p = self.prepared.as_ref().expect("admitted I10");
        let args = EndpointArgs {
            states: states.tensor().data_ptr().cast(),
            storage: p.storage.data_ptr().cast(),
            prepared_status: p.status.data_ptr().cast(),
            values: output.values.data_ptr().cast(),
            feature_status: output.status.data_ptr().cast(),
            oracle_status: output.oracle_status.data_ptr().cast(),
            leading_degrees: output.leading_degrees.data_ptr().cast(),
            leading_counts: output.leading_counts.data_ptr().cast(),
            terminal_counts: output.terminal_counts.data_ptr().cast(),
            evaluations: output.evaluations.data_ptr().cast(),
            stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
            state_words: states.tensor().numel() as u64,
            storage_words: p.storage.numel() as u64,
            value_elements: output.values.numel() as u64,
            count_elements: output.leading_counts.numel() as u64,
            batch_elements: b as u64,
            capacity: p.capacity,
            buckets: p.buckets,
            history_buckets: p.history_buckets,
            max_degree: self.config.max_degree as u64,
            batch: u32::try_from(b).map_err(|_| "I10 batch exceeds CUDA range")?,
            dimension: self.dimension,
            words_per_row: self.dimension.div_ceil(64),
        };
        unsafe {
            crate::cuda::launch_endpoint_response(&args)?;
        }
        Ok(output)
    }
}
