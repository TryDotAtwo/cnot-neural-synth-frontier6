//! Resident Stage60 equations and summaries, with the exact source context
//! and resource guards. Host work is limited to metadata and static uploads.
use crate::cuda::QuadraticArgs;
use crate::device_state::DeviceStateBatch;
use crate::i09_quadratic::{equation_bound, variable_count, working_bytes_upper_bound};
use crate::model::DeviceFeatureStatus;
use crate::{I09Context, I09Limits};
use tch::{Device, Kind, Tensor};

#[derive(Debug)]
pub struct DeviceQuadraticResponse {
    terms: Tensor,
    term_counts: Tensor,
    values: Tensor,
    valid: Tensor,
    incidence: Tensor,
    ranks: Tensor,
    status: Tensor,
}
impl DeviceQuadraticResponse {
    /// [batch,equations,capacity,3] int32 (kind,left,right) canonical monomials.
    /// Only the first term_counts slots are terms; the remaining slots are zero.
    pub fn terms(&self) -> &Tensor {
        &self.terms
    }
    pub fn term_counts(&self) -> &Tensor {
        &self.term_counts
    }
    /// [batch,equations,max_variables+4]: equation,alpha,rank_B,mixed_rank,beta.
    pub fn values(&self) -> &Tensor {
        &self.values
    }
    pub fn token_valid(&self) -> &Tensor {
        &self.valid
    }
    pub fn incidence(&self) -> &Tensor {
        &self.incidence
    }
    pub fn ranks(&self) -> &Tensor {
        &self.ranks
    }
    pub fn status(&self) -> &Tensor {
        &self.status
    }
}
struct PreparedContext {
    gates: Tensor,
    prefix: Tensor,
    horizon: u32,
    cut: u32,
    variables: u32,
    equations: u32,
    allowed_count: u32,
    scratch_per_sample: i64,
}
pub struct DeviceQuadraticPlan {
    dimension: u32,
    device_index: usize,
    limits: I09Limits,
    unavailable: Option<DeviceFeatureStatus>,
    context: Option<PreparedContext>,
}
fn allocate(shape: &[i64], kind: Kind, device: Device) -> Result<Tensor, String> {
    shape
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
        .and_then(|count| {
            count.checked_mul(match kind {
                Kind::Int64 => 8,
                Kind::Uint8 => 1,
                _ => 4,
            })
        })
        .ok_or("I09 tensor extent overflow")?;
    Tensor::f_empty(shape, (kind, device)).map_err(|e| e.to_string())
}
impl DeviceQuadraticPlan {
    /// Contexts include every configured width, as in source registration.
    /// Width mismatch is a typed Unsupported result, not an implicit context.
    pub fn new(
        dimension: u32,
        device_index: usize,
        contexts: &[I09Context],
        limits: I09Limits,
    ) -> Result<Self, String> {
        crate::register_i09_quadratic_response(
            &mut crate::FeatureRegistry::new(),
            contexts.to_vec(),
            limits,
        )
        .map_err(|e| e.0)?;
        crate::torch_device::require_cuda_device(device_index)?;
        if dimension == 0 || dimension > (1 << 24) {
            return Err("I09 width must be within the exact float32 integer range".into());
        }
        let source = contexts.iter().find(|c| c.width() == dimension as usize);
        let mut unavailable = None;
        let context = if let Some(c) = source {
            let n = c.width();
            let admitted = (|| {
                let variables = variable_count(n, c.horizon())?;
                let bound = equation_bound(n, c.horizon())?;
                if variables > limits.max_variables
                    || bound > limits.max_equations
                    || variables > (1 << 24)
                    || bound > (1 << 24)
                    || limits.max_variables + 4 > (1 << 24)
                {
                    return None;
                }
                if working_bytes_upper_bound(n, c.horizon(), variables, bound)? > limits.max_bytes {
                    return None;
                }
                let equations = if c.horizon() == 0 {
                    n * n
                } else {
                    c.horizon() * (3 + 3 * n * n - c.allowed_gates().len()) - 1 + n * n
                };
                let future = variables - c.cut();
                let scratch = future.checked_mul(equations.div_ceil(64))?.checked_add(
                    c.cut()
                        .checked_mul(equations.checked_mul(future)?.div_ceil(64))?,
                )?;
                Some((variables, equations, i64::try_from(scratch).ok()?))
            })();
            if let Some((variables, equations, scratch)) = admitted {
                let device = Device::Cuda(device_index);
                let gates: Vec<i32> = c
                    .allowed_gates()
                    .iter()
                    .flat_map(|&(a, b)| [a as i32, b as i32])
                    .collect();
                Some(PreparedContext {
                    gates: Tensor::from_slice(&gates).to_device(device),
                    prefix: Tensor::from_slice(c.prefix_bits()).to_device(device),
                    horizon: c.horizon() as u32,
                    cut: c.cut() as u32,
                    variables: variables as u32,
                    equations: equations as u32,
                    allowed_count: c.allowed_gates().len() as u32,
                    scratch_per_sample: scratch,
                })
            } else {
                unavailable = Some(DeviceFeatureStatus::ResourceLimit);
                None
            }
        } else {
            unavailable = Some(DeviceFeatureStatus::Unsupported);
            None
        };
        Ok(Self {
            dimension,
            device_index,
            limits,
            unavailable,
            context,
        })
    }
    pub fn variable_count(&self) -> Option<u32> {
        self.context.as_ref().map(|c| c.variables)
    }
    pub fn cut(&self) -> Option<u32> {
        self.context.as_ref().map(|c| c.cut)
    }

    /// All equations, prefix evaluation, ranks and physical support are CUDA
    /// work. No numerical result is read on host, including unavailable status.
    pub fn compile(&self, states: &DeviceStateBatch) -> Result<DeviceQuadraticResponse, String> {
        let device = Device::Cuda(self.device_index);
        if states.dimension() != self.dimension || states.tensor().device() != device {
            return Err("I09 plan and states have different widths/devices".into());
        }
        let alphabet = states
            .alphabet_id()
            .ok_or("I09 requires an explicit state alphabet identity")?;
        // Source alphabet guard precedes every resource check.
        let unavailable = if alphabet != format!("all_to_all_unit_cnot:{}", self.dimension) {
            Some(DeviceFeatureStatus::Unsupported)
        } else {
            self.unavailable
        };
        let b = states.batch_size();
        let g = if unavailable.is_some() {
            0
        } else {
            i64::from(self.context.as_ref().expect("admitted I09").equations)
        };
        let cap = if g == 0 {
            0
        } else if self.context.as_ref().unwrap().horizon == 0 {
            1
        } else {
            i64::from(self.dimension) + 2
        };
        let output = DeviceQuadraticResponse {
            terms: allocate(&[b, g, cap, 3], Kind::Int, device)?,
            term_counts: allocate(&[b, g], Kind::Int, device)?,
            values: allocate(
                &[b, g, self.limits.max_variables as i64 + 4],
                Kind::Float,
                device,
            )?,
            valid: allocate(&[b, g], Kind::Uint8, device)?,
            incidence: allocate(&[b, g, i64::from(self.dimension), 1], Kind::Float, device)?,
            ranks: allocate(&[b, 2], Kind::Int, device)?,
            status: allocate(&[b], Kind::Uint8, device)?,
        };
        if let Some(code) = unavailable {
            return Ok(DeviceQuadraticResponse {
                ranks: output.ranks.zeros_like(),
                status: output.status.full_like(code as i64),
                ..output
            });
        }
        let c = self.context.as_ref().expect("admitted I09");
        let scratch_count = b
            .checked_mul(c.scratch_per_sample)
            .ok_or("I09 scratch extent overflow")?
            .max(1);
        let scratch = allocate(&[scratch_count], Kind::Int64, device)?;
        let args = QuadraticArgs {
            states: states.tensor().data_ptr().cast(),
            allowed_gates: c.gates.data_ptr().cast(),
            prefix: c.prefix.data_ptr().cast(),
            terms: output.terms.data_ptr().cast(),
            term_counts: output.term_counts.data_ptr().cast(),
            scratch: scratch.data_ptr().cast(),
            values: output.values.data_ptr().cast(),
            valid: output.valid.data_ptr().cast(),
            incidence: output.incidence.data_ptr().cast(),
            ranks: output.ranks.data_ptr().cast(),
            status: output.status.data_ptr().cast(),
            stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
            state_words: states.tensor().numel() as u64,
            term_elements: output.terms.numel() as u64,
            group_elements: output.valid.numel() as u64,
            scratch_words: scratch.numel() as u64,
            value_elements: output.values.numel() as u64,
            incidence_elements: output.incidence.numel() as u64,
            rank_elements: output.ranks.numel() as u64,
            batch: u32::try_from(b).map_err(|_| "I09 batch exceeds CUDA range")?,
            dimension: self.dimension,
            words_per_row: self.dimension.div_ceil(64),
            horizon: c.horizon,
            cut: c.cut,
            allowed_count: c.allowed_count,
            equations: c.equations,
            max_variables: self.limits.max_variables as u32,
        };
        // Current Torch stream and stream-aware allocator retain scratch until
        // the queued compiler finishes; output handles retain every payload.
        unsafe {
            crate::cuda::launch_quadratic(&args)?;
        }
        Ok(output)
    }
}
