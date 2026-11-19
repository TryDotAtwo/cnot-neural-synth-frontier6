//! Source-defined finite-template responses on resident CUDA states.
//! Setup packs static choices/probabilities and checks source budgets. Every
//! coefficient, assignment, rational moment and conditional choice stays onGPU.
use crate::cuda::TemplateResponseArgs;
use crate::device_state::DeviceStateBatch;
use crate::template_response::{source_evaluation_estimate, template_response_workspace_bytes};
use crate::{TemplateChoiceKind, TemplateResponseConfig};
use tch::{Device, Kind, Tensor};

pub struct DeviceTemplateResponseBatch {
    globals: Tensor,
    mixed: Tensor,
    valid: Tensor,
    incidence: Tensor,
    status: Tensor,
    costs: Tensor,
    rounding: Tensor,
    points: i64,
}
impl DeviceTemplateResponseBatch {
    pub fn feature(&self, ordinal: u8) -> Option<Tensor> {
        let (offset, channels) = match ordinal {
            2 => (0, 1),
            3 => (1, self.points),
            6 => return Some(self.mixed.shallow_clone()),
            7 => (self.points + 1, 1),
            8 => (self.points + 2, 2),
            9 => (self.points + 4, 5),
            _ => return None,
        };
        Some(self.globals.narrow(1, offset, channels))
    }
    pub fn status(&self) -> &Tensor {
        &self.status
    }
    pub fn token_valid(&self) -> &Tensor {
        &self.valid
    }
    pub fn incidence(&self) -> &Tensor {
        &self.incidence
    }
    /// [B,2^k] exact int64 physical costs in ascending assignment-mask order.
    pub fn assignment_costs(&self) -> &Tensor {
        &self.costs
    }
    /// [B,P+2] assignment masks: midpoint forward/reverse, then each probe.
    /// Every payload is zero for any unavailable row, including late overflow.
    pub fn rounding_assignments(&self) -> &Tensor {
        &self.rounding
    }
}
struct Prepared {
    choices: Tensor,
    wires: Tensor,
    probabilities: Tensor,
    invalid: bool,
}
pub struct DeviceTemplateResponsePlan {
    dimension: u32,
    device_index: usize,
    choices: u32,
    points: u32,
    unavailable: Option<u8>,
    prepared: Option<Prepared>,
}
fn allocate(shape: &[i64], kind: Kind, device: Device, zero: bool) -> Result<Tensor, String> {
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
        .ok_or("template-response tensor byte extent overflow")?;
    if zero {
        Tensor::f_zeros(shape, (kind, device))
    } else {
        Tensor::f_empty(shape, (kind, device))
    }
    .map_err(|e| e.to_string())
}
impl DeviceTemplateResponsePlan {
    pub fn new(
        dimension: u32,
        device_index: usize,
        config: &TemplateResponseConfig,
        max_bytes: Option<usize>,
    ) -> Result<Self, String> {
        crate::torch_device::require_cuda_device(device_index)?;
        if dimension < 2 || dimension > (1 << 24) {
            return Err("template-response width exceeds device state contract".into());
        }
        let k = config.choices().len();
        let points = u32::try_from(config.points().len())
            .map_err(|_| "template-response probe count exceeds CUDA range")?;
        let budgets = config.budgets();
        let width = dimension as usize / 2;
        let bytes = template_response_workspace_bytes(dimension, k);
        let evaluations = source_evaluation_estimate(k, config.points().len(), 1usize << k);
        let limited = bytes
            .is_none_or(|n| n > budgets.max_bytes || max_bytes.is_some_and(|limit| n > limit))
            || width > budgets.max_width
            || (1usize << k) > budgets.max_assignments
            || evaluations.is_none_or(|n| n > budgets.max_evaluations);
        let unavailable = if limited { Some(4) } else { None };
        let device = Device::Cuda(device_index);
        let prepared = if unavailable.is_none() {
            let invalid = config
                .choices()
                .iter()
                .any(|c| c.control >= width || c.target >= width);
            // Invalid static indices are interpreted only after a nonempty
            // partial shear exists. Upload safe placeholders for that branch.
            let mut wires = if invalid {
                vec![0usize]
            } else {
                config
                    .choices()
                    .iter()
                    .flat_map(|c| [c.control, c.target])
                    .collect::<Vec<_>>()
            };
            wires.sort_unstable();
            wires.dedup();
            let choices = config
                .choices()
                .iter()
                .flat_map(|c| {
                    [
                        i32::from(c.kind == TemplateChoiceKind::Swap),
                        if invalid {
                            0
                        } else {
                            wires.binary_search(&c.control).expect("packed wire") as i32
                        },
                        if invalid {
                            0
                        } else {
                            wires.binary_search(&c.target).expect("packed wire") as i32
                        },
                    ]
                })
                .collect::<Vec<_>>();
            let wires = wires.iter().map(|&w| w as i32).collect::<Vec<_>>();
            let probabilities = config
                .points()
                .iter()
                .flatten()
                .flat_map(|p| [p.numerator() as i64, p.denominator() as i64])
                .collect::<Vec<_>>();
            Some(Prepared {
                choices: Tensor::f_from_slice(&choices)
                    .and_then(|t| t.f_to_device(device))
                    .map_err(|e| e.to_string())?,
                wires: Tensor::f_from_slice(&wires)
                    .and_then(|t| t.f_to_device(device))
                    .map_err(|e| e.to_string())?,
                probabilities: Tensor::f_from_slice(&probabilities)
                    .and_then(|t| t.f_to_device(device))
                    .map_err(|e| e.to_string())?,
                invalid,
            })
        } else {
            None
        };
        Ok(Self {
            dimension,
            device_index,
            choices: k as u32,
            points,
            unavailable,
            prepared,
        })
    }
    pub fn evaluate(
        &self,
        states: &DeviceStateBatch,
    ) -> Result<DeviceTemplateResponseBatch, String> {
        let device = Device::Cuda(self.device_index);
        if states.dimension() != self.dimension || states.tensor().device() != device {
            return Err("template-response states and plan widths/devices differ".into());
        }
        let b = states.batch_size();
        let n = i64::from(self.dimension);
        let w = n / 2;
        let points = i64::from(self.points);
        let k = i64::from(self.choices);
        let groups = k * (k - 1) / 2;
        let zero = self.unavailable.is_some();
        let output = DeviceTemplateResponseBatch {
            globals: allocate(&[b, points + 9], Kind::Float, device, zero)?,
            mixed: allocate(&[b, groups, 1], Kind::Float, device, zero)?,
            valid: allocate(&[b, groups], Kind::Uint8, device, zero)?,
            incidence: allocate(&[b, groups, n, 4], Kind::Float, device, zero)?,
            status: allocate(&[b], Kind::Uint8, device, false)?,
            costs: allocate(&[b, 1i64 << self.choices], Kind::Int64, device, zero)?,
            rounding: allocate(&[b, points + 2], Kind::Int64, device, zero)?,
            points,
        };
        if let Some(code) = self.unavailable {
            return Ok(DeviceTemplateResponseBatch {
                status: output.status.full_like(i64::from(code)),
                ..output
            });
        }
        let prepared = self.prepared.as_ref().expect("admitted template metadata");
        let coefficients = allocate(&[b, n - w, w], Kind::Uint8, device, false)?;
        let raw_status = allocate(&[b], Kind::Uint8, device, false)?;
        let scratch = allocate(
            &[
                b,
                w,
                i64::from(self.dimension.div_ceil(64)) + (w + 63) / 64 + 1,
            ],
            Kind::Int64,
            device,
            false,
        )?;
        let batch = u32::try_from(b).map_err(|_| "template-response batch exceeds CUDA range")?;
        let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
        unsafe {
            crate::cuda::launch_basis_gradient(
                states.tensor().data_ptr().cast(),
                scratch.data_ptr().cast(),
                coefficients.data_ptr().cast(),
                raw_status.data_ptr().cast(),
                std::ptr::null_mut(),
                batch,
                self.dimension,
                self.dimension.div_ceil(64),
                scratch.numel() as u64,
                coefficients.numel() as u64,
                0,
                false,
                stream,
            )?;
        }
        let args = TemplateResponseArgs {
            coefficients: coefficients.data_ptr().cast(),
            raw_status: raw_status.data_ptr().cast(),
            choice_data: prepared.choices.data_ptr().cast(),
            wires: prepared.wires.data_ptr().cast(),
            probabilities: prepared.probabilities.data_ptr().cast(),
            costs: output.costs.data_ptr().cast(),
            rounding: output.rounding.data_ptr().cast(),
            globals: output.globals.data_ptr().cast(),
            mixed: output.mixed.data_ptr().cast(),
            valid: output.valid.data_ptr().cast(),
            incidence: output.incidence.data_ptr().cast(),
            status: output.status.data_ptr().cast(),
            stream,
            coefficient_elements: coefficients.numel() as u64,
            choice_elements: prepared.choices.numel() as u64,
            wire_elements: prepared.wires.numel() as u64,
            probability_elements: prepared.probabilities.numel() as u64,
            cost_elements: output.costs.numel() as u64,
            rounding_elements: output.rounding.numel() as u64,
            global_elements: output.globals.numel() as u64,
            mixed_elements: output.mixed.numel() as u64,
            token_elements: output.valid.numel() as u64,
            incidence_elements: output.incidence.numel() as u64,
            batch_elements: b as u64,
            batch,
            dimension: self.dimension,
            choices: self.choices,
            points: self.points,
            unique_wires: prepared.wires.numel() as u32,
            invalid_choices: u32::from(prepared.invalid),
        };
        unsafe {
            crate::cuda::launch_template_response(&args)?;
        }
        Ok(output)
    }
}
