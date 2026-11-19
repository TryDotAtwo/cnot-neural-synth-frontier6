use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::{FeatureSpec, MatrixBoundarySpec};
use tch::{Device, Kind, Tensor};

/// Immutable physical port data. Construction runs once, before the GPU loop.
pub(super) struct BoundaryPlan {
    dimension: u32,
    device_index: usize,
    input_count: u32,
    output_count: u32,
    equations: u32,
    port_wires: Tensor,
    protected: Tensor,
    equation_outputs: Tensor,
    port_values: Tensor,
    port_incidence: Tensor,
    weighted_unpinned: bool,
}

impl BoundaryPlan {
    pub(super) fn new(spec: &MatrixBoundarySpec, device_index: usize) -> Result<Self, String> {
        let input_count =
            u32::try_from(spec.inputs().len()).map_err(|_| "boundary input count overflow")?;
        let output_count =
            u32::try_from(spec.outputs().len()).map_err(|_| "boundary output count overflow")?;
        let ports = u64::from(input_count) + u64::from(output_count);
        let equations = output_count
            .checked_add(
                u32::try_from(spec.zero_outputs().len())
                    .map_err(|_| "boundary zero-output count overflow")?,
            )
            .ok_or("boundary equation count overflow")?;
        if ports > (1 << 24) || equations > (1 << 24) {
            return Err("boundary descriptors exceed exact f32 integer capacity".into());
        }
        let device = Device::Cuda(device_index);
        let physical: Vec<i32> = spec
            .inputs()
            .iter()
            .chain(spec.outputs())
            .map(|wire| i32::try_from(*wire).map_err(|_| "boundary wire exceeds CUDA index range"))
            .collect::<Result<_, _>>()?;
        let mut protected = vec![0u8; ports as usize];
        let mut port_values = vec![0f32; ports as usize * 2];
        let incidence_count = (ports as usize)
            .checked_mul(spec.dimension() as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or("boundary port incidence size overflow")?;
        let mut incidence = vec![0f32; incidence_count];
        for (port, &wire) in physical.iter().enumerate() {
            let input = port < input_count as usize;
            let is_protected = input
                && spec
                    .protected_inputs()
                    .binary_search(&(wire as u32))
                    .is_ok();
            protected[port] = u8::from(is_protected);
            port_values[port * 2] = f32::from(!input);
            port_values[port * 2 + 1] = f32::from(is_protected);
            let offset = (port * spec.dimension() as usize + wire as usize) * 3;
            incidence[offset + usize::from(!input)] = 1.0;
            if is_protected {
                incidence[offset + 2] = 1.0;
            }
        }
        let equation_outputs: Vec<i32> = (0..output_count as i32)
            .chain(spec.zero_outputs().iter().map(|wire| {
                spec.outputs()
                    .binary_search(wire)
                    .expect("validated zero-output subset") as i32
            }))
            .collect();
        Ok(Self {
            dimension: spec.dimension(),
            device_index,
            input_count,
            output_count,
            equations,
            port_wires: Tensor::from_slice(&physical).to_device(device),
            protected: Tensor::from_slice(&protected).to_device(device),
            equation_outputs: Tensor::from_slice(&equation_outputs).to_device(device),
            port_values: Tensor::from_slice(&port_values)
                .view([1, ports as i64, 2])
                .to_device(device),
            port_incidence: Tensor::from_slice(&incidence)
                .view([1, ports as i64, i64::from(spec.dimension()), 3])
                .to_device(device),
            weighted_unpinned: spec.has_weights() && spec.zero_outputs().is_empty(),
        })
    }

    pub(super) fn append_entries(
        &self,
        states: &DeviceStateBatch,
        specs: &[FeatureSpec],
        computed: &Tensor,
        nullity: &Tensor,
        limited: bool,
        entries: &mut Vec<DeviceFeatureEntry>,
    ) -> Result<(), String> {
        let batch = states.batch_size();
        let groups = i64::from(self.equations);
        let ports = i64::from(self.input_count) + i64::from(self.output_count);
        let device = Device::Cuda(self.device_index);
        if states.dimension() != self.dimension || states.tensor().device() != device {
            return Err("boundary relation context and state batch differ".into());
        }
        let allocate = |shape: &[i64], kind| {
            if limited {
                Tensor::zeros(shape, (kind, device))
            } else {
                Tensor::empty(shape, (kind, device))
            }
        };
        let values = allocate(&[batch, groups, 3], Kind::Float);
        let valid = allocate(&[batch, groups], Kind::Uint8);
        let incidence = allocate(&[batch, groups, i64::from(self.dimension), 3], Kind::Float);
        let ranks = allocate(&[batch, 2], Kind::Float);
        if !limited {
            let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
            // SAFETY: immutable metadata was checked at setup; tensors have the
            // exact ABI shapes, own disjoint live storage and use Torch's stream.
            unsafe {
                crate::cuda::launch_boundary_relations(
                    states.tensor().data_ptr().cast::<u64>(),
                    self.port_wires.data_ptr().cast::<u32>(),
                    self.protected.data_ptr().cast::<u8>(),
                    self.equation_outputs.data_ptr().cast::<u32>(),
                    u32::try_from(batch).map_err(|_| "boundary batch exceeds kernel range")?,
                    self.dimension,
                    self.dimension.div_ceil(64),
                    self.input_count,
                    self.output_count,
                    self.equations,
                    values.data_ptr().cast::<f32>(),
                    valid.data_ptr().cast::<u8>(),
                    incidence.data_ptr().cast::<f32>(),
                    ranks.data_ptr().cast::<f32>(),
                    valid.numel() as u64,
                    incidence.numel() as u64,
                    stream,
                )
            }
            .map_err(str::to_owned)?;
        }
        let status = if limited {
            computed.full_like(DeviceFeatureStatus::ResourceLimit as i64)
        } else {
            computed.shallow_clone()
        };
        let visible_rank = nullity.full_like(f64::from(self.input_count)) - nullity;
        let protected_rank = ranks.select(1, 1);
        let relation_rank = ranks.select(1, 0);
        for spec in specs.iter().filter(|spec| spec.id.family() == 'I') {
            let entry = match spec.id.ordinal() {
                1 => {
                    let port_values = self.port_values.expand([batch, ports, 2], true);
                    let port_incidence = self
                        .port_incidence
                        .expand([batch, ports, i64::from(self.dimension), 3], true);
                    DeviceFeatureEntry::new(
                        spec.clone(),
                        if limited {
                            Tensor::zeros_like(&port_values)
                        } else {
                            port_values
                        },
                        status.shallow_clone(),
                        Some(Tensor::full(
                            [batch, ports],
                            i64::from(!limited),
                            (Kind::Uint8, device),
                        )),
                        Some(if limited {
                            Tensor::zeros_like(&port_incidence)
                        } else {
                            port_incidence
                        }),
                    )
                }
                2 => DeviceFeatureEntry::new(
                    spec.clone(),
                    values.shallow_clone(),
                    status.shallow_clone(),
                    Some(valid.shallow_clone()),
                    Some(incidence.shallow_clone()),
                ),
                3 => DeviceFeatureEntry::new(
                    spec.clone(),
                    Tensor::stack(
                        &[&protected_rank.gt(0).to_kind(Kind::Float), &protected_rank],
                        1,
                    ),
                    status.shallow_clone(),
                    None,
                    None,
                ),
                4 => DeviceFeatureEntry::new(
                    spec.clone(),
                    if limited || !self.weighted_unpinned {
                        Tensor::zeros([batch, 3], (Kind::Float, device))
                    } else {
                        Tensor::cat(&[&visible_rank, &visible_rank, nullity], 1)
                    },
                    if !limited && !self.weighted_unpinned {
                        computed.full_like(DeviceFeatureStatus::Unsupported as i64)
                    } else {
                        status.shallow_clone()
                    },
                    None,
                    None,
                ),
                11 => DeviceFeatureEntry::new(
                    spec.clone(),
                    if limited {
                        Tensor::zeros([batch, 4], (Kind::Float, device))
                    } else {
                        Tensor::stack(
                            &[
                                &relation_rank.full_like(ports),
                                &(ports as f64 - &relation_rank),
                                &Tensor::ones_like(&relation_rank),
                                &Tensor::zeros_like(&relation_rank),
                            ],
                            1,
                        )
                    },
                    status.shallow_clone(),
                    None,
                    None,
                ),
                _ => continue,
            };
            entries.push(entry);
        }
        Ok(())
    }
}
