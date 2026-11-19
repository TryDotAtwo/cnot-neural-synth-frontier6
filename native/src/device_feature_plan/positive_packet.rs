use crate::cuda::PositivePacketArgs;
use crate::device_state::DeviceStateBatch;
use crate::model::{DeviceFeatureEntry, DeviceFeatureStatus};
use crate::{FeatureSpec, OptionalGatePacket};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy)]
pub struct PositivePacketPlanInputs<'a> {
    pub contexts: &'a [OptionalGatePacket],
    pub max_n: usize,
    pub max_slots: usize,
    pub max_monomials: usize,
    pub max_factor_entries: usize,
}
struct PreparedPacket {
    values: Tensor,
    valid: Tensor,
    incidence: Tensor,
    status: Tensor,
}
pub(super) struct PositivePacketPlan {
    dimension: u32,
    device_index: usize,
    pre_alphabet: Option<DeviceFeatureStatus>,
    post_alphabet: Option<DeviceFeatureStatus>,
    prepared: Option<PreparedPacket>,
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
        .ok_or("I08 tensor extent overflow")?;
    Tensor::f_empty(shape, (kind, device)).map_err(|e| e.to_string())
}
impl PositivePacketPlan {
    pub(super) fn new(
        dimension: u32,
        device_index: usize,
        input: PositivePacketPlanInputs<'_>,
        max_bytes: Option<usize>,
    ) -> Result<Self, String> {
        let context = input.contexts.iter().find(|c| c.dimension() == dimension);
        let mut post_alphabet = None;
        let pre_alphabet = if let Some(c) = context {
            if dimension as usize > input.max_n
                || c.gates().len() > input.max_slots
                || crate::positive_packet::factor_entry_count(c.factors())
                    .is_none_or(|count| count > input.max_factor_entries)
            {
                Some(DeviceFeatureStatus::ResourceLimit)
            } else {
                if crate::positive_packet::working_bytes_upper_bound(
                    dimension as usize,
                    c.gates().len(),
                    input.max_monomials,
                )
                .is_none_or(|bytes| {
                    bytes > max_bytes.unwrap_or(crate::positive_packet::DEFAULT_MAX_BYTES)
                }) {
                    post_alphabet = Some(DeviceFeatureStatus::ResourceLimit);
                }
                None
            }
        } else {
            Some(DeviceFeatureStatus::Unsupported)
        };
        let prepared = if pre_alphabet.is_none() && post_alphabet.is_none() {
            let c = context.expect("matched positive packet");
            let device = Device::Cuda(device_index);
            let s = i64::try_from(c.gates().len()).map_err(|_| "I08 slot count overflow")?;
            let factors = u32::try_from(c.factors().len())
                .map_err(|_| "I08 factor count exceeds CUDA range")?;
            let gates = Tensor::from_slice(
                &c.gates()
                    .iter()
                    .flat_map(|&(a, b)| [a as i32, b as i32])
                    .collect::<Vec<_>>(),
            )
            .to_device(device);
            let mut scopes = Vec::new();
            let mut offsets = vec![0i64];
            for factor in c.factors() {
                scopes.extend(factor.scope().iter().map(|&v| v as i32));
                offsets.push(i64::try_from(scopes.len()).map_err(|_| "I08 factor scope overflow")?);
            }
            let factor_offsets = Tensor::from_slice(&offsets).to_device(device);
            let factor_scopes = Tensor::from_slice(&scopes).to_device(device);
            let counts = allocate(&[i64::from(dimension)], Kind::Int64, device)?;
            let indices = allocate(&[(i64::from(factors) + 3 * s).max(1)], Kind::Int, device)?;
            let values = allocate(&[s + 1, 4], Kind::Float, device)?;
            let valid = allocate(&[s + 1], Kind::Uint8, device)?;
            let incidence = allocate(&[s + 1, i64::from(dimension), 2], Kind::Float, device)?;
            let status = allocate(&[1], Kind::Uint8, device)?;
            let args = PositivePacketArgs {
                gates: gates.data_ptr().cast(),
                factor_offsets: factor_offsets.data_ptr().cast(),
                factor_scopes: factor_scopes.data_ptr().cast(),
                counts: counts.data_ptr().cast(),
                indices: indices.data_ptr().cast(),
                values: values.data_ptr().cast(),
                valid: valid.data_ptr().cast(),
                incidence: incidence.data_ptr().cast(),
                status: status.data_ptr().cast(),
                stream: crate::cuda::current_stream(device_index).map_err(str::to_owned)?,
                max_monomials: input.max_monomials as u64,
                scope_elements: factor_scopes.numel() as u64,
                count_words: counts.numel() as u64,
                index_elements: indices.numel() as u64,
                value_elements: values.numel() as u64,
                group_elements: valid.numel() as u64,
                incidence_elements: incidence.numel() as u64,
                dimension,
                slots: u32::try_from(s).map_err(|_| "I08 slots exceed CUDA range")?,
                factors,
            };
            unsafe {
                crate::cuda::prepare_positive_packet(&args)?;
            }
            Some(PreparedPacket {
                values,
                valid,
                incidence,
                status,
            })
        } else {
            None
        };
        Ok(Self {
            dimension,
            device_index,
            pre_alphabet,
            post_alphabet,
            prepared,
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
            return Err("I08 plan and states have different widths/devices".into());
        }
        let alphabet = states
            .alphabet_id()
            .ok_or("I08 requires an explicit state alphabet identity")?;
        // Preserve source priority: context limits, then alphabet, then memory.
        let unavailable = self.pre_alphabet.or_else(|| {
            if alphabet != format!("all_to_all_unit_cnot:{}", self.dimension) {
                Some(DeviceFeatureStatus::Unsupported)
            } else {
                self.post_alphabet
            }
        });
        let b = states.batch_size();
        let (values, valid, incidence, status) = if let Some(code) = unavailable {
            (
                allocate(&[b, 0, 4], Kind::Float, device)?,
                allocate(&[b, 0], Kind::Uint8, device)?,
                allocate(&[b, 0, i64::from(self.dimension), 2], Kind::Float, device)?,
                allocate(&[b], Kind::Uint8, device)?.full_like(code as i64),
            )
        } else {
            let p = self.prepared.as_ref().expect("prepared positive packet");
            let g = p.values.size()[0];
            // Pure device broadcasts. The template's dynamic classification and
            // masks are not read back, even for nonlinear/resource-limited cases.
            (
                p.values.unsqueeze(0).expand([b, g, 4], true).contiguous(),
                p.valid.unsqueeze(0).expand([b, g], true).contiguous(),
                p.incidence
                    .unsqueeze(0)
                    .expand([b, g, i64::from(self.dimension), 2], true)
                    .contiguous(),
                p.status.expand([b], true).contiguous(),
            )
        };
        let spec = specs
            .iter()
            .find(|s| (s.id.family(), s.id.ordinal()) == ('I', 8))
            .expect("selected I08");
        entries.push(DeviceFeatureEntry::new(
            spec.clone(),
            values,
            status,
            Some(valid),
            Some(incidence),
        ));
        Ok(())
    }
}
