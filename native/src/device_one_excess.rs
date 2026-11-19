//! Exact resident Stage38 enumeration, complete count/recovery certificate and
//! verified F08. The negative class excludes only changed-row lengths r/r+1.
use crate::cuda::OneExcessArgs;
use crate::device_state::DeviceStateBatch;
use crate::device_zero_excess::{DeviceZeroExcessBatch, DeviceZeroExcessPlan};
use tch::{Device, Kind, Tensor};

pub struct DeviceOneExcessBatch {
    headers: Tensor,
    counts: Tensor,
    gates: Tensor,
    lengths: Tensor,
    status: Tensor,
    feature: Tensor,
}
impl DeviceOneExcessBatch {
    /// [B,6]: class1Exact/2NoOneExcess/3Stage37Exact/4Resource, changed,
    /// templates, max_bag, transport_free(exact only), distance/excluded_length.
    /// An outer width/alphabet unavailability clears the complete header.
    pub fn headers(&self) -> &Tensor {
        &self.headers
    }
    /// [B,limbs] little-endian u32 magnitudes, carried in Int tensors.
    pub fn counts(&self) -> &Tensor {
        &self.counts
    }
    /// [B,n+1,2] exact source encoder I->M; unused physical gates are zero.
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
pub struct DeviceOneExcessPlan {
    dimension: u32,
    device_index: usize,
    max_n: u32,
    max_entries: usize,
    max_changed_rows: Option<u32>,
    max_component_events: Option<u32>,
    max_history_counts: Option<u32>,
    zero_excess: DeviceZeroExcessPlan,
}
fn allocate(shape: &[u64], kind: Kind, device: Device, zero: bool) -> Result<Tensor, String> {
    let count = shape
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or("one-excess allocation extent overflow")?;
    let bytes = count
        .checked_mul(match kind {
            Kind::Int64 => 8,
            Kind::Uint8 => 1,
            _ => 4,
        })
        .ok_or("one-excess allocation byte overflow")?;
    if bytes > i64::MAX as u64 {
        return Err("one-excess allocation exceeds tensor index range".into());
    }
    let dimensions = shape
        .iter()
        .map(|&v| i64::try_from(v).map_err(|_| "one-excess dimension overflow"))
        .collect::<Result<Vec<_>, _>>()?;
    if zero {
        Tensor::f_zeros(&dimensions, (kind, device))
    } else {
        Tensor::f_empty(&dimensions, (kind, device))
    }
    .map_err(|e| e.to_string())
}
fn limb_capacity(n: u64) -> u64 {
    let bits = |x: u64| u64::from(x.ilog2() + 1);
    let k = n + 1;
    (1 + 2 * bits(n) + k + 2 * k * bits(k) + (k + 1) * bits(3 * n)).div_ceil(32) + 1
}
impl DeviceOneExcessPlan {
    pub fn new(
        dimension: u32,
        device_index: usize,
        max_n: u32,
        max_entries: usize,
    ) -> Result<Self, String> {
        Self::new_with_limits(dimension, device_index, max_n, max_entries, None, None, None)
    }
    pub(crate) fn new_with_limits(
        dimension: u32, device_index: usize, max_n: u32, max_entries: usize,
        max_changed_rows: Option<u32>, max_component_events: Option<u32>, max_history_counts: Option<u32>,
    ) -> Result<Self, String> {
        if !(2..=1 << 24).contains(&dimension) || max_n < 2 || max_entries == 0 {
            return Err("one-excess width/budget exceeds device/configuration contract".into());
        }
        if max_changed_rows.is_some_and(|limit| limit == 0 || limit > max_n) {
            return Err("one-excess changed-row limit must be between one and max_n".into());
        }
        if max_component_events.is_some_and(|limit| limit == 0 || u64::from(limit) > u64::from(max_n) + 1) {
            return Err("one-excess component limit must be between one and max_n plus one".into());
        }
        if max_history_counts == Some(0) {
            return Err("one-excess history count limit must be positive".into());
        }
        let zero_excess = DeviceZeroExcessPlan::new(dimension, device_index, max_n)?;
        Ok(Self {
            dimension,
            device_index,
            max_n,
            max_entries,
            max_changed_rows,
            max_component_events,
            max_history_counts,
            zero_excess,
        })
    }
    fn unavailable(&self, states: &DeviceStateBatch) -> Result<Option<i64>, String> {
        if states.dimension() != self.dimension
            || states.tensor().device() != Device::Cuda(self.device_index)
        {
            return Err("one-excess states and plan differ in width/device".into());
        }
        let label = states
            .alphabet_id()
            .ok_or("one-excess requires an explicit state alphabet identity")?;
        Ok(if self.dimension > self.max_n {
            Some(4)
        } else if label != format!("all_to_all_unit_cnot:{}", self.dimension) {
            Some(3)
        } else {
            None
        })
    }
    fn output(
        &self,
        states: &DeviceStateBatch,
        zero: bool,
    ) -> Result<DeviceOneExcessBatch, String> {
        let b = states.batch_size() as u64;
        let n = u64::from(self.dimension);
        let d = Device::Cuda(self.device_index);
        Ok(DeviceOneExcessBatch {
            headers: allocate(&[b, 6], Kind::Int64, d, zero)?,
            counts: allocate(&[b, limb_capacity(n)], Kind::Int, d, zero)?,
            gates: allocate(&[b, n + 1, 2], Kind::Int, d, zero)?,
            lengths: allocate(&[b], Kind::Int64, d, zero)?,
            status: allocate(&[b], Kind::Uint8, d, false)?,
            feature: allocate(&[b, 3], Kind::Float, d, zero)?,
        })
    }
    pub fn evaluate(&self, states: &DeviceStateBatch) -> Result<DeviceOneExcessBatch, String> {
        if let Some(code) = self.unavailable(states)? {
            let output = self.output(states, true)?;
            return Ok(DeviceOneExcessBatch {
                status: output.status.full_like(code),
                ..output
            });
        }
        let zero = self.zero_excess.evaluate(states)?;
        self.evaluate_with_zero_excess(states, &zero)
    }
    // Private reuse within one feature-plan evaluation; callers cannot supply a
    // mutable external alias as the prerequisite. CUDA also checks its counts.
    pub(crate) fn evaluate_with_zero_excess(
        &self,
        states: &DeviceStateBatch,
        zero: &DeviceZeroExcessBatch,
    ) -> Result<DeviceOneExcessBatch, String> {
        if let Some(code) = self.unavailable(states)? {
            let output = self.output(states, true)?;
            return Ok(DeviceOneExcessBatch {
                status: output.status.full_like(code),
                ..output
            });
        }
        let output = self.output(states, false)?;
        let b = states.batch_size() as u64;
        let n = u64::from(self.dimension);
        let w = n.div_ceil(64);
        let k = n + 1;
        let f = n + 2;
        let ew = k.div_ceil(64);
        let limbs = limb_capacity(n);
        let device = Device::Cuda(self.device_index);
        if zero.headers().size() != [b as i64, 3]
            || zero.status().size() != [b as i64]
            || zero.headers().device() != device
            || zero.status().device() != device
        {
            return Err("one-excess prerequisite shape/device mismatch".into());
        }
        // Shape-only scheduling/storage choices. Cache eviction changes work,
        // never exact values or source resource outcomes. No throughput claim.
        let workers = b.min(8);
        let lanes = 8;
        let cache = (4 * k * k).min(4096);
        let nodes = f + k;
        let depth = k + 1;
        let meta = 9 + 10 * k + f + 15 * n * f;
        let bit_words = (f + 2 * k) * ew;
        let scratch = n * w + k * ew + 7 * k + 16;
        let program = 16
            + 8 * nodes
            + nodes * ew
            + k * lanes
            + 2 * cache * lanes
            + 5 * depth * lanes
            + 2 * lanes;
        let arithmetic = limbs * ((2 * depth + 3 + cache) * lanes + k + 6) + limbs;
        let make = |size, kind| allocate(&[workers, size], kind, device, false);
        let histories = make(3 * n * w, Kind::Int64)?;
        let lengths = make(n, Kind::Int)?;
        let protected = make(n, Kind::Uint8)?;
        let sm = make(meta, Kind::Int)?;
        let sb = make(bit_words, Kind::Int64)?;
        let ss = make(scratch, Kind::Int64)?;
        let pg = make(program, Kind::Int64)?;
        let ar = make(arithmetic, Kind::Int)?;
        let classifier = make(16 + 4 * w, Kind::Int64)?;
        let work_gates = make(2 * k, Kind::Int)?;
        let verified = allocate(&[b], Kind::Uint8, device, false)?;
        let word_scratch = allocate(&[b, n, w], Kind::Int64, device, false)?;
        let args = OneExcessArgs {
            states: states.tensor().data_ptr().cast(),
            zero_headers: zero.headers().data_ptr().cast(),
            zero_status: zero.status().data_ptr().cast(),
            histories: histories.data_ptr().cast(),
            history_lengths: lengths.data_ptr().cast(),
            protected_rows: protected.data_ptr().cast(),
            structure_meta: sm.data_ptr().cast(),
            structure_bits: sb.data_ptr().cast(),
            structure_scratch: ss.data_ptr().cast(),
            program: pg.data_ptr().cast(),
            arithmetic: ar.data_ptr().cast(),
            classifier: classifier.data_ptr().cast(),
            workspace_gates: work_gates.data_ptr().cast(),
            headers: output.headers.data_ptr().cast(),
            counts: output.counts.data_ptr().cast(),
            gates: output.gates.data_ptr().cast(),
            lengths: output.lengths.data_ptr().cast(),
            verified: verified.data_ptr().cast(),
            status: output.status.data_ptr().cast(),
            feature: output.feature.data_ptr().cast(),
            word_scratch: word_scratch.data_ptr().cast(),
            stream: crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?,
            state_words: states.tensor().numel() as u64,
            zero_header_words: zero.headers().numel() as u64,
            zero_status_elements: zero.status().numel() as u64,
            history_words: histories.numel() as u64,
            length_elements: lengths.numel() as u64,
            protected_elements: protected.numel() as u64,
            meta_elements: sm.numel() as u64,
            bit_words: sb.numel() as u64,
            scratch_words: ss.numel() as u64,
            program_words: pg.numel() as u64,
            arithmetic_elements: ar.numel() as u64,
            classifier_words: classifier.numel() as u64,
            workspace_gate_elements: work_gates.numel() as u64,
            header_elements: output.headers.numel() as u64,
            count_elements: output.counts.numel() as u64,
            gate_elements: output.gates.numel() as u64,
            batch_elements: b,
            feature_elements: output.feature.numel() as u64,
            word_scratch_words: word_scratch.numel() as u64,
            max_entries: self.max_entries as u64,
            batch: u32::try_from(b).map_err(|_| "one-excess batch exceeds CUDA range")?,
            dimension: self.dimension,
            words_per_row: w as u32,
            workers: workers as u32,
            lanes: lanes as u32,
            cache_slots: cache as u32,
            max_changed_rows: self.max_changed_rows.unwrap_or(0),
            max_component_events: self.max_component_events.unwrap_or(0),
            max_history_counts: self.max_history_counts.unwrap_or(0),
        };
        unsafe {
            crate::cuda::launch_one_excess(&args)?;
        }
        Ok(output)
    }
}
