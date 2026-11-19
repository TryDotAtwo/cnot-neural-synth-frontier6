use std::ffi::c_void;

mod boundary;
pub use boundary::launch_boundary_relations;
mod block_histogram;
pub(crate) mod column_diffusion;
pub(crate) mod pair_diffusion;
pub(crate) mod shared_word;
pub(crate) use block_histogram::{launch_block_histograms, BlockHistogramArgs};
mod basis_gradient;
pub(crate) use basis_gradient::launch_basis_gradient;
mod general_compiler;
mod stage26;
pub(crate) use general_compiler::{
    launch_general_compiler, verify_cnot_words, GeneralCompilerArgs,
};
pub(crate) use stage26::{launch_stage26, Stage26Args};
mod frequency;
pub(crate) use frequency::{launch_frequency, prepare_frequency, FrequencyArgs};
mod shared_relation;
pub(crate) use shared_relation::{launch_shared_relation, SharedRelationArgs};
mod bilinear;
pub(crate) use bilinear::{launch_bilinear, prepare_bilinear, BilinearArgs, BilinearPrepareArgs};
mod positive_packet;
pub(crate) use positive_packet::{prepare_positive_packet, PositivePacketArgs};
mod quadratic;
pub(crate) use quadratic::{launch_quadratic, QuadraticArgs};
mod endpoint_response;
mod guarded_pair;
mod zero_excess;
pub(crate) use zero_excess::{launch_zero_excess, ZeroExcessArgs};
mod one_excess;
pub(crate) use one_excess::{launch_one_excess, OneExcessArgs};
mod paid_basis_atlas;
mod paid_basis_features;
mod template_response;
pub(crate) use guarded_pair::{launch_guarded_pair, GuardedPairArgs};
pub(crate) use template_response::{launch_template_response, TemplateResponseArgs};
mod paid_shear;
pub(crate) use endpoint_response::{
    launch_endpoint_response, prepare_endpoint_response, EndpointArgs, EndpointPrepareArgs,
};
pub(crate) use paid_basis_atlas::{
    optimize_paid_atlas, prepare_paid_atlas, PaidAtlasArgs, PaidAtlasPrepareArgs,
};
pub(crate) use paid_basis_features::{
    paid_basis_append, paid_basis_finish, paid_basis_init, PaidBasisAppendArgs, PaidBasisArgs,
};
pub(crate) use paid_shear::{launch_paid_shear, PaidShearArgs};

unsafe extern "C" {
    fn cnot103_cuda_exchange_device(device_index: i32, previous: &mut i32) -> i32;
    fn cnot103_cuda_current_stream(device_index: i32, stream: *mut *mut c_void) -> i32;
    fn cnot103_cuda_launch_children(
        input_device: *const u64,
        batch: u32,
        dimension: u32,
        words_per_row: u32,
        output_device: *mut u64,
        output_capacity_words: u64,
        stream: *mut c_void,
    ) -> i32;
    fn cnot103_cuda_launch_family_a(
        input_device: *const u64,
        batch: u32,
        dimension: u32,
        words_per_row: u32,
        output_device: *mut f32,
        output_capacity_elements: u64,
        stream: *mut c_void,
    ) -> i32;
    fn cnot103_cuda_launch_residual_rank(
        input_device: *const u64,
        batch: u32,
        dimension: u32,
        words_per_row: u32,
        output_device: *mut f32,
        output_capacity_elements: u64,
        stream: *mut c_void,
    ) -> i32;
    fn cnot103_cuda_launch_block_node_frequencies(
        input_device: *const u64,
        source_masks_device: *const u64,
        target_masks_device: *const u64,
        batch: u32,
        context_count: u32,
        max_patterns: u64,
        dimension: u32,
        words_per_row: u32,
        output_device: *mut f32,
        status_device: *mut u8,
        output_capacity_elements: u64,
        stream: *mut c_void,
    ) -> i32;
    fn cnot103_cuda_launch_block_parity_weight(
        input_device: *const u64,
        source_masks_device: *const u64,
        target_masks_device: *const u64,
        batch: u32,
        context_count: u32,
        max_patterns: u64,
        dimension: u32,
        words_per_row: u32,
        output_device: *mut f32,
        status_device: *mut u8,
        token_valid_device: *mut u8,
        output_capacity_elements: u64,
        stream: *mut c_void,
    ) -> i32;
    fn cnot103_cuda_launch_partial_shear_ranks(
        input_device: *const u64,
        batch: u32,
        dimension: u32,
        words_per_row: u32,
        output_device: *mut f32,
        status_device: *mut u8,
        output_capacity_elements: u64,
        stream: *mut c_void,
    ) -> i32;
    fn cnot103_cuda_launch_projected_ranks(
        input_device: *const u64,
        row_masks_device: *const u64,
        column_masks_device: *const u64,
        candidate_masks_device: *const u64,
        gate_controls_device: *const u32,
        gate_targets_device: *const u32,
        batch: u32,
        cut_count: u32,
        dimension: u32,
        words_per_row: u32,
        subtract_identity: u32,
        append_candidate: u32,
        apply_gate: u32,
        output_device: *mut f32,
        output_capacity_elements: u64,
        stream: *mut c_void,
    ) -> i32;
    fn cnot103_cuda_synchronize(stream: *mut c_void) -> i32;
}

pub fn current_stream(device_index: usize) -> Result<*mut c_void, &'static str> {
    let device_index =
        i32::try_from(device_index).map_err(|_| "CUDA device index is out of range")?;
    let mut stream = std::ptr::null_mut();
    // SAFETY: the bridge validates its output pointer and only returns a borrowed stream handle.
    let status = unsafe { cnot103_cuda_current_stream(device_index, &mut stream) };
    if status == 0 {
        Ok(stream)
    } else {
        Err("failed to obtain the current PyTorch CUDA stream")
    }
}

/// A raw CUDA launch uses the calling thread's current device, including on
/// the default stream. Torch guards its own calls and restores that device.
/// Keep the guard on the same thread until the entire packet is queued.
pub(crate) struct DeviceGuard {
    previous: Option<i32>,
    thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl DeviceGuard {
    pub(crate) fn enter(index: usize) -> Result<Self, String> {
        let target = i32::try_from(index).map_err(|_| "CUDA device index exceeds i32")?;
        let mut previous = -1;
        if unsafe { cnot103_cuda_exchange_device(target, &mut previous) } != 0 {
            return Err(format!("failed to select CUDA device {index}"));
        }
        Ok(Self {
            previous: (previous != target).then_some(previous),
            thread_bound: std::marker::PhantomData,
        })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            let mut current = -1;
            unsafe { cnot103_cuda_exchange_device(previous, &mut current) };
        }
    }
}

/// Launches all ordered one-gate CNOT successors for a batch of packed states.
///
/// # Safety
/// input_device must point to a row-major packed CUDA allocation of
/// [batch, dimension, words_per_row]. output_device must point to a distinct CUDA
/// allocation large enough for [batch, dimension * (dimension - 1), dimension,
/// words_per_row]. The buffers and stream must remain live until synchronize returns.
pub unsafe fn launch_all_to_all_children(
    input_device: *const u64,
    output_device: *mut u64,
    batch: u32,
    dimension: u32,
    words_per_row: u32,
    output_capacity_words: u64,
    stream: *mut c_void,
) -> Result<u64, &'static str> {
    if batch == 0 || dimension < 2 || words_per_row != dimension.div_ceil(64) {
        return Err("invalid CNOT child launch dimensions");
    }
    let actions = u64::from(dimension)
        .checked_mul(u64::from(dimension - 1))
        .ok_or("CNOT child count overflow")?;
    let required = u64::from(batch)
        .checked_mul(actions)
        .and_then(|value| value.checked_mul(u64::from(dimension)))
        .and_then(|value| value.checked_mul(u64::from(words_per_row)))
        .ok_or("CNOT output size overflow")?;
    if output_capacity_words < required {
        return Err("CNOT child output allocation is too small");
    }
    // SAFETY: allocation provenance and lifetime are the caller's documented obligation.
    let status = unsafe {
        cnot103_cuda_launch_children(
            input_device,
            batch,
            dimension,
            words_per_row,
            output_device,
            output_capacity_words,
            stream,
        )
    };
    if status != 0 {
        return Err("CUDA CNOT child launch failed");
    }
    Ok(required)
}

/// Launches A01-A20 structural features directly into a CUDA output tensor.
///
/// # Safety
/// Input and output must be valid allocations on one CUDA device and stay live until
/// the stream work finishes. Capacity is measured in f32 elements.
pub unsafe fn launch_family_a(
    input_device: *const u64,
    batch: u32,
    dimension: u32,
    words_per_row: u32,
    output_device: *mut f32,
    output_capacity_elements: u64,
    stream: *mut c_void,
) -> Result<u64, &'static str> {
    if input_device.is_null()
        || output_device.is_null()
        || batch == 0
        || dimension < 2
        || words_per_row != dimension.div_ceil(64)
    {
        return Err("invalid A-family CUDA launch arguments");
    }
    let cells = u64::from(dimension)
        .checked_mul(u64::from(dimension))
        .ok_or("A-family matrix extent overflow")?;
    let required_per_state = cells
        .checked_mul(17)
        .and_then(|value| {
            u64::from(dimension)
                .checked_mul(10)
                .and_then(|nodes| value.checked_add(nodes))
        })
        .ok_or("A-family feature buffer size overflow")?;
    let required = u64::from(batch)
        .checked_mul(required_per_state)
        .ok_or("A-family feature batch size overflow")?;
    if output_capacity_elements < required {
        return Err("A-family feature output allocation is too small");
    }
    // SAFETY: allocation provenance and lifetime are the caller's documented obligation.
    let status = unsafe {
        cnot103_cuda_launch_family_a(
            input_device,
            batch,
            dimension,
            words_per_row,
            output_device,
            output_capacity_elements,
            stream,
        )
    };
    if status != 0 {
        return Err("CUDA A-family feature kernel failed");
    }
    Ok(required)
}

/// Launches exact GF(2) ranks of M-I for packed CUDA-resident states.
///
/// # Safety
/// Input and output must be live CUDA allocations on the active context. The
/// kernel only reads the packed input and writes one f32 rank per state.
pub unsafe fn launch_residual_ranks(
    input_device: *const u64,
    batch: u32,
    dimension: u32,
    words_per_row: u32,
    output_device: *mut f32,
    output_capacity_elements: u64,
    stream: *mut c_void,
) -> Result<u64, &'static str> {
    if input_device.is_null()
        || output_device.is_null()
        || batch == 0
        || dimension < 2
        || words_per_row != dimension.div_ceil(64)
    {
        return Err("invalid residual-rank launch arguments");
    }
    if output_capacity_elements < u64::from(batch) {
        return Err("residual-rank output allocation is too small");
    }
    // SAFETY: allocation provenance and lifetime are the caller's documented obligation.
    let status = unsafe {
        cnot103_cuda_launch_residual_rank(
            input_device,
            batch,
            dimension,
            words_per_row,
            output_device,
            output_capacity_elements,
            stream,
        )
    };
    if status != 0 {
        return Err("CUDA residual-rank kernel launch failed");
    }
    Ok(u64::from(batch))
}

/// Launches exact D05 node frequencies for target-row and source-column patterns.
/// Status is 1 when computed and 2 when the block-pattern or exact-f32 budget is exceeded.
/// No feature value or status is read back here.
///
/// # Safety
/// Input, masks, outputs and stream must be live CUDA allocations in the active context.
pub unsafe fn launch_block_node_frequencies(
    input_device: *const u64,
    source_masks_device: *const u64,
    target_masks_device: *const u64,
    batch: u32,
    context_count: u32,
    max_patterns: u64,
    dimension: u32,
    words_per_row: u32,
    output_device: *mut f32,
    status_device: *mut u8,
    output_capacity_elements: u64,
    stream: *mut c_void,
) -> Result<u64, &'static str> {
    if input_device.is_null()
        || source_masks_device.is_null()
        || target_masks_device.is_null()
        || output_device.is_null()
        || status_device.is_null()
        || batch == 0
        || context_count == 0
        || max_patterns == 0
        || dimension < 2
        || words_per_row != dimension.div_ceil(64)
    {
        return Err("invalid D05 CUDA launch arguments");
    }
    let output_per_task = u64::from(dimension)
        .checked_mul(2)
        .ok_or("D05 feature width overflow")?;
    let required = u64::from(batch)
        .checked_mul(u64::from(context_count))
        .and_then(|tasks| tasks.checked_mul(output_per_task))
        .ok_or("D05 output size overflow")?;
    if output_capacity_elements < required {
        return Err("D05 output allocation is too small");
    }
    let status = unsafe {
        cnot103_cuda_launch_block_node_frequencies(
            input_device,
            source_masks_device,
            target_masks_device,
            batch,
            context_count,
            max_patterns,
            dimension,
            words_per_row,
            output_device,
            status_device,
            output_capacity_elements,
            stream,
        )
    };
    if status != 0 {
        return Err("CUDA D05 node-frequency kernel failed");
    }
    Ok(required)
}

/// Launches exact D08 physical-block parity weights and per-state range status.
///
/// Status is 1 for a computed feature and 2 when the exact CPU contract would
/// reject its distinct-pattern budget or f32 integer range. No value or
/// status is read back here.
///
/// # Safety
/// Input, masks, outputs and stream must be live CUDA allocations in the active
/// context. Masks have shape [context_count, ceil(dimension / 64)].
pub unsafe fn launch_block_parity_weight(
    input_device: *const u64,
    source_masks_device: *const u64,
    target_masks_device: *const u64,
    batch: u32,
    context_count: u32,
    max_patterns: u64,
    dimension: u32,
    words_per_row: u32,
    output_device: *mut f32,
    status_device: *mut u8,
    token_valid_device: *mut u8,
    output_capacity_elements: u64,
    stream: *mut c_void,
) -> Result<u64, &'static str> {
    if input_device.is_null()
        || source_masks_device.is_null()
        || target_masks_device.is_null()
        || output_device.is_null()
        || status_device.is_null()
        || token_valid_device.is_null()
        || batch == 0
        || context_count == 0
        || dimension < 2
        || words_per_row != dimension.div_ceil(64)
    {
        return Err("invalid D08 CUDA launch arguments");
    }
    let required = u64::from(batch)
        .checked_mul(u64::from(context_count))
        .ok_or("D08 output count overflow")?;
    if output_capacity_elements < required {
        return Err("D08 output allocation is too small");
    }
    let status = unsafe {
        cnot103_cuda_launch_block_parity_weight(
            input_device,
            source_masks_device,
            target_masks_device,
            batch,
            context_count,
            max_patterns,
            dimension,
            words_per_row,
            output_device,
            status_device,
            token_valid_device,
            output_capacity_elements,
            stream,
        )
    };
    if status != 0 {
        return Err("CUDA D08 block-parity kernel failed");
    }
    Ok(required)
}

/// Launches the exact deterministic C05 partial-shear rank on CUDA.
/// Status bytes are 0 for inapplicable, 1 for computed, and 2 for an invalid
/// dependent source-row subset. No feature value or status is read back here.
///
/// # Safety
/// Input, output, status and stream must be live CUDA allocations in the active
/// context. Input shape is [batch, dimension, words_per_row], with packed rows.
pub unsafe fn launch_partial_shear_ranks(
    input_device: *const u64,
    batch: u32,
    dimension: u32,
    words_per_row: u32,
    output_device: *mut f32,
    status_device: *mut u8,
    output_capacity_elements: u64,
    stream: *mut c_void,
) -> Result<u64, &'static str> {
    if input_device.is_null()
        || output_device.is_null()
        || status_device.is_null()
        || batch == 0
        || dimension < 2
        || words_per_row != dimension.div_ceil(64)
    {
        return Err("invalid partial-shear launch arguments");
    }
    let required = u64::from(batch);
    if output_capacity_elements < required {
        return Err("partial-shear output allocation is too small");
    }
    let status = unsafe {
        cnot103_cuda_launch_partial_shear_ranks(
            input_device,
            batch,
            dimension,
            words_per_row,
            output_device,
            status_device,
            output_capacity_elements,
            stream,
        )
    };
    if status != 0 {
        return Err("CUDA partial-shear kernel launch failed");
    }
    Ok(required)
}

/// Waits until work previously launched on this stream has completed.
///
/// # Safety
/// The stream must belong to the active CUDA context and must not be destroyed
/// until this call returns.
pub unsafe fn synchronize(stream: *mut c_void) -> Result<(), &'static str> {
    // SAFETY: stream validity is the caller's documented obligation.
    let status = unsafe { cnot103_cuda_synchronize(stream) };
    if status == 0 {
        Ok(())
    } else {
        Err("CUDA stream synchronization failed")
    }
}

/// Launches exact GF(2) ranks of row/column-masked projections.
///
/// # Safety
/// State, masks, optional candidates/actions, and output must be live CUDA
/// allocations on the active context. Masks are [cut_count, words_per_row]
/// bitsets; action arrays each contain `cut_count` in-range, distinct gate indices.
/// Output has at least [batch, cut_count] f32 elements.
pub unsafe fn launch_projected_ranks(
    input_device: *const u64,
    row_masks_device: *const u64,
    column_masks_device: *const u64,
    candidate_masks_device: *const u64,
    gate_controls_device: *const u32,
    gate_targets_device: *const u32,
    batch: u32,
    cut_count: u32,
    dimension: u32,
    words_per_row: u32,
    subtract_identity: bool,
    append_candidate: bool,
    apply_gate: bool,
    output_device: *mut f32,
    output_capacity_elements: u64,
    stream: *mut c_void,
) -> Result<u64, &'static str> {
    if input_device.is_null()
        || row_masks_device.is_null()
        || column_masks_device.is_null()
        || (append_candidate && candidate_masks_device.is_null())
        || (apply_gate && (gate_controls_device.is_null() || gate_targets_device.is_null()))
        || output_device.is_null()
        || batch == 0
        || cut_count == 0
        || dimension < 2
        || words_per_row != dimension.div_ceil(64)
    {
        return Err("invalid projected-rank launch arguments");
    }
    let required = u64::from(batch)
        .checked_mul(u64::from(cut_count))
        .ok_or("projected-rank output size overflow")?;
    if output_capacity_elements < required {
        return Err("projected-rank output allocation is too small");
    }
    let status = unsafe {
        cnot103_cuda_launch_projected_ranks(
            input_device,
            row_masks_device,
            column_masks_device,
            candidate_masks_device,
            gate_controls_device,
            gate_targets_device,
            batch,
            cut_count,
            dimension,
            words_per_row,
            u32::from(subtract_identity),
            u32::from(append_candidate),
            u32::from(apply_gate),
            output_device,
            output_capacity_elements,
            stream,
        )
    };
    if status != 0 {
        return Err("CUDA projected-rank kernel launch failed");
    }
    Ok(required)
}
