use std::ffi::c_void;

/// ABI shared with cnot_cuda.h. Capacities count elements, never bytes.
#[repr(C)]
pub(crate) struct BlockHistogramArgs {
    pub states: *const u64,
    pub source_masks: *const u64,
    pub target_masks: *const u64,
    pub frequencies: *mut u32,
    pub statistics: *mut u64,
    pub ordered_ids: *mut u32,
    pub pattern_values: *mut f32,
    pub pattern_valid: *mut u8,
    pub pattern_incidence: *mut f32,
    pub pair_values: *mut f32,
    pub pair_valid: *mut u8,
    pub pair_incidence: *mut f32,
    pub stream: *mut c_void,
    pub pair_capacity: u64,
    pub frequency_elements: u64,
    pub statistic_elements: u64,
    pub pattern_groups: u64,
    pub pattern_incidence_elements: u64,
    pub pair_groups: u64,
    pub pair_incidence_elements: u64,
    pub batch: u32,
    pub contexts: u32,
    pub dimension: u32,
    pub words_per_row: u32,
    pub pattern_capacity: u32,
    pub emit_patterns: u32,
}

unsafe extern "C" {
    fn cnot103_cuda_launch_block_histograms(args: *const BlockHistogramArgs) -> i32;
}

/// # Safety
/// All pointers refer to distinct live CUDA allocations on the stream's device.
/// States are packed [batch,dimension,words_per_row]; masks are [contexts,words].
/// Frequencies have [batch,contexts,dimension+1] u32 elements, statistics have
/// [batch,contexts,4] u64 elements. Pattern buffers have P groups per context,
/// pair buffers Q. Values/masks/incidence have respectively [groups,3]/[groups]/
/// [groups,dimension,3] for patterns and [groups]/[groups]/[groups,dimension,3]
/// for pairs. Unused optional buffers may be null. Buffers remain live until
/// their work on this stream completes; this call does not synchronize/read back.
pub(crate) unsafe fn launch_block_histograms(args: &BlockHistogramArgs) -> Result<(), String> {
    let status = unsafe { cnot103_cuda_launch_block_histograms(args) };
    match status {
        0 => Ok(()),
        1 => Err("invalid block-histogram kernel arguments/capacity".into()),
        2 => Err("block-histogram kernel extent overflow".into()),
        _ => Err("CUDA block-histogram kernel launch failed".into()),
    }
}
