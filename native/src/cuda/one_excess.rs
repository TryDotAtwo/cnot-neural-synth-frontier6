use std::ffi::c_void;
#[repr(C)]
pub(crate) struct OneExcessArgs {
    pub states: *const u64,
    pub zero_headers: *const u64,
    pub zero_status: *const u8,
    pub histories: *mut u64,
    pub history_lengths: *mut u32,
    pub protected_rows: *mut u8,
    pub structure_meta: *mut u32,
    pub structure_bits: *mut u64,
    pub structure_scratch: *mut u64,
    pub program: *mut u64,
    pub arithmetic: *mut u32,
    pub classifier: *mut u64,
    pub workspace_gates: *mut u32,
    pub headers: *mut u64,
    pub counts: *mut u32,
    pub gates: *mut u32,
    pub lengths: *mut i64,
    pub verified: *mut u8,
    pub status: *mut u8,
    pub feature: *mut f32,
    pub word_scratch: *mut u64,
    pub stream: *mut c_void,
    pub state_words: u64,
    pub zero_header_words: u64,
    pub zero_status_elements: u64,
    pub history_words: u64,
    pub length_elements: u64,
    pub protected_elements: u64,
    pub meta_elements: u64,
    pub bit_words: u64,
    pub scratch_words: u64,
    pub program_words: u64,
    pub arithmetic_elements: u64,
    pub classifier_words: u64,
    pub workspace_gate_elements: u64,
    pub header_elements: u64,
    pub count_elements: u64,
    pub gate_elements: u64,
    pub batch_elements: u64,
    pub feature_elements: u64,
    pub word_scratch_words: u64,
    pub max_entries: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
    pub workers: u32,
    pub lanes: u32,
    pub cache_slots: u32,
    pub max_changed_rows: u32,
    pub max_component_events: u32,
    pub max_history_counts: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_one_excess(args: *const OneExcessArgs) -> i32;
}
/// Safety: all disjoint allocations and their extents stay live on the current
/// CUDA stream. Inputs are resident; no numerical host synchronization occurs.
pub(crate) unsafe fn launch_one_excess(args: &OneExcessArgs) -> Result<(), String> {
    match unsafe { cnot103_cuda_launch_one_excess(args) } {
        0 => Ok(()),
        1 => Err("invalid one-excess CUDA metadata".into()),
        2 => Err("one-excess CUDA extents are too small or overflow".into()),
        n => Err(format!("one-excess CUDA launch failed: {n}")),
    }
}
