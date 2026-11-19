use std::ffi::c_void;
#[repr(C)]
pub(crate) struct ZeroExcessArgs {
    pub states: *const u64,
    pub scratch: *mut u64,
    pub graph: *mut u32,
    pub parents: *mut u32,
    pub changed_rows: *mut u32,
    pub missing_rows: *mut u32,
    pub update_order: *mut u32,
    pub gates: *mut u32,
    pub headers: *mut u64,
    pub lengths: *mut i64,
    pub verified: *mut u8,
    pub status: *mut u8,
    pub feature: *mut f32,
    pub stream: *mut c_void,
    pub state_words: u64,
    pub scratch_words: u64,
    pub graph_elements: u64,
    pub parent_elements: u64,
    pub changed_elements: u64,
    pub missing_elements: u64,
    pub order_elements: u64,
    pub gate_elements: u64,
    pub header_elements: u64,
    pub feature_elements: u64,
    pub batch_elements: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_zero_excess(args: *const ZeroExcessArgs) -> i32;
}
/// Safety: disjoint live CUDA allocations, extents and current stream meet
/// Cnot103ZeroExcessArgs. No numerical host synchronization/readback occurs.
pub(crate) unsafe fn launch_zero_excess(args: &ZeroExcessArgs) -> Result<(), String> {
    match unsafe { cnot103_cuda_launch_zero_excess(args) } {
        0 => Ok(()),
        1 => Err("invalid zero-excess CUDA metadata".into()),
        2 => Err("zero-excess CUDA extents are too small or overflow".into()),
        n => Err(format!("zero-excess CUDA launch failed: {n}")),
    }
}
