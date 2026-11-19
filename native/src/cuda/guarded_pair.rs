use std::ffi::c_void;
#[repr(C)]
pub(crate) struct GuardedPairArgs {
    pub states: *const u64,
    pub scratch: *mut u64,
    pub graph: *mut u64,
    pub gates: *mut u32,
    pub lengths: *mut i64,
    pub certificate: *mut u64,
    pub status: *mut u8,
    pub verified: *mut u8,
    pub f02: *mut f32,
    pub f06: *mut f32,
    pub valid: *mut u8,
    pub incidence: *mut f32,
    pub stream: *mut c_void,
    pub word_capacity: u64,
    pub state_words: u64,
    pub scratch_words: u64,
    pub graph_words: u64,
    pub gate_elements: u64,
    pub certificate_words: u64,
    pub f06_elements: u64,
    pub incidence_elements: u64,
    pub batch_elements: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_guarded_pair(args: *const GuardedPairArgs) -> i32;
}
/// Safety: disjoint live CUDA allocations, extents, metadata and current stream
/// satisfy cnot_cuda.h. Enqueues proof construction/check/publication, no readback.
pub(crate) unsafe fn launch_guarded_pair(args: &GuardedPairArgs) -> Result<(), String> {
    match unsafe { cnot103_cuda_launch_guarded_pair(args) } {
        0 => Ok(()),
        1 => Err("invalid guarded-pair CUDA metadata".into()),
        2 => Err("guarded-pair CUDA extents are too small or overflow".into()),
        n => Err(format!("guarded-pair CUDA launch failed: {n}")),
    }
}
