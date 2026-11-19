use std::ffi::c_void;

#[repr(C)]
pub(crate) struct PaidAtlasPrepareArgs {
    pub storage: *mut u32,
    pub stream: *mut c_void,
    pub storage_words: u64,
    pub width: u32,
    pub side: u32,
}
#[repr(C)]
pub(crate) struct PaidAtlasArgs {
    pub storage: *const u32,
    pub vectors: *const u8,
    pub transformed: *mut u8,
    pub operations: *mut i32,
    pub lengths: *mut u64,
    pub direct_cost: *mut u64,
    pub paid_cost: *mut u64,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub storage_words: u64,
    pub vector_elements: u64,
    pub gate_elements: u64,
    pub batch_elements: u64,
    pub word_capacity: u64,
    pub rows: u64,
    pub batch: u32,
    pub width: u32,
    pub side: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_prepare_paid_atlas(args: *const PaidAtlasPrepareArgs) -> i32;
    fn cnot103_cuda_optimize_paid_atlas(args: *const PaidAtlasArgs) -> i32;
}
fn check(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("invalid paid-basis atlas CUDA metadata".into()),
        2 => Err("paid-basis atlas CUDA allocation is too small".into()),
        code => Err(format!("paid-basis atlas CUDA launch failed: {code}")),
    }
}
/// # Safety
/// Storage obeys the documented C ABI extents; its lifetime covers execution.
pub(crate) unsafe fn prepare_paid_atlas(args: &PaidAtlasPrepareArgs) -> Result<(), String> {
    check(unsafe { cnot103_cuda_prepare_paid_atlas(args) })
}
/// # Safety
/// Immutable prepared atlas matches width/side; all output buffers are disjoint.
/// Pointers and lifetimes follow cnot_cuda.h and the current Torch stream.
pub(crate) unsafe fn optimize_paid_atlas(args: &PaidAtlasArgs) -> Result<(), String> {
    check(unsafe { cnot103_cuda_optimize_paid_atlas(args) })
}
