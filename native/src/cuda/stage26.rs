use std::ffi::c_void;

#[repr(C)]
pub(crate) struct Stage26Args {
    pub states: *const u64,
    pub certificate: *mut u64,
    pub scratch: *mut u64,
    pub verified: *mut u8,
    pub status: *mut u8,
    pub f03: *mut f32,
    pub f04: *mut f32,
    pub f05: *mut f32,
    pub stream: *mut c_void,
    pub certificate_words: u64,
    pub workspace_words: u64,
    pub pair_elements: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_stage26(args: *const Stage26Args) -> i32;
}
/// # Safety
/// Every pointer must refer to the documented nonoverlapping CUDA allocation,
/// on the current stream's device, with the specified capacity. The allocations
/// must remain live through execution or be freed by the stream-aware allocator.
/// A rejected certificate causes a device trap, not an unavailable feature.
pub(crate) unsafe fn launch_stage26(args: &Stage26Args) -> Result<(), String> {
    match unsafe { cnot103_cuda_launch_stage26(args) } {
        0 => Ok(()),
        1 => Err("invalid Stage26 CUDA launch metadata".into()),
        2 => Err("Stage26 CUDA allocation is too small".into()),
        code => Err(format!("Stage26 CUDA launch failed: {code}")),
    }
}
