use std::ffi::c_void;
#[repr(C)]
pub(crate) struct BilinearPrepareArgs {
    pub u: *const u64,
    pub v: *const u64,
    pub a0: *const u64,
    pub d0: *const u64,
    pub matrix_template: *mut u64,
    pub scratch: *mut u64,
    pub statistics: *mut u64,
    pub stream: *mut c_void,
    pub template_words: u64,
    pub scratch_words: u64,
    pub m: u32,
    pub r: u32,
    pub p: u32,
    pub t: u32,
    pub q: u32,
}
#[repr(C)]
pub(crate) struct BilinearArgs {
    pub states: *const u64,
    pub row_wires: *const u32,
    pub column_wires: *const u32,
    pub matrix_template: *const u64,
    pub statistics: *const u64,
    pub scratch: *mut u64,
    pub values: *mut f32,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub template_words: u64,
    pub scratch_words: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
    pub m: u32,
    pub r: u32,
    pub p: u32,
    pub q: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_prepare_bilinear(args: *const BilinearPrepareArgs) -> i32;
    fn cnot103_cuda_launch_bilinear(args: *const BilinearArgs) -> i32;
}
fn result(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("invalid bilinear CUDA metadata".into()),
        2 => Err("bilinear CUDA allocation is too small".into()),
        code => Err(format!("bilinear CUDA launch failed: {code}")),
    }
}
/// # Safety
/// Validated port matrices and allocated device pointers follow the header;
/// lifetimes extend through stream completion or use a stream-aware allocator.
pub(crate) unsafe fn prepare_bilinear(args: &BilinearPrepareArgs) -> Result<(), String> {
    result(unsafe { cnot103_cuda_prepare_bilinear(args) })
}
/// # Safety
/// Validated physical indices, prepared buffers, capacities and stream follow
/// the header. Lifetimes extend through execution or use a stream-aware allocator.
pub(crate) unsafe fn launch_bilinear(args: &BilinearArgs) -> Result<(), String> {
    result(unsafe { cnot103_cuda_launch_bilinear(args) })
}
