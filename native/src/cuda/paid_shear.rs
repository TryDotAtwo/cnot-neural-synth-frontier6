use std::ffi::c_void;

#[repr(C)]
pub(crate) struct PaidShearArgs {
    pub states: *const u64,
    pub sources: *const u32,
    pub targets: *const u32,
    pub atlases: [*const u32; 6],
    pub block: *mut u8,
    pub source_basis: *mut u8,
    pub target_basis: *mut u8,
    pub preparation: *mut u32,
    pub restoration: *mut u32,
    pub core: *mut u32,
    pub word: *mut u32,
    pub cancellations: *mut u32,
    pub moves: *mut u64,
    pub statistics: *mut u64,
    pub lengths: *mut i64,
    pub endpoint: *mut u64,
    pub replay: *mut u64,
    pub scratch: *mut u64,
    pub status: *mut u8,
    pub verified: *mut u8,
    pub stream: *mut c_void,
    pub state_words: u64,
    pub block_elements: u64,
    pub source_basis_elements: u64,
    pub target_basis_elements: u64,
    pub preparation_elements: u64,
    pub core_elements: u64,
    pub move_elements: u64,
    pub stat_elements: u64,
    pub scratch_words: u64,
    pub batch_elements: u64,
    pub source_elements: u64,
    pub target_elements: u64,
    pub atlas_words: [u64; 6],
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
    pub source_count: u32,
    pub target_count: u32,
    pub local_width: u32,
    pub sweeps: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_paid_shear(args: *const PaidShearArgs) -> i32;
}
/// # Safety
/// Static contexts, atlas identities, disjoint extents and lifetimes satisfy
/// cnot_cuda.h. All allocations use the same current Torch CUDA stream.
pub(crate) unsafe fn launch_paid_shear(args: &PaidShearArgs) -> Result<(), String> {
    match unsafe { cnot103_cuda_launch_paid_shear(args) } {
        0 => Ok(()),
        1 => Err("invalid paid-shear CUDA metadata".into()),
        2 => Err("paid-shear CUDA extent is too small or overflows".into()),
        code => Err(format!("paid-shear CUDA launch failed: {code}")),
    }
}
