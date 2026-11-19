use std::ffi::c_void;
#[repr(C)]
pub(crate) struct QuadraticArgs {
    pub states: *const u64,
    pub allowed_gates: *const u32,
    pub prefix: *const u8,
    pub terms: *mut u32,
    pub term_counts: *mut u32,
    pub scratch: *mut u64,
    pub values: *mut f32,
    pub valid: *mut u8,
    pub incidence: *mut f32,
    pub ranks: *mut u32,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub state_words: u64,
    pub term_elements: u64,
    pub group_elements: u64,
    pub scratch_words: u64,
    pub value_elements: u64,
    pub incidence_elements: u64,
    pub rank_elements: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
    pub horizon: u32,
    pub cut: u32,
    pub allowed_count: u32,
    pub equations: u32,
    pub max_variables: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_quadratic(args: *const QuadraticArgs) -> i32;
}
/// # Safety
/// Pointers, validated static context, capacities and current stream follow
/// cnot_cuda.h. Allocations survive execution or use its stream-aware allocator.
pub(crate) unsafe fn launch_quadratic(args: &QuadraticArgs) -> Result<(), String> {
    match unsafe { cnot103_cuda_launch_quadratic(args) } {
        0 => Ok(()),
        1 => Err("invalid quadratic CUDA metadata".into()),
        2 => Err("quadratic CUDA allocation is too small".into()),
        code => Err(format!("quadratic CUDA launch failed: {code}")),
    }
}
