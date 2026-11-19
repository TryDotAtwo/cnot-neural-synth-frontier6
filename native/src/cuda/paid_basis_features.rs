use std::ffi::c_void;
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PaidBasisArgs {
    pub word: *mut u32,
    pub labels: *mut u64,
    pub lengths: *mut i64,
    pub endpoint: *mut u64,
    pub replay: *mut u64,
    pub counts: *mut u64,
    pub groups: *mut f32,
    pub incidence: *mut f32,
    pub valid: *mut u8,
    pub pairs: *mut f32,
    pub status: *mut u8,
    pub verified: *mut u8,
    pub overflow: *mut u32,
    pub stream: *mut c_void,
    pub capacity: u64,
    pub state_words: u64,
    pub gate_elements: u64,
    pub label_words: u64,
    pub count_elements: u64,
    pub group_elements: u64,
    pub incidence_elements: u64,
    pub token_elements: u64,
    pub pair_elements: u64,
    pub batch_elements: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
    pub context_count: u32,
}
#[repr(C)]
pub(crate) struct PaidBasisAppendArgs {
    pub output: PaidBasisArgs,
    pub states: *const u64,
    pub sources: *const u32,
    pub targets: *const u32,
    pub word: *const u32,
    pub lengths: *const i64,
    pub statistics: *const u64,
    pub status: *const u8,
    pub previous_word: *const u32,
    pub previous_lengths: *const i64,
    pub capacity: u64,
    pub gate_elements: u64,
    pub stat_elements: u64,
    pub source_elements: u64,
    pub target_elements: u64,
    pub previous_capacity: u64,
    pub previous_gate_elements: u64,
    pub source_count: u32,
    pub target_count: u32,
    pub context_index: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_paid_basis_init(args: *const PaidBasisArgs) -> i32;
    fn cnot103_cuda_paid_basis_append(args: *const PaidBasisAppendArgs) -> i32;
    fn cnot103_cuda_paid_basis_finish(args: *const PaidBasisArgs) -> i32;
}
fn result(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("invalid paid-basis CUDA metadata".into()),
        2 => Err("paid-basis CUDA extent is too small or overflows".into()),
        n => Err(format!("paid-basis CUDA launch failed: {n}")),
    }
}
// Safety for all three calls: cnot_cuda.h extents, ordering, static contexts,
// disjoint allocations and same-stream lifetimes must hold.
pub(crate) unsafe fn paid_basis_init(args: &PaidBasisArgs) -> Result<(), String> {
    result(unsafe { cnot103_cuda_paid_basis_init(args) })
}
pub(crate) unsafe fn paid_basis_append(args: &PaidBasisAppendArgs) -> Result<(), String> {
    result(unsafe { cnot103_cuda_paid_basis_append(args) })
}
pub(crate) unsafe fn paid_basis_finish(args: &PaidBasisArgs) -> Result<(), String> {
    result(unsafe { cnot103_cuda_paid_basis_finish(args) })
}
