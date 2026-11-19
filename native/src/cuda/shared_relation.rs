use std::ffi::c_void;

#[repr(C)]
pub(crate) struct SharedRelationArgs {
    pub states: *const u64,
    pub row_wires: *const u32,
    pub coordinate_wires: *const u32,
    pub covectors: *const u64,
    pub scratch: *mut u64,
    pub values: *mut f32,
    pub valid: *mut u8,
    pub incidence: *mut f32,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub scratch_words: u64,
    pub value_elements: u64,
    pub group_elements: u64,
    pub incidence_elements: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
    pub vectors: u32,
    pub coordinates: u32,
    pub ports: u32,
    pub max_vectors: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_shared_relation(args: *const SharedRelationArgs) -> i32;
}
/// # Safety
/// Pointers follow the header's allocation, capacity and stream contract.
/// Physical indices and masks must come from a validated SharedRelationContext.
/// Allocations remain live through execution or use a stream-aware allocator.
pub(crate) unsafe fn launch_shared_relation(args: &SharedRelationArgs) -> Result<(), String> {
    match unsafe { cnot103_cuda_launch_shared_relation(args) } {
        0 => Ok(()),
        1 => Err("invalid shared-relation CUDA metadata".into()),
        2 => Err("shared-relation CUDA allocation is too small".into()),
        code => Err(format!("shared-relation CUDA launch failed: {code}")),
    }
}
