use std::ffi::c_void;
#[repr(C)]
pub(crate) struct PositivePacketArgs {
    pub gates: *const u32,
    pub factor_offsets: *const u64,
    pub factor_scopes: *const u32,
    pub counts: *mut u64,
    pub indices: *mut u32,
    pub values: *mut f32,
    pub valid: *mut u8,
    pub incidence: *mut f32,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub max_monomials: u64,
    pub scope_elements: u64,
    pub count_words: u64,
    pub index_elements: u64,
    pub value_elements: u64,
    pub group_elements: u64,
    pub incidence_elements: u64,
    pub dimension: u32,
    pub slots: u32,
    pub factors: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_prepare_positive_packet(args: *const PositivePacketArgs) -> i32;
}
/// # Safety
/// Device pointers, validated context data, capacities and stream follow the
/// header; allocations survive execution or use a stream-aware allocator.
pub(crate) unsafe fn prepare_positive_packet(args: &PositivePacketArgs) -> Result<(), String> {
    match unsafe { cnot103_cuda_prepare_positive_packet(args) } {
        0 => Ok(()),
        1 => Err("invalid positive-packet CUDA metadata".into()),
        2 => Err("positive-packet CUDA allocation is too small".into()),
        code => Err(format!("positive-packet CUDA launch failed: {code}")),
    }
}
