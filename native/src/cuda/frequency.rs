use std::ffi::c_void;

#[repr(C)]
pub(crate) struct FrequencyArgs {
    pub states: *const u64,
    pub roles: *const u8,
    pub internal_wires: *const u32,
    pub statistics: *const u64,
    pub scratch: *mut u64,
    pub status: *mut u8,
    pub values: *mut f32,
    pub stream: *mut c_void,
    pub scratch_words: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
    pub internal_count: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_prepare_frequency(
        central: *const u64,
        scratch: *mut u64,
        statistics: *mut u64,
        rows: u32,
        columns: u32,
        internal_count: u32,
        scratch_words: u64,
        stream: *mut c_void,
    ) -> i32;
    fn cnot103_cuda_launch_frequency(args: *const FrequencyArgs) -> i32;
}
fn result(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("invalid frequency CUDA launch metadata".into()),
        2 => Err("frequency CUDA scratch is too small".into()),
        _ => Err(format!("frequency CUDA launch failed: {code}")),
    }
}
/// # Safety
/// Inputs/outputs are distinct CUDA allocations on the current stream's device.
/// Central/scratch have rows*ceil(columns/64) words; statistics has two u64s.
/// Allocations remain live through execution or use a stream-aware allocator.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn prepare_frequency(
    central: *const u64,
    scratch: *mut u64,
    statistics: *mut u64,
    rows: u32,
    columns: u32,
    internal_count: u32,
    scratch_words: u64,
    stream: *mut c_void,
) -> Result<(), String> {
    result(unsafe {
        cnot103_cuda_prepare_frequency(
            central,
            scratch,
            statistics,
            rows,
            columns,
            internal_count,
            scratch_words,
            stream,
        )
    })
}
/// # Safety
/// Pointers follow the header's distinct allocation, shape and stream contract;
/// immutable roles and internal wire IDs come from a validated FrequencyContext.
pub(crate) unsafe fn launch_frequency(args: &FrequencyArgs) -> Result<(), String> {
    result(unsafe { cnot103_cuda_launch_frequency(args) })
}
