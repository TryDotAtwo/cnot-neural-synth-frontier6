use std::ffi::c_void;
#[repr(C)]
pub(crate) struct EndpointPrepareArgs {
    pub storage: *mut u64,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub storage_words: u64,
    pub capacity: u64,
    pub buckets: u64,
    pub history_buckets: u64,
    pub max_degree: u64,
    pub max_layer_states: u64,
    pub max_transitions: u64,
    pub dimension: u32,
    pub words_per_row: u32,
}
#[repr(C)]
pub(crate) struct EndpointArgs {
    pub states: *const u64,
    pub storage: *const u64,
    pub prepared_status: *const u8,
    pub values: *mut f32,
    pub feature_status: *mut u8,
    pub oracle_status: *mut u8,
    pub leading_degrees: *mut u64,
    pub leading_counts: *mut u64,
    pub terminal_counts: *mut u64,
    pub evaluations: *mut u64,
    pub stream: *mut c_void,
    pub state_words: u64,
    pub storage_words: u64,
    pub value_elements: u64,
    pub count_elements: u64,
    pub batch_elements: u64,
    pub capacity: u64,
    pub buckets: u64,
    pub history_buckets: u64,
    pub max_degree: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_prepare_endpoint_response(args: *const EndpointPrepareArgs) -> i32;
    fn cnot103_cuda_launch_endpoint_response(args: *const EndpointArgs) -> i32;
}
fn check(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("invalid endpoint-response CUDA metadata".into()),
        2 => Err("endpoint-response CUDA allocation is too small".into()),
        code => Err(format!("endpoint-response CUDA launch failed: {code}")),
    }
}
/// # Safety
/// Pointers/capacities obey cnot_cuda.h. Buffers survive execution or use the
/// current Torch stream's allocator; no simultaneous mutation of the tables.
pub(crate) unsafe fn prepare_endpoint_response(args: &EndpointPrepareArgs) -> Result<(), String> {
    check(unsafe { cnot103_cuda_prepare_endpoint_response(args) })
}
/// # Safety
/// Prepared tables and states are immutable; output buffers are disjoint.
/// Preparation/consumption follow the current-stream lifetime contract.
pub(crate) unsafe fn launch_endpoint_response(args: &EndpointArgs) -> Result<(), String> {
    check(unsafe { cnot103_cuda_launch_endpoint_response(args) })
}
