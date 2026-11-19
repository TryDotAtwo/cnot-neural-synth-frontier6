use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub tau: f64,
    pub seed: u64,
    pub samples: u64,
    pub dimension: u32,
    pub max_n: u32,
}
#[repr(C)]
pub(crate) struct Prepare {
    pub config: Config,
    pub transforms: *mut u64,
    pub total_gates: *mut u64,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub transform_len: u64,
    pub total_len: u64,
    pub status_len: u64,
}
#[repr(C)]
pub(crate) struct Eval {
    pub config: Config,
    pub states: *const u64,
    pub transforms: *const u64,
    pub total_gates: *const u64,
    pub profile_status: *const u8,
    pub input_columns: *mut u32,
    pub sample_columns: *mut u32,
    pub weights: *mut u32,
    pub values: [*mut f32; 4],
    pub statuses: [*mut u8; 4],
    pub stream: *mut c_void,
    pub state_len: u64,
    pub transform_len: u64,
    pub total_len: u64,
    pub profile_status_len: u64,
    pub input_len: u64,
    pub sample_len: u64,
    pub weight_len: u64,
    pub value_len: [u64; 4],
    pub status_len: [u64; 4],
    pub batch: u32,
    pub workers: u32,
    pub selected: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_shared_word_layout(config: &Config, words: &mut u64) -> i32;
    fn cnot103_cuda_prepare_shared_word(args: &Prepare) -> i32;
    fn cnot103_cuda_eval_shared_word(args: &Eval) -> i32;
}
fn check(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("invalid CUDA shared-word diffusion arguments".into()),
        2 => Err("invalid CUDA shared-word diffusion extent".into()),
        _ => Err("CUDA shared-word diffusion launch failed".into()),
    }
}
pub(crate) fn layout(config: &Config) -> Result<u64, String> {
    let mut words = 0;
    check(unsafe { cnot103_cuda_shared_word_layout(config, &mut words) })?;
    Ok(words)
}
/// Buffers must be distinct, correctly sized allocations on the selected stream.
pub(crate) unsafe fn prepare(args: &Prepare) -> Result<(), String> {
    check(unsafe { cnot103_cuda_prepare_shared_word(args) })
}
/// Packed states, the immutable transform cache and the queued workspace must remain live on this stream.
pub(crate) unsafe fn evaluate(args: &Eval) -> Result<(), String> {
    check(unsafe { cnot103_cuda_eval_shared_word(args) })
}
