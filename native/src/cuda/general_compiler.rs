use std::ffi::c_void;

#[repr(C)]
pub(crate) struct GeneralCompilerArgs {
    pub states: *const u64,
    pub scratch: *mut u64,
    pub gates: *mut u32,
    pub lengths: *mut i64,
    pub status: *mut u8,
    pub verified: *mut u8,
    pub lower_bounds: *mut i64,
    pub f09: *mut f32,
    pub f10: *mut f32,
    pub stream: *mut c_void,
    pub word_capacity: u64,
    pub workspace_words: u64,
    pub gate_elements: u64,
    pub batch: u32,
    pub dimension: u32,
    pub words_per_row: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_general_compiler(args: *const GeneralCompilerArgs) -> i32;
    fn cnot103_cuda_verify_cnot_words(
        states: *const u64,
        gates: *const u32,
        lengths: *const i64,
        scratch: *mut u64,
        verified: *mut u8,
        batch: u32,
        dimension: u32,
        words_per_row: u32,
        word_capacity: u64,
        workspace_words: u64,
        gate_elements: u64,
        stream: *mut c_void,
    ) -> i32;
}
fn result(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("invalid general-compiler CUDA arguments/capacity".into()),
        2 => Err("general-compiler CUDA extent overflow".into()),
        _ => Err("general-compiler CUDA launch failed".into()),
    }
}
/// # Safety
/// All buffers are distinct live CUDA allocations on the current stream device.
/// states/scratch are [batch,n,ceil(n/64)] u64; gates [batch,Q,2] u32; lengths and
/// lower_bounds [batch] i64; status/verified [batch] u8; f09/f10 [batch,4/5] f32.
/// Buffers remain live until all queued uses finish. No readback or sync occurs.
pub(crate) unsafe fn launch_general_compiler(args: &GeneralCompilerArgs) -> Result<(), String> {
    result(unsafe { cnot103_cuda_launch_general_compiler(args) })
}
/// # Safety
/// The same allocation and lifetime contract as launch_general_compiler; gates
/// and lengths may contain invalid data, which the verifier rejects on device.
pub(crate) unsafe fn verify_cnot_words(
    states: *const u64,
    gates: *const u32,
    lengths: *const i64,
    scratch: *mut u64,
    verified: *mut u8,
    batch: u32,
    dimension: u32,
    words_per_row: u32,
    word_capacity: u64,
    workspace_words: u64,
    gate_elements: u64,
    stream: *mut c_void,
) -> Result<(), String> {
    result(unsafe {
        cnot103_cuda_verify_cnot_words(
            states,
            gates,
            lengths,
            scratch,
            verified,
            batch,
            dimension,
            words_per_row,
            word_capacity,
            workspace_words,
            gate_elements,
            stream,
        )
    })
}
