use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub taus: [f64; 4],
    pub floor: f64,
    pub max_pair_states: u64,
    pub dimension: u32,
}
#[repr(C)]
#[derive(Default)]
pub(crate) struct Layout {
    pub classes: u64,
    pub weight_stride: u32,
}
#[repr(C)]
pub(crate) struct Prepare {
    pub config: Config,
    pub work: *mut f64,
    pub predecessors: *mut i64,
    pub log_factorial: *mut f64,
    pub weights: *mut f64,
    pub phase_status: *mut u8,
    pub descriptors: *mut f32,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub work_len: u64,
    pub predecessor_len: u64,
    pub factorial_len: u64,
    pub weight_len: u64,
    pub phase_len: u64,
    pub descriptor_len: u64,
    pub status_len: u64,
}
#[repr(C)]
pub(crate) struct Project {
    pub column_weights: *const f32,
    pub column_intersections: *const f32,
    pub descriptors: *const f32,
    pub profile_status: *const u8,
    pub class_ids: *mut i64,
    pub values: *mut f32,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub max_pair_states: u64,
    pub weight_batch_stride: u64,
    pub intersection_batch_stride: u64,
    pub weight_len: u64,
    pub intersection_len: u64,
    pub descriptor_len: u64,
    pub profile_status_len: u64,
    pub class_len: u64,
    pub value_len: u64,
    pub status_len: u64,
    pub batch: u32,
    pub dimension: u32,
}

unsafe extern "C" {
    fn cnot103_cuda_pair_diffusion_layout(config: &Config, layout: &mut Layout) -> i32;
    fn cnot103_cuda_prepare_pair_diffusion(args: &Prepare) -> i32;
    fn cnot103_cuda_project_pair_diffusion(args: &Project) -> i32;
}
fn check(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("invalid CUDA pair diffusion arguments".into()),
        2 => Err("invalid CUDA pair diffusion extent".into()),
        _ => Err("CUDA pair diffusion launch failed".into()),
    }
}
pub(crate) fn layout(config: &Config) -> Result<Layout, String> {
    let mut l = Layout::default();
    check(unsafe { cnot103_cuda_pair_diffusion_layout(config, &mut l) })?;
    Ok(l)
}
/// All buffers use the types, element counts and active CUDA stream in pair_diffusion.h.
pub(crate) unsafe fn prepare(args: &Prepare) -> Result<(), String> {
    check(unsafe { cnot103_cuda_prepare_pair_diffusion(args) })
}
/// A06/A14 are strided views of one live allocation. Class IDs and outputs own distinct storage.
pub(crate) unsafe fn project(args: &Project) -> Result<(), String> {
    check(unsafe { cnot103_cuda_project_pair_diffusion(args) })
}
