use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub taus: [f64; 4],
    pub floor: f64,
    pub dimension: u32,
    pub max_n: u32,
}

#[repr(C)]
#[derive(Default)]
pub(crate) struct Layout {
    pub classes: u32,
    pub weight_stride: u32,
}

#[repr(C)]
pub(crate) struct Prepare {
    pub config: Config,
    pub work: *mut f64,
    pub predecessors: *mut i32,
    pub weights: *mut f64,
    pub phase_status: *mut u8,
    pub descriptors: *mut f32,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub work_len: u64,
    pub predecessor_len: u64,
    pub weight_len: u64,
    pub phase_len: u64,
    pub descriptor_len: u64,
    pub status_len: u64,
}

#[repr(C)]
pub(crate) struct Project {
    pub weight: *const f32,
    pub missing_diagonal: *const f32,
    pub descriptors: *const f32,
    pub profile_status: *const u8,
    pub values: *mut f32,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub weight_batch_stride: u64,
    pub weight_wire_stride: u64,
    pub diagonal_batch_stride: u64,
    pub weight_len: u64,
    pub diagonal_len: u64,
    pub descriptor_len: u64,
    pub profile_status_len: u64,
    pub value_len: u64,
    pub status_len: u64,
    pub batch: u32,
    pub dimension: u32,
    pub max_n: u32,
    pub ordinal: u32,
}

unsafe extern "C" {
    fn cnot103_cuda_column_layout(config: &Config, layout: &mut Layout) -> i32;
    fn cnot103_cuda_prepare_column_diffusion(args: &Prepare) -> i32;
    fn cnot103_cuda_project_column_diffusion(args: &Project) -> i32;
}

fn check(code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err("invalid CUDA column diffusion arguments".into()),
        2 => Err("invalid CUDA column diffusion extent".into()),
        _ => Err("CUDA column diffusion launch failed".into()),
    }
}

pub(crate) fn layout(config: &Config) -> Result<Layout, String> {
    let mut layout = Layout::default();
    check(unsafe { cnot103_cuda_column_layout(config, &mut layout) })?;
    Ok(layout)
}

/// Buffers must be live on the current device/stream with the capacities in
/// column_diffusion.h. Only the descriptors and final status survive setup.
pub(crate) unsafe fn prepare(args: &Prepare) -> Result<(), String> {
    check(unsafe { cnot103_cuda_prepare_column_diffusion(args) })
}

/// The two read-only A-family views use their real storage strides. The
/// immutable prepared profile and distinct outputs must be live on its stream.
pub(crate) unsafe fn project(args: &Project) -> Result<(), String> {
    check(unsafe { cnot103_cuda_project_column_diffusion(args) })
}
