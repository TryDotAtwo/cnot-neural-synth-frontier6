use std::ffi::c_void;
#[repr(C)]
pub(crate) struct TemplateResponseArgs {
    pub coefficients: *const u8,
    pub raw_status: *const u8,
    pub choice_data: *const u32,
    pub wires: *const u32,
    pub probabilities: *const u64,
    pub costs: *mut u64,
    pub rounding: *mut u64,
    pub globals: *mut f32,
    pub mixed: *mut f32,
    pub valid: *mut u8,
    pub incidence: *mut f32,
    pub status: *mut u8,
    pub stream: *mut c_void,
    pub coefficient_elements: u64,
    pub choice_elements: u64,
    pub wire_elements: u64,
    pub probability_elements: u64,
    pub cost_elements: u64,
    pub rounding_elements: u64,
    pub global_elements: u64,
    pub mixed_elements: u64,
    pub token_elements: u64,
    pub incidence_elements: u64,
    pub batch_elements: u64,
    pub batch: u32,
    pub dimension: u32,
    pub choices: u32,
    pub points: u32,
    pub unique_wires: u32,
    pub invalid_choices: u32,
}
unsafe extern "C" {
    fn cnot103_cuda_launch_template_response(args: *const TemplateResponseArgs) -> i32;
}
/// Safety: disjoint extents, normalized probability/config metadata and
/// stream-ordered coefficient inputs satisfy cnot_cuda.h; all buffers remain
/// alive through their queued consumers. No host synchronization or readback.
pub(crate) unsafe fn launch_template_response(args: &TemplateResponseArgs) -> Result<(), String> {
    match unsafe { cnot103_cuda_launch_template_response(args) } {
        0 => Ok(()),
        1 => Err("invalid template-response CUDA metadata".into()),
        2 => Err("template-response CUDA extents are too small or overflow".into()),
        code => Err(format!("template-response CUDA launch failed: {code}")),
    }
}
