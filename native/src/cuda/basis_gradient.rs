use std::ffi::c_void;

unsafe extern "C" {
    fn cnot103_cuda_launch_basis_gradient(
        states: *const u64,
        scratch: *mut u64,
        coefficients: *mut u8,
        status: *mut u8,
        gradients: *mut i64,
        batch: u32,
        dimension: u32,
        words_per_row: u32,
        scratch_words: u64,
        coefficient_elements: u64,
        gradient_elements: u64,
        emit_gradients: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// # Safety
/// Distinct live contiguous CUDA buffers on the current stream's device:
/// states [batch,n,ceil(n/64)], scratch [batch,w*(ceil(n/64)+ceil(w/64)+1)] u64,
/// coefficients [batch,n-w,w] u8, status [batch] u8, optional gradients
/// [batch,w,w] i64, where w=floor(n/2). Keep storage live through queued use.
/// No host readback or synchronization is performed.
pub(crate) unsafe fn launch_basis_gradient(
    states: *const u64,
    scratch: *mut u64,
    coefficients: *mut u8,
    status: *mut u8,
    gradients: *mut i64,
    batch: u32,
    dimension: u32,
    words_per_row: u32,
    scratch_words: u64,
    coefficient_elements: u64,
    gradient_elements: u64,
    emit_gradients: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    let result = unsafe {
        cnot103_cuda_launch_basis_gradient(
            states,
            scratch,
            coefficients,
            status,
            gradients,
            batch,
            dimension,
            words_per_row,
            scratch_words,
            coefficient_elements,
            gradient_elements,
            u32::from(emit_gradients),
            stream,
        )
    };
    match result {
        0 => Ok(()),
        1 => Err("invalid basis-gradient CUDA arguments/capacity".into()),
        2 => Err("basis-gradient CUDA extent overflow".into()),
        _ => Err("basis-gradient CUDA launch failed".into()),
    }
}
