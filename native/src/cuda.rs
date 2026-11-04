use std::ffi::c_void;

unsafe extern "C" {
    fn cnot103_cuda_launch_children(
        input_device: *const u64,
        batch: u32,
        dimension: u32,
        words_per_row: u32,
        output_device: *mut u64,
        output_capacity_words: u64,
        stream: *mut c_void,
    ) -> i32;
    fn cnot103_cuda_synchronize(stream: *mut c_void) -> i32;
}

/// Launches every ordered one-gate CNOT successor for a batch of equal-width states.
///
/// # Safety
/// `input_device` contains row-major packed states with shape
/// `[batch, dimension, words_per_row]`; bit `j` of row `i` is matrix entry `(i,j)`,
/// and unused high bits in each final word are zero. `output_device` contains
/// `[batch, dimension * (dimension - 1), dimension, words_per_row]`, with words
/// contiguous and actions ordered control-major while omitting the diagonal target.
/// The declared output capacity must cover the full checked output size, and the
/// actual allocation must be at least that large.
///
/// The two allocations must not overlap. This launch is asynchronous: both buffers
/// must remain live until `synchronize(stream)` completes, the input must not be
/// modified before then, and the output must not be read or reused before then.
/// `stream` must belong to the same CUDA context and remain live through synchronization.
pub unsafe fn launch_all_to_all_children(
    input_device: *const u64,
    output_device: *mut u64,
    batch: u32,
    dimension: u32,
    words_per_row: u32,
    output_capacity_words: u64,
    stream: *mut c_void,
) -> Result<u64, &'static str> {
    if batch == 0 || dimension < 2 || words_per_row != dimension.div_ceil(64) {
        return Err("invalid CNOT child launch dimensions");
    }
    let actions = u64::from(dimension)
        .checked_mul(u64::from(dimension - 1))
        .ok_or("CNOT child count overflow")?;
    let required = u64::from(batch)
        .checked_mul(actions)
        .and_then(|value| value.checked_mul(u64::from(dimension)))
        .and_then(|value| value.checked_mul(u64::from(words_per_row)))
        .ok_or("CNOT output size overflow")?;
    if output_capacity_words < required {
        return Err("CNOT child output allocation is too small");
    }
    // SAFETY: allocation provenance and lifetime are the caller's documented obligation.
    let status = unsafe {
        cnot103_cuda_launch_children(
            input_device,
            batch,
            dimension,
            words_per_row,
            output_device,
            output_capacity_words,
            stream,
        )
    };
    if status != 0 {
        return Err("CUDA CNOT child launch failed");
    }
    Ok(required)
}

/// Waits until work previously launched on this stream has completed.
///
/// # Safety
/// The stream must belong to the active CUDA context and must not be destroyed
/// until this call returns.
pub unsafe fn synchronize(stream: *mut c_void) -> Result<(), &'static str> {
    // SAFETY: stream validity is the caller's documented obligation.
    let status = unsafe { cnot103_cuda_synchronize(stream) };
    if status == 0 {
        Ok(())
    } else {
        Err("CUDA stream synchronization failed")
    }
}
