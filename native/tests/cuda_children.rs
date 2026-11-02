use cnot103_native::cuda;
use std::ffi::c_void;
use tch::{Device, Kind, Tensor};

#[test]
fn rust_cuda_ffi_writes_the_ordered_cnot_successors_on_device() {
    if let Err(error) = cnot103_native::torch_device::require_cuda_device(0) {
        assert!(
            std::env::var_os("CNOT103_REQUIRE_CUDA").is_none(),
            "CNOT103_REQUIRE_CUDA is set, but CUDA initialization failed: {error}"
        );
        eprintln!("CUDA unavailable; skipping the Rust CUDA FFI smoke test: {error}");
        return;
    }

    let dimension = 3u32;
    let words_per_row = 1u32;
    let rows = [0b011i64, 0b101, 0b110];
    let input = Tensor::from_slice(&rows)
        .to_device(Device::Cuda(0))
        .contiguous();
    let actions = dimension * (dimension - 1);
    let output_words = actions * dimension * words_per_row;
    let output = Tensor::empty([i64::from(output_words)], (Kind::Int64, Device::Cuda(0)));

    // Ensure the tensor copy to the device has finished before the project
    // kernel uses the allocation on CUDA's default stream.
    tch::Cuda::synchronize(0);
    let written = unsafe {
        cuda::launch_all_to_all_children(
            input.data_ptr().cast::<u64>(),
            output.data_ptr().cast::<u64>(),
            1,
            dimension,
            words_per_row,
            u64::from(output_words),
            std::ptr::null_mut::<c_void>(),
        )
    }
    .unwrap();
    assert_eq!(written, u64::from(output_words));
    unsafe { cuda::synchronize(std::ptr::null_mut::<c_void>()) }.unwrap();

    let actual = output.to_device(Device::Cpu).contiguous();
    let actual: Vec<_> = (0..i64::from(output_words))
        .map(|index| actual.int64_value(&[index]))
        .collect();
    let mut expected = Vec::with_capacity(output_words as usize);
    for control in 0..dimension {
        for target_offset in 0..dimension - 1 {
            let target = if target_offset < control {
                target_offset
            } else {
                target_offset + 1
            };
            expected.extend(rows.iter().enumerate().map(|(row, value)| {
                if row as u32 == target {
                    value ^ rows[control as usize]
                } else {
                    *value
                }
            }));
        }
    }
    assert_eq!(actual, expected);
}
