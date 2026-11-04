use cnot103_native::cuda;
use std::ffi::c_void;
use tch::{Device, Kind, Tensor};

fn input_state(dimension: u32, seed: u32) -> Vec<u64> {
    let words_per_row = dimension.div_ceil(64) as usize;
    let mut rows = vec![0u64; dimension as usize * words_per_row];
    for row in 0..dimension as usize {
        rows[row * words_per_row + row / 64] |= 1u64 << (row % 64);
    }

    for step in 0..dimension + 3 + seed {
        let control = (step * 17 + seed + 1) % dimension;
        let mut target = (step * 29 + seed + 2) % dimension;
        if target == control {
            target = (target + 1) % dimension;
        }
        let control_base = control as usize * words_per_row;
        let target_base = target as usize * words_per_row;
        for word in 0..words_per_row {
            rows[target_base + word] ^= rows[control_base + word];
        }
    }
    rows
}

fn cpu_children(states: &[Vec<u64>], dimension: u32, words_per_row: u32) -> Vec<u64> {
    let words_per_row = words_per_row as usize;
    let mut output = Vec::new();
    for rows in states {
        for control in 0..dimension {
            for target_offset in 0..dimension - 1 {
                let target = if target_offset < control {
                    target_offset
                } else {
                    target_offset + 1
                };
                for row in 0..dimension {
                    for word in 0..words_per_row {
                        let value = rows[row as usize * words_per_row + word];
                        output.push(if row == target {
                            value ^ rows[control as usize * words_per_row + word]
                        } else {
                            value
                        });
                    }
                }
            }
        }
    }
    output
}

#[test]
fn rust_cuda_ffi_writes_ordered_cnot_successors_across_packed_widths() {
    if let Err(error) = cnot103_native::torch_device::require_cuda_device(0) {
        assert!(
            std::env::var_os("CNOT103_REQUIRE_CUDA").is_none(),
            "CNOT103_REQUIRE_CUDA is set, but CUDA initialization failed: {error}"
        );
        eprintln!("CUDA unavailable; skipping the Rust CUDA FFI test: {error}");
        return;
    }

    let batch = 2u32;
    for dimension in [2u32, 3, 7, 64, 65] {
        let words_per_row = dimension.div_ceil(64);
        let states = (0..batch)
            .map(|seed| input_state(dimension, seed))
            .collect::<Vec<_>>();
        let flat_input = states.iter().flatten().copied().collect::<Vec<_>>();
        let input_i64 = flat_input
            .iter()
            .map(|word| *word as i64)
            .collect::<Vec<_>>();
        let expected = cpu_children(&states, dimension, words_per_row);
        let output_words = expected.len();
        let output_capacity = output_words as u64;

        let input = Tensor::from_slice(&input_i64)
            .to_device(Device::Cuda(0))
            .contiguous();
        let output = Tensor::empty([output_words as i64], (Kind::Int64, Device::Cuda(0)));

        // Ensure the tensor copy to the device has finished before the project
        // kernel uses the allocation on CUDA's default stream.
        tch::Cuda::synchronize(0);
        let written = unsafe {
            cuda::launch_all_to_all_children(
                input.data_ptr().cast::<u64>(),
                output.data_ptr().cast::<u64>(),
                batch,
                dimension,
                words_per_row,
                output_capacity,
                std::ptr::null_mut::<c_void>(),
            )
        }
        .unwrap();
        assert_eq!(written, output_capacity, "dimension {dimension}");
        unsafe { cuda::synchronize(std::ptr::null_mut::<c_void>()) }.unwrap();

        let actual_tensor = output.to_device(Device::Cpu).contiguous();
        let mut actual_i64 = vec![0i64; output_words];
        actual_tensor.copy_data(&mut actual_i64, output_words);
        let actual = actual_i64
            .into_iter()
            .map(|word| word as u64)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "dimension {dimension}");

        if dimension == 3 {
            assert!(unsafe {
                cuda::launch_all_to_all_children(
                    input.data_ptr().cast::<u64>(),
                    output.data_ptr().cast::<u64>(),
                    batch,
                    dimension,
                    words_per_row,
                    output_capacity - 1,
                    std::ptr::null_mut::<c_void>(),
                )
            }
            .is_err());
            assert!(unsafe {
                cuda::launch_all_to_all_children(
                    input.data_ptr().cast::<u64>(),
                    output.data_ptr().cast::<u64>(),
                    batch,
                    dimension,
                    words_per_row + 1,
                    output_capacity,
                    std::ptr::null_mut::<c_void>(),
                )
            }
            .is_err());
        }
    }
}
