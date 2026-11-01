#pragma once

#include <cstdint>

#if defined(_WIN32)
#  define CNOT103_CUDA_API __declspec(dllexport)
#else
#  define CNOT103_CUDA_API __attribute__((visibility("default")))
#endif

extern "C" {

// All matrix buffers are caller-owned CUDA device allocations. State words are
// row-major; bit j of row i is matrix entry (i,j), with bit 0 in the low bit.
// Output is [batch, ordered_actions, rows, words_per_row]. Ordered actions use
// control-major order and omit the diagonal target for each control.
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_children(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint64_t* output_device,
    std::uint64_t output_capacity_words,
    void* stream);

// Explicit completion point for asynchronous launch errors.
CNOT103_CUDA_API std::int32_t cnot103_cuda_synchronize(void* stream);

}
