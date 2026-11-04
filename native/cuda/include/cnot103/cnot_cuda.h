#pragma once

#include <cstdint>

#if defined(_WIN32)
#  define CNOT103_CUDA_API __declspec(dllexport)
#else
#  define CNOT103_CUDA_API __attribute__((visibility("default")))
#endif

extern "C" {

// Both matrix buffers are caller-owned CUDA device allocations. Input has shape
// [batch, dimension, words_per_row], row-major, with bit j of row i representing
// matrix entry (i,j) and unused high bits of each final word zero. Output has shape
// [batch, dimension*(dimension-1), dimension, words_per_row]; words are contiguous,
// and actions are control-major with the diagonal target omitted. Every output word
// has one writer. Input/output allocations must not overlap.
//
// Launch is asynchronous. Both allocations and the stream must remain valid until
// cnot103_cuda_synchronize(stream) completes; input must not be modified, and output
// must not be read or reused before completion. output_capacity_words is checked
// against the full output size, but the caller remains responsible for truthful
// device-allocation capacities.
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
