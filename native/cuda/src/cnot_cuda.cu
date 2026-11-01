#include "cnot103/cnot_cuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr std::int32_t kOk = 0;
constexpr std::int32_t kInvalidArgument = 1;
constexpr std::int32_t kSizeOverflow = 2;
constexpr std::int32_t kLaunchFailure = 3;
constexpr std::int32_t kSyncFailure = 4;

__global__ void expand_children_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    std::uint64_t total_words,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint64_t actions_per_state) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_words) return;

    std::uint64_t cursor = index;
    const std::uint32_t word = static_cast<std::uint32_t>(cursor % words_per_row);
    cursor /= words_per_row;
    const std::uint32_t row = static_cast<std::uint32_t>(cursor % dimension);
    cursor /= dimension;
    const std::uint64_t action = cursor % actions_per_state;
    const std::uint64_t batch_index = cursor / actions_per_state;

    const std::uint32_t control = static_cast<std::uint32_t>(action / (dimension - 1));
    const std::uint32_t target_offset = static_cast<std::uint32_t>(action % (dimension - 1));
    const std::uint32_t target = target_offset < control ? target_offset : target_offset + 1;

    const std::uint64_t state_base = batch_index * dimension * words_per_row;
    const std::uint64_t source_index = state_base + static_cast<std::uint64_t>(control) * words_per_row + word;
    const std::uint64_t row_index = state_base + static_cast<std::uint64_t>(row) * words_per_row + word;
    std::uint64_t value = input[row_index];
    if (row == target) value ^= input[source_index];
    output[index] = value;
}

bool checked_mul(std::uint64_t left, std::uint64_t right, std::uint64_t* output) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) return false;
    *output = left * right;
    return true;
}

}  // namespace

extern "C" std::int32_t cnot103_cuda_launch_children(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint64_t* output_device,
    std::uint64_t output_capacity_words,
    void* stream) {
    if (input_device == nullptr || output_device == nullptr || batch == 0 || dimension < 2 ||
        words_per_row != dimension / 64u + static_cast<std::uint32_t>(dimension % 64u != 0u)) {
        return kInvalidArgument;
    }

    const std::uint64_t actions = static_cast<std::uint64_t>(dimension) * (dimension - 1u);
    std::uint64_t total_words = batch;
    if (!checked_mul(total_words, actions, &total_words) ||
        !checked_mul(total_words, dimension, &total_words) ||
        !checked_mul(total_words, words_per_row, &total_words)) {
        return kSizeOverflow;
    }
    if (output_capacity_words < total_words) return kInvalidArgument;

    constexpr std::uint32_t kThreads = 256;
    const std::uint64_t blocks64 = total_words / kThreads + static_cast<std::uint64_t>(total_words % kThreads != 0u);
    if (blocks64 == 0 || blocks64 > std::numeric_limits<unsigned int>::max()) return kSizeOverflow;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    expand_children_kernel<<<static_cast<unsigned int>(blocks64), kThreads, 0, cuda_stream>>>(
        input_device, output_device, total_words, dimension, words_per_row, actions);
    return cudaGetLastError() == cudaSuccess ? kOk : kLaunchFailure;
}

extern "C" std::int32_t cnot103_cuda_synchronize(void* stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return cudaStreamSynchronize(cuda_stream) == cudaSuccess ? kOk : kSyncFailure;
}
