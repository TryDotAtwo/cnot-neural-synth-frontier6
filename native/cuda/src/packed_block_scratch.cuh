#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cnot103 {
struct PackedBlockScratch {
    std::uint64_t* global;
    std::uint64_t words;
    std::uint64_t first;

    __device__ std::uint64_t item() const { return first + blockIdx.x; }
    __device__ std::uint64_t* rows(std::uint64_t* shared) const {
        return global ? global + static_cast<std::uint64_t>(blockIdx.x) * words : shared;
    }
};

// One CTA owns one packed matrix. Small matrices use shared memory. A large
// batch leases a single flat region from the caller stream's CUDA memory pool.
// Later chunks reuse that region only after the previous kernel in that stream.
// One very large matrix can exceed the normal 64 MiB scheduling budget.
template <std::size_t GlobalBudget = 64ull << 20, typename Launch>
std::int32_t launch_packed_blocks(std::uint64_t count, std::uint64_t words,
                                 cudaStream_t stream, Launch launch) {
    if (!words || !count || words > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t))
        return 2;
    const auto bytes = static_cast<std::size_t>(words * sizeof(std::uint64_t));
    // Reserve one alignment unit for each kernel's static pivot counters.
    constexpr std::size_t shared_limit = 48 * 1024 - 256;
    const bool use_global = bytes > shared_limit;
    std::uint64_t workers = std::min<std::uint64_t>(count, 0x7fffffffu);
    std::uint64_t* global = nullptr;
    if (use_global) {
        workers = std::min<std::uint64_t>(workers, std::max<std::size_t>(1, GlobalBudget / bytes));
        if (cudaMallocAsync(reinterpret_cast<void**>(&global), workers * bytes, stream) != cudaSuccess)
            return 3;
    }
    cudaError_t status = cudaSuccess;
    for (std::uint64_t first = 0; first < count; first += workers) {
        const auto active = static_cast<unsigned int>(std::min(workers, count - first));
        launch(active, use_global ? 0 : bytes, PackedBlockScratch{global, words, first});
        status = cudaGetLastError();
        if (status != cudaSuccess) break;
    }
    const auto released = global ? cudaFreeAsync(global, stream) : cudaSuccess;
    return status == cudaSuccess && released == cudaSuccess ? 0 : 3;
}
} // namespace cnot103
