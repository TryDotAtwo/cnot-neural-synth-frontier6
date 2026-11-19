// Exact sparse and permutation fixtures expose no-pivot iterations.
#include "cnot103/cnot_cuda.h"
#include "../src/packed_block_scratch.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    std::size_t size;
    explicit DeviceBuffer(std::size_t count) : size(count) {
        checked(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    void upload(const std::vector<T>& values) {
        checked(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void equals(const std::vector<T>& expected) {
        std::vector<T> values(size);
        checked(cudaMemcpy(values.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
        if (values != expected) throw std::runtime_error("literal rank/status mismatch");
    }
};
void launched(int status) {
    if (status != 0) throw std::runtime_error("rank launch rejected");
    checked(cudaDeviceSynchronize());
}

__global__ void verify_block_slots(std::uint64_t* out, std::uint64_t tag,
                                   cnot103::PackedBlockScratch scratch) {
    extern __shared__ std::uint64_t shared[];
    auto* rows = scratch.rows(shared);
    if (threadIdx.x == 0) rows[0] = tag + scratch.item();
    if (threadIdx.x == 1) rows[scratch.words - 1] = tag + 3 * scratch.item();
    __syncthreads();
    if (threadIdx.x == 0) out[scratch.item()] = rows[0] + rows[scratch.words - 1];
}

void scratch_chunks_and_streams() {
    // The normal kernel budget is 64 MiB. A small test budget forces the exact
    // same chunk-offset and reuse path with only seven independent matrices.
    constexpr unsigned count = 7;
    DeviceBuffer<std::uint64_t> a(count), b(count);
    cudaStream_t left, right;
    checked(cudaStreamCreateWithFlags(&left, cudaStreamNonBlocking));
    checked(cudaStreamCreateWithFlags(&right, cudaStreamNonBlocking));
    auto dispatch = [&](std::uint64_t words, std::uint64_t tag, cudaStream_t stream,
                        DeviceBuffer<std::uint64_t>& output, bool global) {
        std::vector<unsigned> chunks;
        bool layout_ok = true;
        const int code = cnot103::launch_packed_blocks<100000>(count, words, stream,
            [&](unsigned n, std::size_t bytes, cnot103::PackedBlockScratch scratch) {
                layout_ok &= ((scratch.global != nullptr) == global && (bytes == 0) == global);
                chunks.push_back(n);
                verify_block_slots<<<n, 32, bytes, stream>>>(output.data, tag, scratch);
            });
        if (!layout_ok || code || chunks != (global ? std::vector<unsigned>{2,2,2,1} : std::vector<unsigned>{7}))
            throw std::runtime_error("wrong packed scratch chunk schedule");
    };
    dispatch(6112, 50, left, a, false);
    dispatch(6113, 3, left, a, true);
    dispatch(6113, 97, right, b, true);
    checked(cudaStreamSynchronize(left)); checked(cudaStreamSynchronize(right));
    std::vector<std::uint64_t> first, second;
    for (unsigned i = 0; i < count; ++i) { first.push_back(6 + 4*i); second.push_back(194 + 4*i); }
    a.equals(first); b.equals(second);
    checked(cudaStreamDestroy(left)); checked(cudaStreamDestroy(right));
    std::puts("6112 shared words / 6113 global words, 2+2+2+1 chunks and two nonblocking streams PASS");
}

void fixture(std::uint32_t n) {
    const auto words = (n + 63u) / 64u;
    std::vector<std::uint64_t> states(static_cast<std::size_t>(3) * n * words, 0);
    for (std::uint32_t state = 0; state < 3; ++state) {
        for (std::uint32_t row = 0; row < n; ++row) {
            const auto column = state == 2 ? (row + 1) % n : row;
            states[(static_cast<std::size_t>(state) * n + row) * words + column / 64u] |= std::uint64_t{1} << (column % 64u);
        }
    }
    states[(static_cast<std::size_t>(n) + n - 1) * words] ^= 1;
    std::vector<std::uint64_t> full(words, ~std::uint64_t{0});
    if (n % 64u != 0u) full.back() = (std::uint64_t{1} << (n % 64u)) - 1;
    DeviceBuffer<std::uint64_t> matrices(states.size()), masks(words), empty(words);
    matrices.upload(states); masks.upload(full); empty.upload(std::vector<std::uint64_t>(words, 0));
    DeviceBuffer<float> values(3);
    DeviceBuffer<std::uint8_t> status(3);
    std::printf("rank fixture launch n=%u\n", n); std::fflush(stdout);
    launched(cnot103_cuda_launch_residual_rank(matrices.data, 3, n, words, values.data, 3, nullptr));
    values.equals({0.0f, 1.0f, static_cast<float>(n - 1)});
    launched(cnot103_cuda_launch_projected_ranks(matrices.data, masks.data, masks.data, nullptr, nullptr, nullptr,
        3, 1, n, words, 0, 0, 0, values.data, 3, nullptr));
    values.equals({static_cast<float>(n), static_cast<float>(n), static_cast<float>(n)});
    launched(cnot103_cuda_launch_projected_ranks(matrices.data, masks.data, masks.data, nullptr, nullptr, nullptr,
        3, 1, n, words, 1, 0, 0, values.data, 3, nullptr));
    values.equals({0.0f, 1.0f, static_cast<float>(n - 1)});
    launched(cnot103_cuda_launch_projected_ranks(matrices.data, empty.data, masks.data, nullptr, nullptr, nullptr,
        3, 1, n, words, 0, 0, 0, values.data, 3, nullptr));
    values.equals({0.0f, 0.0f, 0.0f});
    launched(cnot103_cuda_launch_partial_shear_ranks(matrices.data, 3, n, words, values.data, status.data, 3, nullptr));
    values.equals({0.0f, 1.0f, 0.0f});
    status.equals({0, 1, 0});
    std::printf("rank fixture complete n=%u\n", n); std::fflush(stdout);
}

int main() {
    try {
        scratch_chunks_and_streams();
        for (const auto n : {4u, 65u, 144u, 705u}) fixture(n);
        checked(cudaDeviceReset());
        std::puts("rank literal fixtures PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
