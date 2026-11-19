#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace diffusion_test {
using U32 = std::uint32_t;
using U64 = std::uint64_t;
using Byte = std::uint8_t;

inline void require(bool ok, const char* reason) {
    if (!ok)
        throw std::runtime_error(reason);
}
inline void check(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}

template <class T> struct Buffer {
    T* data = nullptr;
    size_t size;
    explicit Buffer(size_t n) : size(n) {
        if (n)
            check(cudaMalloc(&data, n * sizeof(T)));
    }
    Buffer(const Buffer&) = delete;
    ~Buffer() {
        if (data)
            cudaFree(data);
    }
    void upload(const std::vector<T>& values) {
        require(values.size() == size, "fixture upload size");
        check(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> read(size_t offset, size_t count) const {
        require(offset <= size && count <= size - offset, "fixture read size");
        std::vector<T> result(count);
        if (count)
            check(cudaMemcpy(result.data(), data + offset, count * sizeof(T),
                             cudaMemcpyDeviceToHost));
        return result;
    }
};

} // namespace diffusion_test
