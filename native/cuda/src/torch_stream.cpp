#include "cnot103/cnot_cuda.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

extern "C" std::int32_t cnot103_cuda_exchange_device(
    std::int32_t device_index, std::int32_t* previous) {
    if (device_index < 0 || !previous) return 1;
    int current;
    if (cudaGetDevice(&current) != cudaSuccess) return 2;
    if (current != device_index && cudaSetDevice(device_index) != cudaSuccess) return 2;
    *previous = current;
    return 0;
}

extern "C" std::int32_t cnot103_cuda_current_stream(
    std::int32_t device_index,
    void** stream) {
    if (device_index < 0 || stream == nullptr) {
        return 1;
    }

    try {
        *stream = c10::cuda::getCurrentCUDAStream(device_index).stream();
        return 0;
    } catch (...) {
        return 2;
    }
}
