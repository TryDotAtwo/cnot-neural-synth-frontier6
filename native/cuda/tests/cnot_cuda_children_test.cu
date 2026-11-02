#include "cnot103/cnot_cuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

std::uint64_t required_words(std::uint32_t batch, std::uint32_t dimension,
                            std::uint32_t words_per_row) {
    return static_cast<std::uint64_t>(batch) * dimension * (dimension - 1u) * dimension *
           words_per_row;
}

bool verify_dimension(std::uint32_t dimension) {
    constexpr std::uint32_t batch = 2;
    const std::uint32_t words_per_row = dimension / 64u + (dimension % 64u != 0u);
    const std::uint64_t actions = static_cast<std::uint64_t>(dimension) * (dimension - 1u);
    const std::size_t input_words = static_cast<std::size_t>(batch) * dimension * words_per_row;
    const std::uint64_t output_word_count = required_words(batch, dimension, words_per_row);

    std::vector<std::uint64_t> input(input_words, 0);
    for (std::uint32_t state = 0; state < batch; ++state) {
        for (std::uint32_t row = 0; row < dimension; ++row) {
            input[(static_cast<std::size_t>(state) * dimension + row) * words_per_row + row / 64u] |=
                std::uint64_t{1} << (row % 64u);
        }
        for (std::uint32_t step = 0; step < dimension + 3u + state; ++step) {
            const std::uint32_t control = (step * 17u + state + 1u) % dimension;
            std::uint32_t target = (step * 29u + state + 2u) % dimension;
            if (target == control) target = (target + 1u) % dimension;
            for (std::uint32_t word = 0; word < words_per_row; ++word) {
                auto& target_word = input[(static_cast<std::size_t>(state) * dimension + target) *
                                              words_per_row +
                                          word];
                const auto source_word = input[(static_cast<std::size_t>(state) * dimension + control) *
                                                   words_per_row +
                                               word];
                target_word ^= source_word;
            }
        }
    }

    std::vector<std::uint64_t> expected(static_cast<std::size_t>(output_word_count));
    for (std::uint32_t state = 0; state < batch; ++state) {
        for (std::uint32_t control = 0; control < dimension; ++control) {
            for (std::uint32_t target_offset = 0; target_offset < dimension - 1u; ++target_offset) {
                const std::uint32_t target = target_offset < control ? target_offset : target_offset + 1u;
                const std::uint64_t action = static_cast<std::uint64_t>(control) * (dimension - 1u) +
                                             target_offset;
                for (std::uint32_t row = 0; row < dimension; ++row) {
                    for (std::uint32_t word = 0; word < words_per_row; ++word) {
                        const auto source = input[(static_cast<std::size_t>(state) * dimension + row) *
                                                      words_per_row +
                                                  word];
                        const auto control_row = input[(static_cast<std::size_t>(state) * dimension + control) *
                                                           words_per_row +
                                                       word];
                        const std::uint64_t value = row == target ? source ^ control_row : source;
                        const std::uint64_t offset =
                            (((static_cast<std::uint64_t>(state) * actions + action) * dimension + row) *
                             words_per_row) +
                            word;
                        expected[static_cast<std::size_t>(offset)] = value;
                    }
                }
            }
        }
    }

    std::uint64_t* device_input = nullptr;
    std::uint64_t* device_output = nullptr;
    const auto input_bytes = input.size() * sizeof(std::uint64_t);
    const auto output_bytes = expected.size() * sizeof(std::uint64_t);
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes), "cudaMalloc input")) {
        return false;
    }
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes), "cudaMalloc output")) {
        cudaFree(device_input);
        return false;
    }
    bool ok = check_cuda(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice),
                         "copy input to GPU");
    if (ok) {
        const auto status = cnot103_cuda_launch_children(
            device_input, batch, dimension, words_per_row, device_output, output_word_count, nullptr);
        if (status != 0) {
            std::cerr << "launch failed for dimension " << dimension << ": " << status << '\n';
            ok = false;
        }
    }
    if (ok && cnot103_cuda_synchronize(nullptr) != 0) {
        std::cerr << "stream synchronization failed for dimension " << dimension << '\n';
        ok = false;
    }
    std::vector<std::uint64_t> actual(expected.size());
    if (ok) {
        ok = check_cuda(cudaMemcpy(actual.data(), device_output, output_bytes, cudaMemcpyDeviceToHost),
                        "copy output from GPU");
    }
    if (!ok) {
        cudaFree(device_output);
        cudaFree(device_input);
        return false;
    }

    if (actual != expected) {
        for (std::size_t index = 0; index < actual.size(); ++index) {
            if (actual[index] != expected[index]) {
                std::cerr << "dimension " << dimension << " first mismatch at word " << index
                          << ": expected " << expected[index] << ", got " << actual[index] << '\n';
                break;
            }
        }
        cudaFree(device_output);
        cudaFree(device_input);
        return false;
    }

    if (cnot103_cuda_launch_children(device_input, batch, dimension, words_per_row,
                                     device_output, output_word_count - 1u, nullptr) != 1) {
        std::cerr << "insufficient output capacity was not rejected\n";
        cudaFree(device_output);
        cudaFree(device_input);
        return false;
    }
    if (cnot103_cuda_launch_children(device_input, batch, dimension, words_per_row + 1u,
                                     device_output, output_word_count, nullptr) != 1) {
        std::cerr << "invalid packed row width was not rejected\n";
        cudaFree(device_output);
        cudaFree(device_input);
        return false;
    }
    cudaFree(device_output);
    cudaFree(device_input);
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    const auto device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA device unavailable: " << cudaGetErrorString(device_status) << '\n';
        return 2;
    }

    for (const std::uint32_t dimension : {2u, 3u, 7u, 64u, 65u}) {
        if (!verify_dimension(dimension)) return 1;
    }
    std::cout << "CUDA CNOT successor kernel matches CPU oracle for dimensions 2, 3, 7, 64, 65\n";
    return 0;
}
