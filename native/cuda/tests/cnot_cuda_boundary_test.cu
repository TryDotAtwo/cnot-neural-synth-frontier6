// Independent literal fixtures for focused sanitizer runs, without Torch.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template <typename T> struct Buffer {
    T* pointer = nullptr;
    std::size_t count;
    explicit Buffer(std::size_t size) : count(size) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
    }
    ~Buffer() { cudaFree(pointer); }
    void upload(const std::vector<T>& value) {
        if (value.size() != count) throw std::runtime_error("upload shape mismatch");
        check(cudaMemcpy(pointer, value.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() {
        std::vector<T> result(count);
        check(cudaMemcpy(result.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
};
template <typename T> void expect(const std::vector<T>& actual, const std::vector<T>& expected,
                                 const char* name) {
    if (actual.size() != expected.size()) throw std::runtime_error("comparison shape mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) throw std::runtime_error(std::string(name) + " differs at " + std::to_string(i));
    }
}

void fixture(std::uint32_t n, bool wide, bool protected_inputs) {
    const std::uint32_t words = (n + 63u) / 64u;
    const std::uint32_t inputs = wide ? n : 2;
    const std::uint32_t outputs = wide ? n : 2;
    const std::uint32_t equations = outputs + (wide ? 2 : 1);
    const std::uint32_t batch = wide && n < 448 ? 1 : 2;
    const std::uint32_t ports = inputs + outputs;
    const std::uint64_t groups = static_cast<std::uint64_t>(batch) * equations;
    const std::uint64_t incidence_count = groups * n * 3;
    std::vector<std::uint64_t> matrices(static_cast<std::size_t>(batch) * n * words, 0);
    for (std::uint32_t state = 0; state < batch; ++state) {
        for (std::uint32_t row = 0; row < n; ++row)
            matrices[(static_cast<std::size_t>(state) * n + row) * words + row / 64] |= std::uint64_t{1} << (row % 64);
    }
    if (!wide) {
        matrices[(static_cast<std::size_t>(n) + n - 2) * words] ^= 1;
        matrices[(static_cast<std::size_t>(n) + n - 1) * words] ^= 3;
    } else if (batch == 2) {
        matrices[(static_cast<std::size_t>(n) + 1) * words] ^= 4;
    }
    std::vector<std::uint32_t> wires;
    for (std::uint32_t i = 0; i < inputs; ++i) wires.push_back(i);
    for (std::uint32_t i = 0; i < outputs; ++i) wires.push_back(wide ? i : n - 2 + i);
    std::vector<std::uint8_t> protected_flags(ports, 0);
    for (std::uint32_t i = 0; i < inputs; ++i) protected_flags[i] = protected_inputs ? 1 : 0;
    std::vector<std::uint32_t> eq_outputs;
    for (std::uint32_t i = 0; i < outputs; ++i) eq_outputs.push_back(i);
    if (wide) eq_outputs.push_back(0);
    eq_outputs.push_back(outputs - 1);
    std::vector<float> expected_values(groups * 3, 0), expected_incidence(incidence_count, 0), expected_ranks(batch * 2, 0);
    std::vector<std::uint8_t> expected_valid(groups, 0);
    auto input = [&](std::uint32_t state, std::uint32_t row, std::uint32_t wire) {
        const auto offset = ((static_cast<std::uint64_t>(state) * equations + row) * n + wire) * 3;
        expected_incidence[offset] = 1;
        if (protected_inputs) expected_incidence[offset + 2] = 1;
    };
    auto output = [&](std::uint32_t state, std::uint32_t row, std::uint32_t wire) {
        expected_incidence[((static_cast<std::uint64_t>(state) * equations + row) * n + wire) * 3 + 1] = 1;
    };
    for (std::uint32_t state = 0; state < batch; ++state) {
        const auto rank = wide ? n + 2 : (state == 0 ? 2u : 3u);
        expected_ranks[state * 2] = static_cast<float>(rank);
        expected_ranks[state * 2 + 1] = protected_inputs ? static_cast<float>(wide ? 2 : state) : 0;
        for (std::uint32_t row = 0; row < rank; ++row) {
            const auto group = static_cast<std::uint64_t>(state) * equations + row;
            expected_valid[group] = 1;
            expected_values[group * 3] = static_cast<float>(rank);
            expected_values[group * 3 + 1] = static_cast<float>(ports - rank);
        }
        if (wide) {
            // Canonical pivots: outputs n-1..0, then inputs n-1 and 0.
            for (std::uint32_t row = 0; row < n; ++row) {
                const auto wire = n - 1 - row;
                output(state, row, wire);
                if (wire != 0 && wire != n - 1) input(state, row, wire);
            }
            input(state, n, n - 1);
            input(state, n + 1, 0);
            if (state == 1) input(state, n - 2, 2);
        } else {
            // Identity rows: [y_last, y_previous, 0]. Changed rows:
            // [y_last, x0+y_previous, x0+x1], in descending pivot order.
            output(state, 0, n - 1);
            output(state, 1, n - 2);
            if (state == 1) { input(state, 1, 0); input(state, 2, 0); input(state, 2, 1); }
        }
    }
    Buffer<std::uint64_t> state_device(matrices.size()); state_device.upload(matrices);
    Buffer<std::uint32_t> wires_device(wires.size()); wires_device.upload(wires);
    Buffer<std::uint8_t> protected_device(protected_flags.size()); protected_device.upload(protected_flags);
    Buffer<std::uint32_t> equations_device(eq_outputs.size()); equations_device.upload(eq_outputs);
    Buffer<float> values(groups * 3), incidence(incidence_count), ranks(batch * 2);
    Buffer<std::uint8_t> valid(groups);
    std::printf("boundary fixture launch n=%u wide=%u protected=%u\n", n, unsigned(wide), unsigned(protected_inputs));
    std::fflush(stdout);
    const auto status = cnot103_cuda_launch_boundary_relations(
        state_device.pointer, wires_device.pointer, protected_device.pointer, equations_device.pointer,
        batch, n, words, inputs, outputs, equations, values.pointer, valid.pointer, incidence.pointer,
        ranks.pointer, groups, incidence_count, nullptr);
    if (status != 0) throw std::runtime_error("boundary launch failed: " + std::to_string(status));
    check(cudaDeviceSynchronize());
    expect(values.download(), expected_values, "values");
    expect(valid.download(), expected_valid, "valid");
    expect(incidence.download(), expected_incidence, "incidence");
    expect(ranks.download(), expected_ranks, "ranks");
    std::printf("boundary fixture complete n=%u\n", n);
    std::fflush(stdout);
}

int main() {
    try {
        fixture(4, false, false);
        fixture(4, false, true);
        fixture(65, false, true);
        fixture(65, true, true);
        fixture(144, true, true);
        fixture(448, true, true);
        check(cudaDeviceReset());
        std::puts("boundary literal fixtures PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
