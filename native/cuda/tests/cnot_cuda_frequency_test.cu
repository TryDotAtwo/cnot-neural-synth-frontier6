// Independent literal frequency ranks and every Stage52 interface guard.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using Word = std::uint64_t;
using Index = std::uint32_t;
void checked(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b, const char* error) { if (!b) throw std::runtime_error(error); }
template<typename T> struct Buffer {
    T* data = nullptr; std::size_t size;
    explicit Buffer(std::size_t n) : size(n) { checked(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
    void upload(const std::vector<T>& values) {
        require(values.size() == size, "upload extent mismatch");
        checked(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void equals(const std::vector<T>& expected, const char* message) {
        std::vector<T> values(size); checked(cudaMemcpy(values.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
        require(values == expected, message);
    }
};
void fixture(Index n, Index r, Index t, Index mode) {
    const Index m = n - r - t, w = (n + 63) / 64, rw = (r + 63) / 64, tw = (t + 63) / 64;
    std::vector<Index> order(n);
    for (Index i = 0; i < n; ++i) order[i] = n - 1 - i;
    std::rotate(order.begin(), order.begin() + 2, order.end());
    const Index source = order[0], source_last = order[r - 1];
    const Index inner = order[r], inner_last = order[r + t - 1];
    const Index target = order[r + t], target_next = order[r + t + 1];
    std::vector<std::uint8_t> roles(n, 0);
    for (Index i = r; i < r + t; ++i) roles[order[i]] = 1;
    for (Index i = r + t; i < n; ++i) roles[order[i]] = 2;
    std::vector<Index> internal(order.begin() + r, order.begin() + r + t);
    std::vector<Word> central(Word{m} * rw, 0);
    if (mode) {
        central[(r - 1) / 64] = Word{1} << ((r - 1) % 64);
        central[rw + (r - 1) / 64] = Word{1} << ((r - 1) % 64);
        if (mode == 2) { std::fill(central.begin() + rw, central.begin() + 2 * rw, 0); central[rw] = 1; }
    }
    const Index kappa = mode == 3 ? std::min(m, r) : mode;
    if (mode == 3) {
        std::fill(central.begin(), central.end(), 0);
        for (Index i = 0; i < kappa; ++i) central[Word{i} * rw + i / 64] = Word{1} << (i % 64);
    }
    Buffer<Word> dcentral(central.size()), rank_scratch(central.size()), statistics(2);
    dcentral.upload(central);
    require(cnot103_cuda_prepare_frequency(dcentral.data, rank_scratch.data, statistics.data, m, r, t, rank_scratch.size, nullptr) == 0, "frequency preparation failed");
    checked(cudaDeviceSynchronize());
    statistics.equals({kappa, Word{t} * kappa}, "literal kappa/exponent mismatch");
    dcentral.equals(central, "immutable central frequency changed");
    const Word cells = Word{n} * w;
    std::vector<Word> input(7 * cells, 0);
    for (Word sample = 0; sample < 7; ++sample) for (Index row = 0; row < n; ++row)
        input[sample * cells + Word{row} * w + row / 64] |= Word{1} << (row % 64);
    const auto gate = [&](Word sample, Index control, Index destination) {
        for (Index k = 0; k < w; ++k) input[sample * cells + Word{destination} * w + k] ^= input[sample * cells + Word{control} * w + k];
    };
    gate(1, source, inner); gate(1, inner, inner_last); gate(1, inner_last, target); gate(1, source_last, target_next);
    // Invertible permutation inside the internal block forces row exchanges.
    gate(1, inner, inner_last); gate(1, inner_last, inner); gate(1, inner, inner_last);
    gate(2, inner, source); gate(3, target, inner); gate(4, target, target_next);
    input[5 * cells + Word{target} * w + target / 64] = 0; // target diagonal missing
    input[6 * cells + Word{inner} * w + inner / 64] = 0; // singular internal block
    Buffer<Word> states(input.size()), scratch(7 * Word{t} * tw);
    Buffer<std::uint8_t> droles(n), status(7);
    Buffer<Index> dinternal(t);
    Buffer<float> values(14);
    states.upload(input); droles.upload(roles); dinternal.upload(internal);
    Cnot103FrequencyArgs a{};
    a.states = states.data; a.roles = droles.data; a.internal_wires = dinternal.data; a.statistics = statistics.data;
    a.scratch = scratch.data; a.status = status.data; a.values = values.data; a.scratch_words = scratch.size;
    a.batch = 7; a.dimension = n; a.words_per_row = w; a.internal_count = t;
    require(cnot103_cuda_launch_frequency(&a) == 0, "I07 evaluation failed");
    checked(cudaDeviceSynchronize());
    status.equals({1,1,2,2,2,2,2}, "I07 interface statuses differ");
    std::vector<float> expected(14, 0);
    for (Index sample = 0; sample < 2; ++sample) { expected[2 * sample] = kappa; expected[2 * sample + 1] = t * kappa; }
    values.equals(expected, "I07 interface values differ");
    states.equals(input, "I07 changed the input states");
    // Inject prepared statistics only to exercise exact-f32 status boundaries;
    // these are ABI guard tests, not claims about this small context's rank.
    for (const auto& stats : {std::vector<Word>{(Word{1} << 24) + 1, 0}, std::vector<Word>{0, (Word{1} << 24) + 1}}) {
        statistics.upload(stats);
        require(cnot103_cuda_launch_frequency(&a) == 0, "I07 range-guard launch");
        checked(cudaDeviceSynchronize());
        status.equals({4,4,2,2,2,2,2}, "I07 exact-f32 range guard priority differs");
        values.equals(std::vector<float>(14, 0), "unavailable I07 payload was not zero");
    }
    statistics.upload({Word{1} << 24, Word{1} << 24});
    require(cnot103_cuda_launch_frequency(&a) == 0, "I07 exact-boundary launch");
    checked(cudaDeviceSynchronize());
    status.equals({1,1,2,2,2,2,2}, "exact-f32 boundary should be admitted");
    for (Index k = 0; k < 4; ++k) expected[k] = 1 << 24;
    values.equals(expected, "exact-f32 boundary rounding differs");
    auto short_scratch = a; --short_scratch.scratch_words;
    require(cnot103_cuda_launch_frequency(&short_scratch) == 2, "short internal scratch accepted");
    require(cnot103_cuda_prepare_frequency(dcentral.data, rank_scratch.data, statistics.data, m, r, t, rank_scratch.size - 1, nullptr) == 2, "short central scratch accepted");
    std::printf("frequency n=%u r=%u t=%u rank=%u: values, guards, range, capacities PASS\n", n, r, t, kappa);
}
int main() {
    try {
        const std::vector<std::vector<Index>> shapes{{6,2,2}, {65,2,61}, {144,2,140}, {144,65,4}};
        for (const auto& shape : shapes)
            for (Index mode = 0; mode < 3; ++mode) fixture(shape[0], shape[1], shape[2], mode);
        fixture(144, 65, 4, 3);
        checked(cudaDeviceReset()); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
