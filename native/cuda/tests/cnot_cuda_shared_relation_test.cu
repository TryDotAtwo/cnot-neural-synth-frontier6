// Independent finite enumeration and literal wide relations; no host elimination.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using Word = std::uint64_t;
using Index = std::uint32_t;
using Relation = std::vector<Index>;
using Basis = std::vector<Relation>;
void checked(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b, const char* error) { if (!b) throw std::runtime_error(error); }
template<typename T> struct Buffer {
    T* data = nullptr; std::size_t size;
    explicit Buffer(std::size_t n) : size(n) {
        checked(cudaMalloc(reinterpret_cast<void**>(&data), std::max(n, std::size_t{1}) * sizeof(T)));
        checked(cudaMemset(data, 0xa5, std::max(n, std::size_t{1}) * sizeof(T)));
    }
    ~Buffer() { cudaFree(data); }
    void upload(const std::vector<T>& values) {
        require(values.size() == size, "upload extent mismatch");
        if (size) checked(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void equals(const std::vector<T>& expected, const char* message) {
        std::vector<T> values(size);
        if (size) checked(cudaMemcpy(values.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
        require(values == expected, message);
    }
};
std::vector<Word> identities(Index n, Index batch) {
    const Index w = (n + 63) / 64;
    std::vector<Word> result(Word{batch} * n * w, 0);
    for (Index b = 0; b < batch; ++b) for (Index i = 0; i < n; ++i)
        result[(Word{b} * n + i) * w + i / 64] = Word{1} << (i % 64);
    return result;
}
bool bit(const std::vector<Word>& state, Word offset, Index index) {
    return (state[offset + index / 64] >> (index % 64)) & 1;
}
void run(Index n, Index batch, const std::vector<Word>& input,
         const std::vector<Index>& rows, const std::vector<Index>& coordinates,
         const std::vector<Word>& covectors, const std::vector<Basis>& bases) {
    const Index w = (n + 63) / 64, v = rows.size(), c = coordinates.size(), p = covectors.size() / w;
    const Index max_vectors = v + 2;
    const Word capacity = Word{v} + p, channels = Word{max_vectors} + 2;
    Buffer<Word> states(input.size()), masks(covectors.size()), scratch(Word{batch} * ((Word{c} + v) * ((v + 63) / 64) + v));
    Buffer<Index> drows(v), dcoordinates(c);
    Buffer<float> values(Word{batch} * capacity * channels), incidence(Word{batch} * capacity * n * 2);
    Buffer<std::uint8_t> valid(Word{batch} * capacity), status(batch);
    states.upload(input); masks.upload(covectors); drows.upload(rows); dcoordinates.upload(coordinates);
    Cnot103SharedRelationArgs a{};
    a.states = states.data; a.row_wires = drows.data; a.coordinate_wires = dcoordinates.data;
    a.covectors = p ? masks.data : nullptr; a.scratch = scratch.data; a.values = values.data;
    a.valid = valid.data; a.incidence = incidence.data; a.status = status.data;
    a.scratch_words = scratch.size; a.value_elements = values.size;
    a.group_elements = valid.size; a.incidence_elements = incidence.size;
    a.batch = batch; a.dimension = n; a.words_per_row = w; a.vectors = v;
    a.coordinates = c; a.ports = p; a.max_vectors = max_vectors;
    // Short capacities and malformed dimensions must fail before any launch.
    for (Index field = 0; field < 4; ++field) {
        auto short_buffer = a;
        switch (field) {
            case 0: --short_buffer.scratch_words; break;
            case 1: --short_buffer.value_elements; break;
            case 2: --short_buffer.group_elements; break;
            case 3: --short_buffer.incidence_elements; break;
        }
        require(cnot103_cuda_launch_shared_relation(&short_buffer) == 2, "short I06 capacity accepted");
    }
    auto invalid = a; invalid.words_per_row = 0;
    require(cnot103_cuda_launch_shared_relation(&invalid) == 1, "invalid I06 shape accepted");
    invalid = a; invalid.max_vectors = v - 1;
    require(cnot103_cuda_launch_shared_relation(&invalid) == 1, "invalid I06 channel capacity accepted");
    require(cnot103_cuda_launch_shared_relation(nullptr) == 1, "null I06 args accepted");
    require(cnot103_cuda_launch_shared_relation(&a) == 0, "I06 launch failed");
    checked(cudaDeviceSynchronize());
    std::vector<float> expected(values.size, 0), expected_incidence(incidence.size, 0);
    std::vector<std::uint8_t> expected_valid(valid.size, 0);
    for (Index b = 0; b < batch; ++b) {
        const auto& basis = bases[b];
        const Word groups = basis.size() + p;
        for (Word group = 0; group < groups; ++group) {
            const Word token = Word{b} * capacity + group;
            expected_valid[token] = 1;
            const bool port = group >= basis.size();
            const Word label = port ? group - basis.size() : group;
            expected[token * channels] = port;
            expected[token * channels + 1] = label;
            for (Index i = 0; i < v; ++i) {
                bool coefficient = false;
                if (!port) coefficient = std::find(basis[group].begin(), basis[group].end(), i) != basis[group].end();
                else for (Index wire : coordinates)
                    coefficient ^= bit(input, (Word{b} * n + rows[i]) * w, wire) && bit(covectors, label * w, wire);
                expected[token * channels + 2 + i] = coefficient;
                if (coefficient) expected_incidence[(token * n + rows[i]) * 2] = 1;
            }
            if (port) for (Index wire : coordinates)
                if (bit(covectors, label * w, wire)) expected_incidence[(token * n + wire) * 2 + 1] = 1;
        }
    }
    values.equals(expected, "I06 full canonical relation/port payload differs");
    valid.equals(expected_valid, "I06 dynamic token mask differs");
    incidence.equals(expected_incidence, "I06 labelled physical incidence differs");
    status.equals(std::vector<std::uint8_t>(batch, 1), "I06 status differs");
    states.equals(input, "I06 changed states"); masks.equals(covectors, "I06 changed covectors");
    drows.equals(rows, "I06 changed row labels"); dcoordinates.equals(coordinates, "I06 changed coordinate labels");
    std::printf("I06 n=%u batch=%u vectors=%u coordinates=%u ports=%u: full payload, masks, incidence, capacities PASS\n", n, batch, v, c, p);
}
void exhaustive_small(Index n) {
    const Index w = (n + 63) / 64;
    const std::vector<Index> rows{n - 1, n - 3, n - 2}, coordinates{1, 0};
    auto input = identities(n, 64);
    std::vector<Basis> bases(64);
    for (Index pattern = 0; pattern < 64; ++pattern) {
        Index projected[3]{};
        for (Index i = 0; i < 3; ++i) for (Index j = 0; j < 2; ++j)
            if ((pattern >> (2 * i + j)) & 1) {
                projected[i] |= 1u << j;
                input[(Word{pattern} * n + rows[i]) * w] |= Word{1} << coordinates[j];
            }
        // Enumerate all relations, then select the smallest integer at each
        // highest-set-bit position. This characterizes the unique reduced
        // basis without implementing rank, elimination or nullspace routines.
        for (Index pivot = 3; pivot > 0; --pivot) {
            for (Index mask = 1u << (pivot - 1); mask < (1u << pivot); ++mask) {
                Index sum = 0;
                for (Index i = 0; i < 3; ++i) if ((mask >> i) & 1) sum ^= projected[i];
                if (sum) continue;
                Relation relation;
                for (Index i = 0; i < 3; ++i) if ((mask >> i) & 1) relation.push_back(i);
                bases[pattern].push_back(relation); break;
            }
        }
    }
    std::vector<Word> covectors(2 * w, 0); covectors[0] = 2; covectors[w] = 1;
    run(n, 64, input, rows, coordinates, covectors, bases);
}
void wide(Index mode) {
    const Index n = 144, w = 3, v = 65;
    std::vector<Index> rows(v), coordinates(v);
    for (Index i = 0; i < v; ++i) { rows[i] = 129 - i; coordinates[i] = 64 - i; }
    auto input = identities(n, 1);
    Basis basis;
    if (mode == 0) for (Index i = v; i > 0; --i) basis.push_back({i - 1});
    if (mode == 1) {
        for (Index i = 0; i < v; ++i) input[Word{rows[i]} * w] |= 1;
        for (Index i = v; i > 1; --i) basis.push_back({0, i - 1});
    }
    if (mode == 2) for (Index i = 0; i < v; ++i)
        input[Word{rows[i]} * w + coordinates[i] / 64] |= Word{1} << (coordinates[i] % 64);
    std::vector<Word> covectors(2 * w, 0); covectors[1] = 1; covectors[w] = 1;
    run(n, 1, input, rows, coordinates, covectors, {basis});
}
int main() {
    try {
        for (Index n : {6, 65, 144}) exhaustive_small(n);
        for (Index mode = 0; mode < 3; ++mode) wide(mode);
        run(2, 2, {1, 2, 2, 1}, {1, 0}, {0, 1}, {}, {{}, {}});
        checked(cudaDeviceReset()); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
