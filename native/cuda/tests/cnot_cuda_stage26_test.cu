// Literal certificates, physical relabels and destructive corruption checks.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using Word = std::uint64_t;
void checked(cudaError_t result) { if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); }
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<typename T> struct Buffer {
    T* data = nullptr; std::size_t size;
    explicit Buffer(std::size_t count) : size(count) { checked(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
    void upload(const std::vector<T>& values) {
        require(values.size() == size, "upload extent mismatch");
        checked(cudaMemcpy(data, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> read() const {
        std::vector<T> values(size); checked(cudaMemcpy(values.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost)); return values;
    }
    void equals(const std::vector<T>& expected, const char* label) { require(read() == expected, label); }
};

void fixture(std::uint32_t n, bool moved, const std::string& fault = "") {
    const std::uint32_t w = (n + 63) / 64;
    const Word cells = Word{n} * w, cw = 6 + Word{n} + (4 * Word{n} + 2) * w;
    const Word sw = (7 * Word{n} + 4) * w + n + 1;
    const std::array<std::uint32_t, 6> p = moved ?
        std::array<std::uint32_t, 6>{0, n - 1, 1, n - 2, 2, n - 3} :
        std::array<std::uint32_t, 6>{0, 1, 2, 3, 4, 5};
    const auto set = [w](std::vector<Word>& matrix, Word row, std::uint32_t column) { matrix[row * w + column / 64] |= Word{1} << (column % 64); };
    const auto replace = [&](std::vector<Word>& matrix, Word row, std::initializer_list<std::uint32_t> columns) {
        std::fill(matrix.begin() + row * w, matrix.begin() + (row + 1) * w, 0);
        for (auto column : columns) set(matrix, row, column);
    };
    std::vector<Word> input(4 * cells, 0);
    for (Word sample = 0; sample < 4; ++sample) for (std::uint32_t row = 0; row < n; ++row) set(input, sample * n + row, row);
    replace(input, n + p[1], {p[0], p[1]}); replace(input, n + p[2], {p[0], p[1], p[2]});
    replace(input, 2 * n + p[0], {p[1]}); replace(input, 2 * n + p[1], {p[0]});
    replace(input, 3 * n + p[0], {p[1], p[2], p[3]}); replace(input, 3 * n + p[1], {p[0], p[4], p[5]});
    Buffer<Word> states(input.size()), certificate(4 * cw), scratch(4 * sw);
    Buffer<std::uint8_t> verified(4), status(4);
    Buffer<float> f03(20), f04(4 * Word{n} * n * 3), f05(8);
    states.upload(input);
    Cnot103Stage26Args a{};
    a.states = states.data; a.certificate = certificate.data; a.scratch = scratch.data;
    a.verified = verified.data; a.status = status.data; a.f03 = f03.data; a.f04 = f04.data; a.f05 = f05.data;
    a.certificate_words = certificate.size; a.workspace_words = scratch.size; a.pair_elements = f04.size;
    a.batch = 4; a.dimension = n; a.words_per_row = w;
    if (fault == "padding") {
        require(n % 64, "padding fault requires partial word");
        input[w - 1] |= Word{1} << (n % 64); states.upload(input);
        require(cnot103_cuda_stage26_construct(&a) == 0, "padding fault launch");
        const auto error = cudaDeviceSynchronize();
        require(error == cudaErrorIllegalInstruction || error == cudaErrorLaunchFailure, "malformed padding did not trap");
        std::printf("expected malformed-padding device trap: %s\n", cudaGetErrorString(error)); return;
    }
    require(cnot103_cuda_launch_stage26(&a) == 0, "Stage26 launch rejected");
    checked(cudaDeviceSynchronize());
    verified.equals({1, 1, 1, 1}, "literal certificate rejected");
    status.equals({1, 1, 1, 1}, "literal status mismatch");
    const std::vector<float> f03_literal{0,0,0,0,0, 1,1,0,1,1, 0,0,0,2,2, 0,2,2,2,0};
    f03.equals(f03_literal, "literal F03 mismatch");
    f05.equals({0,0, 2,0, 0,-2, 4,-2}, "literal F05 mismatch");

    std::vector<Word> projected = input;
    projected[(n + p[2]) * w + p[2] / 64] &= ~(Word{1} << (p[2] % 64));
    std::vector<Word> cosets = projected;
    for (Word sample = 1; sample < 4; ++sample) {
        const auto left = sample == 1 ? p[1] : p[0];
        const auto right = sample == 1 ? p[2] : p[1];
        replace(cosets, sample * n + left, {}); replace(cosets, sample * n + right, {});
    }
    std::vector<float> expected_pairs(f04.size);
    for (Word sample = 0; sample < 4; ++sample) for (Word i = 0; i < n; ++i) for (Word j = 0; j < n; ++j) {
        const auto same = [&](const std::vector<Word>& values, Word a, Word b) {
            return std::equal(values.begin() + (sample * n + a) * w, values.begin() + (sample * n + a + 1) * w, values.begin() + (sample * n + b) * w);
        };
        bool nonzero = false;
        for (Word k = 0; k < w; ++k) nonzero |= cosets[(sample * n + i) * w + k] != 0;
        const Word offset = ((sample * n + i) * n + j) * 3;
        expected_pairs[offset] = nonzero && same(cosets, i, j);
        expected_pairs[offset + 1] = same(projected, i, j);
        expected_pairs[offset + 2] = sample == 1 && (i == p[1] || i == p[2]) ? 2 : 1;
    }
    f04.equals(expected_pairs, "literal F04 multiplicity/coset mismatch");

    // Full literal certificate comparison: physical rows, witnesses, high-pivot
    // H and ascending targets, including every unused word and target slot.
    std::vector<Word> expected_certificate(certificate.size, 0);
    const Word retained_offset = 6 + n, deleted_offset = retained_offset + w;
    const Word rows_offset = deleted_offset + w, projected_offset = rows_offset + cells;
    const Word h_offset = projected_offset + cells, targets_offset = h_offset + cells;
    for (Word sample = 0; sample < 4; ++sample) {
        const Word base = sample * cw;
        for (Word k = 0; k < 5; ++k) expected_certificate[base + k] = static_cast<Word>(f03_literal[sample * 5 + k]);
        expected_certificate[base + 5] = sample == 1 ? 2 : (sample == 3 ? 4 : 0);
        std::fill(expected_certificate.begin() + base + 6, expected_certificate.begin() + base + 6 + n, ~Word{0});
        for (std::uint32_t wire = 0; wire < n; ++wire) expected_certificate[base + retained_offset + wire / 64] |= Word{1} << (wire % 64);
        if (sample == 1) {
            expected_certificate[base + 6 + p[2]] = p[2];
            expected_certificate[base + retained_offset + p[2] / 64] &= ~(Word{1} << (p[2] % 64));
            expected_certificate[base + deleted_offset + p[2] / 64] |= Word{1} << (p[2] % 64);
        }
        std::copy(input.begin() + sample * cells, input.begin() + (sample + 1) * cells, expected_certificate.begin() + base + rows_offset);
        std::copy(projected.begin() + sample * cells, projected.begin() + (sample + 1) * cells, expected_certificate.begin() + base + projected_offset);
        std::vector<std::vector<Word>> h, targets;
        const auto vector = [&](std::initializer_list<std::uint32_t> columns) {
            std::vector<Word> v(w, 0); for (auto col : columns) v[col / 64] |= Word{1} << (col % 64); return v;
        };
        if (sample == 1) { h.push_back(vector({p[0], p[1]})); targets = h; }
        if (sample == 2) { h.push_back(vector({p[0]})); h.push_back(vector({p[1]})); }
        if (sample == 3) { h.push_back(vector({p[1], p[2], p[3]})); h.push_back(vector({p[0], p[4], p[5]})); targets = h; }
        for (const auto& v : h) {
            std::uint32_t pivot = 0;
            for (std::uint32_t col = 0; col < n; ++col) if ((v[col / 64] >> (col % 64)) & 1) pivot = col;
            std::copy(v.begin(), v.end(), expected_certificate.begin() + base + h_offset + Word{pivot} * w);
        }
        std::sort(targets.begin(), targets.end(), [](const auto& l, const auto& r) { return std::lexicographical_compare(l.rbegin(), l.rend(), r.rbegin(), r.rend()); });
        for (std::size_t i = 0; i < targets.size(); ++i)
            std::copy(targets[i].begin(), targets[i].end(), expected_certificate.begin() + base + targets_offset + i * w);
    }
    certificate.equals(expected_certificate, "literal certificate payload mismatch");
    auto mutate = [&](Word offset, Word replacement) {
        auto corrupted = expected_certificate; corrupted[cw + offset] = replacement; certificate.upload(corrupted);
        require(cnot103_cuda_stage26_check(&a) == 0, "corruption checker launch rejected");
        checked(cudaDeviceSynchronize()); verified.equals({1,0,1,1}, "corrupt certificate was accepted");
    };
    mutate(5, 999);
    if (fault == "certificate") {
        require(cnot103_cuda_stage26_publish(&a) == 0, "fatal publisher launch");
        const auto error = cudaDeviceSynchronize();
        require(error == cudaErrorIllegalInstruction || error == cudaErrorLaunchFailure, "rejected certificate did not trap");
        std::printf("expected rejected-certificate device trap: %s\n", cudaGetErrorString(error)); return;
    }
    for (Word k = 0; k < 6; ++k) mutate(k, expected_certificate[cw + k] + 1);
    mutate(6 + p[2], n); mutate(6 + p[2], ~Word{0}); mutate(6 + p[0], 0);
    for (Word offset : {retained_offset, deleted_offset, rows_offset, projected_offset, targets_offset})
        mutate(offset, expected_certificate[cw + offset] ^ 1);
    mutate(targets_offset + w, 1); // unused target slot
    const Word pivot_offset = h_offset + Word{std::max(p[0], p[1])} * w + p[2] / 64;
    mutate(pivot_offset, expected_certificate[cw + pivot_offset] ^ (Word{1} << (p[2] % 64)));
    certificate.upload(expected_certificate);
    require(cnot103_cuda_stage26_check(&a) == 0, "restored checker launch");
    require(cnot103_cuda_stage26_publish(&a) == 0, "restored publisher launch");
    checked(cudaDeviceSynchronize()); f04.equals(expected_pairs, "restored payload mismatch");
    verified.equals({1,1,1,1}, "restored certificate rejected");
    auto short_buffer = a; --short_buffer.certificate_words;
    require(cnot103_cuda_launch_stage26(&short_buffer) == 2, "short certificate accepted");
    short_buffer = a; --short_buffer.workspace_words;
    require(cnot103_cuda_launch_stage26(&short_buffer) == 2, "short scratch accepted");
    short_buffer = a; --short_buffer.pair_elements;
    require(cnot103_cuda_launch_stage26(&short_buffer) == 2, "short F04 accepted");
    std::printf("stage26 literal n=%u moved=%d: payloads, witnesses, obstruction, corruptions PASS\n", n, moved);
}
int main(int argc, char** argv) {
    try {
        if (argc == 2) { fixture(6, false, argv[1]); return 0; }
        for (auto n : {6u, 65u, 144u}) for (bool moved : {false, true}) fixture(n, moved);
        checked(cudaDeviceReset()); return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
