#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace {
using Word = std::uint64_t;
using Index = std::uint32_t;

// Full reduced basis, descending highest-set-bit pivots. This is the unique
// canonical basis used by Stage56; a rank-equivalent basis is insufficient.
__device__ Index canonicalize(Word* matrix, Index rows, Index columns, Index words,
    Word* pivot_columns, Index& rank, Index& pivot) {
    if (threadIdx.x == 0) rank = 0;
    __syncthreads();
    for (Index column = columns; column > 0; --column) {
        const Index active = rank;
        if (active == rows) break;
        if (threadIdx.x == 0) pivot = rows;
        __syncthreads();
        const Index p = column - 1;
        const Word bit = Word{1} << (p % 64);
        for (Index row = active + threadIdx.x; row < rows; row += blockDim.x)
            if (matrix[Word{row} * words + p / 64] & bit) atomicMin(&pivot, row);
        __syncthreads();
        if (pivot < rows) {
            for (Index k = threadIdx.x; k < words; k += blockDim.x) {
                const Word first = Word{active} * words + k, second = Word{pivot} * words + k;
                const Word tmp = matrix[first]; matrix[first] = matrix[second]; matrix[second] = tmp;
            }
            __syncthreads();
            for (Index row = threadIdx.x; row < rows; row += blockDim.x)
                if (row != active && (matrix[Word{row} * words + p / 64] & bit))
                    for (Index k = 0; k < words; ++k) matrix[Word{row} * words + k] ^= matrix[Word{active} * words + k];
            __syncthreads();
            if (threadIdx.x == 0) { pivot_columns[active] = p; ++rank; }
        }
        __syncthreads(); // includes every no-pivot column
    }
    return rank;
}
__global__ void shared_relation(Cnot103SharedRelationArgs a) {
    const Index n = a.dimension, w = a.words_per_row, v = a.vectors, c = a.coordinates;
    const Index vw = (v + 63) / 64;
    const Word sample = blockIdx.x, capacity = Word{v} + a.ports, channels = Word{a.max_vectors} + 2;
    const Word scratch_size = (Word{c} + v) * vw + v;
    const Word* input = a.states + sample * Word{n} * w;
    Word* equations = a.scratch + sample * scratch_size;
    Word* kernel = equations + Word{c} * vw;
    Word* pivots = kernel + Word{v} * vw;
    float* values = a.values + sample * capacity * channels;
    float* incidence = a.incidence + sample * capacity * n * 2;
    auto* valid = a.valid + sample * capacity;
    for (Word i = threadIdx.x; i < scratch_size; i += blockDim.x) equations[i] = 0;
    for (Word i = threadIdx.x; i < capacity * channels; i += blockDim.x) values[i] = 0;
    for (Word i = threadIdx.x; i < capacity * n * 2; i += blockDim.x) incidence[i] = 0;
    for (Word i = threadIdx.x; i < capacity; i += blockDim.x) valid[i] = 0;
    __syncthreads();
    for (Word k = threadIdx.x; k < Word{c} * vw; k += blockDim.x) {
        const Index coordinate = k / vw, word = k % vw;
        const Index physical_column = a.coordinate_wires[coordinate];
        Word packed = 0;
        for (Index bit = 0; bit < 64 && word * 64 + bit < v; ++bit) {
            const Index row = a.row_wires[word * 64 + bit];
            if ((input[Word{row} * w + physical_column / 64] >> (physical_column % 64)) & 1) packed |= Word{1} << bit;
        }
        equations[k] = packed;
    }
    __syncthreads();
    __shared__ Index rank, pivot;
    const Index equation_rank = canonicalize(equations, c, v, vw, pivots, rank, pivot);
    for (Index free = threadIdx.x; free < v; free += blockDim.x) {
        bool is_pivot = false;
        for (Index j = 0; j < equation_rank; ++j) is_pivot |= pivots[j] == free;
        if (is_pivot) continue;
        Word* vector = kernel + Word{free} * vw;
        vector[free / 64] |= Word{1} << (free % 64);
        for (Index j = 0; j < equation_rank; ++j)
            if ((equations[Word{j} * vw + free / 64] >> (free % 64)) & 1)
                vector[pivots[j] / 64] |= Word{1} << (pivots[j] % 64);
    }
    __syncthreads();
    // Stage56 canonicalizes the nullspace basis itself as well. The zero rows
    // at non-free coordinates do not alter that unique resulting basis.
    const Index nullity = canonicalize(kernel, v, v, vw, pivots, rank, pivot);
    const Word groups = Word{nullity} + a.ports;
    for (Word group = threadIdx.x; group < groups; group += blockDim.x) {
        valid[group] = 1;
        values[group * channels] = group >= nullity ? 1.0f : 0.0f;
        values[group * channels + 1] = static_cast<float>(group >= nullity ? group - nullity : group);
    }
    for (Word item = threadIdx.x; item < groups * v; item += blockDim.x) {
        const Word group = item / v;
        const Index index = item % v, physical_row = a.row_wires[index];
        bool coefficient;
        if (group < nullity) coefficient = (kernel[group * vw + index / 64] >> (index % 64)) & 1;
        else {
            unsigned parity = 0;
            const Word* mask = a.covectors + (group - nullity) * w;
            for (Index k = 0; k < w; ++k) parity ^= __popcll(input[Word{physical_row} * w + k] & mask[k]) & 1;
            coefficient = parity != 0;
        }
        values[group * channels + 2 + index] = coefficient;
        if (coefficient) incidence[(group * n + physical_row) * 2] = 1;
    }
    for (Word item = threadIdx.x; item < Word{a.ports} * n; item += blockDim.x) {
        const Word label = item / n;
        const Index physical_coordinate = item % n;
        if ((a.covectors[label * w + physical_coordinate / 64] >> (physical_coordinate % 64)) & 1)
            incidence[((Word{nullity} + label) * n + physical_coordinate) * 2 + 1] = 1;
    }
    if (threadIdx.x == 0) a.status[sample] = 1;
}
}
extern "C" std::int32_t cnot103_cuda_launch_shared_relation(const Cnot103SharedRelationArgs* args) {
    if (!args) return 1;
    const auto& a = *args;
    if (!a.states || !a.row_wires || !a.coordinate_wires || (a.ports && !a.covectors) || !a.scratch ||
        !a.values || !a.valid || !a.incidence || !a.status || !a.batch || a.batch > 0x7fffffffu ||
        a.dimension < 2 || a.dimension > (1u << 24) || a.words_per_row != (a.dimension + 63) / 64 ||
        !a.vectors || a.vectors > a.dimension || !a.coordinates || a.coordinates > a.dimension ||
        a.ports > a.coordinates || a.vectors > a.max_vectors || a.max_vectors > (1u << 24)) return 1;
    const Word groups = Word{a.batch} * (Word{a.vectors} + a.ports);
    const Word scratch_per_state = (Word{a.coordinates} + a.vectors) * ((a.vectors + 63) / 64) + a.vectors;
    if (a.batch > a.scratch_words / scratch_per_state || groups > a.group_elements ||
        groups > a.value_elements / (Word{a.max_vectors} + 2) ||
        groups > a.incidence_elements / (Word{a.dimension} * 2)) return 2;
    shared_relation<<<a.batch, 128, 0, reinterpret_cast<cudaStream_t>(a.stream)>>>(a);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
