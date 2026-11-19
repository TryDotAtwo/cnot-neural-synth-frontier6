#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace {
using Word = std::uint64_t;
using Index = std::uint32_t;

// Rank only: ascending-column elimination below each pivot is equivalent to
// source Gauss-Jordan. Global scratch avoids a hidden shared-memory width cap.
__device__ Index rank_rows(Word* rows, Index m, Index columns, Index w, Index& rank, Index& pivot) {
    if (threadIdx.x == 0) rank = 0;
    __syncthreads();
    for (Index column = 0; column < columns; ++column) {
        const Index active = rank;
        if (active == m) break;
        if (threadIdx.x == 0) pivot = m;
        __syncthreads();
        const Word mask = Word{1} << (column % 64);
        for (Index row = active + threadIdx.x; row < m; row += blockDim.x)
            if (rows[Word{row} * w + column / 64] & mask) atomicMin(&pivot, row);
        __syncthreads();
        if (pivot < m) {
            for (Index k = threadIdx.x; k < w; k += blockDim.x) {
                const Word first = Word{active} * w + k, second = Word{pivot} * w + k;
                const Word tmp = rows[first]; rows[first] = rows[second]; rows[second] = tmp;
            }
            __syncthreads();
            for (Index row = active + 1 + threadIdx.x; row < m; row += blockDim.x)
                if (rows[Word{row} * w + column / 64] & mask)
                    for (Index k = 0; k < w; ++k) rows[Word{row} * w + k] ^= rows[Word{active} * w + k];
            __syncthreads();
            if (threadIdx.x == 0) ++rank;
        }
        __syncthreads(); // includes absent-pivot iterations
    }
    return rank;
}
__global__ void prepare(const Word* central, Word* scratch, Word* statistics, Index m, Index r, Index t) {
    const Index w = (r + 63) / 64;
    for (Word k = threadIdx.x; k < Word{m} * w; k += blockDim.x) scratch[k] = central[k];
    __syncthreads();
    __shared__ Index rank, pivot;
    const Index kappa = rank_rows(scratch, m, r, w, rank, pivot);
    if (threadIdx.x == 0) { statistics[0] = kappa; statistics[1] = Word{t} * kappa; }
}
__global__ void evaluate(Cnot103FrequencyArgs a) {
    const Index n = a.dimension, w = a.words_per_row, t = a.internal_count, tw = (t + 63) / 64;
    const Word sample = blockIdx.x;
    const Word* input = a.states + sample * Word{n} * w;
    Word* internal = a.scratch + sample * Word{t} * tw;
    __shared__ Index valid, rank, pivot;
    if (threadIdx.x == 0) {
        valid = 1; a.status[sample] = 2;
        a.values[2 * sample] = 0; a.values[2 * sample + 1] = 0;
    }
    __syncthreads();
    for (Word cell = threadIdx.x; cell < Word{n} * n; cell += blockDim.x) {
        const Index row = cell / n, column = cell % n;
        const auto row_role = a.roles[row], column_role = a.roles[column];
        const bool value = (input[Word{row} * w + column / 64] >> (column % 64)) & 1;
        if ((row_role == 0 && value != (row == column)) ||
            (row_role == 1 && column_role == 2 && value) ||
            (row_role == 2 && column_role == 2 && value != (row == column))) atomicExch(&valid, 0);
    }
    for (Index i = threadIdx.x; i < t; i += blockDim.x) if (a.internal_wires[i] >= n) atomicExch(&valid, 0);
    __syncthreads();
    if (!valid) return;
    for (Word k = threadIdx.x; k < Word{t} * tw; k += blockDim.x) {
        const Index row = k / tw, word = k % tw;
        Word packed = 0;
        for (Index bit = 0; bit < 64 && word * 64 + bit < t; ++bit) {
            const Index physical = a.internal_wires[word * 64 + bit];
            if ((input[Word{a.internal_wires[row]} * w + physical / 64] >> (physical % 64)) & 1) packed |= Word{1} << bit;
        }
        internal[k] = packed;
    }
    __syncthreads();
    const Index internal_rank = rank_rows(internal, t, t, tw, rank, pivot);
    if (threadIdx.x == 0 && internal_rank == t) {
        if (a.statistics[0] > (Word{1} << 24) || a.statistics[1] > (Word{1} << 24)) a.status[sample] = 4;
        else {
            a.status[sample] = 1;
            a.values[2 * sample] = static_cast<float>(a.statistics[0]);
            a.values[2 * sample + 1] = static_cast<float>(a.statistics[1]);
        }
    }
}
bool width(Index value) { return value > 0 && value <= (1u << 24); }
}
extern "C" std::int32_t cnot103_cuda_prepare_frequency(
    const Word* central, Word* scratch, Word* statistics, Index rows, Index columns, Index internal_count,
    Word scratch_words, void* stream) {
    if (!central || !scratch || !statistics || !width(rows) || !width(columns) || !width(internal_count)) return 1;
    if (scratch_words < Word{rows} * ((columns + 63) / 64)) return 2;
    prepare<<<1, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(central, scratch, statistics, rows, columns, internal_count);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
extern "C" std::int32_t cnot103_cuda_launch_frequency(const Cnot103FrequencyArgs* args) {
    if (!args) return 1;
    const auto& a = *args;
    if (!a.states || !a.roles || !a.internal_wires || !a.statistics || !a.scratch || !a.status || !a.values ||
        !a.batch || a.batch > 0x7fffffffu || !width(a.dimension) || a.dimension < 3 ||
        a.words_per_row != (a.dimension + 63) / 64 || !a.internal_count || a.internal_count > a.dimension - 2) return 1;
    const Word per_sample = Word{a.internal_count} * ((a.internal_count + 63) / 64);
    if (a.batch > a.scratch_words / per_sample) return 2;
    evaluate<<<a.batch, 128, 0, reinterpret_cast<cudaStream_t>(a.stream)>>>(a);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
