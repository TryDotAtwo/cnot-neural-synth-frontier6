#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace {
using Word = std::uint64_t;
using Index = std::uint32_t;
// For B=UV and T=[[B,A0],[D0,C]], the source Stage55 invariant is
// rank(T)+rank(B)-rank([B,A0])-rank([B;D0]). Reducing B to diag(I_h,0)
// leaves [[0,A3],[D2,C-D1*A1]] after h pivots. Quotienting A3/D2 gives
// exactly rank(L*(C-D1*A1)*R). I05 emits this rank, not a particular basis.
// U/V independence is guaranteed by the validated context; all products and
// ranks, including immutable setup statistics, are evaluated on the device.
__device__ Index rank_rows(Word* rows, Index m, Index columns, Index w, Index& rank, Index& pivot) {
    if (threadIdx.x == 0) rank = 0;
    __syncthreads();
    for (Index column = 0; column < columns; ++column) {
        const Index active = rank;
        if (active == m) break;
        if (threadIdx.x == 0) pivot = m;
        __syncthreads();
        const Word mask = Word{1} << (column % 64);
        for (Word row = Word{active} + threadIdx.x; row < m; row += blockDim.x)
            if (rows[row * w + column / 64] & mask) atomicMin(&pivot, static_cast<Index>(row));
        __syncthreads();
        if (pivot < m) {
            for (Index k = threadIdx.x; k < w; k += blockDim.x) {
                const Word first = Word{active} * w + k, second = Word{pivot} * w + k;
                const Word tmp = rows[first]; rows[first] = rows[second]; rows[second] = tmp;
            }
            __syncthreads();
            for (Word row = Word{active} + 1 + threadIdx.x; row < m; row += blockDim.x)
                if (rows[row * w + column / 64] & mask)
                    for (Index k = 0; k < w; ++k) rows[row * w + k] ^= rows[Word{active} * w + k];
            __syncthreads();
            if (threadIdx.x == 0) ++rank;
        }
        __syncthreads();
    }
    return rank;
}
__device__ bool bit(const Word* matrix, Word row, Index stride, Index column) {
    return (matrix[row * stride + column / 64] >> (column % 64)) & 1;
}
__global__ void prepare(Cnot103BilinearPrepareArgs a) {
    const Index w = (Word{a.q} + a.r + 63) / 64;
    const Index uw = (Word{a.t} + 63) / 64, qw = (Word{a.q} + 63) / 64, rw = (Word{a.r} + 63) / 64;
    const Word cells = (Word{a.p} + a.m) * w;
    for (Word item = threadIdx.x; item < cells; item += blockDim.x) {
        const Index row = item / w, word = item % w;
        Word packed = 0;
        for (Index k = 0; k < 64 && Word{word} * 64 + k < Word{a.q} + a.r; ++k) {
            const Index column = Word{word} * 64 + k;
            bool value = false;
            if (row < a.p) {
                if (column < a.q) for (Index inner = 0; inner < a.t; ++inner)
                    value ^= bit(a.u, row, uw, inner) && bit(a.v, inner, qw, column);
                else value = bit(a.a0, row, rw, column - a.q);
            } else if (column < a.q) value = bit(a.d0, row - a.p, qw, column);
            if (value) packed |= Word{1} << k;
        }
        a.matrix_template[item] = packed;
    }
    __syncthreads();
    __shared__ Index rank, pivot;
    for (Index phase = 0; phase < 3; ++phase) {
        for (Word item = threadIdx.x; item < cells; item += blockDim.x) a.scratch[item] = a.matrix_template[item];
        __syncthreads();
        const Index result = rank_rows(a.scratch, phase == 2 ? a.p + a.m : a.p,
            phase == 1 ? a.q + a.r : a.q, w, rank, pivot);
        if (threadIdx.x == 0) a.statistics[phase] = result;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const Word total = Word{a.t} + a.statistics[0], ports = Word{a.p} + a.q;
        a.statistics[3] = total >= ports ? total - ports : ~Word{0};
    }
}
__global__ void evaluate(Cnot103BilinearArgs a) {
    const Index w = (Word{a.q} + a.r + 63) / 64;
    const Word cells = (Word{a.p} + a.m) * w, sample = blockIdx.x;
    const Word* input = a.states + sample * Word{a.dimension} * a.words_per_row;
    Word* matrix = a.scratch + sample * cells;
    for (Word item = threadIdx.x; item < cells; item += blockDim.x) {
        const Index row = item / w, word = item % w;
        Word packed = a.matrix_template[item];
        if (row >= a.p) for (Index k = 0; k < 64 && Word{word} * 64 + k < Word{a.q} + a.r; ++k) {
            const Index column = Word{word} * 64 + k;
            if (column >= a.q && bit(input, a.row_wires[row - a.p], a.words_per_row, a.column_wires[column - a.q]))
                packed |= Word{1} << k;
        }
        matrix[item] = packed;
    }
    __syncthreads();
    __shared__ Index rank, pivot;
    const Word total = rank_rows(matrix, a.p + a.m, a.q + a.r, w, rank, pivot);
    if (threadIdx.x == 0) {
        const Word added = total + a.statistics[0], removed = a.statistics[1] + a.statistics[2];
        a.status[sample] = 4;
        a.values[2 * sample] = 0; a.values[2 * sample + 1] = 0;
        if (added >= removed && added - removed <= (Word{1} << 24) && a.statistics[3] <= (Word{1} << 24)) {
            a.status[sample] = 1;
            a.values[2 * sample] = static_cast<float>(added - removed);
            a.values[2 * sample + 1] = static_cast<float>(a.statistics[3]);
        }
    }
}
bool shape(Index m, Index r, Index p, Index q) {
    return m && r && Word{m} + p <= std::numeric_limits<Index>::max() && Word{r} + q <= std::numeric_limits<Index>::max();
}
}
extern "C" std::int32_t cnot103_cuda_prepare_bilinear(const Cnot103BilinearPrepareArgs* args) {
    if (!args) return 1;
    const auto& a = *args;
    if (!shape(a.m, a.r, a.p, a.q) || a.p > a.t || a.q > a.t ||
        (a.p && a.t && !a.u) || (a.t && a.q && !a.v) || (a.p && !a.a0) || (a.q && !a.d0) ||
        !a.matrix_template || !a.scratch || !a.statistics) return 1;
    const Word cells = (Word{a.p} + a.m) * ((Word{a.q} + a.r + 63) / 64);
    if (a.template_words < cells || a.scratch_words < cells) return 2;
    prepare<<<1, 128, 0, reinterpret_cast<cudaStream_t>(a.stream)>>>(a);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
extern "C" std::int32_t cnot103_cuda_launch_bilinear(const Cnot103BilinearArgs* args) {
    if (!args) return 1;
    const auto& a = *args;
    if (!shape(a.m, a.r, a.p, a.q) || !a.states || !a.row_wires || !a.column_wires ||
        !a.matrix_template || !a.statistics || !a.scratch || !a.values || !a.status ||
        !a.batch || a.batch > 0x7fffffffu || a.dimension < 2 || a.dimension > (1u << 24) ||
        a.words_per_row != (a.dimension + 63) / 64 || a.m > a.dimension || a.r > a.dimension) return 1;
    const Word cells = (Word{a.p} + a.m) * ((Word{a.q} + a.r + 63) / 64);
    if (a.template_words < cells || a.batch > a.scratch_words / cells) return 2;
    evaluate<<<a.batch, 128, 0, reinterpret_cast<cudaStream_t>(a.stream)>>>(a);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
