#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace {
using Word = std::uint64_t;
using Index = std::uint32_t;
constexpr Word absent = ~Word{0};

__host__ __device__ Word certificate_size(Index n, Index w) {
    return 6 + Word{n} + (4 * Word{n} + 2) * w;
}
__host__ __device__ Word workspace_size(Index n, Index w) {
    return (7 * Word{n} + 4) * w + n + 1;
}
struct Certificate {
    Word *counts, *witnesses, *retained, *deleted, *rows, *projected, *h, *targets;
    __device__ Certificate(Word* base, Index n, Index w) {
        counts = base; witnesses = counts + 6; retained = witnesses + n;
        deleted = retained + w; rows = deleted + w; projected = rows + Word{n} * w;
        h = projected + Word{n} * w; targets = h + Word{n} * w;
    }
};
struct Workspace {
    Word *pool, *cosets, *basis, *low_h, *row_cosets, *tmp, *other, *changed, *active, *multiplicity, *cheap;
    __device__ Workspace(Word* base, Index n, Index w) {
        pool = base; cosets = pool + 2 * Word{n} * w; basis = cosets + 2 * Word{n} * w;
        low_h = basis + Word{n} * w; row_cosets = low_h + Word{n} * w;
        tmp = row_cosets + Word{n} * w; other = tmp + w; changed = other + w;
        active = changed + w; multiplicity = active + w; cheap = multiplicity + n;
    }
};
__device__ void clear(Word* v, Word size) {
    for (Word k = 0; k < size; ++k) v[k] = 0;
}
__device__ void copy(Word* dst, const Word* src, Index w) {
    for (Index k = 0; k < w; ++k) dst[k] = src[k];
}
__device__ void add(Word* dst, const Word* src, Index w) {
    for (Index k = 0; k < w; ++k) dst[k] ^= src[k];
}
__device__ bool zero(const Word* v, Index w) {
    for (Index k = 0; k < w; ++k) if (v[k]) return false;
    return true;
}
__device__ bool equal(const Word* a, const Word* b, Index w) {
    for (Index k = 0; k < w; ++k) if (a[k] != b[k]) return false;
    return true;
}
__device__ bool bit(const Word* v, Index p) { return (v[p / 64] >> (p % 64)) & 1; }
__device__ void set(Word* v, Index p) { v[p / 64] |= Word{1} << (p % 64); }
__device__ bool intersects(const Word* a, const Word* b, Index w) {
    for (Index k = 0; k < w; ++k) if (a[k] & b[k]) return true;
    return false;
}
__device__ int highest(const Word* v, Index w) {
    for (Index k = w; k > 0; --k) if (v[k - 1])
        return static_cast<int>(64 * (k - 1) + 63 - __clzll(v[k - 1]));
    return -1;
}
__device__ int lowest(const Word* v, Index w) {
    for (Index k = 0; k < w; ++k) if (v[k])
        return static_cast<int>(64 * k + __ffsll(v[k]) - 1);
    return -1;
}
__device__ bool unit(const Word* v, Index w) {
    bool found = false;
    for (Index k = 0; k < w; ++k) if (v[k]) {
        if (found || (v[k] & (v[k] - 1))) return false;
        found = true;
    }
    return found;
}
__device__ bool less(const Word* a, const Word* b, Index w) {
    for (Index k = w; k > 0; --k) if (a[k - 1] != b[k - 1]) return a[k - 1] < b[k - 1];
    return false;
}
__device__ void sort(Word* rows, Index count, Index w, Word* tmp) {
    for (Index i = 1; i < count; ++i) {
        copy(tmp, rows + Word{i} * w, w);
        Index j = i;
        while (j && less(tmp, rows + Word{j - 1} * w, w)) {
            copy(rows + Word{j} * w, rows + Word{j - 1} * w, w); --j;
        }
        copy(rows + Word{j} * w, tmp, w);
    }
}
__device__ bool high_insert(Word* basis, Word* value, Index w) {
    for (int p = highest(value, w); p >= 0; p = highest(value, w)) {
        Word* row = basis + Word{static_cast<Index>(p)} * w;
        if (bit(row, p)) add(value, row, w);
        else { copy(row, value, w); return true; }
    }
    return false;
}
__device__ void high_canonicalize(Word* basis, Index n, Index w) {
    for (Index low = 0; low < n; ++low) if (bit(basis + Word{low} * w, low))
        for (Index high = low + 1; high < n; ++high)
            if (bit(basis + Word{high} * w, low)) add(basis + Word{high} * w, basis + Word{low} * w, w);
}
__device__ void high_remainder(Word* value, const Word* basis, Index n, Index w) {
    for (Index p = n; p > 0; --p) if (bit(value, p - 1)) add(value, basis + Word{p - 1} * w, w);
}
// Checker deliberately uses low pivots and all outside-pool pairs. Its
// remainder stops at the first absent low pivot, exactly like the Rust source.
__device__ bool low_insert(Word* basis, Word* value, Index w) {
    for (int p = lowest(value, w); p >= 0; p = lowest(value, w)) {
        Word* row = basis + Word{static_cast<Index>(p)} * w;
        if (bit(row, p)) add(value, row, w);
        else { copy(row, value, w); return true; }
    }
    return false;
}
__device__ bool in_low_span(Word* value, const Word* basis, Index w) {
    for (int p = lowest(value, w); p >= 0; p = lowest(value, w)) {
        const Word* row = basis + Word{static_cast<Index>(p)} * w;
        if (!bit(row, p)) return false;
        add(value, row, w);
    }
    return true;
}

__global__ void construct(Cnot103Stage26Args a) {
    const Index n = a.dimension, w = a.words_per_row;
    const Word sample = blockIdx.x, cw = certificate_size(n, w), sw = workspace_size(n, w);
    Word* base = a.certificate + sample * cw;
    Word* work = a.scratch + sample * sw;
    const Word* input = a.states + sample * Word{n} * w;
    Certificate c(base, n, w); Workspace s(work, n, w);
    for (Word k = threadIdx.x; k < cw; k += blockDim.x) base[k] = 0;
    for (Word k = threadIdx.x; k < sw; k += blockDim.x) work[k] = 0;
    for (Word k = threadIdx.x; k < 3 * Word{n} * n; k += blockDim.x) a.f04[sample * 3 * Word{n} * n + k] = 0;
    if (threadIdx.x == 0) {
        a.verified[sample] = 0; a.status[sample] = 0;
        for (Index k = 0; k < 5; ++k) a.f03[sample * 5 + k] = 0;
        a.f05[sample * 2] = 0; a.f05[sample * 2 + 1] = 0;
    }
    __syncthreads();
    for (Word k = threadIdx.x; k < Word{n} * w; k += blockDim.x) c.rows[k] = input[k];
    for (Index row = threadIdx.x; row < n; row += blockDim.x) {
        c.witnesses[row] = absent;
        bool changed = false, off_diagonal = false;
        for (Index k = 0; k < w; ++k) {
            const Word identity = k == row / 64 ? Word{1} << (row % 64) : 0;
            changed |= input[Word{row} * w + k] != identity;
        }
        for (Index i = 0; i < n; ++i) off_diagonal |= i != row && bit(input + Word{i} * w, row);
        const bool retained = !changed || off_diagonal;
        const auto mask = static_cast<unsigned long long>(Word{1} << (row % 64));
        if (retained) atomicOr(reinterpret_cast<unsigned long long*>(c.retained + row / 64), mask);
        if (changed && retained) atomicOr(reinterpret_cast<unsigned long long*>(s.active + row / 64), mask);
    }
    __syncthreads();
    for (Word k = threadIdx.x; k < Word{n} * w; k += blockDim.x) c.projected[k] = input[k] & c.retained[k % w];
    __syncthreads();
    if (threadIdx.x != 0) return;

    // DeviceStateBatch checks metadata; malformed packed input must not turn
    // a high padding bit into an out-of-bounds basis pivot. Valid MatrixState
    // inputs have zero padding. Keep this invariant check on device as well.
    if (n % 64) for (Index row = 0; row < n; ++row)
        if (input[Word{row} * w + w - 1] >> (n % 64)) { asm volatile("trap;"); return; }

    for (Index row = 0; row < n; ++row) {
        const Word* value = c.projected + Word{row} * w;
        if (!zero(value, w)) for (Index column = 0; column < n; ++column)
            if (!bit(c.retained, column) && bit(c.rows + Word{row} * w, column) && !bit(c.deleted, column)) {
                set(c.deleted, column); c.witnesses[column] = row; ++c.counts[0];
            }
        if (zero(value, w) || unit(value, w)) continue;
        bool duplicate = false;
        for (Index i = 0; i < c.counts[1]; ++i) duplicate |= equal(value, c.targets + Word{i} * w, w);
        if (!duplicate) { copy(c.targets + c.counts[1] * w, value, w); ++c.counts[1]; }
    }
    sort(c.targets, static_cast<Index>(c.counts[1]), w, s.tmp);
    for (Index row = 0; row < n; ++row) if (intersects(c.rows + Word{row} * w, s.active, w)) {
        copy(s.tmp, c.projected + Word{row} * w, w);
        c.counts[3] += high_insert(c.h, s.tmp, w);
    }
    high_canonicalize(c.h, n, w);
    Index pool_count = 0;
    for (Index wire = 0; wire < n; ++wire) if (bit(c.retained, wire)) set(s.pool + Word{pool_count++} * w, wire);
    for (Index i = 0; i < c.counts[1]; ++i) copy(s.pool + Word{pool_count++} * w, c.targets + Word{i} * w, w);
    sort(s.pool, pool_count, w, s.tmp);
    for (Index i = 0; i < pool_count; ++i) {
        Word* coset = s.cosets + Word{i} * w;
        const Word* value = s.pool + Word{i} * w;
        copy(coset, value, w); high_remainder(coset, c.h, n, w);
        if (zero(coset, w)) {
            if (unit(value, w)) { copy(s.tmp, value, w); c.counts[4] += high_insert(s.basis, s.tmp, w); }
        } else {
            for (Index j = 0; j < i; ++j) if (equal(coset, s.cosets + Word{j} * w, w)) {
                copy(s.tmp, value, w); add(s.tmp, s.pool + Word{j} * w, w);
                c.counts[4] += high_insert(s.basis, s.tmp, w); break;
            }
        }
    }
    // Canonicalization does not change ranks, but preserves the source basis.
    high_canonicalize(s.basis, n, w);
    for (Index i = 0; i < c.counts[1]; ++i) {
        copy(s.tmp, c.targets + Word{i} * w, w); high_remainder(s.tmp, c.h, n, w);
        if (zero(s.tmp, w)) {
            copy(s.tmp, c.targets + Word{i} * w, w);
            c.counts[2] += high_insert(s.basis, s.tmp, w);
        }
    }
    c.counts[5] = c.counts[0] + c.counts[1] + c.counts[2];
    // Elementary bound: changed rows/columns and rank(M xor I), in integer GF(2).
    clear(s.basis, Word{n} * w); clear(s.changed, w);
    Index changed_rows = 0, rank = 0;
    for (Index row = 0; row < n; ++row) {
        copy(s.tmp, c.rows + Word{row} * w, w); s.tmp[row / 64] ^= Word{1} << (row % 64);
        changed_rows += !zero(s.tmp, w);
        for (Index k = 0; k < w; ++k) s.changed[k] |= s.tmp[k];
        rank += high_insert(s.basis, s.tmp, w);
    }
    Index changed_columns = 0;
    for (Index k = 0; k < w; ++k) changed_columns += __popcll(s.changed[k]);
    *s.cheap = max(rank, max(changed_rows, changed_columns));
}

__global__ void independently_check(Cnot103Stage26Args a) {
    if (threadIdx.x != 0) return;
    const Index n = a.dimension, w = a.words_per_row;
    const Word sample = blockIdx.x;
    Certificate c(a.certificate + sample * certificate_size(n, w), n, w);
    Workspace s(a.scratch + sample * workspace_size(n, w), n, w);
    const Word* input = a.states + sample * Word{n} * w;
    a.verified[sample] = 0;
    if (c.counts[0] > n || c.counts[1] > n || c.counts[2] > n || c.counts[3] > n || c.counts[4] > n) return;
    if (n % 64 && ((c.retained[w - 1] | c.deleted[w - 1]) >> (n % 64))) return;
    for (Word k = 0; k < Word{n} * w; ++k) {
        if (c.rows[k] != input[k] || c.projected[k] != (input[k] & c.retained[k % w])) return;
    }
    Word deletions = 0;
    for (Index column = 0; column < n; ++column) {
        bool deleted = false;
        if (!bit(c.retained, column)) for (Index row = 0; row < n; ++row)
            deleted |= bit(input + Word{row} * w, column) && intersects(input + Word{row} * w, c.retained, w);
        if (deleted != bit(c.deleted, column)) return;
        if (deleted) {
            ++deletions;
            const Word witness = c.witnesses[column];
            if (witness >= n || !bit(input + witness * w, column) || !intersects(input + witness * w, c.retained, w)) return;
        } else if (c.witnesses[column] != absent) return;
    }
    if (deletions != c.counts[0]) return;
    // Reconstruct unique targets from the independently reprojected original
    // rows. Do not accept the constructor's pool or target counts as evidence.
    clear(s.pool, 2 * Word{n} * w);
    Index targets = 0;
    for (Index row = 0; row < n; ++row) {
        for (Index k = 0; k < w; ++k) s.tmp[k] = input[Word{row} * w + k] & c.retained[k];
        if (zero(s.tmp, w) || unit(s.tmp, w)) continue;
        bool duplicate = false;
        for (Index j = 0; j < targets; ++j) duplicate |= equal(s.tmp, s.pool + Word{j} * w, w);
        if (!duplicate) copy(s.pool + Word{targets++} * w, s.tmp, w);
    }
    if (targets != c.counts[1]) return;
    sort(s.pool, targets, w, s.tmp);
    for (Index i = 0; i < n; ++i) {
        if (i < targets) { if (!equal(s.pool + Word{i} * w, c.targets + Word{i} * w, w)) return; }
        else if (!zero(c.targets + Word{i} * w, w)) return;
    }
    clear(s.low_h, Word{n} * w);
    Index h_dimension = 0;
    for (Index p = 0; p < n; ++p) {
        const Word* value = c.h + Word{p} * w;
        if (zero(value, w)) continue;
        if (highest(value, w) != static_cast<int>(p)) return;
        for (Index k = 0; k < w; ++k) if (value[k] & ~c.retained[k]) return;
        copy(s.tmp, value, w); h_dimension += low_insert(s.low_h, s.tmp, w);
    }
    if (h_dimension != c.counts[3]) return;
    Index pool_count = targets;
    for (Index wire = 0; wire < n; ++wire) if (bit(c.retained, wire)) set(s.pool + Word{pool_count++} * w, wire);
    sort(s.pool, pool_count, w, s.tmp);
    clear(s.basis, Word{n} * w);
    Index w0_dimension = 0;
    for (Index i = 0; i < pool_count; ++i) {
        Word* remainder = s.cosets + Word{i} * w;
        copy(remainder, s.pool + Word{i} * w, w);
        const bool inside = in_low_span(remainder, s.low_h, w);
        if (inside && unit(s.pool + Word{i} * w, w)) {
            copy(s.tmp, s.pool + Word{i} * w, w); w0_dimension += low_insert(s.basis, s.tmp, w);
        }
    }
    for (Index left = 0; left < pool_count; ++left) if (!zero(s.cosets + Word{left} * w, w))
        for (Index right = left + 1; right < pool_count; ++right) if (!zero(s.cosets + Word{right} * w, w)) {
            copy(s.tmp, s.pool + Word{left} * w, w); add(s.tmp, s.pool + Word{right} * w, w);
            copy(s.other, s.tmp, w);
            if (in_low_span(s.other, s.low_h, w)) w0_dimension += low_insert(s.basis, s.tmp, w);
        }
    if (w0_dimension != c.counts[4]) return;
    Index obstruction = 0;
    for (Index i = 0; i < targets; ++i) {
        copy(s.tmp, c.targets + Word{i} * w, w);
        if (in_low_span(s.tmp, s.low_h, w)) {
            copy(s.tmp, c.targets + Word{i} * w, w); obstruction += low_insert(s.basis, s.tmp, w);
        }
    }
    if (obstruction != c.counts[2] || c.counts[5] != deletions + targets + obstruction) return;
    a.verified[sample] = 1;
}

__global__ void publish(Cnot103Stage26Args a) {
    const Word sample = blockIdx.x;
    if (!a.verified[sample]) {
        // Unlike assert(), this is not removed by NDEBUG. A rejected proof is
        // fatal, matching the source panic, without numerical host readback.
        if (threadIdx.x == 0) asm volatile("trap;");
        return;
    }
    const Index n = a.dimension, w = a.words_per_row;
    Certificate c(a.certificate + sample * certificate_size(n, w), n, w);
    Workspace s(a.scratch + sample * workspace_size(n, w), n, w);
    for (Index row = threadIdx.x; row < n; row += blockDim.x) {
        Word* coset = s.row_cosets + Word{row} * w;
        copy(coset, c.projected + Word{row} * w, w); high_remainder(coset, c.h, n, w);
        Word count = 0;
        for (Index j = 0; j < n; ++j) count += equal(c.projected + Word{row} * w, c.projected + Word{j} * w, w);
        s.multiplicity[row] = count;
    }
    __syncthreads();
    for (Word pair = threadIdx.x; pair < Word{n} * n; pair += blockDim.x) {
        const Index row = pair / n, column = pair % n;
        const Word* coset = s.row_cosets + Word{row} * w;
        float* dst = a.f04 + (sample * Word{n} * n + pair) * 3;
        dst[0] = !zero(coset, w) && equal(coset, s.row_cosets + Word{column} * w, w);
        dst[1] = equal(c.projected + Word{row} * w, c.projected + Word{column} * w, w);
        dst[2] = static_cast<float>(s.multiplicity[row]);
    }
    if (threadIdx.x == 0) {
        for (Index k = 0; k < 5; ++k) a.f03[sample * 5 + k] = static_cast<float>(c.counts[k]);
        a.f05[sample * 2] = static_cast<float>(c.counts[5]);
        a.f05[sample * 2 + 1] = static_cast<float>(c.counts[5]) - static_cast<float>(*s.cheap);
        a.status[sample] = 1;
    }
}

bool fits(Word a, Word b, Word available) { return !b || a <= available / b; }
int validate(const Cnot103Stage26Args* args) {
    if (!args) return 1;
    const auto& a = *args;
    if (!a.states || !a.certificate || !a.scratch || !a.verified || !a.status || !a.f03 || !a.f04 || !a.f05) return 1;
    const Index n = a.dimension, w = a.words_per_row;
    if (!a.batch || a.batch > 0x7fffffffu || n < 2 || n > (1u << 24) || w != (n + 63) / 64) return 1;
    if (!fits(a.batch, certificate_size(n, w), a.certificate_words) ||
        !fits(a.batch, workspace_size(n, w), a.workspace_words) ||
        !fits(a.batch, 3 * Word{n} * n, a.pair_elements)) return 2;
    return 0;
}
}
extern "C" std::int32_t cnot103_cuda_stage26_construct(const Cnot103Stage26Args* args) {
    const int error = validate(args); if (error) return error;
    construct<<<args->batch, 128, 0, reinterpret_cast<cudaStream_t>(args->stream)>>>(*args);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
extern "C" std::int32_t cnot103_cuda_stage26_check(const Cnot103Stage26Args* args) {
    const int error = validate(args); if (error) return error;
    independently_check<<<args->batch, 1, 0, reinterpret_cast<cudaStream_t>(args->stream)>>>(*args);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
extern "C" std::int32_t cnot103_cuda_stage26_publish(const Cnot103Stage26Args* args) {
    const int error = validate(args); if (error) return error;
    publish<<<args->batch, 128, 0, reinterpret_cast<cudaStream_t>(args->stream)>>>(*args);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
extern "C" std::int32_t cnot103_cuda_launch_stage26(const Cnot103Stage26Args* args) {
    int error = cnot103_cuda_stage26_construct(args); if (error) return error;
    error = cnot103_cuda_stage26_check(args); if (error) return error;
    return cnot103_cuda_stage26_publish(args);
}
