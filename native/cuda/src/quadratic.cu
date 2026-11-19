#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace {
using U64 = std::uint64_t;
using U32 = std::uint32_t;
constexpr U32 threads = 128;
constexpr U64 exact = 1ULL << 24;

// Canonical monomials use the Rust enum order: constant, variable, product.
// Every equation has a single writer; insertion also implements x*x=x and XOR.
__device__ void toggle(U32* terms, U32& count, U32 kind, U32 left = 0, U32 right = 0) {
    if (kind == 2) {
        if (left == right) { kind = 1; right = 0; }
        else if (left > right) { const U32 tmp = left; left = right; right = tmp; }
    }
    U32 at = 0;
    while (at < count) {
        const U32* t = terms + 3 * at;
        if (t[0] > kind || (t[0] == kind && (t[1] > left || (t[1] == left && t[2] >= right)))) break;
        ++at;
    }
    const bool found = at < count && terms[3 * at] == kind && terms[3 * at + 1] == left && terms[3 * at + 2] == right;
    if (found) {
        for (U32 i = at; i + 1 < count; ++i)
            for (U32 j = 0; j < 3; ++j) terms[3 * i + j] = terms[3 * (i + 1) + j];
        --count;
        for (U32 j = 0; j < 3; ++j) terms[3 * count + j] = 0;
    } else {
        for (U32 i = count; i > at; --i)
            for (U32 j = 0; j < 3; ++j) terms[3 * i + j] = terms[3 * (i - 1) + j];
        terms[3 * at] = kind; terms[3 * at + 1] = left; terms[3 * at + 2] = right;
        ++count;
    }
}

__device__ bool allowed(const Cnot103QuadraticArgs& a, U32 control, U32 target) {
    U32 lo = 0, hi = a.allowed_count;
    const U64 key = U64(control) * a.dimension + target;
    while (lo < hi) {
        const U32 mid = lo + (hi - lo) / 2;
        const U64 other = U64(a.allowed_gates[2ULL * mid]) * a.dimension + a.allowed_gates[2ULL * mid + 1];
        if (other < key) lo = mid + 1; else hi = mid;
    }
    return lo < a.allowed_count && a.allowed_gates[2ULL * lo] == control && a.allowed_gates[2ULL * lo + 1] == target;
}

__device__ U32 emit(const Cnot103QuadraticArgs& a, const U64* state, U32 eq, U32* terms) {
    const U32 n = a.dimension, n2 = n * n, sw = n2 + 3 * n + 1;
    const U32 per = 3 + 3 * n2 - a.allowed_count;
    const U32 boundary = a.horizon ? a.horizon * per - 1 : 0;
    U32 count = 0;
    if (eq >= boundary) {
        const U32 cell = eq - boundary, row = cell / n, col = cell % n;
        const bool bit = (state[U64(row) * a.words_per_row + col / 64] >> (col % 64)) & 1;
        if (a.horizon) {
            toggle(terms, count, 1, (a.horizon - 1) * sw + 1 + 3 * n + cell);
            if (bit) toggle(terms, count, 0);
        } else if (bit != (row == col)) toggle(terms, count, 0);
        return count;
    }
    const U32 slot = eq < per - 1 ? 0 : 1 + (eq - (per - 1)) / per;
    U32 local = eq - (slot ? slot * per - 1 : 0);
    const U32 base = slot * sw;
    if (local < 2) {
        toggle(terms, count, 1, base);
        for (U32 wire = 0; wire < n; ++wire) toggle(terms, count, 1, base + 1 + local * n + wire);
        return count;
    }
    local -= 2;
    const U32 pairs = n * (n - 1) / 2;
    if (local < 2 * pairs) {
        const U32 offset = local < pairs ? 1 : 1 + n;
        U32 pair = local < pairs ? local : local - pairs;
        U32 left = 0;
        while (pair >= n - left - 1) { pair -= n - left - 1; ++left; }
        toggle(terms, count, 2, base + offset + left, base + offset + left + 1 + pair);
        return count;
    }
    local -= 2 * pairs;
    const U32 forbidden = n2 - a.allowed_count;
    if (local < forbidden) {
        for (U32 control = 0; control < n; ++control) for (U32 target = 0; target < n; ++target) {
            if (!allowed(a, control, target)) {
                if (!local) {
                    toggle(terms, count, 2, base + 1 + control, base + 1 + n + target);
                    return count;
                }
                --local;
            }
        }
    }
    local -= forbidden;
    if (slot) {
        if (!local) {
            toggle(terms, count, 1, base);
            toggle(terms, count, 2, base - sw, base);
            return count;
        }
        --local;
    }
    if (local < n) {
        const U32 col = local;
        toggle(terms, count, 1, base + 1 + 2 * n + col);
        if (!slot) toggle(terms, count, 1, base + 1 + col);
        else for (U32 row = 0; row < n; ++row)
            toggle(terms, count, 2, base + 1 + row, base - sw + 1 + 3 * n + row * n + col);
    } else {
        const U32 cell = local - n, row = cell / n, col = cell % n;
        toggle(terms, count, 1, base + 1 + 3 * n + cell);
        if (slot) toggle(terms, count, 1, base - sw + 1 + 3 * n + cell);
        else if (row == col) toggle(terms, count, 0);
        toggle(terms, count, 2, base + 1 + n + row, base + 1 + 2 * n + col);
    }
    return count;
}

__device__ void support(float* row, U32 index, U32 n, U32 sw) {
    const U32 within = index % sw;
    if (!within) for (U32 wire = 0; wire < n; ++wire) row[wire] = 1.0f;
    else if (within <= 3 * n) row[(within - 1) % n] = 1.0f;
    else { const U32 cell = within - 1 - 3 * n; row[cell / n] = row[cell % n] = 1.0f; }
}

// Exact rank, skipping stretches of empty columns. Each elimination row has
// one writer, so its pivot-bit test cannot race a write to that same word.
// Every pivot/no-pivot and row-swap path reaches the same block barriers.
__device__ U32 rank(U64* matrix, U32 rows, U64 words, unsigned long long& column_plus_one, U32& pivot_row) {
    U32 result = 0;
    while (result < rows && words) {
        if (!threadIdx.x) { column_plus_one = 0; pivot_row = rows; }
        __syncthreads();
        U64 local = 0;
        for (U64 i = U64(result) * words + threadIdx.x; i < U64(rows) * words; i += blockDim.x) {
            const U64 value = matrix[i];
            if (value) {
                const U64 col = (i % words) * 64 + 64 - __clzll(value);
                if (col > local) local = col;
            }
        }
        atomicMax(&column_plus_one, static_cast<unsigned long long>(local));
        __syncthreads();
        if (!column_plus_one) break;
        const U64 col = column_plus_one - 1;
        for (U32 row = result + threadIdx.x; row < rows; row += blockDim.x)
            if ((matrix[U64(row) * words + col / 64] >> (col % 64)) & 1) atomicMin(&pivot_row, row);
        __syncthreads();
        if (pivot_row != result) for (U64 w = threadIdx.x; w < words; w += blockDim.x) {
            const U64 tmp = matrix[U64(result) * words + w];
            matrix[U64(result) * words + w] = matrix[U64(pivot_row) * words + w];
            matrix[U64(pivot_row) * words + w] = tmp;
        }
        __syncthreads();
        for (U32 row = result + 1 + threadIdx.x; row < rows; row += blockDim.x) {
            U64* target = matrix + U64(row) * words;
            if ((target[col / 64] >> (col % 64)) & 1)
                for (U64 w = 0; w < words; ++w) target[w] ^= matrix[U64(result) * words + w];
        }
        __syncthreads();
        ++result;
    }
    // The next rank call reuses these shared scalars. In particular, finish
    // every thread's no-pivot branch before any thread can reset them.
    __syncthreads();
    return result;
}

__global__ void compile(Cnot103QuadraticArgs a) {
    const U64 b = blockIdx.x;
    const U32 n = a.dimension, sw = n * n + 3 * n + 1;
    const U32 vars = a.horizon * sw, future = vars - a.cut, channels = a.max_variables + 4;
    const U64 g = a.equations, cap = a.horizon ? n + 2 : 1;
    const U64 beta_words = (g + 63) / 64, mixed_words = (g * future + 63) / 64;
    const U64 scratch_per = U64(future) * beta_words + U64(a.cut) * mixed_words;
    const U64* state = a.states + b * n * a.words_per_row;
    U32* terms = a.terms + b * g * cap * 3;
    U32* counts = a.term_counts + b * g;
    float* values = a.values + b * g * channels;
    float* incidence = a.incidence + b * g * n;
    U64* beta = a.scratch + b * scratch_per;
    U64* mixed = beta + U64(future) * beta_words;
    for (U64 i = threadIdx.x; i < g * cap * 3; i += blockDim.x) terms[i] = 0;
    for (U64 i = threadIdx.x; i < g * channels; i += blockDim.x) values[i] = 0;
    for (U64 i = threadIdx.x; i < g * n; i += blockDim.x) incidence[i] = 0;
    for (U64 i = threadIdx.x; i < scratch_per; i += blockDim.x) beta[i] = 0;
    __syncthreads();
    for (U32 eq = threadIdx.x; eq < g; eq += blockDim.x) {
        U32* row = terms + U64(eq) * cap * 3;
        const U32 count = emit(a, state, eq, row);
        counts[eq] = count;
        a.valid[b * g + eq] = 1;
        float* v = values + U64(eq) * channels;
        float* inc = incidence + U64(eq) * n;
        v[0] = static_cast<float>(eq);
        U32 alpha = 0;
        for (U32 t = 0; t < count; ++t) {
            const U32 kind = row[3 * t], left = row[3 * t + 1], right = row[3 * t + 2];
            if (!kind) alpha ^= 1;
            else {
                support(inc, left, n, sw);
                if (kind == 1) { if (left < a.cut) alpha ^= a.prefix[left]; }
                else {
                    support(inc, right, n, sw);
                    if (right < a.cut) alpha ^= a.prefix[left] & a.prefix[right];
                    else if (left < a.cut) {
                        const U32 f = right - a.cut;
                        const U64 bit = U64(eq) * future + f;
                        atomicXor(reinterpret_cast<unsigned long long*>(mixed + U64(left) * mixed_words + bit / 64), 1ULL << (bit % 64));
                        // Stage60 beta deliberately omits linear future terms.
                        if (a.prefix[left]) {
                            v[4 + f] = 1.0f - v[4 + f];
                            atomicXor(reinterpret_cast<unsigned long long*>(beta + U64(f) * beta_words + eq / 64), 1ULL << (eq % 64));
                        }
                    }
                }
            }
        }
        v[1] = static_cast<float>(alpha);
    }
    __syncthreads();
    __shared__ unsigned long long column_plus_one;
    __shared__ U32 pivot_row;
    const U32 rank_b = rank(beta, future, beta_words, column_plus_one, pivot_row);
    const U32 mixed_rank = rank(mixed, a.cut, mixed_words, column_plus_one, pivot_row);
    for (U64 eq = threadIdx.x; eq < g; eq += blockDim.x) {
        values[eq * channels + 2] = static_cast<float>(rank_b);
        values[eq * channels + 3] = static_cast<float>(mixed_rank);
    }
    if (!threadIdx.x) { a.ranks[b * 2] = rank_b; a.ranks[b * 2 + 1] = mixed_rank; a.status[b] = 1; }
}

bool product(U64 a, U64 b, U64& result) {
    if (b && a > std::numeric_limits<U64>::max() / b) return false;
    result = a * b;
    return true;
}
}

extern "C" std::int32_t cnot103_cuda_launch_quadratic(const Cnot103QuadraticArgs* a) {
    if (!a || !a->dimension || a->dimension > exact || !a->batch || a->batch > 0x7fffffffU ||
        a->words_per_row != (U64(a->dimension) + 63) / 64 || !a->states || !a->terms || !a->term_counts ||
        !a->scratch || !a->values || !a->valid || !a->incidence || !a->ranks || !a->status ||
        (a->allowed_count && !a->allowed_gates) || (a->cut && !a->prefix)) return 1;
    const U64 n = a->dimension, n2 = n * n, sw = n2 + 3 * n + 1;
    U64 variables = 0, bound = 0;
    if (!product(sw, a->horizon, variables) || !product(3 + 3 * n2, a->horizon, bound) ||
        variables > exact || bound > exact || n2 > exact - bound || a->cut > variables ||
        !a->max_variables || a->max_variables > exact - 4 || variables > a->max_variables ||
        a->allowed_count > n * (n - 1)) return 1;
    const U64 g = a->horizon ? U64(a->horizon) * (3 + 3 * n2 - a->allowed_count) - 1 + n2 : n2;
    if (a->equations != g) return 1;
    const U64 future = variables - a->cut, cap = a->horizon ? n + 2 : 1;
    U64 beta_size = 0, mixed_size = 0;
    if (!product(future, (g + 63) / 64, beta_size) ||
        !product(a->cut, (g * future + 63) / 64, mixed_size) ||
        beta_size > std::numeric_limits<U64>::max() - mixed_size) return 1;
    const U64 scratch_per = beta_size + mixed_size;
    U64 groups = 0, terms = 0, values = 0, incidence = 0, scratch = 0, states = 0;
    if (!product(a->batch, g, groups) || !product(groups, cap * 3, terms) ||
        !product(groups, U64(a->max_variables) + 4, values) || !product(groups, n, incidence) ||
        !product(a->batch, scratch_per, scratch) || !product(U64(a->batch) * n, a->words_per_row, states)) return 1;
    if (a->state_words < states || a->term_elements < terms || a->group_elements < groups ||
        a->value_elements < values || a->incidence_elements < incidence || a->scratch_words < scratch ||
        a->rank_elements < U64(a->batch) * 2) return 2;
    compile<<<a->batch, threads, 0, static_cast<cudaStream_t>(a->stream)>>>(*a);
    return static_cast<std::int32_t>(cudaGetLastError());
}
