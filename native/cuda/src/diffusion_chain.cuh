#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace cnot103_diffusion {
using U32 = std::uint32_t;
using U64 = std::uint64_t;
using Byte = std::uint8_t;
constexpr Byte computed = 1, inapplicable = 2, unsupported = 3, limited = 4;
constexpr int threads = 128, max_terms = 1'000'000;
constexpr U64 byte_limit = std::numeric_limits<std::int64_t>::max();

template <int Edges, class Index> struct ProfileWork {
    double *log_size, *stationary, *probability, *steps, *mass;
    __device__ ProfileWork(double* base, Index count)
        : log_size(base), stationary(base + count), probability(base + 2 * count),
          steps(base + (2 + Edges) * count), mass(base + (10 + Edges) * count) {}
};

inline bool weight_capacity(const double (&taus)[4], U32 dimension, bool available, U32& stride) {
    stride = available ? 1 : 0;
    for (double tau : taus) {
        if (!std::isfinite(tau) || tau < 0)
            return false;
        // Size scratch using static config only. The device independently
        // decides whether the source truncation fits its one-million-term cap.
        const double time = dimension * tau;
        if (!available || time == 0 || time > max_terms)
            continue;
        const double bound = std::ceil(time + 12. * std::sqrt(time) + 32.) + 2.;
        stride = std::max(stride, static_cast<U32>(std::min(bound, double(max_terms + 1))));
    }
    return true;
}

// The centered recurrence and its normalization follow the source operation
// order. The omitted prefix [0,lower) has zero Poisson weights.
__device__ inline int poisson(double time, double* weights, int stride) {
    const double margin = __dadd_rn(__dmul_rn(12.0, __dsqrt_rn(time)), 32.0);
    const double last = ceil(__dadd_rn(time, margin));
    if (!isfinite(last) || last > max_terms)
        return -1;
    const int upper = static_cast<int>(last);
    if (upper >= stride)
        asm volatile("trap;");
    const int mode = static_cast<int>(floor(time));
    const int lower = static_cast<int>(fmax(0., floor(__dsub_rn(time, margin))));
    for (int i = 0; i < lower; ++i)
        weights[i] = 0.;
    weights[mode] = 1.;
    for (int i = mode; i > lower; --i)
        weights[i - 1] = __ddiv_rn(__dmul_rn(weights[i], i), time);
    for (int i = mode; i < upper; ++i)
        weights[i + 1] = __ddiv_rn(__dmul_rn(weights[i], time), i + 1);
    double sum = 0.;
    for (int i = lower; i <= upper; ++i)
        sum = __dadd_rn(sum, weights[i]);
    if (!isfinite(sum) || sum <= 0)
        return -1;
    for (int i = lower; i <= upper; ++i)
        weights[i] = __ddiv_rn(weights[i], sum);
    return upper;
}

template <int Edges, class Args, class Index>
__global__ void propagate(Args a, Index classes, Index initial, int stride) {
    const int horizon = blockIdx.x;
    const double time = __dmul_rn(a.config.dimension, a.config.taus[horizon]);
    ProfileWork<Edges, Index> w(a.work, classes);
    double* weights = a.weights + horizon * stride;
    double* mass = w.mass + horizon * classes;
    double* current = w.steps + 2 * horizon * classes;
    double* next = current + classes;
    __shared__ int upper;
    if (threadIdx.x == 0) {
        upper = -1;
        if (a.phase_status[0] == computed && isfinite(time)) {
            if (time == 0.) {
                weights[0] = 1.;
                upper = 0;
            } else
                upper = poisson(time, weights, stride);
        }
        a.phase_status[horizon + 1] = upper >= 0 ? computed : limited;
    }
    __syncthreads();
    if (upper < 0)
        return;
    for (Index i = threadIdx.x; i < classes; i += blockDim.x) {
        current[i] = i == initial ? 1. : 0.;
        mass[i] = 0.;
    }
    __syncthreads();
    for (int step = 0; step <= upper; ++step) {
        const double weight = weights[step];
        for (Index dest = threadIdx.x; dest < classes; dest += blockDim.x) {
            if (weight != 0.)
                mass[dest] = __dadd_rn(mass[dest], __dmul_rn(weight, current[dest]));
            if (step == upper)
                continue;
            double sum = 0.;
            for (int slot = 0; slot < Edges; ++slot) {
                const Index edge = slot * classes + dest;
                const Index source = a.predecessors[edge];
                if (source < 0)
                    break;
                sum = __dadd_rn(sum, __dmul_rn(current[source], w.probability[edge]));
            }
            next[dest] = sum;
        }
        __syncthreads();
        double* previous = current;
        current = next;
        next = previous;
    }
    __shared__ double total;
    if (threadIdx.x == 0) {
        double sum = 0.;
        for (Index i = 0; i < classes; ++i)
            sum = __dadd_rn(sum, mass[i]);
        total = sum;
        if (!isfinite(sum) || sum <= 0)
            a.phase_status[horizon + 1] = limited;
    }
    __syncthreads();
    if (a.phase_status[horizon + 1] == computed)
        for (Index i = threadIdx.x; i < classes; i += blockDim.x)
            mass[i] = __ddiv_rn(mass[i], total);
}

inline bool extent(U64 required, U64 supplied, U64 size) {
    return required <= supplied && supplied <= byte_limit / size;
}

inline bool total_fits(std::initializer_list<U64> byte_counts) {
    U64 total = 0;
    for (const U64 bytes : byte_counts) {
        if (bytes > byte_limit - total)
            return false;
        total += bytes;
    }
    return true;
}

inline bool readable_view(U32 batch, U64 n, U64 batch_stride, U64 wire_stride, U64 supplied) {
    if (!wire_stride || (n > 1 && wire_stride > (byte_limit / sizeof(float) - 1) / (n - 1)))
        return false;
    const U64 row = (n - 1) * wire_stride + 1;
    if (batch > 1 &&
        (batch_stride < row || batch_stride > (byte_limit / sizeof(float) - row) / (batch - 1)))
        return false;
    return extent((batch - 1) * batch_stride + row, supplied, sizeof(float));
}
} // namespace cnot103_diffusion
