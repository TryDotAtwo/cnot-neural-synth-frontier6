#pragma once
#include "cnot103/shared_word.h"
#include <cuda_runtime.h>

namespace cnot103_shared_word {
constexpr int workers_per_block = 128;
constexpr unsigned full_warp = 0xffffffffu;
using Eval = Cnot103SharedWordEval;

__device__ int warp_sum(int x) {
    for (int offset = 16; offset; offset /= 2)
        x += __shfl_down_sync(full_warp, x, offset);
    return x;
}
enum Moment {
    weight1,
    weight2,
    character1,
    character2,
    weight_product,
    weight_intersection,
    weight_xor,
    intersection,
    intersection_squared,
    xor_weight,
    xor_squared,
    character_product,
    moment_count
};
struct Totals {
    int value[moment_count];
};

__device__ double sample_cov(int product, int first_sum, double second_mean, double divisor) {
    return __ddiv_rn(__dsub_rn(double(product), __dmul_rn(first_sum, second_mean)), divisor);
}
__device__ double standard_error(int sum, int squares, double samples) {
    const double centered = __dsub_rn(squares, __ddiv_rn(__dmul_rn(sum, sum), samples));
    const double variance = fmax(0., __ddiv_rn(centered, samples - 1.));
    return __ddiv_rn(__dsqrt_rn(variance), __dsqrt_rn(samples));
}

__global__ void node_means(Eval a, U32 start, U32 count) {
    const U32 n = a.config.dimension, samples = a.config.samples;
    __shared__ int partial[workers_per_block / 32];
    for (U64 node = blockIdx.x; node < U64(count) * n; node += gridDim.x) {
        int sum = 0;
        if (*a.profile_status == 1) {
            for (U32 s = threadIdx.x; s < samples; s += blockDim.x)
                sum += a.weights[node * samples + s];
        }
        sum = warp_sum(sum);
        if ((threadIdx.x & 31) == 0)
            partial[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            int total = 0;
            for (int v : partial)
                total += v;
            a.values[0][U64(start) * n + node] =
                __double2float_rn(__ddiv_rn(total, double(samples)));
        }
        __syncthreads();
    }
}

__device__ void observe(Totals& totals, const Eval& a, U64 left, U64 right, U32 s, U32 words,
                        U32 samples) {
    const int x = a.weights[left * samples + s], y = a.weights[right * samples + s];
    int both = 0;
    for (U32 k = 0; k < words; ++k)
        both += __popc(a.sample_columns[(left * words + k) * samples + s] &
                       a.sample_columns[(right * words + k) * samples + s]);
    const int different = x + y - 2 * both, lx = (x & 1) ? -1 : 1, ry = (y & 1) ? -1 : 1;
    auto& v = totals.value;
    v[weight1] += x;
    v[weight2] += y;
    v[character1] += lx;
    v[character2] += ry;
    v[weight_product] += x * y;
    v[weight_intersection] += x * both;
    v[weight_xor] += x * different;
    v[intersection] += both;
    v[intersection_squared] += both * both;
    v[xor_weight] += different;
    v[xor_squared] += different * different;
    v[character_product] += lx * ry;
}

__device__ void publish(const Eval& a, const Totals& totals, U64 pair, U32 first, U32 second) {
    const auto& v = totals.value;
    const double samples = a.config.samples, divisor = samples - 1.;
    const double mean_inter = __ddiv_rn(v[intersection], samples);
    const double mean_xor = __ddiv_rn(v[xor_weight], samples);
    const double mean_char = __ddiv_rn(v[character_product], samples);
    if (a.selected & 2) {
        float* g = a.values[1] + 5 * pair;
        g[0] = sample_cov(v[weight_product], v[weight1], __ddiv_rn(v[weight2], samples), divisor);
        g[1] = mean_inter;
        g[2] = mean_xor;
        g[3] = sample_cov(v[weight_intersection], v[weight1], mean_inter, divisor);
        g[4] = sample_cov(v[weight_xor], v[weight1], mean_xor, divisor);
    }
    if (a.selected & 4) {
        float* g = a.values[2] + 2 * pair;
        g[0] = mean_char;
        g[1] = __dsub_rn(mean_char, __dmul_rn(__ddiv_rn(v[character1], samples),
                                              __ddiv_rn(v[character2], samples)));
    }
    if (a.selected & 8) {
        float* g = a.values[3] + 7 * pair;
        const U64 sample = pair / (U64(a.config.dimension) * a.config.dimension);
        if (a.statuses[3][sample] != 1) {
            for (int k = 0; k < 7; ++k)
                g[k] = 0.f;
        } else {
            const double stationary =
                first == second ? 1. : __ddiv_rn(-1., __dsub_rn(ldexp(1., a.config.dimension), 1.));
            g[0] = standard_error(v[intersection], v[intersection_squared], samples);
            g[1] = standard_error(v[xor_weight], v[xor_squared], samples);
            g[2] = standard_error(v[character_product], int(a.config.samples), samples);
            g[3] = fabs(__dsub_rn(mean_char, stationary));
            g[4] = samples;
            g[5] = __ddiv_rn(double(*a.total_gates), samples);
            g[6] = a.config.tau;
        }
    }
}

__global__ void pair_moments(Eval a, U32 start, U32 count) {
    const U32 n = a.config.dimension, samples = a.config.samples, words = (n + 31) / 32;
    const U64 cells = U64(n) * n;
    const U32 lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    __shared__ Totals partial[workers_per_block / 32];
    for (U64 pair = blockIdx.x; pair < U64(count) * cells; pair += gridDim.x) {
        const U64 output = U64(start) * cells + pair;
        if (*a.profile_status != 1) {
            constexpr int widths[] = {1, 5, 2, 7};
            for (int f = 1; f < 4; ++f)
                if ((a.selected & (1u << f)) && threadIdx.x < widths[f])
                    a.values[f][output * widths[f] + threadIdx.x] = 0.f;
            continue;
        }
        const U64 base = (pair / cells) * n;
        const U32 first = (pair / n) % n, second = pair % n;
        Totals values{};
        for (U32 s = threadIdx.x; s < samples; s += blockDim.x)
            observe(values, a, base + first, base + second, s, words, samples);
        for (int k = 0; k < moment_count; ++k) {
            const int sum = warp_sum(values.value[k]);
            if (lane == 0)
                partial[warp].value[k] = sum;
        }
        __syncthreads();
        if (warp == 0) {
            Totals result{};
            for (int k = 0; k < moment_count; ++k)
                result.value[k] =
                    warp_sum(lane < workers_per_block / 32 ? partial[lane].value[k] : 0);
            if (lane == 0)
                publish(a, result, output, first, second);
        }
        __syncthreads();
    }
}
} // namespace cnot103_shared_word
