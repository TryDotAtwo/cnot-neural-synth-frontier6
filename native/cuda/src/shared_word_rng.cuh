#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cnot103_shared_word {
using U32 = std::uint32_t;
using U64 = std::uint64_t;

// NumPy 2.2.6 SeedSequence + PCG64, with the same interleaving of cached
// 32-bit bounded integers and 64-bit Poisson draws as the Rust provider.
struct NumpyRng {
    U64 lo = 0, hi = 0, inc_lo, inc_hi;
    U32 cached = 0;
    bool has_cached = false;

    __device__ static U32 hash_mix(U32 value, U32& factor) {
        value ^= factor;
        factor *= 0x931e8875u;
        value *= factor;
        return value ^ (value >> 16);
    }
    __device__ explicit NumpyRng(U64 seed) {
        U32 pool[4], words[8], factor = 0x43b0d7e5u;
        for (int i = 0; i < 4; ++i) {
            const U32 entropy = i == 0 ? U32(seed) : (i == 1 ? U32(seed >> 32) : 0);
            pool[i] = hash_mix(entropy, factor);
        }
        for (int i = 0; i < 4; ++i) {
            const U32 source = pool[i];
            for (int j = 0; j < 4; ++j)
                if (i != j) {
                    const U32 mixed =
                        0xca01f9ddu * pool[j] - 0x4973f715u * hash_mix(source, factor);
                    pool[j] = mixed ^ (mixed >> 16);
                }
        }
        factor = 0x8b51f9ddu;
        for (int i = 0; i < 8; ++i) {
            U32 x = pool[i % 4] ^ factor;
            factor *= 0x58f38dedu;
            x *= factor;
            words[i] = x ^ (x >> 16);
        }
        const U64 state_hi = U64(words[0]) | (U64(words[1]) << 32);
        const U64 state_lo = U64(words[2]) | (U64(words[3]) << 32);
        const U64 seq_hi = U64(words[4]) | (U64(words[5]) << 32);
        const U64 seq_lo = U64(words[6]) | (U64(words[7]) << 32);
        inc_lo = (seq_lo << 1) | 1;
        inc_hi = (seq_hi << 1) | (seq_lo >> 63);
        step();
        const U64 previous = lo;
        lo += state_lo;
        hi += state_hi + (lo < previous);
        step();
    }
    __device__ void step() {
        constexpr U64 ml = 0x4385df649fccf645ULL, mh = 0x2360ed051fc65da4ULL;
        const U64 product = lo * ml;
        hi = __umul64hi(lo, ml) + hi * ml + lo * mh + inc_hi + (product + inc_lo < product);
        lo = product + inc_lo;
    }
    __device__ U64 u64() {
        step();
        const U64 x = hi ^ lo;
        const U32 rot = hi >> 58;
        return (x >> rot) | (x << ((-rot) & 63));
    }
    __device__ U32 u32() {
        if (has_cached) {
            has_cached = false;
            return cached;
        }
        const U64 x = u64();
        cached = U32(x >> 32);
        has_cached = true;
        return U32(x);
    }
    __device__ double uniform() { return __dmul_rn(double(u64() >> 11), 0x1p-53); }
    __device__ U32 integer(U32 bound) {
        if (bound == 1)
            return 0;
        const U32 threshold = (-bound) % bound;
        for (;;) {
            const U64 product = U64(u32()) * bound;
            if (U32(product) >= threshold)
                return U32(product >> 32);
        }
    }
    __device__ static double log_gamma(double x) {
        constexpr double coefficients[] = {8.333333333333333e-02, -2.777777777777778e-03,
                                           7.936507936507937e-04, -5.952380952380952e-04,
                                           8.417508417508418e-04, -1.917526917526918e-03,
                                           6.41025641025641e-3,   -2.955065359477124e-02,
                                           1.796443723688307e-01, -1.39243221690590};
        if (x == 1. || x == 2.)
            return 0.;
        const int n = x < 7. ? int(7. - x) : 0;
        double x0 = __dadd_rn(x, double(n));
        const double inv = __ddiv_rn(1., x0), x2 = __dmul_rn(inv, inv);
        double correction = coefficients[9];
        for (int i = 8; i >= 0; --i)
            correction = __dadd_rn(__dmul_rn(correction, x2), coefficients[i]);
        double result = __dadd_rn(__ddiv_rn(correction, x0), 0.5 * 1.8378770664093453);
        result = __dsub_rn(__dadd_rn(result, __dmul_rn(__dsub_rn(x0, 0.5), log(x0))), x0);
        for (int i = 0; i < n; ++i) {
            result = __dsub_rn(result, log(__dsub_rn(x0, 1.)));
            x0 = __dsub_rn(x0, 1.);
        }
        return result;
    }
    __device__ static bool to_steps(double k, U64& out) {
        // Rust converts float to saturating i64, then uses u64::try_from.
        if (k < 0.)
            return false;
        if (isnan(k))
            out = 0;
        else if (k >= 0x1p63)
            out = 0x7fffffffffffffffULL;
        else
            out = U64(k);
        return true;
    }
    __device__ bool poisson(double lambda, U64& out) {
        if (!isfinite(lambda) || lambda < 0. || lambda > 0x1p64)
            return false;
        out = 0;
        if (lambda == 0.)
            return true;
        if (lambda < 10.) {
            const double threshold = exp(-lambda);
            double product = 1.;
            for (;;) {
                product = __dmul_rn(product, uniform());
                if (product <= threshold)
                    return true;
                if (out == ~U64(0))
                    return false;
                ++out;
            }
        }
        const double b = __dadd_rn(0.931, __dmul_rn(2.53, __dsqrt_rn(lambda)));
        const double a = __dadd_rn(-0.059, __dmul_rn(0.02483, b));
        const double inv_alpha = __dadd_rn(1.1239, __ddiv_rn(1.1328, __dsub_rn(b, 3.4)));
        const double quick = __dsub_rn(0.9277, __ddiv_rn(3.6224, __dsub_rn(b, 2.)));
        for (;;) {
            const double u = __dsub_rn(uniform(), 0.5), v = uniform(), us = __dsub_rn(0.5, fabs(u));
            const double scale = __dadd_rn(__ddiv_rn(__dmul_rn(2., a), us), b);
            const double k = floor(__dadd_rn(__dadd_rn(__dmul_rn(scale, u), lambda), 0.43));
            if (us >= 0.07 && v <= quick)
                return to_steps(k, out);
            if (k < 0. || (us < 0.013 && v > us))
                continue;
            const double lhs = __dsub_rn(__dadd_rn(log(v), log(inv_alpha)),
                                         log(__dadd_rn(__ddiv_rn(a, __dmul_rn(us, us)), b)));
            const double rhs = __dsub_rn(__dadd_rn(-lambda, __dmul_rn(k, log(lambda))),
                                         log_gamma(__dadd_rn(k, 1.)));
            if (lhs <= rhs)
                return to_steps(k, out);
        }
    }
};
} // namespace cnot103_shared_word
