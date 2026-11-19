#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace cnot103_i10 {
using U64 = std::uint64_t;
// Two little-endian limbs, with exact carry and overflow detection. No
// saturation: callers invalidate the entire finite-horizon response on false.
__device__ inline bool add(U64& low, U64& high, U64 other_low, U64 other_high) {
    const U64 next_low = low + other_low, carry = next_low < low;
    const U64 sum_high = high + other_high, next_high = sum_high + carry;
    const bool fits = sum_high >= high && next_high >= sum_high;
    low = next_low; high = next_high;
    return fits;
}
// One correctly rounded integer-to-double conversion, ties to even. Rounding
// high and low limbs separately before adding can give a different answer.
__device__ inline double as_double(U64 low, U64 high) {
    if (!high) return static_cast<double>(low);
    const unsigned shift = 127 - __clzll(high) - 52;
    U64 significand;
    bool greater, tie;
    if (shift < 64) {
        significand = (high << (64 - shift)) | (low >> shift);
        const U64 remainder = low & ((U64{1} << shift) - 1), half = U64{1} << (shift - 1);
        greater = remainder > half; tie = remainder == half;
    } else if (shift == 64) {
        significand = high;
        greater = low > (U64{1} << 63); tie = low == (U64{1} << 63);
    } else {
        const unsigned upper_shift = shift - 64;
        significand = high >> upper_shift;
        const U64 remainder = high & ((U64{1} << upper_shift) - 1), half = U64{1} << (upper_shift - 1);
        greater = remainder > half || (remainder == half && low != 0);
        tie = remainder == half && low == 0;
    }
    if (greater || (tie && (significand & 1))) ++significand;
    return ldexp(static_cast<double>(significand), shift);
}
}
