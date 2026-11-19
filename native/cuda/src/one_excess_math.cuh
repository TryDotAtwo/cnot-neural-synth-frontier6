#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Stage38 uses unbounded source integers. The admitted dimension supplies a
// sufficient capacity, not a new numerical/resource cutoff. Each CUDA lane can
// own an integer in a limb-major pool: adjacent lanes access adjacent words.
// No allocation, host arithmetic on payloads or device pointer graph is used.
namespace cnot103_one_excess {
using Word = std::uint32_t;
using Wide = std::uint64_t;

struct ConstLimbs {
    const Word* data;
    unsigned count;
    std::size_t stride;
    __device__ Word at(unsigned i) const { return data[std::size_t(i)*stride]; }
};
struct Limbs {
    Word* data;
    unsigned count;
    std::size_t stride;
    __device__ Word& at(unsigned i) const { return data[std::size_t(i)*stride]; }
    __device__ operator ConstLimbs() const { return {data,count,stride}; }
};

__host__ __device__ inline unsigned bit_length(Wide x) {
    unsigned bits=0;
    for (;x;x>>=1) ++bits;
    return bits;
}

// k <= n+1 events, at most k+1 factors, <=3n terms per factor and
// <=2n(n-1) templates. All partial nonnegative tables and the absolute signed
// finite-difference/interleaving prefixes are bounded by
// 2*n^2 * 2^k * k^(2k) * (3n)^(k+1). Using bit lengths bounds that product
// strictly by 2^B. One spare limb also covers small construction intermediates.
// The dimension domain is the existing resident-state/exact-float contract.
__host__ __device__ inline unsigned limbs_for_dimension(unsigned n) {
    if (n<2 || n>(1U<<24)) return 0;
    const Wide k=Wide(n)+1;
    const Wide bits=1+2*bit_length(n)+k+2*k*bit_length(k)+(k+1)*bit_length(3*Wide(n));
    return static_cast<unsigned>((bits+31)/32+1);
}

// Views in an operation have the same positive capacity and positive stride.
// Except multiply, dst may be exactly either input view. Partial overlap is
// never allowed. On numerical overflow/underflow the low capacity bits are
// written and false is returned; callers must not publish that truncated value.
__device__ inline void clear(Limbs dst) {
    for(unsigned i=0;i<dst.count;++i) dst.at(i)=0;
}
__device__ inline void set_small(Limbs dst, Word value) {
    clear(dst); dst.at(0)=value;
}
__device__ inline void copy(Limbs dst, ConstLimbs src) {
    for(unsigned i=0;i<dst.count;++i) dst.at(i)=src.at(i);
}
__device__ inline unsigned used(ConstLimbs value) {
    unsigned size=value.count;
    while(size && value.at(size-1)==0) --size;
    return size;
}
__device__ inline int compare(ConstLimbs a, ConstLimbs b) {
    for(unsigned i=a.count;i-->0;) {
        const Word av=a.at(i),bv=b.at(i);
        if(av!=bv) return av<bv?-1:1;
    }
    return 0;
}
__device__ inline bool add(Limbs dst, ConstLimbs a, ConstLimbs b) {
    Wide carry=0;
    for(unsigned i=0;i<dst.count;++i) {
        const Wide value=Wide(a.at(i))+b.at(i)+carry;
        dst.at(i)=static_cast<Word>(value); carry=value>>32;
    }
    return carry==0;
}
__device__ inline bool subtract(Limbs dst, ConstLimbs a, ConstLimbs b) {
    Wide borrow=0;
    for(unsigned i=0;i<dst.count;++i) {
        const Wide av=a.at(i),rhs=Wide(b.at(i))+borrow;
        dst.at(i)=static_cast<Word>(av-rhs); borrow=av<rhs;
    }
    return borrow==0;
}
__device__ inline bool multiply_small(Limbs dst, ConstLimbs a, Word b) {
    Wide carry=0;
    for(unsigned i=0;i<dst.count;++i) {
        const Wide value=Wide(a.at(i))*b+carry;
        dst.at(i)=static_cast<Word>(value); carry=value>>32;
    }
    return carry==0;
}

// dst is disjoint from both inputs. a and b may be the same view.
__device__ inline bool multiply(Limbs dst, ConstLimbs a, ConstLimbs b) {
    const unsigned na=used(a),nb=used(b);
    clear(dst); bool fits=true;
    for(unsigned i=0;i<na;++i) {
        const Word av=a.at(i);
        if(av==0) continue;
        const unsigned end=nb<dst.count-i?nb:dst.count-i;
        Wide carry=0;
        for(unsigned j=0;j<end;++j) {
            // (2^32-1)^2 + (2^32-1) + (2^32-1) fits exactly in u64.
            const Wide value=Wide(av)*b.at(j)+dst.at(i+j)+carry;
            dst.at(i+j)=static_cast<Word>(value); carry=value>>32;
        }
        if(i+end<dst.count) dst.at(i+end)=static_cast<Word>(carry);
        else if(carry) fits=false;
        for(unsigned j=end;j<nb;++j) if(b.at(j)) fits=false;
    }
    return fits;
}

// Exact quotient and remainder for a positive u32 divisor. A zero divisor
// clears dst/remainder and returns false. Callers needing an integral result
// additionally require remainder==0; Stage38 divides factorials in small steps.
__device__ inline bool divide_small(Limbs dst, ConstLimbs a, Word divisor, Word& remainder) {
    if(divisor==0) { clear(dst); remainder=0; return false; }
    Wide rem=0;
    for(unsigned i=dst.count;i-->0;) {
        const Wide value=(rem<<32)|a.at(i);
        dst.at(i)=static_cast<Word>(value/divisor); rem=value%divisor;
    }
    remainder=static_cast<Word>(rem); return true;
}

// Sign is separate from magnitude. Inputs use -1/0/1, with sign0 permitted only
// for zero magnitude; any sign on zero is normalized. Successful output always
// has sign0 exactly when its magnitude is zero. Same-view aliases are supported.
__device__ inline bool signed_add(Limbs dst, int& sign, ConstLimbs a, int sa, ConstLimbs b, int sb) {
    const unsigned na=used(a),nb=used(b);
    if(sa<-1 || sa>1 || sb<-1 || sb>1 || (na && sa==0) || (nb && sb==0)) {
        clear(dst); sign=0; return false;
    }
    if(!na) sa=0;
    if(!nb) sb=0;
    if(sa==0) { copy(dst,b); sign=sb; return true; }
    if(sb==0) { copy(dst,a); sign=sa; return true; }
    if(sa==sb) { const bool ok=add(dst,a,b); sign=used(dst)?sa:0; return ok; }
    const int order=compare(a,b);
    if(order==0) { clear(dst); sign=0; return true; }
    const bool ok=order>0?subtract(dst,a,b):subtract(dst,b,a);
    sign=order>0?sa:sb; return ok;
}

// A failed exact operation under the proven capacity is an implementation or
// proof defect. It must not be converted into a source ResourceLimit/negative.
__device__ inline void require_exact(bool ok) {
    if(!ok) asm volatile("trap;");
}
}
