#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cnot103_template {
using U64=std::uint64_t;using U32=std::uint32_t;
struct Wide {U64 lo,hi;};
__device__ inline bool zero(Wide x){return !(x.lo|x.hi);}
__device__ inline int compare(Wide a,Wide b){return a.hi!=b.hi?(a.hi<b.hi?-1:1):(a.lo<b.lo?-1:a.lo>b.lo?1:0);}
__device__ inline Wide subtract(Wide a,Wide b){return {a.lo-b.lo,a.hi-b.hi-(a.lo<b.lo)};}
__device__ inline bool add(Wide a,Wide b,Wide& out){
    const U64 low=a.lo+b.lo,carry=low<a.lo,high=a.hi+b.hi,next=high+carry;
    out={low,next};return high>=a.hi&&next>=high;
}
__device__ inline bool multiply(Wide a,U64 factor,Wide& out){
    const U64 low=a.lo*factor,carry=__umul64hi(a.lo,factor),high=a.hi*factor,next=high+carry;
    out={low,next};return __umul64hi(a.hi,factor)==0&&next>=high;
}
__device__ inline bool shift(Wide a,U32 bits,Wide& out){
    if(bits>=128)return false;
    if(!bits){out=a;return true;}
    if(bits>=64){out={0,a.lo<<(bits-64)};return !a.hi&&(bits==64||!(a.lo>>(128-bits)));}
    out={a.lo<<bits,(a.hi<<bits)|(a.lo>>(64-bits))};return !(a.hi>>(64-bits));
}
// Restoring division with a 129th carry bit; remainder stays <denominator.
// A carried shift followed by wrapping subtraction is exact because the
// mathematical difference is nonnegative and strictly below denominator.
__device__ inline bool divide(Wide value,Wide denominator,Wide& quotient,Wide& remainder){
    if(zero(denominator))return false;quotient={0,0};remainder={0,0};
    for(int bit=127;bit>=0;--bit){
        const U64 carry=remainder.hi>>63;
        const U64 next=bit>=64?(value.hi>>(bit-64))&1:(value.lo>>bit)&1;
        remainder={(remainder.lo<<1)|next,(remainder.hi<<1)|(remainder.lo>>63)};
        if(carry||compare(remainder,denominator)>=0){remainder=subtract(remainder,denominator);if(bit>=64)quotient.hi|=U64{1}<<(bit-64);else quotient.lo|=U64{1}<<bit;}
    }
    return true;
}
__device__ inline int bit_length(Wide x){return x.hi?128-__clzll(x.hi):x.lo?64-__clzll(x.lo):0;}
// Exact source rational_to_f32 contract. In particular, source refuses a
// negative fractional-bit count (floor(log2(ratio))>23); do not silently
// replace this with a broader approximate floating-point conversion.
__device__ inline bool rational_bits(Wide numerator,Wide denominator,U32& bits){
    if(zero(denominator))return false;if(zero(numerator)){bits=0;return true;}
    int exponent=bit_length(numerator)-bit_length(denominator);Wide scaled{};
    bool below;
    if(exponent>=0){const bool fits=shift(denominator,exponent,scaled);below=!fits||compare(numerator,scaled)<0;}
    else {const bool fits=shift(numerator,-exponent,scaled);below=fits&&compare(scaled,denominator)<0;}
    if(below)--exponent;if(exponent>127)return false;
    const bool subnormal=exponent< -126;const int fractional=subnormal?149:23-exponent;if(fractional<0)return false;
    Wide integer{},remainder{},whole{},fraction{0,0};divide(numerator,denominator,integer,remainder);
    if(!zero(integer)&&!shift(integer,fractional,whole))return false;
    for(int i=0;i<fractional;++i){
        if(!multiply(fraction,2,fraction))return false;
        const Wide complement=subtract(denominator,remainder);
        if(compare(remainder,complement)>=0){remainder=subtract(remainder,complement);if(!add(fraction,{1,0},fraction))return false;}
        else if(!multiply(remainder,2,remainder))return false;
    }
    Wide significand{};if(!add(whole,fraction,significand))return false;
    const int rounding=compare(remainder,subtract(denominator,remainder));
    if(rounding>0||(rounding==0&&(significand.lo&1)))if(!add(significand,{1,0},significand))return false;
    if(significand.hi)return false;
    if(subnormal){if(significand.lo>(U64{1}<<23))return false;bits=U32(significand.lo);return true;}
    if(significand.lo>=(U64{1}<<24)){significand.lo>>=1;++exponent;}
    if(exponent>127||significand.lo<(U64{1}<<23))return false;
    bits=(U32(exponent+127)<<23)|U32(significand.lo-(U64{1}<<23));return true;
}
struct Probability{U64 numerator,denominator;};
__device__ inline bool expectation(const U64* costs,U32 k,const Probability* point,Wide& numerator,Wide& denominator){
    denominator={1,0};numerator={0,0};
    for(U32 i=0;i<k;++i)if(!multiply(denominator,point[i].denominator,denominator))return false;
    for(U32 mask=0;mask<(1U<<k);++mask){Wide weight{1,0};
        for(U32 i=0;i<k;++i){const auto p=point[i];const U64 factor=(mask>>i)&1?p.numerator:p.denominator-p.numerator;if(!multiply(weight,factor,weight))return false;}
        if(!multiply(weight,costs[mask],weight)||!add(numerator,weight,numerator))return false;
    }return true;
}
__device__ inline bool conditional(const U64* costs,U32 k,const Probability* point,U32 fixed,Wide& off,Wide& on){
    off={0,0};on={0,0};
    for(U32 mask=0;mask<(1U<<k);++mask){if((mask>>fixed)&1)continue;Wide weight{1,0};
        for(U32 i=0;i<k;++i){if(i==fixed)continue;const auto p=point[i];const U64 factor=(mask>>i)&1?p.numerator:p.denominator-p.numerator;if(!multiply(weight,factor,weight))return false;}
        Wide left{},right{};if(!multiply(weight,costs[mask],left)||!add(off,left,off)||!multiply(weight,costs[mask|(1U<<fixed)],right)||!add(on,right,on))return false;
    }return true;
}
__device__ inline bool rounding_assignment(const U64* costs,U32 k,const Probability* input,bool reverse,U32& assignment){
    Probability point[8];for(U32 i=0;i<k;++i)point[i]=input[i];
    for(U32 step=0;step<k;++step){const U32 i=reverse?k-1-step:step;Wide off{},on{};if(!conditional(costs,k,point,i,off,on))return false;point[i]={compare(on,off)<0?U64{1}:0,1};}
    assignment=0;for(U32 i=0;i<k;++i)assignment|=U32(point[i].numerator==1)<<i;return true;
}
}
