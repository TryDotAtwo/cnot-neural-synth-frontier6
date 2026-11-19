#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cnot103_paid {
using U32 = std::uint32_t;
using U64 = std::uint64_t;
constexpr U32 lanes = 128;
__host__ __device__ constexpr U32 space(U32 width) { return U32{1} << (width * width); }
__host__ __device__ constexpr U64 storage_words(U32 width) { return U64(space(width)) * (6 + (U32{1} << width)) + 1; }
// Gauss-Jordan needs at most w*(w-1) row adds and w-1 three-CNOT swaps.
// Every shortest atlas path is no longer; width1 retains one unused output slot.
__host__ __device__ constexpr U32 path_capacity(U32 width) { return width == 1 ? 1 : (width + 3) * (width - 1); }
template<typename T> struct Atlas {
    T *maps, *parents, *controls, *targets, *distances, *seen, *population, *count;
    __device__ Atlas(T* p, U32 width) {
        const U32 size=space(width);
        maps=p;p+=size; parents=p;p+=size; controls=p;p+=size;
        targets=p;p+=size; distances=p;p+=size; seen=p;p+=size;
        population=p;p+=U64(size)*(U32{1}<<width); count=p;
    }
};
// Complete linear map: row-major w*w bits, not a hash/truncated state key.
__device__ inline U32 image(U32 map, U32 width, U32 code) {
    U32 output=0;
    const U32 mask=(U32{1}<<width)-1;
    for(U32 row=0;row<width;++row) output|=(__popc(((map>>(row*width))&mask)&code)&1)<<row;
    return output;
}
__device__ inline bool better(U64 cost,U32 index,U64 other_cost,U32 other_index) {
    return cost<other_cost || (cost==other_cost && index<other_index);
}
// All 128 threads call this with a complete histogram and shared arrays. The
// final barrier precedes every read; caller must finish reads before reuse.
__device__ inline U32 choose(const Atlas<const U32>& atlas,U32 width,
    const unsigned long long* histogram,U64* costs,U32* indices) {
    const U32 codes=U32{1}<<width,count=atlas.count[0];
    U64 best=~U64{0};U32 best_index=~U32{0};
    for(U32 index=threadIdx.x;index<count;index+=blockDim.x) {
        U64 cost=U64(atlas.distances[index])*2;
        for(U32 code=0;code<codes;++code)cost+=histogram[code]*atlas.population[U64(index)*codes+code];
        if(better(cost,index,best,best_index)){best=cost;best_index=index;}
    }
    costs[threadIdx.x]=best;indices[threadIdx.x]=best_index;
    __syncthreads();
    for(U32 stride=lanes/2;stride;stride/=2) {
        if(threadIdx.x<stride && better(costs[threadIdx.x+stride],indices[threadIdx.x+stride],costs[threadIdx.x],indices[threadIdx.x])) {
            costs[threadIdx.x]=costs[threadIdx.x+stride];indices[threadIdx.x]=indices[threadIdx.x+stride];
        }
        __syncthreads();
    }
    return indices[0];
}
// Single-owner path reconstruction. Capacity follows the constructive bound.
__device__ inline U32 path(const Atlas<const U32>& atlas,U32 selected,U32* operations,U64 capacity) {
    const U32 length=atlas.distances[selected];
    if(length>capacity)asm volatile("trap;");
    U32 current=selected;
    for(U32 i=0;i<length;++i) {
        if(!current || atlas.parents[current]>=current)asm volatile("trap;");
        operations[2*(length-1-i)]=atlas.controls[current];
        operations[2*(length-1-i)+1]=atlas.targets[current];
        current=atlas.parents[current];
    }
    if(current)asm volatile("trap;");
    return length;
}
}
