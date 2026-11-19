#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace {
using Word=std::uint64_t;
using Index=std::uint32_t;
// Every nonconstant term in product(I+x_j E_j) is a single elementary matrix
// with a unique selector history. Left multiplication by E_(target,control)
// preserves exactly the terms whose current target equals control. Counts by
// endpoint therefore reproduce the source's expansion budget exactly, without
// storing all matrices. No monomials collide: each new term has a fresh bit j.
__global__ void classify(Cnot103PositivePacketArgs a) {
    __shared__ Index valid;
    if(threadIdx.x==0)valid=1;
    for(Word wire=threadIdx.x;wire<a.dimension;wire+=blockDim.x)a.counts[wire]=0;
    __syncthreads();
    for(Word f=threadIdx.x;f<a.factors;f+=blockDim.x){
        const Word begin=a.factor_offsets[f],end=a.factor_offsets[f+1];
        Index last=0;
        if(begin>end||end>a.scope_elements)atomicExch(&valid,0);
        else for(Word i=begin;i<end;++i){
            const Index slot=a.factor_scopes[i];
            if(slot>=a.slots)atomicExch(&valid,0);
            else if(slot>last)last=slot;
        }
        a.indices[f]=last;
    }
    __syncthreads();
    if(threadIdx.x!=0)return;
    if(!valid){*a.status=3;return;}
    Word total=1;
    bool nonlinear=false;
    for(Index slot=0;slot<a.slots;++slot){
        const Index control=a.gates[Word{slot}*2],target=a.gates[Word{slot}*2+1];
        if(control>=a.dimension||target>=a.dimension||control==target){*a.status=3;return;}
        // created=1+counts[control]. Check before adding, including UINT64_MAX
        // budgets. ResourceLimit takes priority even after nonaffinity is known.
        if(a.counts[control]>=a.max_monomials-total){*a.status=4;return;}
        const Word created=1+a.counts[control];
        nonlinear|=created>1;
        a.counts[target]+=created;total+=created;
    }
    *a.status=nonlinear?3:1;
}
__global__ void intervals(Cnot103PositivePacketArgs a){
    if(*a.status!=1)return;
    Index* first=a.indices+a.factors;
    Index* last=first+a.slots;
    Index* live_until=last+a.slots;
    for(Word slot=Word{blockIdx.x}*blockDim.x+threadIdx.x;slot<a.slots;slot+=Word{gridDim.x}*blockDim.x){
        const Index control=a.gates[slot*2],target=a.gates[slot*2+1];
        Index start=slot,finish=slot,until=slot;
        for(Index other=0;other<a.slots;++other)
            if(a.gates[Word{other}*2]==control&&a.gates[Word{other}*2+1]==target){
                if(other<start)start=other;if(other>finish)finish=other;
            }
        for(Index f=0;f<a.factors;++f){
            const Index end=a.indices[f];
            if(end<=until)continue;
            for(Word i=a.factor_offsets[f];i<a.factor_offsets[Word{f}+1];++i)
                if(a.factor_scopes[i]==slot){until=end;break;}
        }
        first[slot]=start;last[slot]=finish;live_until[slot]=until;
    }
}
__global__ void publish(Cnot103PositivePacketArgs a){
    const Index cut=blockIdx.x;
    for(Word i=threadIdx.x;i<Word{a.dimension}*2;i+=blockDim.x)a.incidence[Word{cut}*a.dimension*2+i]=0;
    if(threadIdx.x<4)a.values[Word{cut}*4+threadIdx.x]=0;
    if(threadIdx.x==0)a.valid[cut]=0;
    __syncthreads();
    if(*a.status!=1)return;
    const Index* first=a.indices+a.factors;
    const Index* last=first+a.slots;
    const Index* live_until=last+a.slots;
    // Affine singleton columns are E_j. Distinct physical gate pairs form
    // independent coordinates, so cut width counts pairs present on both sides.
    if(threadIdx.x==0){
        Index width=0,live=0;
        for(Index slot=0;slot<a.slots;++slot){
            width+=first[slot]==slot&&slot<cut&&last[slot]>=cut;
            live+=slot<cut&&live_until[slot]>=cut;
        }
        a.values[Word{cut}*4]=cut;a.values[Word{cut}*4+1]=width;
        a.values[Word{cut}*4+2]=live;a.values[Word{cut}*4+3]=width+live;
        a.valid[cut]=1;
    }
    // Each physical role cell has one writer, including shared controls/targets.
    for(Index wire=threadIdx.x;wire<a.dimension;wire+=blockDim.x){
        bool control=false,target=false;
        for(Index slot=0;slot<cut;++slot)if(live_until[slot]>=cut){
            control|=a.gates[Word{slot}*2]==wire;target|=a.gates[Word{slot}*2+1]==wire;
        }
        const Word offset=(Word{cut}*a.dimension+wire)*2;
        a.incidence[offset]=control;a.incidence[offset+1]=target;
    }
}
}
extern "C" std::int32_t cnot103_cuda_prepare_positive_packet(const Cnot103PositivePacketArgs* args){
    if(!args)return 1;
    const auto& a=*args;
    if(a.dimension<2||a.dimension>(1u<<24)||a.slots>(1u<<23)||!a.max_monomials||
        (a.slots&&!a.gates)||!a.factor_offsets||(a.scope_elements&&!a.factor_scopes)||
        !a.counts||!a.indices||!a.values||!a.valid||!a.incidence||!a.status)return 1;
    const Word groups=Word{a.slots}+1;
    if(a.count_words<a.dimension||a.index_elements<Word{a.factors}+Word{a.slots}*3||
        a.group_elements<groups||groups>a.value_elements/4||groups>a.incidence_elements/(Word{a.dimension}*2))return 2;
    const auto stream=reinterpret_cast<cudaStream_t>(a.stream);
    classify<<<1,128,0,stream>>>(a);
    if(cudaGetLastError()!=cudaSuccess)return 3;
    intervals<<<a.slots?(a.slots+127)/128:1,128,0,stream>>>(a);
    if(cudaGetLastError()!=cudaSuccess)return 3;
    publish<<<static_cast<Index>(groups),128,0,stream>>>(a);
    return cudaGetLastError()==cudaSuccess?0:3;
}
