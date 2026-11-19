#include "cnot103/cnot_cuda.h"
#include "paid_basis_atlas.cuh"

namespace {
using namespace cnot103_paid;

__global__ void prepare(Cnot103PaidAtlasPrepareArgs a) {
    for(U64 i=threadIdx.x;i<storage_words(a.width);i+=blockDim.x)a.storage[i]=0;
    __syncthreads();
    Atlas<U32> atlas(a.storage,a.width);
    if(!threadIdx.x) {
        U32 identity=0;
        for(U32 bit=0;bit<a.width;++bit)identity|=U32{1}<<(bit*a.width+bit);
        atlas.maps[0]=identity;
        atlas.parents[0]=atlas.controls[0]=atlas.targets[0]=~U32{0};
        atlas.seen[identity]=1;
        U32 count=1;
        const U32 mask=(U32{1}<<a.width)-1;
        // The queue is the append-only map array. A single publication owner
        // preserves source first discovery in control-major/target-minor order.
        for(U32 current=0;current<count;++current) {
            const U32 map=atlas.maps[current];
            for(U32 control=0;control<a.width;++control)for(U32 target=0;target<a.width;++target) {
                if(control==target)continue;
                const U32 source=a.side==0?target:control,destination=a.side==0?control:target;
                const U32 next=map^(((map>>(source*a.width))&mask)<<(destination*a.width));
                if(atlas.seen[next])continue;
                if(count>=space(a.width))asm volatile("trap;");
                atlas.seen[next]=count+1;
                atlas.maps[count]=next;atlas.parents[count]=current;
                atlas.controls[count]=control;atlas.targets[count]=target;
                atlas.distances[count]=atlas.distances[current]+1;
                if(atlas.distances[count]>path_capacity(a.width))asm volatile("trap;");
                ++count;
            }
        }
        U32 expected=1;
        for(U32 i=0;i<a.width;++i)expected*=((U32{1}<<a.width)-(U32{1}<<i));
        if(count!=expected)asm volatile("trap;");
        atlas.count[0]=count;
    }
    __syncthreads();
    const U32 codes=U32{1}<<a.width;
    for(U64 index=threadIdx.x;index<U64(atlas.count[0])*codes;index+=blockDim.x)
        atlas.population[index]=__popc(image(atlas.maps[index/codes],a.width,index%codes));
}

__global__ void optimize(Cnot103PaidAtlasArgs a) {
    const U64 batch=blockIdx.x,vector_size=a.rows*a.width;
    const std::uint8_t* input=a.vectors+batch*vector_size;
    std::uint8_t* output=a.transformed+batch*vector_size;
    std::int32_t* gates=a.operations+batch*a.word_capacity*2;
    Atlas<const U32> atlas(a.storage,a.width);
    __shared__ unsigned long long histogram[16],verified_cost;
    __shared__ U64 costs[lanes];
    __shared__ U32 indices[lanes],invalid,selected,length;
    for(U64 i=threadIdx.x;i<vector_size;i+=blockDim.x)output[i]=0;
    for(U64 i=threadIdx.x;i<2*a.word_capacity;i+=blockDim.x)gates[i]=0;
    if(threadIdx.x<16)histogram[threadIdx.x]=0;
    if(!threadIdx.x) {
        invalid=0;a.status[batch]=3;
        a.lengths[batch]=a.direct_cost[batch]=a.paid_cost[batch]=0;
    }
    __syncthreads();
    for(U64 row=threadIdx.x;row<a.rows;row+=blockDim.x) {
        U32 code=0;
        for(U32 bit=0;bit<a.width;++bit) {
            const U32 value=input[row*a.width+bit];
            if(value>1)atomicOr(&invalid,1U);
            code|=(value&1)<<bit;
        }
        atomicAdd(histogram+code,1ULL);
    }
    __syncthreads();
    if(invalid)return;
    const U32 codes=U32{1}<<a.width;
    const U32 winner=choose(atlas,a.width,histogram,costs,indices);
    if(!threadIdx.x) {
        selected=winner;length=path(atlas,selected,reinterpret_cast<U32*>(gates),a.word_capacity);
        verified_cost=2*length;
        for(U32 code=0;code<codes;++code)a.direct_cost[batch]+=histogram[code]*__popc(code);
        if(costs[0]>a.direct_cost[batch])asm volatile("trap;");
    }
    __syncthreads();
    for(U64 row=threadIdx.x;row<a.rows;row+=blockDim.x) {
        U32 original=0;
        for(U32 bit=0;bit<a.width;++bit)original|=U32(input[row*a.width+bit])<<bit;
        U32 transformed=original;
        // Independent path replay checks the selected map and its paid cost.
        for(U32 i=0;i<length;++i) {
            const U32 control=gates[2*i],target=gates[2*i+1];
            if(control>=a.width || target>=a.width || control==target)asm volatile("trap;");
            const U32 source=a.side==0?target:control,destination=a.side==0?control:target;
            transformed^=((transformed>>source)&1)<<destination;
        }
        if(transformed!=image(atlas.maps[selected],a.width,original))asm volatile("trap;");
        for(U32 bit=0;bit<a.width;++bit)output[row*a.width+bit]=(transformed>>bit)&1;
        atomicAdd(&verified_cost,static_cast<unsigned long long>(__popc(transformed)));
    }
    __syncthreads();
    if(!threadIdx.x) {
        if(verified_cost!=costs[0])asm volatile("trap;");
        a.lengths[batch]=length;a.paid_cost[batch]=costs[0];a.status[batch]=1;
    }
}

bool valid_width(U32 width,U32 side){return width>=1 && width<=4 && side<=1;}
bool product(U64 left,U64 right,U64& out){if(right && left>~U64{0}/right)return false;out=left*right;return true;}
}
extern "C" std::int32_t cnot103_cuda_prepare_paid_atlas(const Cnot103PaidAtlasPrepareArgs* a) {
    if(!a || !a->storage || !valid_width(a->width,a->side))return 1;
    if(a->storage_words<cnot103_paid::storage_words(a->width))return 2;
    prepare<<<1,cnot103_paid::lanes,0,static_cast<cudaStream_t>(a->stream)>>>(*a);
    return static_cast<std::int32_t>(cudaGetLastError());
}
extern "C" std::int32_t cnot103_cuda_optimize_paid_atlas(const Cnot103PaidAtlasArgs* a) {
    if(!a || !a->storage || !a->vectors || !a->transformed || !a->operations || !a->lengths ||
        !a->direct_cost || !a->paid_cost || !a->status || !valid_width(a->width,a->side) || !a->rows || !a->batch)return 1;
    U64 vector_size,vector_elements,gate_size,gate_elements;
    if(a->word_capacity<cnot103_paid::path_capacity(a->width) ||
        !product(a->rows,a->width,vector_size) || vector_size>~U64{0}-2*cnot103_paid::path_capacity(a->width) ||
        !product(a->batch,vector_size,vector_elements) || !product(a->word_capacity,2,gate_size) ||
        !product(a->batch,gate_size,gate_elements))return 1;
    if(a->storage_words<cnot103_paid::storage_words(a->width) || a->vector_elements<vector_elements ||
        a->gate_elements<gate_elements || a->batch_elements<a->batch)return 2;
    optimize<<<a->batch,cnot103_paid::lanes,0,static_cast<cudaStream_t>(a->stream)>>>(*a);
    return static_cast<std::int32_t>(cudaGetLastError());
}
