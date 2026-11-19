#include "cnot103/cnot_cuda.h"
#include "one_excess_count.cuh"

namespace {
using namespace cnot103_one_excess;
__global__ void count_batch(Cnot103OneExcessCountArgs a){
    const U64 worker=blockIdx.x;const CountLayout l(a.dimension,a.lanes,a.cache_slots);const StructureLayout sl=l.structure;
    StructureView s{sl,a.structure_meta+worker*sl.meta_count,a.structure_bits+worker*sl.bit_count,a.structure_scratch+worker*sl.scratch_count};
    CountView p{l,a.program+worker*l.program_words,a.arithmetic+worker*l.arithmetic_words};
    for(U64 b=worker;b<a.batch;b+=gridDim.x){
        HistoryView h{a.histories+b*3*sl.n*sl.w,a.lengths+b*sl.n,a.protected_rows+b*sl.n,sl.n,sl.w};
        count_history(h,s,p,a.max_entries);
        for(U32 word=threadIdx.x;word<l.limbs;word+=blockDim.x)a.counts[b*l.limbs+word]=p.global(0).at(word);
        if(threadIdx.x==0){U64 evictions=0;for(U32 lane=0;lane<l.lanes;++lane){const U64 value=p.data[l.evictions+lane];evictions=~U64(0)-evictions<value?~U64(0):evictions+value;}
            a.results[4*b]=p.data[6];a.results[4*b+1]=p.data[5];a.results[4*b+2]=p.data[4];a.results[4*b+3]=evictions;}
        __syncthreads();
    }
}
template<class T> bool extent(U64 count,U64 stride,U64 available){return stride && count<=available/stride && count<=(~U64(0)/sizeof(T))/stride;}
}
extern "C" std::int32_t cnot103_cuda_launch_one_excess_count(const Cnot103OneExcessCountArgs* a){
    if(!a||!a->histories||!a->lengths||!a->protected_rows||!a->structure_meta||!a->structure_bits||!a->structure_scratch||!a->program||!a->arithmetic||!a->counts||!a->results||!a->batch||a->batch>0x7fffffffU||a->dimension<2||a->dimension>(1U<<24)||!a->workers||a->workers>a->batch||!a->lanes||a->lanes>32||!a->max_entries)return 1;
    const CountLayout l(a->dimension,a->lanes,a->cache_slots);const auto& s=l.structure;const U64 b=a->batch,workers=a->workers;
    if(!extent<U64>(b,3*U64(s.n)*s.w,a->history_words)||!extent<U32>(b,s.n,a->length_elements)||!extent<std::uint8_t>(b,s.n,a->protected_elements)||
        !extent<U32>(workers,s.meta_count,a->meta_elements)||!extent<U64>(workers,s.bit_count,a->bit_words)||!extent<U64>(workers,s.scratch_count,a->scratch_words)||
        !extent<U64>(workers,l.program_words,a->program_words)||!extent<U32>(workers,l.arithmetic_words,a->arithmetic_elements)||!extent<U32>(b,l.limbs,a->count_elements)||!extent<U64>(b,4,a->result_words))return 2;
    count_batch<<<a->workers,32,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return cudaGetLastError()==cudaSuccess?0:3;
}
