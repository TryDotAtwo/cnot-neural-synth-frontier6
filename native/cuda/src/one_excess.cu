#include "cnot103/cnot_cuda.h"
#include "one_excess.cuh"

namespace {
using namespace cnot103_one_excess;
__global__ void classify(Cnot103OneExcessArgs a,bool verify){
    const U64 worker=blockIdx.x;const ClassifierLayout l(a.dimension,a.lanes,a.cache_slots);const auto& c=l.count;const auto& s=c.structure;
    ClassifierView v{l,a.histories+worker*l.history_words,a.history_lengths+worker*s.n,a.protected_rows+worker*s.n,a.classifier+worker*l.classifier_words,a.workspace_gates+worker*l.gate_elements,
        {s,a.structure_meta+worker*s.meta_count,a.structure_bits+worker*s.bit_count,a.structure_scratch+worker*s.scratch_count},
        {c,a.program+worker*c.program_words,a.arithmetic+worker*l.arithmetic_words}};
    for(U64 b=worker;b<a.batch;b+=gridDim.x){const bool word_ok=verify&&a.verified[b]==1;__syncthreads();if(!threadIdx.x)a.verified[b]=0;__syncthreads();
        recognize_history_family(a.states+b*U64(s.n)*s.w,a.zero_headers+3*b,a.zero_status[b],v,a.max_entries,a.max_changed_rows,a.max_component_events,a.max_history_counts);canonicalize_result(v);
        if(verify){
            for(U32 i=threadIdx.x;i<c.limbs;i+=blockDim.x)require_exact(a.counts[b*c.limbs+i]==v.total().at(i));
            for(U64 i=threadIdx.x;i<l.gate_elements;i+=blockDim.x)require_exact(a.gates[b*l.gate_elements+i]==v.gates[i]);
            if(!threadIdx.x){for(U32 i=0;i<6;++i)require_exact(a.headers[b*6+i]==result_header(v,i));require_exact(a.lengths[b]==static_cast<std::int64_t>(v.data[5]));require_exact(v.data[0]!=1||word_ok);}
            __syncthreads();if(!threadIdx.x)a.verified[b]=1;
        }else{
            for(U32 i=threadIdx.x;i<c.limbs;i+=blockDim.x)a.counts[b*c.limbs+i]=v.total().at(i);
            for(U64 i=threadIdx.x;i<l.gate_elements;i+=blockDim.x)a.gates[b*l.gate_elements+i]=v.gates[i];
            if(!threadIdx.x){for(U32 i=0;i<6;++i)a.headers[b*6+i]=result_header(v,i);a.lengths[b]=v.data[5];a.status[b]=0;for(U32 i=0;i<3;++i)a.feature[b*3+i]=0;}
        }
        __syncthreads();
    }
}
__global__ void publish(Cnot103OneExcessArgs a){const U64 b=U64(blockIdx.x)*blockDim.x+threadIdx.x;if(b>=a.batch)return;require_exact(a.verified[b]==1);
    const U64 cls=a.headers[6*b];require_exact(cls>=1&&cls<=4);a.status[b]=cls==4?4:1;for(U32 i=0;i<3;++i)a.feature[b*3+i]=cls==i+1;}
template<class T> bool extent(U64 count,U64 stride,U64 available){return stride&&count<=available/stride&&count<=(~U64(0)/sizeof(T))/stride;}
int validate(const Cnot103OneExcessArgs* a){
    if(!a||!a->states||!a->zero_headers||!a->zero_status||!a->histories||!a->history_lengths||!a->protected_rows||!a->structure_meta||!a->structure_bits||!a->structure_scratch||!a->program||!a->arithmetic||!a->classifier||!a->workspace_gates||!a->headers||!a->counts||!a->gates||!a->lengths||!a->verified||!a->status||!a->feature||!a->word_scratch)return 1;
    if(!a->batch||a->batch>0x7fffffffU||a->dimension<2||a->dimension>(1U<<24)||a->words_per_row!=(a->dimension+63)/64||!a->workers||a->workers>a->batch||!a->lanes||a->lanes>32||!a->max_entries)return 1;
    const ClassifierLayout l(a->dimension,a->lanes,a->cache_slots);const auto& c=l.count;const auto& s=c.structure;const U64 b=a->batch,w=a->workers;
    if(!extent<U64>(b,U64(s.n)*s.w,a->state_words)||!extent<U64>(b,3,a->zero_header_words)||!extent<std::uint8_t>(b,1,a->zero_status_elements)||
        !extent<U64>(w,l.history_words,a->history_words)||!extent<U32>(w,s.n,a->length_elements)||!extent<std::uint8_t>(w,s.n,a->protected_elements)||
        !extent<U32>(w,s.meta_count,a->meta_elements)||!extent<U64>(w,s.bit_count,a->bit_words)||!extent<U64>(w,s.scratch_count,a->scratch_words)||
        !extent<U64>(w,c.program_words,a->program_words)||!extent<U32>(w,l.arithmetic_words,a->arithmetic_elements)||!extent<U64>(w,l.classifier_words,a->classifier_words)||!extent<U32>(w,l.gate_elements,a->workspace_gate_elements)||
        !extent<U64>(b,6,a->header_elements)||!extent<U32>(b,c.limbs,a->count_elements)||!extent<U32>(b,l.gate_elements,a->gate_elements)||!extent<std::int64_t>(b,1,a->batch_elements)||!extent<float>(b,3,a->feature_elements)||!extent<U64>(b,U64(s.n)*s.w,a->word_scratch_words))return 2;
    return 0;
}
}
extern "C" std::int32_t cnot103_cuda_one_excess_construct(const Cnot103OneExcessArgs* a){int e=validate(a);if(e)return e;classify<<<a->workers,32,0,static_cast<cudaStream_t>(a->stream)>>>(*a,false);return cudaGetLastError()==cudaSuccess?0:3;}
extern "C" std::int32_t cnot103_cuda_one_excess_check(const Cnot103OneExcessArgs* a){int e=validate(a);if(e)return e;
    e=cnot103_cuda_verify_cnot_words(a->states,a->gates,a->lengths,a->word_scratch,a->verified,a->batch,a->dimension,a->words_per_row,a->dimension+1,a->word_scratch_words,a->gate_elements,a->stream);
    if(e)return e;classify<<<a->workers,32,0,static_cast<cudaStream_t>(a->stream)>>>(*a,true);return cudaGetLastError()==cudaSuccess?0:3;}
extern "C" std::int32_t cnot103_cuda_one_excess_publish(const Cnot103OneExcessArgs* a){int e=validate(a);if(e)return e;publish<<<(a->batch+127)/128,128,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return cudaGetLastError()==cudaSuccess?0:3;}
extern "C" std::int32_t cnot103_cuda_launch_one_excess(const Cnot103OneExcessArgs* a){int e=cnot103_cuda_one_excess_construct(a);if(e)return e;e=cnot103_cuda_one_excess_check(a);return e?e:cnot103_cuda_one_excess_publish(a);}
