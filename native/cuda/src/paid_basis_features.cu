#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <limits>

namespace {
using U32=std::uint32_t; using U64=std::uint64_t; using I64=std::int64_t;
constexpr U32 lanes=128;
constexpr U64 exact_limit=U64{1}<<24;
__device__ void clear(const Cnot103PaidBasisArgs& a,U64 b){
    const U64 n=a.dimension,g=a.context_count,q=a.capacity,w=a.words_per_row;
    for(U64 i=threadIdx.x;i<q*2;i+=blockDim.x)a.word[b*q*2+i]=0;
    for(U64 i=threadIdx.x;i<q;i+=blockDim.x)a.labels[b*q+i]=0;
    for(U64 i=threadIdx.x;i<n*w;i+=blockDim.x)a.endpoint[b*n*w+i]=0;
    for(U64 i=threadIdx.x;i<n*n*3;i+=blockDim.x)a.counts[b*n*n*3+i]=0;
    for(U64 i=threadIdx.x;i<g*18;i+=blockDim.x)a.groups[b*g*18+i]=0;
    for(U64 i=threadIdx.x;i<g*n*2;i+=blockDim.x)a.incidence[b*g*n*2+i]=0;
    for(U64 i=threadIdx.x;i<g;i+=blockDim.x)a.valid[b*g+i]=0;
    for(U64 i=threadIdx.x;i<n*n*4;i+=blockDim.x)a.pairs[b*n*n*4+i]=0;
    if(!threadIdx.x)a.lengths[b]=0;
}
__global__ void init(Cnot103PaidBasisArgs a){
    const U64 b=blockIdx.x;clear(a,b);__syncthreads();
    for(U32 row=threadIdx.x;row<a.dimension;row+=blockDim.x)
        a.endpoint[(b*a.dimension+row)*a.words_per_row+row/64]=U64{1}<<(row%64);
    if(!threadIdx.x){a.status[b]=1;a.overflow[b]=0;a.verified[b]=0;}
}
__device__ bool gate_valid(U32 c,U32 t,U32 n){return c<n&&t<n&&c!=t;}
__global__ void append(Cnot103PaidBasisAppendArgs in){
    const auto& a=in.output;const U64 b=blockIdx.x,n=a.dimension,w=a.words_per_row,g=a.context_count;
    __shared__ U32 run;
    if(!threadIdx.x){
        if(a.status[b]==1&&in.status[b]!=1)a.status[b]=in.status[b]==4?4:3;
        if(a.status[b]==1&&(in.lengths[b]<0||U64(in.lengths[b])>in.capacity||
            a.lengths[b]<0||U64(a.lengths[b])>a.capacity||
            (in.context_index&&(in.previous_lengths[b]<0||U64(in.previous_lengths[b])>in.previous_capacity))))a.status[b]=3;
        run=a.status[b]==1;
    }
    __syncthreads();if(!run)return;
    // Direct composition of the original shear is independent of every
    // compiler word, basis and stack. No target row is also a source row.
    for(U64 i=threadIdx.x;i<U64(in.target_count)*w;i+=blockDim.x){
        const U32 row=in.targets[i/w];const U64 part=i%w;
        if(row>=n)asm volatile("trap;");
        U64 value=a.endpoint[(b*n+row)*w+part];
        for(U32 s=0;s<in.source_count;++s){const U32 col=in.sources[s];
            if(col>=n)asm volatile("trap;");
            if((in.states[(b*n+row)*w+col/64]>>(col%64))&1)
                value^=a.endpoint[(b*n+col)*w+part];
        }
        a.endpoint[(b*n+row)*w+part]=value;
    }
    for(U32 s=threadIdx.x;s<in.source_count;s+=blockDim.x){
        const U32 wire=in.sources[s];if(wire>=n)asm volatile("trap;");
        a.incidence[((b*g+in.context_index)*n+wire)*2]=1;
    }
    for(U32 t=threadIdx.x;t<in.target_count;t+=blockDim.x){
        const U32 wire=in.targets[t];if(wire>=n)asm volatile("trap;");
        a.incidence[((b*g+in.context_index)*n+wire)*2+1]=1;
    }
    if(!threadIdx.x){
        const U64* s=in.statistics+b*16;
        const U64 values[18]={s[3],s[4],s[5],s[6],s[1],s[0]>s[1]?s[0]-s[1]:0,
            s[3],s[11],s[9],s[4],s[12],s[10],s[13],s[13]>~U64{0}/3?~U64{0}:3*s[13],s[0],s[15],s[14],s[1]};
        for(U32 i=0;i<18;++i){if(values[i]>exact_limit)a.overflow[b]=1;a.groups[(b*g+in.context_index)*18+i]=float(values[i]);}
        a.valid[b*g+in.context_index]=1;
        U64* counts=a.counts+b*n*n*3;
        const U32* current=in.word+b*in.capacity*2;const U64 length=in.lengths[b];
        if(in.context_index){
            const U32* previous=in.previous_word+b*in.previous_capacity*2;const U64 plen=in.previous_lengths[b];
            for(U64 i=0;i<length&&i<plen;++i){
                const U32 c=current[2*i],t=current[2*i+1];
                if(c!=previous[2*(plen-1-i)]||t!=previous[2*(plen-1-i)+1])break;
                if(!gate_valid(c,t,a.dimension)){a.status[b]=3;break;}
                ++counts[(U64(c)*n+t)*3+1];
            }
        }
        U64 size=a.lengths[b];U32* stack=a.word+b*a.capacity*2;U64* labels=a.labels+b*a.capacity;
        for(U64 i=0;i<length&&a.status[b]==1;++i){
            const U32 c=current[2*i],t=current[2*i+1];
            if(!gate_valid(c,t,a.dimension)){a.status[b]=3;break;}
            U64* pair=counts+(U64(c)*n+t)*3;
            if(size&&stack[2*(size-1)]==c&&stack[2*(size-1)+1]==t){
                --size;if(!pair[0])asm volatile("trap;");--pair[0];
                if(labels[size]!=in.context_index)++pair[2];
                stack[2*size]=stack[2*size+1]=0;labels[size]=0;
            }else{
                if(size>=a.capacity)asm volatile("trap;");
                stack[2*size]=c;stack[2*size+1]=t;labels[size]=in.context_index;++size;++pair[0];
            }
        }
        a.lengths[b]=size;
    }
}
__global__ void publish(Cnot103PaidBasisArgs a){
    const U64 b=blockIdx.x,n=a.dimension;
    if(!threadIdx.x&&a.status[b]==1){
        if(a.overflow[b])a.status[b]=4;
        else if(!a.verified[b])a.status[b]=3;
    }
    __syncthreads();
    if(a.status[b]==1){
        for(U64 pair=threadIdx.x;pair<n*n;pair+=blockDim.x){
            const U64 c=pair/n,t=pair%n;
            a.pairs[(b*n*n+pair)*4]=float((a.endpoint[(b*n+t)*a.words_per_row+c/64]>>(c%64))&1);
            for(U32 channel=0;channel<3;++channel){
                const U64 count=a.counts[(b*n*n+pair)*3+channel];
                if(count>exact_limit)atomicOr(a.overflow+b,1U);
                a.pairs[(b*n*n+pair)*4+channel+1]=float(count);
            }
        }
    }
    __syncthreads();
    if(!threadIdx.x&&a.status[b]==1&&a.overflow[b])a.status[b]=4;
    __syncthreads();
    if(a.status[b]!=1)clear(a,b);
}
bool extent(U64 provided,U64 batch,U64 count){return (!count||batch<=~U64{0}/count)&&provided>=batch*count;}
int validate(const Cnot103PaidBasisArgs* a){
    if(!a||!a->word||!a->labels||!a->lengths||!a->endpoint||!a->replay||!a->counts||!a->groups||!a->incidence||!a->valid||!a->pairs||!a->status||!a->verified||!a->overflow)return 1;
    if(!a->batch||a->batch>0x7fffffffu||a->dimension<2||a->dimension>(1U<<24)||a->words_per_row!=(U64(a->dimension)+63)/64||
        !a->context_count||a->context_count>1024||!a->capacity||a->capacity>U64(std::numeric_limits<I64>::max())/2)return 1;
    const U64 n=a->dimension,g=a->context_count,b=a->batch;
    if(a->batch_elements<b||!extent(a->state_words,b,n*a->words_per_row)||!extent(a->gate_elements,b,a->capacity*2)||
        !extent(a->label_words,b,a->capacity)||!extent(a->count_elements,b,n*n*3)||!extent(a->group_elements,b,g*18)||
        !extent(a->incidence_elements,b,g*n*2)||!extent(a->token_elements,b,g)||!extent(a->pair_elements,b,n*n*4))return 2;
    return 0;
}
}
extern "C" std::int32_t cnot103_cuda_paid_basis_init(const Cnot103PaidBasisArgs* a){
    const int result=validate(a);if(result)return result;
    init<<<a->batch,lanes,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return static_cast<std::int32_t>(cudaGetLastError());
}
extern "C" std::int32_t cnot103_cuda_paid_basis_append(const Cnot103PaidBasisAppendArgs* in){
    if(!in)return 1;const int result=validate(&in->output);if(result)return result;const auto& a=in->output;
    if(!in->states||!in->sources||!in->targets||!in->word||!in->lengths||!in->statistics||!in->status||
        !in->source_count||!in->target_count||U64(in->source_count)+in->target_count>a.dimension||
        in->context_index>=a.context_count||in->capacity!=U64(in->source_count)*in->target_count||
        (in->context_index&&(!in->previous_word||!in->previous_lengths||!in->previous_capacity||in->previous_capacity>a.capacity)))return 1;
    if(in->capacity>a.capacity||in->source_elements<in->source_count||in->target_elements<in->target_count||
        !extent(in->gate_elements,a.batch,in->capacity*2)||!extent(in->stat_elements,a.batch,16)||
        (in->context_index&&!extent(in->previous_gate_elements,a.batch,in->previous_capacity*2)))return 2;
    append<<<a.batch,lanes,0,static_cast<cudaStream_t>(a.stream)>>>(*in);return static_cast<std::int32_t>(cudaGetLastError());
}
extern "C" std::int32_t cnot103_cuda_paid_basis_finish(const Cnot103PaidBasisArgs* a){
    const int result=validate(a);if(result)return result;
    const auto code=cnot103_cuda_verify_cnot_words(a->endpoint,a->word,a->lengths,a->replay,a->verified,a->batch,a->dimension,a->words_per_row,a->capacity,a->state_words,a->gate_elements,a->stream);
    if(code)return code;
    publish<<<a->batch,lanes,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return static_cast<std::int32_t>(cudaGetLastError());
}
