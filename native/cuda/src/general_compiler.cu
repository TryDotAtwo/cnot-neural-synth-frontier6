#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace {
__device__ std::uint64_t identity_word(std::uint64_t index,std::uint32_t words) {
    const auto row=index/words;
    return index%words==row/64 ? std::uint64_t{1}<<(row%64) : 0;
}
__global__ void compile_words(Cnot103GeneralCompilerArgs a) {
    __shared__ std::uint32_t pivot,error,mismatch;
    __shared__ std::uint64_t count;
    const auto sample=static_cast<std::uint64_t>(blockIdx.x);
    const auto n=a.dimension,words=a.words_per_row;
    const auto cells=static_cast<std::uint64_t>(n)*words;
    const auto* input=a.states+sample*cells;
    auto* matrix=a.scratch+sample*cells;
    auto* gates=a.gates ? a.gates+sample*a.word_capacity*2 : nullptr;
    for(std::uint64_t i=threadIdx.x;i<cells;i+=blockDim.x) matrix[i]=input[i];
    for(std::uint64_t i=threadIdx.x;i<a.word_capacity*2;i+=blockDim.x) gates[i]=0;
    if(threadIdx.x==0){count=0;error=0;mismatch=0;}
    __syncthreads();
    for(std::uint32_t column=0;column<n;++column){
        if(threadIdx.x==0) pivot=n;
        __syncthreads();
        const auto word=column/64;
        const auto bit=std::uint64_t{1}<<(column%64);
        for(std::uint32_t row=column+threadIdx.x;row<n;row+=blockDim.x)
            if(matrix[static_cast<std::uint64_t>(row)*words+word]&bit) atomicMin(&pivot,row);
        __syncthreads();
        if(pivot==n){
            if(threadIdx.x==0) error=3;
            __syncthreads();
            break;
        }
        if(pivot!=column){
            if(threadIdx.x==0){
                if(a.word_capacity-count<3) error=4;
                else{
                    gates[count*2]=pivot;gates[count*2+1]=column;
                    gates[count*2+2]=column;gates[count*2+3]=pivot;
                    gates[count*2+4]=pivot;gates[count*2+5]=column;
                    count+=3;
                }
            }
            __syncthreads();
            if(error) break;
            for(std::uint32_t k=threadIdx.x;k<words;k+=blockDim.x){
                const auto first=static_cast<std::uint64_t>(column)*words+k;
                const auto second=static_cast<std::uint64_t>(pivot)*words+k;
                const auto saved=matrix[first];matrix[first]=matrix[second];matrix[second]=saved;
            }
            __syncthreads();
        }
        // Emission follows the reference's row order. Eliminations commute
        // within this pivot because they share an unchanged control row.
        if(threadIdx.x==0){
            for(std::uint32_t row=0;row<n;++row){
                if(row==column || !(matrix[static_cast<std::uint64_t>(row)*words+word]&bit)) continue;
                if(count==a.word_capacity){error=4;break;}
                gates[count*2]=column;gates[count*2+1]=row;++count;
            }
        }
        __syncthreads();
        if(error) break;
        for(std::uint32_t row=threadIdx.x;row<n;row+=blockDim.x){
            if(row==column || !(matrix[static_cast<std::uint64_t>(row)*words+word]&bit)) continue;
            for(std::uint32_t k=0;k<words;++k)
                matrix[static_cast<std::uint64_t>(row)*words+k]^=matrix[static_cast<std::uint64_t>(column)*words+k];
        }
        __syncthreads();
    }
    for(std::uint64_t i=threadIdx.x;i<cells;i+=blockDim.x)
        if(matrix[i]!=identity_word(i,words)) atomicExch(&mismatch,1u);
    __syncthreads();
    if(threadIdx.x==0 && error==0 && mismatch) error=3;
    __syncthreads();
    if(error==0){
        for(std::uint64_t i=threadIdx.x;i<count/2;i+=blockDim.x){
            const auto other=count-1-i;
            const auto control=gates[i*2],target=gates[i*2+1];
            gates[i*2]=gates[other*2];gates[i*2+1]=gates[other*2+1];
            gates[other*2]=control;gates[other*2+1]=target;
        }
    }else{
        for(std::uint64_t i=threadIdx.x;i<a.word_capacity*2;i+=blockDim.x) gates[i]=0;
    }
    if(threadIdx.x==0){a.lengths[sample]=error?0:static_cast<std::int64_t>(count);a.status[sample]=error?error:1;}
}

// Replay has no dependency on the eliminator's transformed matrix or pivots.
__global__ void replay_words(const std::uint64_t* states,const std::uint32_t* gates,
    const std::int64_t* lengths,std::uint64_t* scratch,std::uint8_t* verified,
    std::uint32_t n,std::uint32_t words,std::uint64_t capacity){
    __shared__ std::uint32_t mismatch;
    const auto sample=static_cast<std::uint64_t>(blockIdx.x);
    const auto cells=static_cast<std::uint64_t>(n)*words;
    const auto length=lengths[sample];
    if(length<0 || static_cast<std::uint64_t>(length)>capacity){
        if(threadIdx.x==0) verified[sample]=0;
        return;
    }
    auto* matrix=scratch+sample*cells;
    const auto* word=gates?gates+sample*capacity*2:nullptr;
    for(std::uint64_t i=threadIdx.x;i<cells;i+=blockDim.x) matrix[i]=identity_word(i,words);
    if(threadIdx.x==0) mismatch=0;
    __syncthreads();
    for(std::int64_t index=0;index<length;++index){
        const auto control=word[index*2],target=word[index*2+1];
        if(control>=n || target>=n || control==target){
            if(threadIdx.x==0) verified[sample]=0;
            return;
        }
        for(std::uint32_t k=threadIdx.x;k<words;k+=blockDim.x)
            matrix[static_cast<std::uint64_t>(target)*words+k]^=matrix[static_cast<std::uint64_t>(control)*words+k];
        __syncthreads();
    }
    for(std::uint64_t i=threadIdx.x;i<cells;i+=blockDim.x)
        if(matrix[i]!=states[sample*cells+i]) atomicExch(&mismatch,1u);
    __syncthreads();
    if(threadIdx.x==0) verified[sample]=mismatch?0:1;
}

__global__ void bounds_and_status(Cnot103GeneralCompilerArgs a){
    __shared__ std::uint32_t rank,pivot,changed_rows,changed_columns;
    const auto sample=static_cast<std::uint64_t>(blockIdx.x);
    const auto n=a.dimension,words=a.words_per_row;
    const auto cells=static_cast<std::uint64_t>(n)*words;
    auto* residual=a.scratch+sample*cells;
    if(threadIdx.x==0){
        a.lower_bounds[sample]=0;
        for(int i=0;i<4;++i) a.f09[sample*4+i]=0;
        for(int i=0;i<5;++i) a.f10[sample*5+i]=0;
        if(a.status[sample]==1 && !a.verified[sample]) a.status[sample]=3;
        rank=0;changed_rows=0;changed_columns=0;
    }
    __syncthreads();
    if(a.status[sample]==1){
        for(std::uint64_t i=threadIdx.x;i<cells;i+=blockDim.x)
            residual[i]=a.states[sample*cells+i]^identity_word(i,words);
        __syncthreads();
        for(std::uint32_t index=threadIdx.x;index<n;index+=blockDim.x){
            bool row_changed=false,column_changed=false;
            for(std::uint32_t k=0;k<words;++k) row_changed|=residual[static_cast<std::uint64_t>(index)*words+k]!=0;
            for(std::uint32_t row=0;row<n;++row) column_changed|=((residual[static_cast<std::uint64_t>(row)*words+index/64]>>(index%64))&1)!=0;
            if(row_changed) atomicAdd(&changed_rows,1u);
            if(column_changed) atomicAdd(&changed_columns,1u);
        }
        __syncthreads();
        for(std::uint32_t column=0;column<n;++column){
            const auto active=rank;
            if(threadIdx.x==0) pivot=n;
            __syncthreads();
            const auto word=column/64;
            const auto bit=std::uint64_t{1}<<(column%64);
            for(std::uint32_t row=active+threadIdx.x;row<n;row+=blockDim.x)
                if(residual[static_cast<std::uint64_t>(row)*words+word]&bit) atomicMin(&pivot,row);
            __syncthreads();
            if(pivot<n){
                for(std::uint32_t k=threadIdx.x;k<words;k+=blockDim.x){
                    const auto first=static_cast<std::uint64_t>(active)*words+k;
                    const auto second=static_cast<std::uint64_t>(pivot)*words+k;
                    const auto saved=residual[first];residual[first]=residual[second];residual[second]=saved;
                }
                __syncthreads();
                for(std::uint32_t row=active+1+threadIdx.x;row<n;row+=blockDim.x){
                    if(!(residual[static_cast<std::uint64_t>(row)*words+word]&bit)) continue;
                    for(std::uint32_t k=0;k<words;++k)
                        residual[static_cast<std::uint64_t>(row)*words+k]^=residual[static_cast<std::uint64_t>(active)*words+k];
                }
                __syncthreads();
                if(threadIdx.x==0) ++rank;
            }
            __syncthreads(); // includes no-pivot iterations
        }
        if(threadIdx.x==0){
            const auto lower=max(rank,max(changed_rows,changed_columns));
            const auto upper=a.lengths[sample];
            if(upper<static_cast<std::int64_t>(lower)) a.status[sample]=3;
            else if(upper>(1<<24)) a.status[sample]=4;
            else{
                a.lower_bounds[sample]=lower;
                const float tight=upper==lower?1.0f:0.0f;
                a.f09[sample*4]=static_cast<float>(upper);
                a.f09[sample*4+1]=static_cast<float>(lower);
                a.f09[sample*4+2]=static_cast<float>(upper-lower);
                a.f09[sample*4+3]=tight;
                a.f10[sample*5]=1;a.f10[sample*5+1]=1;a.f10[sample*5+2]=tight;
                a.f10[sample*5+3]=0;a.f10[sample*5+4]=1;
            }
        }
    }
    __syncthreads();
    if(a.status[sample]!=1){
        for(std::uint64_t i=threadIdx.x;i<a.word_capacity*2;i+=blockDim.x) a.gates[sample*a.word_capacity*2+i]=0;
        if(threadIdx.x==0){a.lengths[sample]=0;a.verified[sample]=0;}
    }
}

bool multiply(std::uint64_t a,std::uint64_t b,std::uint64_t* result){
    if(b && a>std::numeric_limits<std::uint64_t>::max()/b) return false;
    *result=a*b;return true;
}
int validate(std::uint32_t b,std::uint32_t n,std::uint32_t words,std::uint64_t capacity,
    std::uint64_t scratch,std::uint64_t gate_elements){
    if(!b||b>0x7fffffffu||n<2||n>(1u<<24)||words!=(n+63)/64) return 1;
    std::uint64_t cells,groups,gate_count;
    if(!multiply(b,static_cast<std::uint64_t>(n)*words,&cells)||!multiply(b,capacity,&groups)||!multiply(groups,2,&gate_count)) return 2;
    if(scratch<cells||gate_elements<gate_count) return 1;
    return 0;
}
}

extern "C" std::int32_t cnot103_cuda_verify_cnot_words(
    const std::uint64_t* states,const std::uint32_t* gates,const std::int64_t* lengths,
    std::uint64_t* scratch,std::uint8_t* verified,std::uint32_t batch,
    std::uint32_t dimension,std::uint32_t words_per_row,std::uint64_t word_capacity,
    std::uint64_t workspace_words,std::uint64_t gate_elements,void* stream){
    if(!states||!lengths||!scratch||!verified||(word_capacity && !gates)) return 1;
    const auto error=validate(batch,dimension,words_per_row,word_capacity,workspace_words,gate_elements);
    if(error) return error;
    replay_words<<<batch,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(states,gates,lengths,scratch,verified,dimension,words_per_row,word_capacity);
    return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" std::int32_t cnot103_cuda_launch_general_compiler(const Cnot103GeneralCompilerArgs* args){
    if(!args) return 1;
    const auto& a=*args;
    if(!a.states||!a.scratch||!a.lengths||!a.status||!a.verified||!a.lower_bounds||!a.f09||!a.f10||(a.word_capacity && !a.gates)) return 1;
    const auto error=validate(a.batch,a.dimension,a.words_per_row,a.word_capacity,a.workspace_words,a.gate_elements);
    if(error) return error;
    const auto stream=reinterpret_cast<cudaStream_t>(a.stream);
    compile_words<<<a.batch,256,0,stream>>>(a);
    if(cudaGetLastError()!=cudaSuccess) return 3;
    replay_words<<<a.batch,256,0,stream>>>(a.states,a.gates,a.lengths,a.scratch,a.verified,a.dimension,a.words_per_row,a.word_capacity);
    if(cudaGetLastError()!=cudaSuccess) return 3;
    bounds_and_status<<<a.batch,256,0,stream>>>(a);
    return cudaGetLastError()==cudaSuccess?0:3;
}
