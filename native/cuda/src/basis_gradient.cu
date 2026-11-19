#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace {
__global__ void coefficients_kernel(const std::uint64_t* input,
    std::uint64_t* scratch, std::uint8_t* coefficients, std::uint8_t* status,
    std::uint32_t n, std::uint32_t words, std::uint32_t width, std::uint32_t coefficient_words) {
    __shared__ std::uint32_t rank;
    __shared__ std::uint32_t pivot;
    __shared__ std::uint32_t nonzero;
    const auto sample=static_cast<std::uint64_t>(blockIdx.x);
    const auto source_size=static_cast<std::uint64_t>(width)*words;
    const auto inverse_size=static_cast<std::uint64_t>(width)*coefficient_words;
    const auto scratch_size=source_size+inverse_size+width;
    const auto target_count=n-width;
    const auto coefficient_size=static_cast<std::uint64_t>(target_count)*width;
    const auto* state=input+sample*n*words;
    auto* source=scratch+sample*scratch_size;
    auto* inverse=source+source_size;
    auto* pivots=inverse+inverse_size;
    auto* output=coefficients+sample*coefficient_size;
    for (std::uint64_t index=threadIdx.x;index<source_size;index+=blockDim.x) source[index]=state[index];
    for (std::uint64_t index=threadIdx.x;index<inverse_size;index+=blockDim.x) {
        const auto row=index/coefficient_words;
        const auto word=index%coefficient_words;
        inverse[index]=word==row/64 ? std::uint64_t{1}<<(row%64) : 0;
    }
    for (std::uint64_t index=threadIdx.x;index<coefficient_size;index+=blockDim.x) output[index]=0;
    if (threadIdx.x==0) { rank=0; nonzero=0; }
    __syncthreads();
    for (std::uint32_t column=0;column<n;++column) {
        const auto active=rank;
        if (active==width) break;
        if (threadIdx.x==0) pivot=width;
        __syncthreads();
        const auto word=column/64;
        const auto bit=std::uint64_t{1}<<(column%64);
        for (std::uint32_t row=active+threadIdx.x;row<width;row+=blockDim.x)
            if (source[static_cast<std::uint64_t>(row)*words+word]&bit) atomicMin(&pivot,row);
        __syncthreads();
        if (pivot<width) {
            const auto found=pivot;
            for (std::uint32_t k=threadIdx.x;k<words;k+=blockDim.x) {
                const auto a=static_cast<std::uint64_t>(active)*words+k;
                const auto b=static_cast<std::uint64_t>(found)*words+k;
                const auto saved=source[a]; source[a]=source[b]; source[b]=saved;
            }
            for (std::uint32_t k=threadIdx.x;k<coefficient_words;k+=blockDim.x) {
                const auto a=static_cast<std::uint64_t>(active)*coefficient_words+k;
                const auto b=static_cast<std::uint64_t>(found)*coefficient_words+k;
                const auto saved=inverse[a]; inverse[a]=inverse[b]; inverse[b]=saved;
            }
            __syncthreads();
            for (std::uint32_t row=threadIdx.x;row<width;row+=blockDim.x) {
                if (row==active || (source[static_cast<std::uint64_t>(row)*words+word]&bit)==0) continue;
                for (std::uint32_t k=0;k<words;++k)
                    source[static_cast<std::uint64_t>(row)*words+k]^=source[static_cast<std::uint64_t>(active)*words+k];
                for (std::uint32_t k=0;k<coefficient_words;++k)
                    inverse[static_cast<std::uint64_t>(row)*coefficient_words+k]^=inverse[static_cast<std::uint64_t>(active)*coefficient_words+k];
            }
            __syncthreads();
            if (threadIdx.x==0) { pivots[active]=column; ++rank; }
        }
        // Also required on no-pivot columns before resetting pivot/rereading rank.
        __syncthreads();
    }
    if (rank!=width) {
        if (threadIdx.x==0) status[sample]=2;
        return;
    }
    const auto packed_output_size=static_cast<std::uint64_t>(target_count)*coefficient_words;
    for (std::uint64_t index=threadIdx.x;index<packed_output_size;index+=blockDim.x) {
        const auto target=index/coefficient_words;
        const auto word=index%coefficient_words;
        std::uint64_t value=0;
        for (std::uint32_t k=0;k<width;++k) {
            const auto column=pivots[k];
            if ((state[(width+target)*words+column/64]>>(column%64))&1)
                value^=inverse[static_cast<std::uint64_t>(k)*coefficient_words+word];
        }
        if (value) atomicExch(&nonzero,1u);
        for (std::uint32_t k=0;k<64 && word*64+k<width;++k)
            output[target*width+word*64+k]=static_cast<std::uint8_t>((value>>k)&1);
    }
    __syncthreads();
    if (threadIdx.x==0) status[sample]=static_cast<std::uint8_t>(nonzero);
}

__global__ void gradients_kernel(const std::uint8_t* coefficients, const std::uint8_t* status,
    std::int64_t* gradients, std::uint32_t width, std::uint32_t targets) {
    const auto sample=static_cast<std::uint64_t>(blockIdx.x);
    const auto pairs=static_cast<std::uint64_t>(width)*width;
    const auto* block=coefficients+sample*targets*width;
    for (std::uint64_t pair=threadIdx.x;pair<pairs;pair+=blockDim.x) {
        const auto source=pair/width;
        const auto target=pair%width;
        std::int64_t value=0;
        if (status[sample]==1 && source!=target) {
            value=2;
            for (std::uint32_t row=0;row<targets;++row) {
                const auto left=block[static_cast<std::uint64_t>(row)*width+source];
                const auto right=block[static_cast<std::uint64_t>(row)*width+target];
                value+=static_cast<std::int64_t>(right)-2*static_cast<std::int64_t>(left)*right;
            }
        }
        gradients[sample*pairs+pair]=value;
    }
}

bool extent(std::uint64_t left,std::uint64_t right,std::uint64_t* result) {
    if (right && left>std::numeric_limits<std::uint64_t>::max()/right) return false;
    *result=left*right; return true;
}
}

extern "C" std::int32_t cnot103_cuda_launch_basis_gradient(
    const std::uint64_t* states,std::uint64_t* scratch,std::uint8_t* coefficients,
    std::uint8_t* status,std::int64_t* gradients,std::uint32_t batch,
    std::uint32_t dimension,std::uint32_t words_per_row,
    std::uint64_t scratch_words,std::uint64_t coefficient_elements,
    std::uint64_t gradient_elements,std::uint32_t emit_gradients,void* stream) {
    if (!states||!scratch||!coefficients||!status||!batch||batch>0x7fffffffu||
        dimension<2||dimension>(1u<<24)||words_per_row!=(dimension+63)/64||emit_gradients>1) return 1;
    const auto width=dimension/2;
    const auto targets=dimension-width;
    const auto coefficient_words=(width+63)/64;
    const auto per_sample=static_cast<std::uint64_t>(width)*(words_per_row+coefficient_words+1);
    std::uint64_t scratch_required=0,coefficient_required=0,gradient_required=0;
    if (!extent(batch,per_sample,&scratch_required)||!extent(batch,static_cast<std::uint64_t>(targets)*width,&coefficient_required)||
        !extent(batch,static_cast<std::uint64_t>(width)*width,&gradient_required)) return 2;
    if (scratch_words<scratch_required||coefficient_elements<coefficient_required||
        (emit_gradients && (!gradients||gradient_elements<gradient_required))) return 1;
    auto cuda_stream=reinterpret_cast<cudaStream_t>(stream);
    coefficients_kernel<<<batch,256,0,cuda_stream>>>(states,scratch,coefficients,status,dimension,words_per_row,width,coefficient_words);
    if (cudaGetLastError()!=cudaSuccess) return 3;
    if (emit_gradients) {
        gradients_kernel<<<batch,256,0,cuda_stream>>>(coefficients,status,gradients,width,targets);
        if (cudaGetLastError()!=cudaSuccess) return 3;
    }
    return 0;
}
