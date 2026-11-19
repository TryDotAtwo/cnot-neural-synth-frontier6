#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace {
constexpr std::uint32_t kUnused = 0xffffffffu;

__device__ std::uint64_t pattern_word(const std::uint64_t* states, std::uint64_t base,
    const std::uint64_t* sources, std::uint32_t words, std::uint32_t id, std::uint32_t word) {
    return id == 0 ? 0 : (states[base + static_cast<std::uint64_t>(id - 1) * words + word] & sources[word]);
}
__device__ bool same_pattern(const std::uint64_t* states, std::uint64_t base,
    const std::uint64_t* sources, std::uint32_t words, std::uint32_t left, std::uint32_t right) {
    for (std::uint32_t word = 0; word < words; ++word)
        if (pattern_word(states,base,sources,words,left,word) != pattern_word(states,base,sources,words,right,word)) return false;
    return true;
}
__device__ bool pattern_less(const std::uint64_t* states, std::uint64_t base,
    const std::uint64_t* sources, std::uint32_t words, std::uint32_t left, std::uint32_t right) {
    for (std::uint32_t reverse = words; reverse > 0; --reverse) {
        const auto a = pattern_word(states,base,sources,words,left,reverse-1);
        const auto b = pattern_word(states,base,sources,words,right,reverse-1);
        if (a != b) return a < b;
    }
    return false;
}
__device__ bool selected(const std::uint64_t* mask, std::uint32_t wire) {
    return ((mask[wire / 64u] >> (wire % 64u)) & 1) != 0;
}

__global__ void histogram_counts(Cnot103BlockHistogramArgs a) {
    __shared__ std::uint32_t counts[4]; // patterns,zero,total_nonzero,max_frequency
    const auto item = static_cast<std::uint64_t>(blockIdx.x);
    const auto context = static_cast<std::uint32_t>(item % a.contexts);
    const auto base = (item / a.contexts) * a.dimension * a.words_per_row;
    const auto* sources = a.source_masks + static_cast<std::uint64_t>(context) * a.words_per_row;
    const auto* targets = a.target_masks + static_cast<std::uint64_t>(context) * a.words_per_row;
    auto* frequencies = a.frequencies + item * (static_cast<std::uint64_t>(a.dimension) + 1);
    if (threadIdx.x == 0) { counts[0]=1; counts[1]=0; counts[2]=0; counts[3]=0; }
    __syncthreads();
    for (std::uint32_t row = threadIdx.x; row < a.dimension; row += blockDim.x) {
        frequencies[row+1] = 0;
        if (!selected(targets,row)) continue;
        if (same_pattern(a.states,base,sources,a.words_per_row,row+1,0)) {
            atomicAdd(&counts[1],1u);
            continue;
        }
        bool first = true;
        std::uint32_t frequency = 0;
        for (std::uint32_t other = 0; other < a.dimension; ++other) {
            if (selected(targets,other) && same_pattern(a.states,base,sources,a.words_per_row,row+1,other+1)) {
                ++frequency;
                if (other < row) first = false;
            }
        }
        if (first) {
            frequencies[row+1] = frequency;
            atomicAdd(&counts[0],1u);
            atomicAdd(&counts[2],frequency);
            atomicMax(&counts[3],frequency);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        frequencies[0] = counts[1];
        a.statistics[item*4] = counts[0];
        a.statistics[item*4+1] = counts[2];
        a.statistics[item*4+2] = counts[3];
        const auto nonzero = static_cast<std::uint64_t>(counts[0]-1);
        a.statistics[item*4+3] = nonzero < 2 ? 0 : nonzero*(nonzero-1)/2;
    }
}

__global__ void histogram_patterns(Cnot103BlockHistogramArgs a) {
    const auto item = static_cast<std::uint64_t>(blockIdx.x);
    const auto context = static_cast<std::uint32_t>(item % a.contexts);
    const auto base = (item / a.contexts) * a.dimension * a.words_per_row;
    const auto* sources = a.source_masks + static_cast<std::uint64_t>(context) * a.words_per_row;
    const auto* targets = a.target_masks + static_cast<std::uint64_t>(context) * a.words_per_row;
    const auto* frequencies = a.frequencies + item * (static_cast<std::uint64_t>(a.dimension)+1);
    const auto group_base = item * a.pattern_capacity;
    auto* ids = a.ordered_ids + group_base;
    for (std::uint32_t group = threadIdx.x; group < a.pattern_capacity; group += blockDim.x) {
        ids[group] = kUnused;
        if (a.emit_patterns != 0) {
            a.pattern_valid[group_base+group]=0;
            for (std::uint32_t channel=0; channel<3; ++channel) a.pattern_values[(group_base+group)*3+channel]=0;
        }
    }
    __syncthreads();
    if (a.statistics[item*4] <= a.pattern_capacity) {
        if (threadIdx.x == 0) {
            ids[0]=0;
            if (a.emit_patterns != 0) {
                a.pattern_valid[group_base]=1;
                a.pattern_values[group_base*3]=static_cast<float>(frequencies[0]);
                a.pattern_values[group_base*3+1]=1;
            }
        }
        for (std::uint32_t row=threadIdx.x; row<a.dimension; row+=blockDim.x) {
            if (frequencies[row+1]==0) continue;
            std::uint32_t order=1;
            for (std::uint32_t other=0; other<a.dimension; ++other) {
                if (frequencies[other+1]!=0 && pattern_less(a.states,base,sources,a.words_per_row,other+1,row+1)) ++order;
            }
            ids[order]=row+1;
            if (a.emit_patterns != 0) {
                const auto group=group_base+order;
                const auto frequency=static_cast<float>(frequencies[row+1]);
                a.pattern_valid[group]=1;
                a.pattern_values[group*3]=frequency;
                a.pattern_values[group*3+1]=1;
                a.pattern_values[group*3+2]=frequency;
            }
        }
    }
    __syncthreads();
    if (a.emit_patterns != 0) {
        const auto elements=static_cast<std::uint64_t>(a.pattern_capacity)*a.dimension*3;
        for (std::uint64_t index=threadIdx.x; index<elements; index+=blockDim.x) {
            const auto group=static_cast<std::uint32_t>(index/(static_cast<std::uint64_t>(a.dimension)*3));
            const auto wire=static_cast<std::uint32_t>((index/3)%a.dimension);
            const auto role=static_cast<std::uint32_t>(index%3);
            const auto id=ids[group];
            bool value=false;
            if (id!=kUnused) {
                if (role==0) value=selected(sources,wire);
                else if (role==1) value=((pattern_word(a.states,base,sources,a.words_per_row,id,wire/64)>>(wire%64))&1)!=0;
                else value=selected(targets,wire) && same_pattern(a.states,base,sources,a.words_per_row,id,wire+1);
            }
            a.pattern_incidence[group_base*a.dimension*3+index]=value?1.0f:0.0f;
        }
    }
}

__device__ void pair_indices(std::uint64_t index, std::uint64_t count, std::uint32_t* left, std::uint32_t* right) {
    std::uint64_t low=0,high=count-2;
    while (low<high) {
        const auto middle=(low+high+1)/2;
        const auto prefix=middle*(2*count-middle-1)/2;
        if (prefix<=index) low=middle; else high=middle-1;
    }
    const auto prefix=low*(2*count-low-1)/2;
    *left=static_cast<std::uint32_t>(low+1); // skip forced zero pattern
    *right=static_cast<std::uint32_t>(low+2+index-prefix);
}

__global__ void histogram_pairs(Cnot103BlockHistogramArgs a) {
    const auto item=static_cast<std::uint64_t>(blockIdx.x);
    const auto context=static_cast<std::uint32_t>(item%a.contexts);
    const auto base=(item/a.contexts)*a.dimension*a.words_per_row;
    const auto* sources=a.source_masks+static_cast<std::uint64_t>(context)*a.words_per_row;
    const auto* ids=a.ordered_ids+item*a.pattern_capacity;
    const auto patterns=a.statistics[item*4];
    const auto pairs=a.statistics[item*4+3];
    const bool available=patterns<=a.pattern_capacity && pairs<=a.pair_capacity;
    const auto group_base=item*a.pair_capacity;
    for (std::uint64_t pair=threadIdx.x; pair<a.pair_capacity; pair+=blockDim.x) {
        const bool present=available && pair<pairs;
        std::uint32_t intersection=0;
        if (present) {
            std::uint32_t left,right;
            pair_indices(pair,patterns-1,&left,&right);
            for (std::uint32_t word=0; word<a.words_per_row; ++word)
                intersection+=__popcll(pattern_word(a.states,base,sources,a.words_per_row,ids[left],word)&pattern_word(a.states,base,sources,a.words_per_row,ids[right],word));
        }
        a.pair_valid[group_base+pair]=present?1:0;
        a.pair_values[group_base+pair]=static_cast<float>(intersection);
    }
    const auto elements=a.pair_capacity*a.dimension*3;
    for (std::uint64_t index=threadIdx.x; index<elements; index+=blockDim.x) {
        const auto pair=index/(static_cast<std::uint64_t>(a.dimension)*3);
        const auto wire=static_cast<std::uint32_t>((index/3)%a.dimension);
        const auto role=static_cast<std::uint32_t>(index%3);
        bool value=false;
        if (available && pair<pairs) {
            if (role==0) value=selected(sources,wire);
            else {
                std::uint32_t left,right;
                pair_indices(pair,patterns-1,&left,&right);
                value=((pattern_word(a.states,base,sources,a.words_per_row,ids[role==1?left:right],wire/64)>>(wire%64))&1)!=0;
            }
        }
        a.pair_incidence[group_base*a.dimension*3+index]=value?1.0f:0.0f;
    }
}

bool multiply(std::uint64_t left,std::uint64_t right,std::uint64_t* result) {
    if (right!=0 && left>std::numeric_limits<std::uint64_t>::max()/right) return false;
    *result=left*right; return true;
}
} // namespace

extern "C" std::int32_t cnot103_cuda_launch_block_histograms(const Cnot103BlockHistogramArgs* args) {
    if (!args) return 1;
    const auto& a=*args;
    if (!a.states||!a.source_masks||!a.target_masks||!a.frequencies||!a.statistics||
        a.batch==0||a.contexts==0||a.dimension<2||a.dimension>(1u<<24)||
        a.words_per_row!=(a.dimension/64u+(a.dimension%64u!=0))||a.pattern_capacity==0||
        a.pattern_capacity>static_cast<std::uint64_t>(a.dimension)+1||a.emit_patterns>1) return 1;
    const auto items=static_cast<std::uint64_t>(a.batch)*a.contexts;
    if (items>0x7fffffffu) return 2;
    if (a.frequency_elements<items*(static_cast<std::uint64_t>(a.dimension)+1)||a.statistic_elements<items*4) return 1;
    const bool sort=a.emit_patterns!=0||a.pair_capacity!=0;
    const auto pattern_groups=items*a.pattern_capacity;
    std::uint64_t pattern_incidence=0,pair_groups=0,pair_incidence=0;
    if (sort && (!a.ordered_ids||a.pattern_groups<pattern_groups)) return 1;
    if (a.emit_patterns!=0) {
        if (!multiply(pattern_groups,static_cast<std::uint64_t>(a.dimension)*3,&pattern_incidence)) return 2;
        if (!a.pattern_values||!a.pattern_valid||!a.pattern_incidence||a.pattern_incidence_elements<pattern_incidence) return 1;
    }
    if (a.pair_capacity!=0) {
        if (!multiply(items,a.pair_capacity,&pair_groups)||!multiply(pair_groups,static_cast<std::uint64_t>(a.dimension)*3,&pair_incidence)) return 2;
        if (!a.pair_values||!a.pair_valid||!a.pair_incidence||a.pair_groups<pair_groups||a.pair_incidence_elements<pair_incidence) return 1;
    }
    const auto stream=reinterpret_cast<cudaStream_t>(a.stream);
    histogram_counts<<<static_cast<unsigned int>(items),256,0,stream>>>(a);
    if (cudaGetLastError()!=cudaSuccess) return 3;
    if (sort) {
        histogram_patterns<<<static_cast<unsigned int>(items),256,0,stream>>>(a);
        if (cudaGetLastError()!=cudaSuccess) return 3;
    }
    if (a.pair_capacity!=0) {
        histogram_pairs<<<static_cast<unsigned int>(items),256,0,stream>>>(a);
        if (cudaGetLastError()!=cudaSuccess) return 3;
    }
    return 0;
}
