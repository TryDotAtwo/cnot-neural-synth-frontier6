#include "cnot103/cnot_cuda.h"
#include "endpoint_response_math.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace {
using U64 = std::uint64_t;
using U32 = std::uint32_t;
constexpr U64 lanes = 128;
constexpr U64 empty = 0;

template<typename T> struct View {
    T *keys[2], *counts[2], *tables[2], *history_keys, *history_counts, *history_degrees, *history_table;
    T *stage_keys, *stage_counts, *stage_hashes, *statistics;
    __device__ View(T* p, U64 capacity, U64 buckets, U64 history_buckets, U64 words) {
        for (unsigned i=0;i<2;++i) { keys[i]=p;p+=capacity*words; }
        for (unsigned i=0;i<2;++i) { counts[i]=p;p+=2*capacity; }
        for (unsigned i=0;i<2;++i) { tables[i]=p;p+=buckets; }
        history_keys=p;p+=2*capacity*words;
        history_counts=p;p+=4*capacity;
        history_degrees=p;p+=2*capacity;
        history_table=p;p+=history_buckets;
        stage_keys=p;p+=lanes*words;
        stage_counts=p;p+=2*lanes;
        stage_hashes=p;p+=lanes;
        statistics=p;
    }
};
__device__ U64 mix(U64 value) {
    value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
    value=(value^(value>>27))*0x94d049bb133111ebULL;
    return value^(value>>31);
}
__device__ U64 hash_word(U64 hash,U64 value) {
    // Mix all bit positions before reducing to power-of-two buckets. Raw
    // word-FNV leaves high physical columns invisible to low bucket bits.
    return (hash^mix(value))*1099511628211ULL;
}
__device__ U64 hash_key(const U64* key, U64 words) {
    U64 hash=14695981039346656037ULL;
    for(U64 i=0;i<words;++i)hash=hash_word(hash,key[i]);
    return mix(hash);
}
__device__ U64 slot(const U64* keys, const U64* table, U64 buckets, const U64* key, U64 words, U64 hash) {
    for(U64 probe=0;probe<buckets;++probe) {
        const U64 at=(hash+probe)&(buckets-1), entry=table[at];
        if(entry==empty) return at;
        bool same=true;
        for(U64 word=0;word<words;++word) if(keys[(entry-1)*words+word]!=key[word]) {same=false;break;}
        if(same) return at;
    }
    // Tables have at least twice as many buckets as their admitted key count.
    asm volatile("trap;");
    return 0;
}

__global__ void prepare(Cnot103EndpointPrepareArgs a) {
    const U64 words=U64(a.dimension)*a.words_per_row;
    const U64 alphabet_size=U64(a.dimension)*(a.dimension-1);
    View<U64> v(a.storage,a.capacity,a.buckets,a.history_buckets,words);
    __shared__ U64 current_size,next_size,history_size,evaluations,required;
    __shared__ U32 code;
    for(U64 i=threadIdx.x;i<2*a.buckets;i+=blockDim.x) v.tables[0][i]=0;
    for(U64 i=threadIdx.x;i<a.history_buckets;i+=blockDim.x) v.history_table[i]=0;
    if(!threadIdx.x) { current_size=history_size=1;evaluations=0;code=1; }
    __syncthreads();
    if(!threadIdx.x) {
        for(U64 cell=0;cell<words;++cell) {
            const U64 row=cell/a.words_per_row,word=cell%a.words_per_row;
            const U64 value=word==row/64 ? (U64{1}<<(row%64)) : 0;
            v.keys[0][cell]=v.history_keys[cell]=value;
        }
        v.counts[0][0]=v.history_counts[0]=1;v.counts[0][1]=v.history_counts[1]=0;v.history_degrees[0]=0;
        const U64 hash=hash_key(v.keys[0],words);
        v.tables[0][hash&(a.buckets-1)]=1;v.history_table[hash&(a.history_buckets-1)]=1;
    }
    __syncthreads();
    // W_d counts all length-d physical words, including revisited endpoints.
    // Do not replace this recurrence by disjoint first-arrival BFS frontiers.
    for(U64 degree=1;degree<=a.max_degree;++degree) {
        const unsigned previous=(degree-1)&1,next=degree&1;
        if(!threadIdx.x) {
            next_size=0;
            if(current_size>a.max_layer_states || current_size>~U64{0}/alphabet_size) code=4;
            else {
                required=current_size*alphabet_size;
                if(required>a.max_transitions-evaluations) code=4;
            }
        }
        __syncthreads();
        if(code!=1) break;
        for(U64 i=threadIdx.x;i<a.buckets;i+=blockDim.x) v.tables[next][i]=0;
        __syncthreads();
        for(U64 begin=0;begin<required;) {
            const U64 chunk=(required-begin)<lanes ? required-begin : lanes;
            if(threadIdx.x<chunk) {
                const U64 transition=begin+threadIdx.x,parent=transition/alphabet_size,gate=transition%alphabet_size;
                const U32 control=gate/(a.dimension-1);
                U32 target=gate%(a.dimension-1);if(target>=control)++target;
                const U64* source=v.keys[previous]+parent*words;
                U64* candidate=v.stage_keys+U64(threadIdx.x)*words;
                U64 hash=14695981039346656037ULL;
                for(U64 cell=0;cell<words;++cell) {
                    U64 value=source[cell];
                    if(cell/a.words_per_row==target) value^=source[U64(control)*a.words_per_row+cell%a.words_per_row];
                    candidate[cell]=value;hash=hash_word(hash,value);
                }
                v.stage_hashes[threadIdx.x]=mix(hash);
                v.stage_counts[2*threadIdx.x]=v.counts[previous][2*parent];
                v.stage_counts[2*threadIdx.x+1]=v.counts[previous][2*parent+1];
            }
            __syncthreads();
            // Single-owner insertion/count reduction avoids key-publication
            // and two-limb update races. Chunk expansion remains cooperative.
            if(!threadIdx.x) for(U64 lane=0;lane<chunk;++lane) {
                const U64* candidate=v.stage_keys+lane*words;
                const U64 bucket=slot(v.keys[next],v.tables[next],a.buckets,candidate,words,v.stage_hashes[lane]);
                U64 entry=v.tables[next][bucket];
                if(!entry) {
                    if(next_size==a.capacity) {code=4;break;}
                    entry=++next_size;
                    for(U64 word=0;word<words;++word) v.keys[next][(entry-1)*words+word]=candidate[word];
                    v.counts[next][2*(entry-1)]=v.counts[next][2*(entry-1)+1]=0;
                    v.tables[next][bucket]=entry;
                }
                if(!cnot103_i10::add(v.counts[next][2*(entry-1)],v.counts[next][2*(entry-1)+1],
                    v.stage_counts[2*lane],v.stage_counts[2*lane+1])) {code=4;break;}
            }
            __syncthreads();
            if(code!=1) break;
            begin+=chunk;
        }
        __syncthreads();
        if(code!=1) break;
        __syncthreads(); // finish every status read before the owner may change it
        if(!threadIdx.x) {
            evaluations+=required;
            if(next_size>a.max_layer_states) code=4;
            else {
                // Every CNOT is an involution, hence W_i subset W_{i+2}.
                // Union_{i<=d} W_i = W_d union W_{d-1}: at most 2*capacity.
                for(U64 index=0;index<next_size;++index) {
                    const U64* key=v.keys[next]+index*words;
                    const U64 bucket=slot(v.history_keys,v.history_table,a.history_buckets,key,words,hash_key(key,words));
                    if(v.history_table[bucket])continue;
                    if(history_size>=2*a.capacity) {asm volatile("trap;");}
                    for(U64 word=0;word<words;++word)v.history_keys[history_size*words+word]=key[word];
                    v.history_counts[2*history_size]=v.counts[next][2*index];
                    v.history_counts[2*history_size+1]=v.counts[next][2*index+1];
                    v.history_degrees[history_size]=degree;
                    v.history_table[bucket]=++history_size;
                }
                current_size=next_size;
            }
        }
        __syncthreads();
        // Uniform boundary before reusing shared sizes/status at the next
        // degree; never return a previously found leading term on failure.
        if(degree==a.max_degree)break; // avoid UINT64_MAX loop wrap
    }
    __syncthreads();
    if(!threadIdx.x) {
        a.status[0]=static_cast<std::uint8_t>(code);
        v.statistics[0]=code==1?evaluations:0;
        v.statistics[1]=code==1?current_size:0;
        v.statistics[2]=code==1?history_size:0;
    }
}

__global__ void evaluate(Cnot103EndpointArgs a) {
    const U64 sample=U64(blockIdx.x)*blockDim.x+threadIdx.x;if(sample>=a.batch)return;
    a.feature_status[sample]=a.oracle_status[sample]=a.prepared_status[0];
    for(unsigned i=0;i<4;++i)a.values[sample*4+i]=0;
    a.leading_degrees[sample]=a.evaluations[sample]=0;
    for(unsigned i=0;i<2;++i)a.leading_counts[sample*2+i]=a.terminal_counts[sample*2+i]=0;
    if(a.prepared_status[0]!=1)return;
    const U64 words=U64(a.dimension)*a.words_per_row;
    View<const U64> v(a.storage,a.capacity,a.buckets,a.history_buckets,words);
    const U64* key=a.states+sample*words;const U64 hash=hash_key(key,words);
    a.evaluations[sample]=v.statistics[0];
    const U64 first=v.history_table[slot(v.history_keys,v.history_table,a.history_buckets,key,words,hash)];
    const unsigned last=a.max_degree&1;
    const U64 terminal=v.tables[last][slot(v.keys[last],v.tables[last],a.buckets,key,words,hash)];
    if(terminal)for(unsigned i=0;i<2;++i)a.terminal_counts[sample*2+i]=v.counts[last][2*(terminal-1)+i];
    if(!first) {a.feature_status[sample]=3;return;}
    const U64 degree=v.history_degrees[first-1],low=v.history_counts[2*(first-1)],high=v.history_counts[2*(first-1)+1];
    a.leading_degrees[sample]=degree;a.leading_counts[2*sample]=low;a.leading_counts[2*sample+1]=high;
    a.values[sample*4]=static_cast<float>(degree);
    a.values[sample*4+1]=static_cast<float>(log1p(cnot103_i10::as_double(low,high)));
    a.values[sample*4+2]=1.0f;
}

bool multiply(U64 a,U64 b,U64& out){if(b&&a>~U64{0}/b)return false;out=a*b;return true;}
bool plus(U64 a,U64 b,U64& out){if(a>~U64{0}-b)return false;out=a+b;return true;}
bool shape(U32 n,U32 w,U64 capacity,U64 buckets,U64 history_buckets,U64& total) {
    if(n<2 || n>(1U<<24) || w!=(U64(n)+63)/64 || !capacity || capacity>(~U64{0}-1)/4 ||
        buckets<2*capacity+1 || history_buckets<4*capacity+1 || (buckets&(buckets-1)) || (history_buckets&(history_buckets-1)))return false;
    const U64 words=U64(n)*w;
    U64 matrix,counters,tables,stage;
    return multiply(capacity,4*words,matrix) && multiply(capacity,10,counters) &&
        multiply(buckets,2,tables) && plus(tables,history_buckets,tables) && multiply(lanes,words,stage) &&
        plus(stage,3*lanes+3,stage) && plus(matrix,counters,total) && plus(total,tables,total) && plus(total,stage,total);
}
}
extern "C" std::int32_t cnot103_cuda_prepare_endpoint_response(const Cnot103EndpointPrepareArgs* a) {
    if(!a || !a->storage || !a->status)return 1;
    U64 required;
    if(!shape(a->dimension,a->words_per_row,a->capacity,a->buckets,a->history_buckets,required))return 1;
    U64 lower_capacity=1;
    if(a->max_degree) {
        lower_capacity=a->max_layer_states<a->max_transitions?a->max_layer_states:a->max_transitions;
        const U64 bits=U64(a->dimension)*a->dimension;
        if(bits<64 && lower_capacity>(U64{1}<<bits))lower_capacity=U64{1}<<bits;
        if(!lower_capacity)lower_capacity=1;
    }
    if(a->capacity<lower_capacity)return 1;
    if(a->storage_words<required)return 2;
    prepare<<<1,lanes,0,static_cast<cudaStream_t>(a->stream)>>>(*a);
    return static_cast<std::int32_t>(cudaGetLastError());
}
extern "C" std::int32_t cnot103_cuda_launch_endpoint_response(const Cnot103EndpointArgs* a) {
    if(!a || !a->states || !a->storage || !a->prepared_status || !a->values || !a->feature_status || !a->oracle_status ||
        !a->leading_degrees || !a->leading_counts || !a->terminal_counts || !a->evaluations || !a->batch)return 1;
    U64 required,state_words;
    if(!shape(a->dimension,a->words_per_row,a->capacity,a->buckets,a->history_buckets,required) ||
        !multiply(U64(a->batch)*a->dimension,a->words_per_row,state_words))return 1;
    if(a->storage_words<required || a->state_words<state_words || a->value_elements<U64(a->batch)*4 ||
        a->count_elements<U64(a->batch)*2 || a->batch_elements<a->batch)return 2;
    evaluate<<<(U64(a->batch)+lanes-1)/lanes,lanes,0,static_cast<cudaStream_t>(a->stream)>>>(*a);
    return static_cast<std::int32_t>(cudaGetLastError());
}
