#pragma once
#include "one_excess_count.cuh"

namespace cnot103_one_excess {
struct ClassifierLayout {
    CountLayout count;U64 history_words,classifier_words,arithmetic_words,gate_elements;
    __host__ __device__ ClassifierLayout(U32 n,U32 lanes,U32 cache):count(n,lanes,cache){
        history_words=3*U64(n)*count.structure.w;classifier_words=16+4*U64(count.structure.w);
        arithmetic_words=count.arithmetic_words+count.limbs;gate_elements=2*U64(count.k);
    }
};
struct ClassifierView {
    ClassifierLayout l;U64* histories;U32* lengths;std::uint8_t* protected_rows;U64* data;U32* gates;StructureView structure;CountView count;
    __device__ U64& row(U32 physical,U32 position,U32 word)const{return histories[(U64(position)*l.count.structure.w+word)*l.count.structure.n+physical];}
    // Four packed rows: next candidate, previous candidate, first positive,
    // rollback of the first target state during a recovery trial.
    __device__ U64& saved(U32 which,U32 word)const{return data[16+U64(which)*l.count.structure.w+word];}
    __device__ HistoryView history()const{return {histories,lengths,protected_rows,l.count.structure.n,l.count.structure.w};}
    __device__ Limbs total()const{return {count.arithmetic+l.count.arithmetic_words,l.count.limbs,1};}
};
__device__ inline U64 identity_word(U32 row,U32 word){return row/64==word?U64(1)<<(row%64):0;}
__device__ inline bool changed_row(const U64* state,U32 row,U32 w){
    for(U32 word=0;word<w;++word)if(state[U64(row)*w+word]!=identity_word(row,word))return true;return false;
}
__device__ inline U64 intermediate_word(const U64* state,U32 target,U32 endpoint,U32 word,U32 w){
    return state[U64(target)*w+word]^(endpoint%2?identity_word(endpoint/2,word):state[U64(endpoint/2)*w+word]);
}
// Filtering the sorted candidates is equivalent to repeatedly taking the least
// admissible value strictly greater than the previous one. XOR is injective,
// so repeated endpoints/intermediates are deduplicated by that strict bound.
__device__ inline bool endpoint_less(const U64* state, U32 target, U32 a, U32 b, U32 n, U32 w) {
    if (a == 2*n) return false;
    if (b == 2*n) return true;
    for (U32 word = w; word-- > 0;) {
        const U64 x = intermediate_word(state, target, a, word, w);
        const U64 y = intermediate_word(state, target, b, word, w);
        if (x != y) return x < y;
    }
    return a < b;
}

// Endpoint admissibility depends on the target and frozen state, not on the
// previously selected value. Reuse the gate scratch (2*(n+1) u32s per worker)
// while enumerating a target; the output word is cleared before recovery.
// Each lane owns disjoint flags. The barrier publishes them to the entire warp.
__device__ inline void prepare_intermediates(const U64* state,ClassifierView v,U32 target){
    const U32 n=v.l.count.structure.n,w=v.l.count.structure.w;
    for(U32 endpoint=threadIdx.x;endpoint<2*n;endpoint+=32){
        if(endpoint/2==target){v.gates[endpoint]=0;continue;}
        bool nonzero=false,not_initial=false,not_final=false;
        for(U32 word=0;word<w;++word){U64 x=intermediate_word(state,target,endpoint,word,w);nonzero|=x!=0;not_initial|=x!=state[U64(target)*w+word];not_final|=x!=identity_word(target,word);}
        bool admitted=false;
        if(nonzero&&not_initial&&not_final)
            for(U32 control=0;control<n&&!admitted;++control)if(control!=target){bool initial=true,final=true;
                for(U32 word=0;word<w;++word){U64 diff=intermediate_word(state,target,endpoint,word,w)^identity_word(target,word);initial&=diff==state[U64(control)*w+word];final&=diff==identity_word(control,word);}admitted=initial||final;}
        v.gates[endpoint]=admitted?1:0;
    }
    __syncthreads();
}

// classify launches exactly one warp per worker. Each lane checks a disjoint
// set of endpoints. The warp selects the same least packed intermediate as
// the serial scan, including the lowest endpoint on an equal value.
__device__ inline bool next_intermediate(const U64* state,ClassifierView v,U32 target){
    const U32 n=v.l.count.structure.n,w=v.l.count.structure.w;U32 best=2*n;
    const U32 lane = threadIdx.x;
    for(U32 endpoint=lane;endpoint<2*n;endpoint+=32){if(!v.gates[endpoint])continue;
        if(v.data[9]){int comparison=0;for(U32 word=w;word-->0;){U64 x=intermediate_word(state,target,endpoint,word,w),previous=v.saved(1,word);if(x!=previous){comparison=x>previous?1:-1;break;}}
            if(comparison<=0)continue;}
        if(endpoint_less(state,target,endpoint,best,n,w))best=endpoint;
    }
    for (U32 stride=16; stride; stride/=2) {
        const U32 other = __shfl_down_sync(0xffffffffu, best, stride);
        if (lane < stride && endpoint_less(state,target,other,best,n,w)) best=other;
    }
    best=__shfl_sync(0xffffffffu,best,0);
    if(best==2*n)return false;
    for(U32 word=lane;word<w;word+=32)v.saved(0,word)=v.saved(1,word)=intermediate_word(state,target,best,word,w);
    if(lane==0)v.data[9]=1;
    return true;
}
__device__ inline void build_history(const U64* state,ClassifierView v,U32 twice,U32 saved_row){
    const U32 n=v.l.count.structure.n,w=v.l.count.structure.w;
    for(U32 r=threadIdx.x;r<n;r+=blockDim.x){const bool changed=changed_row(state,r,w);v.lengths[r]=r==twice?3:changed?2:1;v.protected_rows[r]=0;
        for(U32 word=0;word<w;++word){v.row(r,0,word)=state[U64(r)*w+word];v.row(r,1,word)=r==twice?v.saved(saved_row,word):changed?identity_word(r,word):0;v.row(r,2,word)=r==twice?identity_word(r,word):0;}}
    __syncthreads();
}
// Count calls, including the selected-template recount and recovery trials.
// This counter is shared across the entire state and reproduced in the checker.
__device__ inline bool acquire_history_count(ClassifierView v,U32 max_history_counts){
    if(!threadIdx.x){
        if(max_history_counts&&v.data[13]>=max_history_counts)v.data[0]=4;
        else ++v.data[13];
    }
    __syncthreads();return v.data[0]!=4;
}
// Full source recognition after the complete-unit alphabet metadata guard.
// data: class, changed, templates, bag, transport-free, gate count, chosen row;
// remaining prefix fields are private loop/decision scalars.
__device__ inline void recognize_history_family(const U64* state,const U64* zero_header,std::uint8_t zero_status,ClassifierView v,U64 max_entries,U32 max_changed_rows,U32 max_component_events,U32 max_history_counts){
    const U32 n=v.l.count.structure.n,w=v.l.count.structure.w;
    for(U64 i=threadIdx.x;i<v.l.classifier_words;i+=blockDim.x)v.data[i]=0;
    for(U64 i=threadIdx.x;i<v.l.gate_elements;i+=blockDim.x)v.gates[i]=0;
    __syncthreads();if(!threadIdx.x){clear(v.total());v.data[6]=n;v.data[4]=1;}
    __syncthreads();build_history(state,v,n,0);require_exact(validate_histories(v.history(),v.structure));
    if(!threadIdx.x){U32 changed=0,missing=0;
        for(U32 r=0;r<n;++r){U32 weight=0;for(U32 word=0;word<w;++word)weight+=__popcll(state[U64(r)*w+word]);
            if(weight==1&&!(state[U64(r)*w+r/64]>>(r%64)&1))v.data[4]=0;
            if(v.lengths[r]==1)continue;++changed;bool parent=false;
            for(U32 c=0;c<n&&!parent;++c)if(c!=r){bool initial=true,final=true;
                for(U32 word=0;word<w;++word){U64 diff=state[U64(r)*w+word]^identity_word(r,word);initial&=diff==state[U64(c)*w+word];final&=diff==identity_word(c,word);}parent=initial||final;}
            missing+=!parent;
        }
        require_exact(zero_status==1&&zero_header[0]==(missing?2:1)&&zero_header[1]==changed&&zero_header[2]==missing);
        v.data[1]=changed;v.data[0]=missing?2:3;
    }
    __syncthreads();
    // Cheap zero-excess classifications remain available at any changed-row count.
    // Expensive one-excess enumeration beyond the configured scope is unknown.
    if(!threadIdx.x&&v.data[0]==2&&max_changed_rows&&v.data[1]>max_changed_rows)v.data[0]=4;
    __syncthreads();if(v.data[0]>=3)return;
    for(U32 target=0;target<n;++target){if(!changed_row(state,target,w))continue;
        if(!threadIdx.x)v.data[9]=0;__syncthreads();
        prepare_intermediates(state,v,target);
        while(true){const bool found=next_intermediate(state,v,target);
            if(!threadIdx.x&&found)++v.data[2];
            __syncthreads();if(!found)break;
            build_history(state,v,target,0);if(!acquire_history_count(v,max_history_counts))return;
            count_history(v.history(),v.structure,v.count,max_entries,max_component_events);
            if(!threadIdx.x){if(v.count.data[6]==4)v.data[0]=4;
                else{if(v.count.data[5]>v.data[3])v.data[3]=v.count.data[5];require_exact(add(v.total(),v.total(),v.count.global(0)));
                    if(v.data[6]==n&&used(v.count.global(0))){v.data[6]=target;for(U32 word=0;word<w;++word)v.saved(2,word)=v.saved(0,word);}}}
            __syncthreads();if(v.data[0]==4)return;
        }
    }
    if(!threadIdx.x)v.data[0]=used(v.total())?1:2;__syncthreads();if(v.data[0]==2)return;
    // Enumeration borrowed the future output-word scratch for flags. Restore
    // canonical zero padding before writing a recovered positive certificate.
    for(U64 i=threadIdx.x;i<v.l.gate_elements;i+=blockDim.x)v.gates[i]=0;
    __syncthreads();
    require_exact(v.data[6]<n);build_history(state,v,static_cast<U32>(v.data[6]),2);
    // The source recounts the chosen template before retaining its double row.
    if(!acquire_history_count(v,max_history_counts))return;
    count_history(v.history(),v.structure,v.count,max_entries,max_component_events);
    if(!threadIdx.x){if(v.count.data[6]==4)v.data[0]=4;else require_exact(used(v.count.global(0))&&v.structure.meta[0]==v.data[1]+1);}
    __syncthreads();if(v.data[0]==4)return;
    for(U32 r=threadIdx.x;r<n;r+=blockDim.x)v.protected_rows[r]=r==v.data[6];__syncthreads();
    const U32 distance=static_cast<U32>(v.data[1]+1);
    for(U32 step=0;step<distance;++step){if(!threadIdx.x)v.data[12]=0;__syncthreads();
        for(U32 target=0;target<n;++target){if(v.lengths[target]==1)continue;
            if(!threadIdx.x){v.data[10]=n;v.data[11]=v.lengths[target];
                for(U32 c=0;c<n;++c)if(c!=target){bool equal=true;for(U32 word=0;word<w;++word)equal&=(v.row(target,0,word)^v.row(target,1,word))==v.row(c,0,word);
                    if(equal){require_exact(v.data[10]==n);v.data[10]=c;}}}
            __syncthreads();if(v.data[10]==n)continue;
            const U32 old_length=static_cast<U32>(v.data[11]);
            for(U32 word=threadIdx.x;word<w;word+=blockDim.x){v.saved(3,word)=v.row(target,0,word);
                for(U32 p=0;p+1<old_length;++p)v.row(target,p,word)=v.row(target,p+1,word);v.row(target,old_length-1,word)=0;}
            if(!threadIdx.x)v.lengths[target]=old_length-1;__syncthreads();
            if(!acquire_history_count(v,max_history_counts))return;
            count_history(v.history(),v.structure,v.count,max_entries,max_component_events);
            if(!threadIdx.x){if(v.count.data[6]==4)v.data[0]=4;else if(used(v.count.global(0))){v.data[12]=1;const U32 at=distance-1-step;
                    v.gates[2*at]=static_cast<U32>(v.data[10]);v.gates[2*at+1]=target;++v.data[5];}}
            __syncthreads();if(v.data[0]==4)return;if(v.data[12])break;
            for(U32 word=threadIdx.x;word<w;word+=blockDim.x){for(U32 p=old_length-1;p>0;--p)v.row(target,p,word)=v.row(target,p-1,word);v.row(target,0,word)=v.saved(3,word);}
            if(!threadIdx.x)v.lengths[target]=old_length;__syncthreads();
        }
        require_exact(v.data[12]==1);
        __syncthreads();
    }
    if(!threadIdx.x){require_exact(v.data[5]==distance);for(U32 r=0;r<n;++r)require_exact(v.lengths[r]==1);
        require_exact(!v.data[4]||v.data[3]<=5);}
    __syncthreads();
}
__device__ inline void canonicalize_result(ClassifierView v){
    if(v.data[0]!=1)for(U64 i=threadIdx.x;i<v.l.gate_elements;i+=blockDim.x)v.gates[i]=0;
    if(!threadIdx.x){if(v.data[0]!=1){v.data[4]=0;v.data[5]=0;}
        if(v.data[0]==4){v.data[2]=v.data[3]=0;clear(v.total());}}
    __syncthreads();
}
__device__ inline U64 result_header(ClassifierView v,U32 field){
    if(field<5)return v.data[field];return v.data[0]<=2?v.data[1]+1:0;
}
}
