#include "cnot103/cnot_cuda.h"
#include "paid_basis_atlas.cuh"

namespace {
using namespace cnot103_paid;
using I64=std::int64_t;
using U8=std::uint8_t;
__host__ __device__ U64 prep_capacity(U64 cells){return cells/2?cells/2:1;}
__device__ U8 vector_bit(const U8* block,U32 sources,U32 side,U32 index,U32 coordinate){
    return side==0?block[U64(coordinate)*sources+index]:block[U64(index)*sources+coordinate];
}
__device__ void append_group(U64* groups,U32& count,const U32* indices,U32 width,U32 limit){
    for(U32 i=0;i<count;++i){
        if(groups[U64(i)*5]!=width)continue;
        bool same=true;for(U32 j=0;j<width;++j)same&=groups[U64(i)*5+1+j]==indices[j];
        if(same)return;
    }
    if(count>=limit)asm volatile("trap;");
    groups[U64(count)*5]=width;
    for(U32 j=0;j<4;++j)groups[U64(count)*5+1+j]=j<width?indices[j]:0;
    ++count;
}
__device__ void sort_group(U32* group,U32 width){
    for(U32 i=1;i<width;++i){const U32 value=group[i];U32 j=i;while(j&&group[j-1]>value){group[j]=group[j-1];--j;}group[j]=value;}
}
// One frozen group list per side/sweep. Scores are computed cooperatively;
// ordered publication and duplicate elimination have a single owner.
__device__ void select_groups(const U8* block,U32 sources,U32 targets,U32 side,U32 max_width,
    U64* weights,U64* active,U64* leaders,I64* scores,U64* groups,U32& active_count,U32& group_count) {
    const U32 width=side==0?sources:targets,coordinates=side==0?targets:sources;
    for(U32 index=threadIdx.x;index<width;index+=blockDim.x){
        U64 weight=0;for(U32 k=0;k<coordinates;++k)weight+=vector_bit(block,sources,side,index,k);weights[index]=weight;
    }
    __syncthreads();
    if(!threadIdx.x){
        active_count=group_count=0;
        for(U32 i=0;i<width;++i)if(weights[i])active[active_count++]=i;
        for(U32 i=0;i<active_count;++i){
            const U64 value=active[i];U32 j=i;
            while(j && (weights[leaders[j-1]]<weights[value] || (weights[leaders[j-1]]==weights[value]&&leaders[j-1]>value))){leaders[j]=leaders[j-1];--j;}
            leaders[j]=value;
        }
    }
    __syncthreads();
    if(active_count<2)return;
    const U32 local=min(max_width,active_count);
    for(U32 position=0;position<active_count;++position){
        const U32 leader=leaders[position];
        for(U32 i=threadIdx.x;i<active_count;i+=blockDim.x){
            const U32 other=active[i];I64 distance=0;
            for(U32 k=0;k<coordinates;++k)distance+=vector_bit(block,sources,side,leader,k)!=vector_bit(block,sources,side,other,k);
            scores[other]=distance-static_cast<I64>(max(weights[leader],weights[other]));
        }
        __syncthreads();
        if(!threadIdx.x){
            U32 group[4]={leader,0,0,0};
            for(U32 slot=1;slot<local;++slot){
                U32 best=~U32{0};I64 best_score=0;
                for(U32 i=0;i<active_count;++i){
                    const U32 other=active[i];bool used=false;for(U32 k=0;k<slot;++k)used|=group[k]==other;if(used)continue;
                    if(best==~U32{0} || scores[other]<best_score || (scores[other]==best_score&&other<best)){best=other;best_score=scores[other];}
                }
                group[slot]=best;
            }
            sort_group(group,local);append_group(groups,group_count,group,local,2*width);
        }
        __syncthreads();
    }
    if(!threadIdx.x)for(U32 start=0;start<active_count;start+=local-1){
        const U32 length=min(local,active_count-start);if(length<2)continue;
        U32 group[4]={0,0,0,0};for(U32 i=0;i<length;++i)group[i]=active[start+i];
        append_group(groups,group_count,group,length,2*width);
    }
    __syncthreads();
}
__device__ void clear_payload(const Cnot103PaidShearArgs& a,U64 sample){
    const U64 cells=U64(a.source_count)*a.target_count,pcap=prep_capacity(cells),matrix=U64(a.dimension)*a.words_per_row;
    for(U64 i=threadIdx.x;i<cells;i+=blockDim.x)a.block[sample*cells+i]=0;
    for(U64 i=threadIdx.x;i<U64(a.source_count)*a.source_count;i+=blockDim.x)a.source_basis[sample*U64(a.source_count)*a.source_count+i]=0;
    for(U64 i=threadIdx.x;i<U64(a.target_count)*a.target_count;i+=blockDim.x)a.target_basis[sample*U64(a.target_count)*a.target_count+i]=0;
    for(U64 i=threadIdx.x;i<pcap*2;i+=blockDim.x){a.preparation[sample*pcap*2+i]=0;a.restoration[sample*pcap*2+i]=0;a.cancellations[sample*pcap*2+i]=0;}
    for(U64 i=threadIdx.x;i<cells*2;i+=blockDim.x){a.core[sample*cells*2+i]=0;a.word[sample*cells*2+i]=0;}
    for(U64 i=threadIdx.x;i<pcap*11;i+=blockDim.x)a.moves[sample*pcap*11+i]=0;
    for(U64 i=threadIdx.x;i<matrix;i+=blockDim.x)a.endpoint[sample*matrix+i]=0;
    if(threadIdx.x<16)a.statistics[sample*16+threadIdx.x]=0;
    if(!threadIdx.x){a.status[sample]=3;a.verified[sample]=0;a.lengths[sample]=0;}
}
__device__ void push_gate(U32 control,U32 target,U64 label,U32* word,U64* labels,U32* events,
    U64& length,U64& canceled,U64 capacity,U64 event_capacity){
    if(length && word[2*(length-1)]==control && word[2*(length-1)+1]==target){
        --length;
        if(labels[length]!=label){if(canceled>=event_capacity)asm volatile("trap;");events[2*canceled]=control;events[2*canceled+1]=target;++canceled;}
        word[2*length]=word[2*length+1]=0;
    }else{
        if(length>=capacity)asm volatile("trap;");word[2*length]=control;word[2*length+1]=target;labels[length]=label;++length;
    }
}

__global__ void compile(Cnot103PaidShearArgs a){
    const U64 sample=blockIdx.x,S=a.source_count,T=a.target_count,C=S*T,P=prep_capacity(C),L=max(S,T),N=U64(a.dimension)*a.words_per_row;
    U8* block=a.block+sample*C;U8* sb=a.source_basis+sample*S*S;U8* tb=a.target_basis+sample*T*T;
    U32* prep=a.preparation+sample*P*2;U32* core=a.core+sample*C*2;U32* restore=a.restoration+sample*P*2;
    U32* word=a.word+sample*C*2;U32* events=a.cancellations+sample*P*2;
    U64* stats=a.statistics+sample*16;U64* moves=a.moves+sample*P*11;
    U64* workspace=a.scratch+sample*(15*L+P+C);
    U64* weights=workspace;U64* active=weights+L;U64* leaders=active+L;I64* scores=reinterpret_cast<I64*>(leaders+L);
    U64* before_codes=reinterpret_cast<U64*>(scores+L);U64* groups=before_codes+L;U64* prep_labels=groups+10*L;U64* stack_labels=prep_labels+P;
    __shared__ U32 active_count,group_count,selected,operation_count,local_ops[42];
    __shared__ unsigned long long histogram[16],current_weight,sum;
    __shared__ U64 costs[lanes],sweep_before,cost_before,prep_start;
    __shared__ U32 indices[lanes];
    clear_payload(a,sample);
    if(!threadIdx.x)sum=0;
    __syncthreads();
    for(U64 i=threadIdx.x;i<N;i+=blockDim.x){const U64 row=i/a.words_per_row,part=i%a.words_per_row;a.endpoint[sample*N+i]=part==row/64?(U64{1}<<(row%64)):0;}
    for(U64 i=threadIdx.x;i<S;++i)sb[i*S+i]=1;
    for(U64 i=threadIdx.x;i<T;++i)tb[i*T+i]=1;
    __syncthreads();
    U64 local_weight=0;
    for(U64 cell=threadIdx.x;cell<C;cell+=blockDim.x){
        const U32 row=a.targets[cell/S],column=a.sources[cell%S];
        if(row>=a.dimension||column>=a.dimension)asm volatile("trap;");
        const U8 bit=(a.states[sample*N+U64(row)*a.words_per_row+column/64]>>(column%64))&1;
        block[cell]=bit;local_weight+=bit;
        if(bit)atomicOr(reinterpret_cast<unsigned long long*>(a.endpoint+sample*N+U64(row)*a.words_per_row+column/64),1ULL<<(column%64));
    }
    atomicAdd(&sum,static_cast<unsigned long long>(local_weight));
    __syncthreads();
    if(!threadIdx.x){current_weight=sum;stats[0]=sum;}
    __syncthreads();
    for(U32 sweep=0;sweep<a.sweeps;++sweep){
        if(!threadIdx.x)sweep_before=2*stats[2]+current_weight;
        __syncthreads();
        for(U32 side=0;side<2;++side){
            select_groups(block,S,T,side,a.local_width,weights,active,leaders,scores,groups,active_count,group_count);
            const U32 coordinates=side==0?T:S;
            for(U32 group=0;group<group_count;++group){
                const U64* record=groups+U64(group)*5;const U32 width=record[0];
                Atlas<const U32> atlas(a.atlases[side*3+width-2],width);
                if(threadIdx.x<16)histogram[threadIdx.x]=0;
                if(!threadIdx.x)++stats[7];
                __syncthreads();
                for(U32 row=threadIdx.x;row<coordinates;row+=blockDim.x){
                    U32 code=0;for(U32 bit=0;bit<width;++bit)code|=U32(vector_bit(block,S,side,record[bit+1],row))<<bit;
                    before_codes[row]=code;atomicAdd(histogram+code,1ULL);
                }
                __syncthreads();
                const U32 winner=choose(atlas,width,histogram,costs,indices);
                if(!threadIdx.x){selected=winner;operation_count=path(atlas,selected,local_ops,21);cost_before=2*stats[2]+current_weight;prep_start=stats[2];}
                __syncthreads();
                if(!operation_count)continue;
                for(U32 op=0;op<operation_count;++op){
                    const U32 control=record[local_ops[2*op]+1],target=record[local_ops[2*op+1]+1];
                    if(side==0){
                        for(U32 row=threadIdx.x;row<T;row+=blockDim.x)block[U64(row)*S+control]^=block[U64(row)*S+target];
                        for(U32 col=threadIdx.x;col<S;col+=blockDim.x)sb[U64(target)*S+col]^=sb[U64(control)*S+col];
                    }else{
                        for(U32 col=threadIdx.x;col<S;col+=blockDim.x)block[U64(target)*S+col]^=block[U64(control)*S+col];
                        for(U32 col=threadIdx.x;col<T;col+=blockDim.x)tb[U64(target)*T+col]^=tb[U64(control)*T+col];
                    }
                    if(!threadIdx.x){
                        if(stats[2]>=P)asm volatile("trap;");const U32* wires=side==0?a.sources:a.targets;
                        prep[2*stats[2]]=wires[control];prep[2*stats[2]+1]=wires[target];prep_labels[stats[2]]=stats[8];++stats[2];
                    }
                    __syncthreads();
                }
                for(U32 row=threadIdx.x;row<coordinates;row+=blockDim.x){
                    U32 actual=0;for(U32 bit=0;bit<width;++bit)actual|=U32(vector_bit(block,S,side,record[bit+1],row))<<bit;
                    if(actual!=image(atlas.maps[selected],width,before_codes[row]))asm volatile("trap;");
                }
                if(!threadIdx.x)sum=0;
                __syncthreads();
                U64 weight=0;for(U64 cell=threadIdx.x;cell<C;cell+=blockDim.x)weight+=block[cell];
                atomicAdd(&sum,static_cast<unsigned long long>(weight));
                __syncthreads();
                if(!threadIdx.x){
                    current_weight=sum;const U64 after=2*stats[2]+current_weight;
                    if(after>=cost_before || after>stats[0] || stats[8]>=P)asm volatile("trap;");
                    U64* move=moves+stats[8]*11;move[0]=sweep;move[1]=side;move[2]=width;
                    for(U32 i=0;i<4;++i)move[3+i]=i<width?record[i+1]:0;
                    move[7]=cost_before;move[8]=after;move[9]=prep_start;move[10]=stats[2];
                    ++stats[8];++stats[9+side];stats[3+side]+=operation_count;
                }
                __syncthreads();
            }
            // Every lane finishes reading frozen groups before next-side reuse.
            __syncthreads();
        }
        const bool unchanged=2*stats[2]+current_weight==sweep_before;
        __syncthreads();
        if(unchanged)break;
    }
    if(!threadIdx.x){
        for(U32 row=0;row<T;++row)for(U32 col=0;col<S;++col)if(block[U64(row)*S+col]){core[2*stats[5]]=a.sources[col];core[2*stats[5]+1]=a.targets[row];++stats[5];}
        stats[6]=stats[2];stats[15]=2*stats[2]+stats[5];
        if(stats[15]>stats[0])asm volatile("trap;");
        for(U64 i=0;i<stats[2];++i){restore[2*i]=prep[2*(stats[2]-1-i)];restore[2*i+1]=prep[2*(stats[2]-1-i)+1];}
        for(U64 i=0;i<stats[2];++i)push_gate(prep[2*i],prep[2*i+1],prep_labels[i],word,stack_labels,events,stats[1],stats[14],C,P);
        for(U64 i=0;i<stats[5];++i)push_gate(core[2*i],core[2*i+1],stats[8],word,stack_labels,events,stats[1],stats[14],C,P);
        for(U64 i=0;i<stats[2];++i)push_gate(restore[2*i],restore[2*i+1],2*stats[8]-prep_labels[stats[2]-1-i],word,stack_labels,events,stats[1],stats[14],C,P);
        for(U64 i=0;i<S*S;++i)stats[11]+=sb[i]!=(i/S==i%S);
        for(U64 i=0;i<T*T;++i)stats[12]+=tb[i]!=(i/T==i%T);
        for(U64 i=0;i+2<stats[1];){
            const bool swap=word[2*i]==word[2*(i+2)]&&word[2*i+1]==word[2*(i+2)+1]&&word[2*i]==word[2*(i+1)+1]&&word[2*i+1]==word[2*(i+1)];
            if(swap){++stats[13];i+=3;}else ++i;
        }
        if(stats[1]>stats[0])asm volatile("trap;");a.lengths[sample]=stats[1];
    }
}
__global__ void publish(Cnot103PaidShearArgs a){
    __shared__ U32 failed;if(!threadIdx.x)failed=!a.verified[blockIdx.x];__syncthreads();
    if(failed)clear_payload(a,blockIdx.x);else if(!threadIdx.x)a.status[blockIdx.x]=1;
}
bool multiply(U64 a,U64 b,U64& out){if(b&&a>~U64{0}/b)return false;out=a*b;return true;}
bool extent(U64 provided,U64 batch,U64 count){U64 required;return multiply(batch,count,required)&&provided>=required;}
}
extern "C" std::int32_t cnot103_cuda_launch_paid_shear(const Cnot103PaidShearArgs* a){
    if(!a||!a->states||!a->sources||!a->targets||!a->block||!a->source_basis||!a->target_basis||!a->preparation||!a->restoration||
        !a->core||!a->word||!a->cancellations||!a->moves||!a->statistics||!a->lengths||!a->endpoint||!a->replay||!a->scratch||!a->status||!a->verified)return 1;
    if(!a->batch||a->batch>0x7fffffffu||a->dimension<2||a->dimension>(1U<<24)||a->words_per_row!=(U64(a->dimension)+63)/64||
        !a->source_count||!a->target_count||U64(a->source_count)+a->target_count>a->dimension||a->local_width<2||a->local_width>4||a->sweeps>16)return 1;
    const U64 S=a->source_count,T=a->target_count,C=S*T,P=prep_capacity(C),L=S>T?S:T,N=U64(a->dimension)*a->words_per_row;
    for(U32 side=0;side<2;++side)for(U32 width=2;width<=a->local_width;++width){const U32 index=side*3+width-2;
        if(!a->atlases[index])return 1;if(a->atlas_words[index]<storage_words(width))return 2;}
    if(a->source_elements<S||a->target_elements<T||a->batch_elements<a->batch||!extent(a->state_words,a->batch,N)||
        !extent(a->block_elements,a->batch,C)||!extent(a->source_basis_elements,a->batch,S*S)||!extent(a->target_basis_elements,a->batch,T*T)||
        !extent(a->preparation_elements,a->batch,2*P)||!extent(a->core_elements,a->batch,2*C)||!extent(a->move_elements,a->batch,11*P)||
        !extent(a->stat_elements,a->batch,16)||!extent(a->scratch_words,a->batch,15*L+P+C))return 2;
    const auto stream=static_cast<cudaStream_t>(a->stream);
    compile<<<a->batch,lanes,0,stream>>>(*a);auto code=cudaGetLastError();if(code!=cudaSuccess)return static_cast<std::int32_t>(code);
    const auto replay=cnot103_cuda_verify_cnot_words(a->endpoint,a->word,a->lengths,a->replay,a->verified,a->batch,a->dimension,a->words_per_row,C,a->state_words,a->core_elements,a->stream);
    if(replay)return replay;
    publish<<<a->batch,lanes,0,stream>>>(*a);return static_cast<std::int32_t>(cudaGetLastError());
}
