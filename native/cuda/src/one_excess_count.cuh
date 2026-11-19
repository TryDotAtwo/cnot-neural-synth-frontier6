#pragma once
#include "one_excess_structure.cuh"

namespace cnot103_one_excess {
// Source tables form an expression forest: each consumed table has one parent.
// A bounded cache affects recomputation only, never the source table budget.
struct CountLayout {
    StructureLayout structure;
    U32 k,limbs,nodes,depth,lanes,cache;
    U64 scopes,digits,tags,indices,frames,evictions,failed,program_words,lane_slots,owner_slots,arithmetic_words;
    __host__ __device__ CountLayout(U32 n,U32 lane_count,U32 cache_slots):structure(n),k(n+1),limbs(limbs_for_dimension(n)),nodes(2*n+3),depth(n+2),lanes(lane_count),cache(cache_slots){
        scopes=16+8*U64(nodes);digits=scopes+U64(nodes)*structure.ew;tags=digits+U64(k)*lanes;
        indices=tags+U64(cache)*lanes;frames=indices+U64(cache)*lanes;
        evictions=frames+5*U64(depth)*lanes;failed=evictions+lanes;program_words=failed+lanes;
        lane_slots=2*U64(depth)+3+cache;owner_slots=U64(k)+6;
        arithmetic_words=U64(limbs)*(lane_slots*lanes+owner_slots);
    }
};
struct CountView {
    CountLayout l;U64* data;U32* arithmetic;
    // Node arrays: kind(0factor/1sum/2free), reference, child_start/count,
    // live, roots, concatenated children, output-scope arity.
    __device__ U64& node(U32 field,U32 id)const{return data[16+U64(field)*l.nodes+id];}
    __device__ U64& scope(U32 id,U32 word)const{return data[l.scopes+U64(word)*l.nodes+id];}
    __device__ U64& digit(U32 event,U32 lane)const{return data[l.digits+U64(event)*l.lanes+lane];}
    __device__ U64& tag(U32 slot,U32 lane)const{return data[l.tags+U64(slot)*l.lanes+lane];}
    __device__ U64& index(U32 slot,U32 lane)const{return data[l.indices+U64(slot)*l.lanes+lane];}
    __device__ U64& frame(U32 depth,U32 field,U32 lane)const{return data[l.frames+(U64(field)*l.depth+depth)*l.lanes+lane];}
    __device__ Limbs number(U64 slot,U32 lane)const{return {arithmetic+slot*l.lanes+lane,l.limbs,l.lane_slots*l.lanes};}
    __device__ Limbs owner(U64 slot)const{return {arithmetic+U64(l.limbs)*l.lane_slots*l.lanes+slot,l.limbs,l.owner_slots};}
    __device__ Limbs global(U32 slot)const{return owner(U64(l.k)+1+slot);}
    __device__ Limbs returned(U32 lane)const{return number(2*U64(l.depth),lane);}
    __device__ Limbs temporary(U32 lane)const{return number(2*U64(l.depth)+1,lane);}
    __device__ Limbs product(U32 lane)const{return number(2*U64(l.depth)+2,lane);}
    __device__ Limbs cached(U32 slot,U32 lane)const{return number(2*U64(l.depth)+3+slot,lane);}
};

// Called by the CTA owner after the exact component order has been computed.
// Creation order is live-table order, so ascending live IDs reproduce source
// drain/remaining/append semantics without a linked pointer graph.
__device__ inline void build_expression(StructureView s,CountView p,U32 component){
    p.data[0]=p.data[1]=p.data[2]=0;
    for(U32 i=0;i<p.l.nodes;++i)p.node(4,i)=0;
    for(U32 f=0;f<s.meta[1];++f){bool nonempty=false,belongs=true;U32 arity=0;
        for(U32 w=0;w<s.l.ew;++w){U64 bits=s.scope(f,w);arity+=__popcll(bits);while(bits){U32 e=w*64+__ffsll(bits)-1;bits&=bits-1;nonempty=true;belongs&=s.meta[s.l.ids+e]==component;}}
        if(!nonempty||!belongs)continue;
        U32 id=static_cast<U32>(p.data[0]++);require_exact(id<p.l.nodes);p.node(0,id)=0;p.node(1,id)=f;p.node(2,id)=0;p.node(3,id)=0;p.node(4,id)=1;p.node(7,id)=arity;
        for(U32 w=0;w<s.l.ew;++w)p.scope(id,w)=s.scope(f,w);
    }
    const U32 begin=s.meta[s.l.offsets+component],end=s.meta[s.l.offsets+component+1];
    for(U32 at=begin;at<end;++at){const U32 variable=s.meta[s.l.order+at],id=static_cast<U32>(p.data[0]++);
        require_exact(id<p.l.nodes);p.node(0,id)=1;p.node(1,id)=variable;p.node(2,id)=p.data[2];p.node(3,id)=0;p.node(4,id)=1;
        for(U32 w=0;w<s.l.ew;++w)p.scope(id,w)=0;
        for(U32 old=0;old<id;++old)if(p.node(4,old)&&(p.scope(old,variable/64)>>(variable%64)&1)){
            const U32 edge=static_cast<U32>(p.data[2]++);require_exact(edge<p.l.nodes);p.node(6,edge)=old;++p.node(3,id);p.node(4,old)=0;
            for(U32 w=0;w<s.l.ew;++w)p.scope(id,w)|=p.scope(old,w);
        }
        if(!p.node(3,id))p.node(0,id)=2;
        else {require_exact(p.scope(id,variable/64)>>(variable%64)&1);p.scope(id,variable/64)&=~(U64(1)<<(variable%64));}
        U32 arity=0;for(U32 w=0;w<s.l.ew;++w)arity+=__popcll(p.scope(id,w));p.node(7,id)=arity;
    }
    for(U32 id=0;id<p.data[0];++id)if(p.node(4,id)){require_exact(p.node(7,id)==0);p.node(5,static_cast<U32>(p.data[1]++))=id;}
}
__device__ inline U64 assignment_index(CountView p,U32 node,U32 lane,U32 q){
    U64 index=0;
    for(U32 w=0;w<p.l.structure.ew;++w){U64 scope=p.scope(node,w);while(scope){U32 e=w*64+__ffsll(scope)-1;scope&=scope-1;const U64 digit=p.digit(e,lane);
        require_exact(digit<q && index<=(~U64(0)-digit)/q);index=index*q+digit;}}
    return index;
}
__device__ inline U32 cache_slot(CountView p,U32 node,U64 index){return static_cast<U32>(((index^(index>>32))+U64(node)*0x9e3779b97f4a7c15ULL)%p.l.cache);}
__device__ inline bool cache_load(CountView p,U32 node,U64 index,U32 lane){
    if(!p.l.cache)return false;const U32 slot=cache_slot(p,node,index);
    if(p.tag(slot,lane)!=node||p.index(slot,lane)!=index)return false;
    copy(p.returned(lane),p.cached(slot,lane));return true;
}
__device__ inline void cache_store(CountView p,U32 node,U64 index,U32 lane){
    if(!p.l.cache)return;const U32 slot=cache_slot(p,node,index);
    if(p.tag(slot,lane)!=p.l.nodes&&(p.tag(slot,lane)!=node||p.index(slot,lane)!=index)){
        U64& evictions=p.data[p.l.evictions+lane];if(evictions!=~U64(0))++evictions;
    }
    copy(p.cached(slot,lane),p.returned(lane));p.index(slot,lane)=index;p.tag(slot,lane)=node;
}

// Explicit depth-first stack. A child's eliminated variables do not occur in
// its parent's output scope or in a sibling's live scope. Cache keys therefore
// need only the node and its complete output-scope assignment for this q.
__device__ inline void evaluate_node(StructureView s,CountView p,U32 root,U32 lane,U32 q){
    int depth=0;p.frame(0,0,lane)=root;p.frame(0,1,lane)=0;
    while(depth>=0){const U32 d=static_cast<U32>(depth),node=static_cast<U32>(p.frame(d,0,lane));
        const U32 phase=static_cast<U32>(p.frame(d,1,lane));require_exact(node<p.data[0]);
        if(phase==0){const U64 index=assignment_index(p,node,lane,q);p.frame(d,4,lane)=index;
            if(cache_load(p,node,index,lane)){--depth;continue;}
            const U32 kind=static_cast<U32>(p.node(0,node));
            if(kind==0){const U32 f=static_cast<U32>(p.node(1,node));U32 value=0;
                for(U32 t=0;t<s.terms(f);++t){bool valid=true;for(U32 pair=0;pair<s.term(f,t,0);++pair)valid&=p.digit(s.term(f,t,1+2*pair),lane)<p.digit(s.term(f,t,2+2*pair),lane);value+=valid;}
                set_small(p.returned(lane),value);cache_store(p,node,index,lane);--depth;continue;
            }
            if(kind==2){set_small(p.returned(lane),q);cache_store(p,node,index,lane);--depth;continue;}
            require_exact(kind==1&&p.node(3,node)!=0);clear(p.number(2*U64(d),lane));set_small(p.number(2*U64(d)+1,lane),1);
            p.frame(d,2,lane)=0;p.frame(d,3,lane)=0;p.digit(static_cast<U32>(p.node(1,node)),lane)=0;p.frame(d,1,lane)=1;
        }else if(phase==1){
            if(p.frame(d,2,lane)<p.node(3,node)){const U32 edge=static_cast<U32>(p.node(2,node)+p.frame(d,2,lane));require_exact(edge<p.data[2]&&d+1<p.l.depth);
                p.frame(d,1,lane)=2;p.frame(d+1,0,lane)=p.node(6,edge);p.frame(d+1,1,lane)=0;++depth;
            }else{
                require_exact(add(p.number(2*U64(d),lane),p.number(2*U64(d),lane),p.number(2*U64(d)+1,lane)));
                if(++p.frame(d,3,lane)<q){p.digit(static_cast<U32>(p.node(1,node)),lane)=p.frame(d,3,lane);set_small(p.number(2*U64(d)+1,lane),1);p.frame(d,2,lane)=0;}
                else{copy(p.returned(lane),p.number(2*U64(d),lane));cache_store(p,node,p.frame(d,4,lane),lane);--depth;}
            }
        }else{require_exact(phase==2);
            if(used(p.returned(lane))==0){clear(p.number(2*U64(d)+1,lane));p.frame(d,2,lane)=p.node(3,node);}
            else{require_exact(multiply(p.temporary(lane),p.number(2*U64(d)+1,lane),p.returned(lane)));copy(p.number(2*U64(d)+1,lane),p.temporary(lane));++p.frame(d,2,lane);}
            p.frame(d,1,lane)=1;
        }
    }
}
__device__ inline bool lattice_sum(StructureView s,CountView p,U32 lane,U32 q,U64 max_entries){
    // Initial factors precede all elimination nodes. Preserve every source
    // q^scope, q^joint and q^reduced guard before numerical zero shortcuts.
    for(U32 node=0;node<p.data[0];++node){U64 size=0;const U32 kind=static_cast<U32>(p.node(0,node)),arity=static_cast<U32>(p.node(7,node));
        if(kind==0){if(!checked_power(q,arity,max_entries,size))return false;}
        else if(kind==1){if(!checked_power(q,arity+1,max_entries,size)||!checked_power(q,arity,max_entries,size))return false;}
    }
    for(U32 slot=0;slot<p.l.cache;++slot)p.tag(slot,lane)=p.l.nodes;
    for(U32 e=0;e<p.l.k;++e)p.digit(e,lane)=0;
    set_small(p.product(lane),1);
    for(U32 root=0;root<p.data[1];++root){evaluate_node(s,p,static_cast<U32>(p.node(5,root)),lane,q);
        if(used(p.returned(lane))==0){clear(p.product(lane));break;}
        require_exact(multiply(p.temporary(lane),p.product(lane),p.returned(lane)));copy(p.product(lane),p.temporary(lane));
    }
    copy(p.owner(q),p.product(lane));return true;
}
__device__ inline void resource_limit(CountView p){p.data[6]=4;p.data[5]=0;clear(p.global(0));}

// Entire history count, called cooperatively by one CTA. Status1 includes
// exact zero; status4 denotes a configured table or component resource limit.
// data[4]=processed components, [5]=source max_bag (0 on resource), [6]=status.
__device__ inline void count_history(HistoryView h,StructureView s,CountView p,U64 max_entries,U32 max_component_events=0){
    for(U64 i=threadIdx.x;i<p.l.program_words;i+=blockDim.x)p.data[i]=0;
    __syncthreads();
    if(threadIdx.x==0){p.data[6]=1;clear(p.global(0));}
    __syncthreads();require_exact(prepare_structure(h,s));if(s.meta[2])return;
    if(threadIdx.x==0){set_small(p.global(0),1);
        for(U32 f=0;f<s.meta[1];++f){bool empty=true;for(U32 w=0;w<s.l.ew;++w)empty&=s.scope(f,w)==0;
            if(empty)require_exact(multiply_small(p.global(0),p.global(0),s.terms(f)));}
        for(U32 factor=2;factor<=s.meta[0];++factor)require_exact(multiply_small(p.global(0),p.global(0),factor));
    }
    __syncthreads();
    for(U32 component=0;component<s.meta[3];++component){
        if(threadIdx.x==0&&max_component_events&&
           s.meta[s.l.offsets+component+1]-s.meta[s.l.offsets+component]>max_component_events)resource_limit(p);
        __syncthreads();if(p.data[6]==4)return;
        if(threadIdx.x==0)++p.data[4];__syncthreads();require_exact(order_component(h,s,component,max_entries));
        if(threadIdx.x==0){if(s.meta[s.l.bag+component]>p.data[5])p.data[5]=s.meta[s.l.bag+component];if(!s.meta[s.l.budget+component])resource_limit(p);}
        __syncthreads();if(p.data[6]==4)return;
        if(threadIdx.x==0)build_expression(s,p,component);__syncthreads();
        const U32 m=s.meta[s.l.offsets+component+1]-s.meta[s.l.offsets+component];
        if(threadIdx.x<p.l.lanes){const U32 lane=threadIdx.x;p.data[p.l.failed+lane]=0;
            for(U32 q=lane+1;q<=m;q+=p.l.lanes)if(!lattice_sum(s,p,lane,q,max_entries)){p.data[p.l.failed+lane]=1;break;}}
        __syncthreads();
        if(threadIdx.x==0){bool failed=false;for(U32 lane=0;lane<p.l.lanes;++lane)failed|=p.data[p.l.failed+lane]!=0;
            if(failed)resource_limit(p);
            else{clear(p.global(1));set_small(p.global(2),1);int sign=0;U32 remainder=0;
                for(U32 q=1;q<=m;++q){require_exact(multiply_small(p.global(2),p.global(2),m-q+1));require_exact(divide_small(p.global(2),p.global(2),q,remainder)&&remainder==0);
                    require_exact(multiply(p.global(3),p.global(2),p.owner(q)));
                    require_exact(signed_add(p.global(1),sign,p.global(1),sign,p.global(3),(m-q)%2?-1:1));}
                require_exact(sign>=0);p.data[7]=sign!=0;
                if(!sign)clear(p.global(0));else{require_exact(multiply(p.global(4),p.global(0),p.global(1)));copy(p.global(0),p.global(4));}
            }
        }
        __syncthreads();if(p.data[6]==4||!p.data[7])return;
    }
    if(threadIdx.x==0){U32 remainder=0;
        for(U32 component=0;component<s.meta[3];++component){const U32 m=s.meta[s.l.offsets+component+1]-s.meta[s.l.offsets+component];
            for(U32 d=2;d<=m;++d)require_exact(divide_small(p.global(0),p.global(0),d,remainder)&&remainder==0);}
    }
    __syncthreads();
}
}
