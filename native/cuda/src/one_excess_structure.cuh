#pragma once
#include "one_excess_math.cuh"

namespace cnot103_one_excess {
using U32=std::uint32_t;
using U64=std::uint64_t;

// Static shape metadata only. Histories use [position,word,physical_row].
// Terms use [field,term,factor], scopes/graphs use [word,factor_or_event].
// Each lane owns a factor/event; no pointer graph or per-state host decision.
struct StructureLayout {
    U32 n,w,k,f,t,ew;
    U64 term,ids,vertices,offsets,order,bag,budget,protected_flags,meta_count,bit_count,scratch_count;
    __host__ __device__ explicit StructureLayout(U32 width):n(width),w((width+63)/64),k(width+1),f(width+2),t(3*width),ew((k+63)/64) {
        term=8+U64(3)*k+f; ids=term+U64(5)*t*f; vertices=ids+k; offsets=vertices+k;
        order=offsets+k+1; bag=order+k; budget=bag+k; protected_flags=budget+k;
        meta_count=protected_flags+k;bit_count=U64(f+2*k)*ew;
        scratch_count=U64(n)*w+U64(k)*ew+7*U64(k)+16;
    }
};
struct HistoryView {
    const U64* data;
    const U32* lengths;
    const std::uint8_t* protected_rows;
    U32 n,w;
    __device__ U64 value(U32 r,U32 p,U32 word)const{return data[(U64(p)*w+word)*n+r];}
};
struct StructureView {
    StructureLayout l;U32* meta;U64* bits;U64* scratch;
    __device__ U32& row(U32 e)const{return meta[8+e];}
    __device__ U32& transition(U32 e)const{return meta[8+l.k+e];}
    __device__ U32& row_offset(U32 r)const{return meta[8+2*U64(l.k)+r];}
    __device__ U32& terms(U32 f)const{return meta[8+3*U64(l.k)+f];}
    __device__ U32& term(U32 f,U32 t,U32 field)const{return meta[l.term+(U64(field)*l.t+t)*l.f+f];}
    __device__ U64& scope(U32 f,U32 w)const{return bits[U64(w)*l.f+f];}
    __device__ U64& graph(U32 e,U32 w)const{return bits[U64(l.f)*l.ew+U64(w)*l.k+e];}
    __device__ U64& mandatory(U32 e,U32 w)const{return bits[U64(l.f+l.k)*l.ew+U64(w)*l.k+e];}
    __device__ U64& rank(U32 r,U32 w)const{return scratch[U64(w)*l.n+r];}
    __device__ U64& adjacency(U32 e,U32 w)const{return scratch[U64(l.n)*l.w+U64(w)*l.k+e];}
    __device__ U64& work(U32 array,U32 e)const{return scratch[U64(l.n)*l.w+U64(l.k)*l.ew+U64(array)*l.k+e];}
    __device__ U64& scalar(U32 i)const{return scratch[U64(l.n)*l.w+U64(l.k)*l.ew+7*U64(l.k)+i];}
    __device__ bool protected_event(const HistoryView& h,U32 e)const{return h.protected_rows[row(e)]||h.lengths[row(e)]==3;}
};

__device__ inline bool checked_power(U64 base,U32 exponent,U64 limit,U64& value) {
    value=1;
    for(U32 i=0;i<exponent;++i){if(base && value>~U64(0)/base)return false;value*=base;if(value>limit)return false;}
    return true;
}
__device__ inline void add_constraint(StructureView s,U32 f,U32 t,U32 p,U32 before,U32 after) {
    s.term(f,t,1+2*p)=before;s.term(f,t,2+2*p)=after;
    s.scope(f,before/64)|=U64(1)<<(before%64);s.scope(f,after/64)|=U64(1)<<(after%64);
}

// Cooperative CTA validation. Both endpoints must be invertible, every active
// history row nonzero, adjacent values distinct, packed padding zero. This
// workspace represents <=one double-update row, including recovery prefixes.
__device__ inline bool validate_histories(HistoryView h,StructureView s) {
    for(U64 i=threadIdx.x;i<s.l.meta_count;i+=blockDim.x)s.meta[i]=0;
    for(U64 i=threadIdx.x;i<s.l.bit_count;i+=blockDim.x)s.bits[i]=0;
    for(U64 i=threadIdx.x;i<s.l.scratch_count;i+=blockDim.x)s.scratch[i]=0;
    __syncthreads();
    for(U32 r=threadIdx.x;r<h.n;r+=blockDim.x){
        U32 len=h.lengths[r];
        if(len<1||len>3||h.protected_rows[r]>1){atomicOr(s.meta+7,1U);continue;}
        for(U32 p=0;p<3;++p){bool nonzero=false,different=false;
            for(U32 w=0;w<h.w;++w){const U64 v=h.value(r,p,w);nonzero|=v!=0;
                if(p<len && p>0)different|=v!=h.value(r,p-1,w);
                if((p>=len && v)||(w==h.w-1 && h.n%64 && (v>>(h.n%64))))atomicOr(s.meta+7,1U);
            }
            if(p<len && (!nonzero||(p>0&&!different)))atomicOr(s.meta+7,1U);
        }
    }
    __syncthreads();if(s.meta[7])return false;
    if(threadIdx.x==0){U32 e=0,double_row=h.n,protected_count=0;
        for(U32 r=0;r<h.n;++r){s.row_offset(r)=e;
            if(h.lengths[r]==3){if(double_row!=h.n)s.meta[7]=1;double_row=r;}
            for(U32 p=0;p<h.lengths[r]-1;++p){if(e>=s.l.k){s.meta[7]=1;break;}
                s.row(e)=r;s.transition(e)=p;protected_count+=h.protected_rows[r]||h.lengths[r]==3;++e;
            }
        }
        s.row_offset(h.n)=e;s.meta[0]=e;s.meta[4]=double_row;s.meta[5]=protected_count;
    }
    __syncthreads();if(s.meta[7])return false;
    for(U32 endpoint=0;endpoint<2;++endpoint){
        for(U32 r=threadIdx.x;r<h.n;r+=blockDim.x)for(U32 w=0;w<h.w;++w)s.rank(r,w)=h.value(r,endpoint?h.lengths[r]-1:0,w);
        __syncthreads();
        for(U32 column=0;column<h.n;++column){
            if(threadIdx.x==0){U32 pivot=column;while(pivot<h.n && !(s.rank(pivot,column/64)>>(column%64)&1))++pivot;
                s.scalar(1)=pivot;if(pivot==h.n)s.meta[7]=1;}
            __syncthreads();if(s.meta[7])return false;
            const U32 pivot=static_cast<U32>(s.scalar(1));
            if(pivot!=column)for(U32 w=threadIdx.x;w<h.w;w+=blockDim.x){const U64 v=s.rank(column,w);s.rank(column,w)=s.rank(pivot,w);s.rank(pivot,w)=v;}
            __syncthreads();
            for(U32 r=threadIdx.x;r<h.n;r+=blockDim.x)if(r!=column && (s.rank(r,column/64)>>(column%64)&1))
                for(U32 w=0;w<h.w;++w)s.rank(r,w)^=s.rank(column,w);
            __syncthreads();
        }
    }
    for(U32 e=threadIdx.x;e<s.l.k;e+=blockDim.x){s.meta[s.l.ids+e]=s.l.k;s.meta[s.l.vertices+e]=s.l.k;s.meta[s.l.order+e]=s.l.k;}
    if(threadIdx.x==0)s.meta[6]=1;
    __syncthreads();return true;
}

__device__ inline void factorize_histories(HistoryView h,StructureView s) {
    const U32 chain=s.meta[4]<h.n,raw_count=s.meta[0]+chain;
    for(U32 f=threadIdx.x;f<raw_count;f+=blockDim.x){
        if(chain && f==0){const U32 e=s.row_offset(s.meta[4]);s.terms(0)=1;s.term(0,0,0)=1;add_constraint(s,0,0,0,e,e+1);continue;}
        const U32 e=f-chain,r=s.row(e),p=s.transition(e);U32 terms=0;
        for(U32 control=0;control<h.n;++control)if(control!=r)for(U32 position=0;position<h.lengths[control];++position){
            bool equal=true;for(U32 w=0;w<h.w;++w)equal&=(h.value(r,p,w)^h.value(r,p+1,w))==h.value(control,position,w);
            if(!equal)continue;
            U32 count=0;
            if(position>0)add_constraint(s,f,terms,count++,s.row_offset(control)+position-1,e);
            if(position+1<h.lengths[control])add_constraint(s,f,terms,count++,e,s.row_offset(control)+position);
            s.term(f,terms,0)=count;
            if(count==2){U32 a=s.term(f,terms,1),b=s.term(f,terms,2),c=s.term(f,terms,3),d=s.term(f,terms,4);
                if(a>c||(a==c&&b>d)){s.term(f,terms,1)=c;s.term(f,terms,2)=d;s.term(f,terms,3)=a;s.term(f,terms,4)=b;}}
            ++terms;
        }
        s.terms(f)=terms;
    }
    __syncthreads();
    if(threadIdx.x==0){U32 kept=0;
        for(U32 f=0;f<raw_count;++f){bool duplicate=false;
            if(s.terms(f)==1)for(U32 previous=0;previous<kept;++previous)if(s.terms(previous)==1){bool same=true;
                for(U32 field=0;field<5;++field)same&=s.term(f,0,field)==s.term(previous,0,field);duplicate|=same;}
            if(duplicate)continue;
            if(kept!=f){s.terms(kept)=s.terms(f);for(U32 field=0;field<5;++field)for(U32 t=0;t<s.l.t;++t)s.term(kept,t,field)=s.term(f,t,field);
                for(U32 w=0;w<s.l.ew;++w)s.scope(kept,w)=s.scope(f,w);}
            if(s.terms(kept)==0)s.meta[2]=1;
            ++kept;
        }
        s.meta[1]=kept;
    }
    __syncthreads();
    for(U32 f=s.meta[1]+threadIdx.x;f<s.l.f;f+=blockDim.x){s.terms(f)=0;
        for(U32 field=0;field<5;++field)for(U32 t=0;t<s.l.t;++t)s.term(f,t,field)=0;
        for(U32 w=0;w<s.l.ew;++w)s.scope(f,w)=0;}
    __syncthreads();
}

__device__ inline void build_graphs(StructureView s) {
    for(U32 e=threadIdx.x;e<s.meta[0];e+=blockDim.x){
        for(U32 f=0;f<s.meta[1];++f){
            if(s.scope(f,e/64)>>(e%64)&1)for(U32 w=0;w<s.l.ew;++w)s.graph(e,w)|=s.scope(f,w);
            for(U32 pair=0;pair<s.term(f,0,0);++pair){const U32 before=s.term(f,0,1+2*pair),after=s.term(f,0,2+2*pair);
                if(before!=e)continue;bool common=true;
                for(U32 t=1;t<s.terms(f);++t){bool found=false;for(U32 p=0;p<s.term(f,t,0);++p)
                    found|=s.term(f,t,1+2*p)==before&&s.term(f,t,2+2*p)==after;common&=found;}
                if(common)s.mandatory(e,after/64)|=U64(1)<<(after%64);
            }
        }
        s.graph(e,e/64)&=~(U64(1)<<(e%64));
    }
    __syncthreads();
    if(threadIdx.x==0){const U32 count=s.meta[0];U32 ready=0,visited=0;
        for(U32 e=0;e<count;++e){U64 indegree=0;for(U32 u=0;u<count;++u)indegree+=s.mandatory(u,e/64)>>(e%64)&1;
            s.work(5,e)=indegree;if(!indegree)s.work(6,ready++)=e;}
        while(ready){U32 e=static_cast<U32>(s.work(6,--ready));++visited;
            for(U32 u=0;u<count;++u)if(s.mandatory(e,u/64)>>(u%64)&1)if(--s.work(5,u)==0)s.work(6,ready++)=u;}
        if(visited!=count)s.meta[2]=2;
    }
    __syncthreads();
}

// Prepare global zero reasons and component partition only. Ordering and
// budget preflights are deliberately separate: a zero leading coefficient in
// an earlier component must short-circuit later component budget failures.
__device__ inline bool prepare_structure(HistoryView h,StructureView s) {
    if(!validate_histories(h,s))return false;
    factorize_histories(h,s);if(s.meta[2])return true;
    build_graphs(s);if(s.meta[2])return true;
    if(threadIdx.x==0){U32 cc=0,offset=0;
        for(U32 start=0;start<s.meta[0];++start)if(!s.work(3,start)){
            U32 head=0,tail=1;s.work(6,0)=start;s.work(3,start)=1;
            while(head<tail){U32 e=static_cast<U32>(s.work(6,head++));s.meta[s.l.ids+e]=cc;
                for(U32 u=0;u<s.meta[0];++u)if((s.graph(e,u/64)>>(u%64)&1)&&!s.work(3,u)){s.work(3,u)=1;s.work(6,tail++)=u;}}
            s.meta[s.l.offsets+cc]=offset;
            for(U32 e=0;e<s.meta[0];++e)if(s.meta[s.l.ids+e]==cc)s.meta[s.l.vertices+offset++]=e;
            s.meta[s.l.offsets+(++cc)]=offset;
        }
        s.meta[3]=cc;
    }
    __syncthreads();return true;
}

__device__ inline bool order_component(HistoryView h,StructureView s,U32 component,U64 max_entries) {
    const U32 start=s.meta[s.l.offsets+component],size=s.meta[s.l.offsets+component+1]-start;
    for(U32 e=threadIdx.x;e<s.l.k;e+=blockDim.x){s.work(3,e)=0;s.work(4,e)=s.meta[s.l.ids+e]==component;
        for(U32 w=0;w<s.l.ew;++w)s.adjacency(e,w)=s.graph(e,w);}
    __syncthreads();
    if(threadIdx.x==0){bool pseudo=true;
        for(U32 root=0;root<s.meta[0];++root)if(s.work(4,root)&&!s.protected_event(h,root)&&!s.work(3,root)){
            U32 head=0,tail=1;s.work(6,0)=root;s.work(3,root)=1;U64 degree=0;
            while(head<tail){U32 e=static_cast<U32>(s.work(6,head++));
                for(U32 u=0;u<s.meta[0];++u)if(s.work(4,u)&&!s.protected_event(h,u)&&(s.graph(e,u/64)>>(u%64)&1)){
                    ++degree;if(!s.work(3,u)){s.work(3,u)=1;s.work(6,tail++)=u;}}}
            if(degree/2>tail)pseudo=false;
        }
        s.meta[s.l.protected_flags+component]=s.meta[5]!=0&&pseudo;
    }
    __syncthreads();
    for(U32 step=0;step<size;++step){
        for(U32 e=threadIdx.x;e<s.meta[0];e+=blockDim.x)if(s.work(4,e)){
            U64 degree=0,unprotected=0,twice_fill=0;
            for(U32 w=0;w<s.l.ew;++w){U64 neighbors=s.adjacency(e,w);degree+=__popcll(neighbors);
                while(neighbors){U32 u=w*64+__ffsll(neighbors)-1;neighbors&=neighbors-1;
                    unprotected+=!s.protected_event(h,u);
                    for(U32 q=0;q<s.l.ew;++q){U64 absent=s.adjacency(e,q)&~s.adjacency(u,q);
                        if(u/64==q)absent&=~(U64(1)<<(u%64));twice_fill+=__popcll(absent);}
                }
            }
            s.work(0,e)=twice_fill/2;s.work(1,e)=degree;s.work(2,e)=unprotected;
        }
        __syncthreads();
        if(threadIdx.x==0){bool pool=false;
            if(s.meta[s.l.protected_flags+component])for(U32 e=0;e<s.meta[0];++e)pool|=s.work(4,e)&&!s.protected_event(h,e);
            U32 selected=s.l.k;
            for(U32 e=0;e<s.meta[0];++e)if(s.work(4,e)&&(!pool||!s.protected_event(h,e))){
                bool less=selected==s.l.k;
                if(!less){U64 a=pool?s.work(2,e):0,b=pool?s.work(2,selected):0;
                    less=a<b||(a==b&&(s.work(0,e)<s.work(0,selected)||(s.work(0,e)==s.work(0,selected)&&s.work(1,e)<s.work(1,selected))));}
                if(less)selected=e;
            }
            s.scalar(2)=selected;
            if(selected==s.l.k||(pool&&s.work(2,selected)>2))s.meta[7]=1;
            else{const U32 bag=static_cast<U32>(s.work(1,selected)+1);if(bag>s.meta[s.l.bag+component])s.meta[s.l.bag+component]=bag;
                s.meta[s.l.order+start+step]=selected;}
        }
        __syncthreads();if(s.meta[7])return false;
        const U32 selected=static_cast<U32>(s.scalar(2));
        for(U32 e=threadIdx.x;e<s.meta[0];e+=blockDim.x)if(e!=selected&&s.work(4,e)){
            const bool neighbor=s.adjacency(e,selected/64)>>(selected%64)&1;
            if(neighbor)for(U32 w=0;w<s.l.ew;++w)s.adjacency(e,w)|=s.adjacency(selected,w);
            s.adjacency(e,e/64)&=~(U64(1)<<(e%64));s.adjacency(e,selected/64)&=~(U64(1)<<(selected%64));
        }
        __syncthreads();if(threadIdx.x==0)s.work(4,selected)=0;__syncthreads();
    }
    if(threadIdx.x==0){U64 entries=0;s.meta[s.l.budget+component]=checked_power(size,s.meta[s.l.bag+component],max_entries,entries);}
    __syncthreads();return true;
}
}
