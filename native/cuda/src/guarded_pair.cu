#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>

namespace {
using U64=std::uint64_t;using U32=std::uint32_t;
constexpr U32 certificate_size=16;
__device__ bool bit(const U64* m,U32 w,U32 r,U32 c){return (m[U64(r)*w+c/64]>>(c%64))&1;}
__device__ bool identity_row(const U64* m,U32 w,U32 row){
    for(U32 j=0;j<w;++j)if(m[U64(row)*w+j]!=(j==row/64?U64{1}<<(row%64):0))return false;return true;
}
__device__ U32 product(U32 f0,U32 f1,U32 v){return (__popc(f0&v)&1)|((__popc(f1&v)&1)<<1);}
__device__ U32 root(U64* parents,U32 v){while(parents[v]!=v){parents[v]=parents[parents[v]];v=parents[v];}return v;}
__device__ void emit(U32* gates,U64& count,U32 control,U32 target){gates[count*2]=control;gates[count*2+1]=target;++count;}
// Constructor uses the source inverse and exact fixed frame table.
__global__ void construct(Cnot103GuardedPairArgs a){
    const U64 b=blockIdx.x;const U32 n=a.dimension,w=a.words_per_row;
    U64* c=a.certificate+b*certificate_size;U32* gates=a.gates+b*a.word_capacity*2;
    for(U64 i=threadIdx.x;i<a.word_capacity*2;i+=blockDim.x)gates[i]=0;
    for(U32 i=threadIdx.x;i<certificate_size;i+=blockDim.x)c[i]=0;
    for(U64 i=threadIdx.x;i<U64(n)*3;i+=blockDim.x)a.incidence[b*n*3+i]=0;
    for(U32 i=threadIdx.x;i<6;i+=blockDim.x)a.f06[b*6+i]=0;
    if(threadIdx.x==0){a.f02[b]=0;a.valid[b]=0;a.lengths[b]=0;a.status[b]=0;a.verified[b]=0;}
    __syncthreads();if(threadIdx.x)return;
    const U64* m=a.states+b*U64(n)*w;U32 changed[2]={n,n},count=0;
    for(U32 r=0;r<n;++r)if(!identity_row(m,w,r)){if(count<2)changed[count]=r;++count;}
    if(count>2){a.status[b]=3;return;}
    U32 active[2]={changed[0],changed[1]},used=count;
    for(U32 r=0;used<2;++r)if(r!=active[0]&&r!=active[1])active[used++]=r;
    const U32 f0=bit(m,w,active[0],active[0])|(bit(m,w,active[0],active[1])<<1);
    const U32 f1=bit(m,w,active[1],active[0])|(bit(m,w,active[1],active[1])<<1);
    if(((f0&1)*((f1>>1)&1)^((f0>>1)&1)*(f1&1))!=1)return;
    const U32 inverse0=((f1>>1)&1)|(f0&2),inverse1=(f1&1)|((f0&1)<<1);
    U64* parents=a.graph+b*U64(n)*2;U64* transformed=parents+n;
    U32 nonzero=0,shared=0,shared_wires[2]={n,n};
    for(U32 r=0;r<n;++r){parents[r]=r;transformed[r]=0;
        if(r==active[0]||r==active[1])continue;
        const U32 value=bit(m,w,active[0],r)|(bit(m,w,active[1],r)<<1);
        transformed[r]=product(inverse0,inverse1,value);
        nonzero+=transformed[r]!=0;
        if(transformed[r]==3){if(shared<2)shared_wires[shared]=r;++shared;}
    }
    const bool identity=f0==1&&f1==2,motion=!identity||shared>2;
    U32 controls[3]={0,0,0},targets[3]={1,1,1},frame_count=0;
    if(motion){
        if(identity)frame_count=2;
        else if(f0==1&&f1==3)frame_count=1;
        else if(f0==3&&f1==2){frame_count=1;controls[0]=1;targets[0]=0;}
        else if(f0==2&&f1==3){frame_count=2;controls[1]=1;targets[1]=0;}
        else if(f0==3&&f1==1){frame_count=2;controls[0]=1;targets[0]=0;controls[1]=0;targets[1]=1;}
        else if(f0==2&&f1==1){frame_count=3;controls[1]=1;targets[1]=0;}
        else return;
    }
    U64 length=0;
    if(!motion){
        for(U32 col=0;col<n;++col)if(col!=active[0]&&col!=active[1])
            for(U32 r=0;r<2;++r)if(bit(m,w,active[r],col))emit(gates,length,col,active[r]);
    }else{
        for(U32 col=0;col<n;++col)if(transformed[col]==1||transformed[col]==2)emit(gates,length,col,active[transformed[col]==2]);
        emit(gates,length,active[controls[0]],active[targets[0]]);
        for(U32 col=0;col<n;++col)if(transformed[col]==3)emit(gates,length,col,active[controls[0]]);
        for(U32 i=1;i<frame_count;++i)emit(gates,length,active[controls[i]],active[targets[i]]);
    }
    const U64 formula=U64(nonzero)+(motion?frame_count:shared);
    if(length!=formula||length>a.word_capacity)return;
    U32 connectivity=0,zero_diagonal=0;
    for(U32 r=0;r<n;++r){zero_diagonal+=!bit(m,w,r,r);
        for(U32 col=0;col<n;++col)if(col!=r&&bit(m,w,r,col)){
            const U32 left=root(parents,r),right=root(parents,col);
            if(left!=right){parents[left]=right;++connectivity;}
        }
    }
    const U32 lower_kind=!identity?0:shared>=2?1:2;
    const U64 lower=U64(connectivity)+(lower_kind==0?zero_diagonal:lower_kind==1?1:0);
    if(lower!=length)return;
    const U64 metadata[16]={count,active[0],active[1],f0,f1,nonzero,shared,U64(motion),connectivity,zero_diagonal,lower,lower_kind,shared_wires[0],shared_wires[1],changed[0],changed[1]};
    for(U32 i=0;i<16;++i)c[i]=metadata[i];
    a.lengths[b]=length;a.status[b]=1;
}
// Checker solves columns by four-vector enumeration and support connectivity
// by BFS. Neither the constructor's inverse nor DSU/frame table is reused.
__device__ U32 solve(const U64* m,U32 w,const U32* active,U32 f0,U32 f1,U32 col){
    const U32 observed=bit(m,w,active[0],col)|(bit(m,w,active[1],col)<<1);
    for(U32 v=0;v<4;++v)if(product(f0,f1,v)==observed)return v;return 4;
}
__device__ bool matches(const U32* gates,U64 length,U64& cursor,U32 c,U32 t){
    if(cursor>=length||gates[cursor*2]!=c||gates[cursor*2+1]!=t)return false;++cursor;return true;
}
__global__ void check(Cnot103GuardedPairArgs a){
    const U64 b=blockIdx.x;const U32 n=a.dimension,w=a.words_per_row;
    const bool word_ok=a.verified[b]==1;a.verified[b]=0;
    const U64* m=a.states+b*U64(n)*w;const U64* cert=a.certificate+b*certificate_size;
    const U32* gates=a.gates+b*a.word_capacity*2;
    U32 changed[2]={n,n},count=0;
    for(U32 row=0;row<n;++row)if(!identity_row(m,w,row)){if(count<2)changed[count]=row;++count;}
    if(count>2){
        if(a.status[b]!=3||a.lengths[b]!=0)return;
        for(U32 i=0;i<16;++i)if(cert[i])return;
        for(U64 i=0;i<a.word_capacity*2;++i)if(gates[i])return;
        a.verified[b]=1;return;
    }
    if(a.status[b]!=1||!word_ok||a.lengths[b]<0||U64(a.lengths[b])>a.word_capacity)return;
    U32 active[2]={changed[0],changed[1]},used=count;
    for(U32 row=0;used<2;++row)if(row!=active[0]&&row!=active[1])active[used++]=row;
    U32 frame[2]={0,0};for(U32 r=0;r<2;++r)for(U32 col=0;col<2;++col)frame[r]|=U32(bit(m,w,active[r],active[col]))<<col;
    if(!frame[0]||!frame[1]||frame[0]==frame[1])return;
    U32 nonzero=0,shared=0,shared_wires[2]={n,n};
    for(U32 col=0;col<n;++col)if(col!=active[0]&&col!=active[1]){
        const U32 v=solve(m,w,active,frame[0],frame[1],col);if(v>3)return;
        nonzero+=v!=0;if(v==3){if(shared<2)shared_wires[shared]=col;++shared;}
    }
    U64* seen=a.graph+b*U64(n)*2;U64* queue=seen+n;for(U32 r=0;r<n;++r)seen[r]=0;
    U32 components=0,zero_diagonal=0;
    for(U32 r=0;r<n;++r){zero_diagonal+=!bit(m,w,r,r);if(seen[r])continue;
        ++components;U32 head=0,tail=0;queue[tail++]=r;seen[r]=1;
        while(head<tail){const U32 v=queue[head++];for(U32 q=0;q<n;++q)if(!seen[q]&&(bit(m,w,v,q)||bit(m,w,q,v))){seen[q]=1;queue[tail++]=q;}}
    }
    const bool identity=frame[0]==1&&frame[1]==2,motion=!identity||shared>2;
    U32 route=0,steps=0;bool found=false;
    if(identity&&motion){steps=2;found=true;}
    else for(U32 len=0;len<=3&&!found;++len)for(U32 path=0;path<(1U<<len)&&!found;++path){
        U32 f[2]={1,2};for(U32 i=0;i<len;++i){const U32 c=(path>>(len-i-1))&1;f[1-c]^=f[c];}
        if(f[0]==frame[0]&&f[1]==frame[1]){found=true;steps=len;route=path;}
    }
    if(!found)return;
    const U32 connectivity=n-components,lower_kind=!identity?0:shared>=2?1:2;
    const U64 lower=U64(connectivity)+(lower_kind==0?zero_diagonal:lower_kind==1?1:0);
    const U64 length=U64(nonzero)+(motion?steps:shared);
    if(U64(a.lengths[b])!=length||lower!=length)return;
    const U64 expected[16]={count,active[0],active[1],frame[0],frame[1],nonzero,shared,U64(motion),connectivity,zero_diagonal,lower,lower_kind,shared_wires[0],shared_wires[1],changed[0],changed[1]};
    for(U32 i=0;i<16;++i)if(cert[i]!=expected[i])return;
    U64 cursor=0;
    if(!motion){
        for(U32 col=0;col<n;++col)if(col!=active[0]&&col!=active[1])for(U32 r=0;r<2;++r)
            if(bit(m,w,active[r],col)&&!matches(gates,length,cursor,col,active[r]))return;
    }else{
        for(U32 col=0;col<n;++col)if(col!=active[0]&&col!=active[1]){
            const U32 v=solve(m,w,active,frame[0],frame[1],col);
            if((v==1||v==2)&&!matches(gates,length,cursor,col,active[v/2]))return;
        }
        const U32 first=(route>>(steps-1))&1;
        if(!matches(gates,length,cursor,active[first],active[1-first]))return;
        U32 intermediate[2]={1,2};intermediate[1-first]^=intermediate[first];
        const U32 unit=product(intermediate[0],intermediate[1],3);if(unit!=1&&unit!=2)return;
        for(U32 col=0;col<n;++col)if(col!=active[0]&&col!=active[1])
            if(solve(m,w,active,frame[0],frame[1],col)==3&&!matches(gates,length,cursor,col,active[unit/2]))return;
        for(U32 i=1;i<steps;++i){const U32 control=(route>>(steps-i-1))&1;if(!matches(gates,length,cursor,active[control],active[1-control]))return;}
    }
    if(cursor!=length)return;
    for(U64 i=length*2;i<a.word_capacity*2;++i)if(gates[i])return;
    a.verified[b]=1;
}
__global__ void publish(Cnot103GuardedPairArgs a){
    const U64 b=blockIdx.x;if(a.verified[b]!=1){if(!threadIdx.x)asm volatile("trap;");return;}
    if(a.status[b]!=1)return;const U64* c=a.certificate+b*certificate_size;
    for(U32 row=threadIdx.x;row<a.dimension;row+=blockDim.x){
        float* roles=a.incidence+(b*a.dimension+row)*3;
        roles[0]=row==c[1];roles[1]=row==c[2];roles[2]=row!=c[1]&&row!=c[2];
    }
    if(!threadIdx.x){
        a.f02[b]=__ull2float_rn(c[8]+c[9]);float* f=a.f06+b*6;
        f[0]=c[3]&1;f[1]=(c[3]>>1)&1;f[2]=c[4]&1;f[3]=(c[4]>>1)&1;f[4]=__ull2float_rn(c[5]);f[5]=__ull2float_rn(c[6]);a.valid[b]=1;
    }
}
bool extent(U64 b,U64 stride,U64 actual){return !stride||b<=actual/stride;}
int validate(const Cnot103GuardedPairArgs* a){
    if(!a||!a->states||!a->scratch||!a->graph||!a->gates||!a->lengths||!a->certificate||!a->status||!a->verified||!a->f02||!a->f06||!a->valid||!a->incidence)return 1;
    const U64 n=a->dimension,b=a->batch,w=a->words_per_row;
    if(!b||b>0x7fffffffU||n<2||n>(1U<<24)||w!=(n+63)/64)return 1;
    if(a->word_capacity<n+1||a->word_capacity>~U64{0}/2||a->batch_elements<b||
        !extent(b,n*w,a->state_words)||!extent(b,n*w,a->scratch_words)||!extent(b,n*2,a->graph_words)||
        !extent(b,a->word_capacity*2,a->gate_elements)||!extent(b,16,a->certificate_words)||!extent(b,6,a->f06_elements)||!extent(b,n*3,a->incidence_elements))return 2;
    return 0;
}
}
extern "C" std::int32_t cnot103_cuda_guarded_pair_construct(const Cnot103GuardedPairArgs* a){
    const int e=validate(a);if(e)return e;construct<<<a->batch,128,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" std::int32_t cnot103_cuda_guarded_pair_check(const Cnot103GuardedPairArgs* a){
    const int e=validate(a);if(e)return e;
    const int replay=cnot103_cuda_verify_cnot_words(a->states,a->gates,a->lengths,a->scratch,a->verified,a->batch,a->dimension,a->words_per_row,a->word_capacity,a->scratch_words,a->gate_elements,a->stream);
    if(replay)return replay;check<<<a->batch,1,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" std::int32_t cnot103_cuda_guarded_pair_publish(const Cnot103GuardedPairArgs* a){
    const int e=validate(a);if(e)return e;publish<<<a->batch,128,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" std::int32_t cnot103_cuda_launch_guarded_pair(const Cnot103GuardedPairArgs* a){
    int e=cnot103_cuda_guarded_pair_construct(a);if(e)return e;e=cnot103_cuda_guarded_pair_check(a);return e?e:cnot103_cuda_guarded_pair_publish(a);
}
