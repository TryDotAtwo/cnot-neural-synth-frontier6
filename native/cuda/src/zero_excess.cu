#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
namespace {
using U32=std::uint32_t;using U64=std::uint64_t;
__device__ U64 unit(U32 row,U32 word){return row/64==word?U64{1}<<(row%64):0;}
__device__ bool equal_difference(const U64* row,const U64* control,U32 target,U32 w){
    for(U32 j=0;j<w;++j)if((row[j]^unit(target,j))!=control[j])return false;return true;
}
__device__ bool edge(const U32* parents,const U32* changed,U32 source,U32 destination){
    return changed[source]&&changed[destination]&&((parents[source*2+1]==0&&parents[source*2]==destination)||(parents[destination*2+1]==1&&parents[destination*2]==source));
}
__global__ void construct(Cnot103ZeroExcessArgs a){
    const U64 b=blockIdx.x;const U32 n=a.dimension,w=a.words_per_row;
    const U64* m=a.states+b*U64(n)*w;U64* header=a.headers+b*3;
    U32* parents=a.parents+b*U64(n)*2;U32* changed=a.changed_rows+b*n;U32* missing=a.missing_rows+b*n;U32* order=a.update_order+b*n;
    U32* gates=a.gates+b*U64(n)*2;U32* flags=a.graph+b*U64(n)*3;U32* indegree=flags+n;U32* queue=indegree+n;
    for(U32 row=threadIdx.x;row<n;row+=blockDim.x){
        changed[row]=missing[row]=order[row]=n;gates[row*2]=gates[row*2+1]=0;parents[row*2]=n;parents[row*2+1]=2;indegree[row]=0;
        const U64* value=m+U64(row)*w;bool same=true;for(U32 j=0;j<w;++j)same&=value[j]==unit(row,j);flags[row]=!same;
        if(same)continue;
        U32 control=n,mode=2;
        for(U32 c=0;c<n;++c)if(c!=row&&equal_difference(value,m+U64(c)*w,row,w)){control=c;mode=0;break;}
        if(control==n){
            U32 ones=0,coordinate=n;
            for(U32 j=0;j<w;++j){const U64 v=value[j]^unit(row,j);ones+=__popcll(v);if(v)coordinate=j*64+__ffsll(v)-1;}
            if(ones==1&&coordinate<n&&coordinate!=row){control=coordinate;mode=1;}
        }
        parents[row*2]=control;parents[row*2+1]=mode;
    }
    if(!threadIdx.x){for(U32 j=0;j<3;++j){header[j]=0;a.feature[b*3+j]=0;}a.status[b]=0;a.verified[b]=0;a.lengths[b]=0;}
    __syncthreads();if(threadIdx.x)return;
    // Packed MatrixState has no bits beyond n. Reject malformed raw ABI input
    // before it can masquerade as a genuine missing-parent witness.
    if(n%64)for(U32 row=0;row<n;++row)if(m[U64(row)*w+w-1]>>(n%64))return;
    U32 count=0,absent=0;
    for(U32 row=0;row<n;++row)if(flags[row]){changed[count++]=row;if(parents[row*2]==n)missing[absent++]=row;}
    header[1]=count;header[2]=absent;
    // Missing parents precede all graph reasoning in the source.
    if(absent){header[0]=2;a.status[b]=1;return;}
    for(U32 src=0;src<n;++src)for(U32 dst=0;dst<n;++dst)if(edge(parents,flags,src,dst))++indegree[dst];
    U32 head=0,tail=0;
    for(U32 row=0;row<n;++row)if(flags[row]&&!indegree[row])queue[tail++]=row;
    while(head<tail){
        const U32 row=queue[head];order[head]=row;const U32 position=count-1-head;
        gates[position*2]=parents[row*2];gates[position*2+1]=row;++head;
        for(U32 dst=0;dst<n;++dst)if(edge(parents,flags,row,dst)&&!--indegree[dst])queue[tail++]=dst;
    }
    if(head!=count)return; // integrity failure; checker/publisher trap
    a.lengths[b]=count;header[0]=1;a.status[b]=1;
}
// Independent readiness checker: no indegrees, edge counters or source FIFO
// scratch are trusted. It discovers newly ready destinations in physical order
// and verifies their exact append positions in the supplied FIFO sequence.
__device__ bool ready(U32 node,U32 n,const U32* parents,const U32* changed,const U32* processed){
    if(parents[node*2+1]==1){const U32 c=parents[node*2];if(changed[c]&&!processed[c])return false;}
    for(U32 row=0;row<n;++row)if(changed[row]&&parents[row*2+1]==0&&parents[row*2]==node&&!processed[row])return false;
    return true;
}
__global__ void check(Cnot103ZeroExcessArgs a){
    const U64 b=blockIdx.x;const U32 n=a.dimension,w=a.words_per_row;const bool word_ok=a.verified[b]==1;a.verified[b]=0;
    if(a.status[b]!=1)return;
    const U64* m=a.states+b*U64(n)*w;const U64* header=a.headers+b*3;
    if(n%64)for(U32 row=0;row<n;++row)if(m[U64(row)*w+w-1]&(~U64{0}<<(n%64)))return;
    const U32* parents=a.parents+b*U64(n)*2;const U32* changed=a.changed_rows+b*n;const U32* missing=a.missing_rows+b*n;
    const U32* order=a.update_order+b*n;const U32* gates=a.gates+b*U64(n)*2;
    U32* flags=a.graph+b*U64(n)*3;U32* queued=flags+n;U32* processed=queued+n;
    U32 count=0,absent=0;
    for(U32 row=0;row<n;++row){
        U32 weight=0;for(U32 j=0;j<w;++j)weight+=__popcll(m[U64(row)*w+j]);
        const bool diagonal=(m[U64(row)*w+row/64]>>(row%64))&1;
        flags[row]=!(weight==1&&diagonal);queued[row]=processed[row]=0;
        U32 control=n,mode=2;
        if(flags[row]){
            if(changed[count++]!=row)return;
            // M[target]+M[control]=e_target, checked by Hamming weight and
            // the distinguished bit, rather than the constructor comparison.
            for(U32 candidate=0;candidate<n;++candidate)if(candidate!=row){
                U32 different=0;for(U32 j=0;j<w;++j)different+=__popcll(m[U64(row)*w+j]^m[U64(candidate)*w+j]);
                const bool target_bit=((m[U64(row)*w+row/64]^m[U64(candidate)*w+row/64])>>(row%64))&1;
                if(different==1&&target_bit){control=candidate;mode=0;break;}
            }
            if(control==n&&diagonal&&weight==2){
                for(U32 col=0;col<n;++col)if(col!=row&&((m[U64(row)*w+col/64]>>(col%64))&1)){control=col;mode=1;break;}
            }
            if(control==n&&missing[absent++]!=row)return;
        }
        if(parents[row*2]!=control||parents[row*2+1]!=mode)return;
    }
    if(header[1]!=count||header[2]!=absent||header[0]!=(absent?2:1))return;
    for(U32 i=count;i<n;++i)if(changed[i]!=n)return;
    for(U32 i=absent;i<n;++i)if(missing[i]!=n)return;
    if(absent){
        if(a.lengths[b]!=0)return;
        for(U32 i=0;i<n;++i)if(order[i]!=n||gates[i*2]||gates[i*2+1])return;
        a.verified[b]=1;return;
    }
    if(a.lengths[b]!=count||!word_ok)return;
    U32 head=0,tail=0;
    for(U32 row=0;row<n;++row)if(flags[row]&&ready(row,n,parents,flags,processed)){
        if(tail>=count||order[tail++]!=row)return;queued[row]=1;
    }
    while(head<tail){
        const U32 row=order[head];if(row>=n||processed[row])return;processed[row]=1;
        const U32 position=count-1-head;if(gates[position*2]!=parents[row*2]||gates[position*2+1]!=row)return;++head;
        for(U32 dst=0;dst<n;++dst)if(flags[dst]&&!queued[dst]&&ready(dst,n,parents,flags,processed)){
            if(tail>=count||order[tail++]!=dst)return;queued[dst]=1;
        }
    }
    if(head!=count)return;
    for(U32 i=count;i<n;++i)if(order[i]!=n||gates[i*2]||gates[i*2+1])return;
    a.verified[b]=1;
}
__global__ void publish(Cnot103ZeroExcessArgs a){
    const U64 b=blockIdx.x;if(a.verified[b]!=1){asm volatile("trap;");return;}
    a.feature[b*3]=a.headers[b*3]==1;a.feature[b*3+1]=a.headers[b*3]==2;a.feature[b*3+2]=0;
}
bool extent(U64 b,U64 stride,U64 actual){return !stride||b<=actual/stride;}
int validate(const Cnot103ZeroExcessArgs* a){
    if(!a||!a->states||!a->scratch||!a->graph||!a->parents||!a->changed_rows||!a->missing_rows||!a->update_order||!a->gates||!a->headers||!a->lengths||!a->verified||!a->status||!a->feature)return 1;
    const U64 n=a->dimension,b=a->batch,w=a->words_per_row;
    if(!b||b>0x7fffffffU||n<2||n>(1U<<24)||w!=(n+63)/64)return 1;
    if(a->batch_elements<b||!extent(b,n*w,a->state_words)||!extent(b,n*w,a->scratch_words)||!extent(b,n*3,a->graph_elements)||
        !extent(b,n*2,a->parent_elements)||!extent(b,n,a->changed_elements)||!extent(b,n,a->missing_elements)||!extent(b,n,a->order_elements)||
        !extent(b,n*2,a->gate_elements)||!extent(b,3,a->header_elements)||!extent(b,3,a->feature_elements))return 2;
    return 0;
}
}
extern "C" std::int32_t cnot103_cuda_zero_excess_construct(const Cnot103ZeroExcessArgs* a){
    const int e=validate(a);if(e)return e;construct<<<a->batch,128,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" std::int32_t cnot103_cuda_zero_excess_check(const Cnot103ZeroExcessArgs* a){
    const int e=validate(a);if(e)return e;
    const int replay=cnot103_cuda_verify_cnot_words(a->states,a->gates,a->lengths,a->scratch,a->verified,a->batch,a->dimension,a->words_per_row,a->dimension,a->scratch_words,a->gate_elements,a->stream);
    if(replay)return replay;check<<<a->batch,1,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" std::int32_t cnot103_cuda_zero_excess_publish(const Cnot103ZeroExcessArgs* a){
    const int e=validate(a);if(e)return e;publish<<<a->batch,1,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" std::int32_t cnot103_cuda_launch_zero_excess(const Cnot103ZeroExcessArgs* a){
    int e=cnot103_cuda_zero_excess_construct(a);if(e)return e;e=cnot103_cuda_zero_excess_check(a);return e?e:cnot103_cuda_zero_excess_publish(a);
}
