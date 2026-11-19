#include "cnot103/cnot_cuda.h"
#include "template_response_math.cuh"

namespace {
using namespace cnot103_template;using I64=std::int64_t;
constexpr U32 lanes=128;
__device__ void clear(const Cnot103TemplateResponseArgs& a,U64 b){
    const U64 groups=U64(a.choices)*(a.choices-1)/2,assignments=1U<<a.choices;
    for(U64 i=threadIdx.x;i<assignments;i+=blockDim.x)a.costs[b*assignments+i]=0;
    for(U64 i=threadIdx.x;i<U64(a.points)+2;i+=blockDim.x)a.rounding[b*(U64(a.points)+2)+i]=0;
    for(U64 i=threadIdx.x;i<U64(a.points)+9;i+=blockDim.x)a.globals[b*(U64(a.points)+9)+i]=0;
    for(U64 i=threadIdx.x;i<groups;i+=blockDim.x){a.mixed[b*groups+i]=0;a.valid[b*groups+i]=0;}
    for(U64 i=threadIdx.x;i<groups*a.dimension*4;i+=blockDim.x)a.incidence[b*groups*a.dimension*4+i]=0;
}
__device__ bool outputs(const Cnot103TemplateResponseArgs& a,U64 b){
    const U32 k=a.choices,assignments=1U<<k;const U64 groups=U64(k)*(k-1)/2;
    const U64* costs=a.costs+b*assignments;float* globals=a.globals+b*(U64(a.points)+9);U64* rounded=a.rounding+b*(U64(a.points)+2);
    Probability midpoint[8],point[8];for(U32 i=0;i<k;++i)midpoint[i]={1,2};Wide numerator{},denominator{};U32 bits;
    if(!expectation(costs,k,midpoint,numerator,denominator)||!rational_bits(numerator,denominator,bits))return false;globals[0]=__uint_as_float(bits);
    for(U32 p=0;p<a.points;++p){
        for(U32 i=0;i<k;++i)point[i]={a.probabilities[(U64(p)*k+i)*2],a.probabilities[(U64(p)*k+i)*2+1]};
        if(!expectation(costs,k,point,numerator,denominator)||!rational_bits(numerator,denominator,bits))return false;globals[1+p]=__uint_as_float(bits);
    }
    U32 group=0;
    for(U32 first=0;first<k;++first)for(U32 second=first+1;second<k;++second){
        // n<=2^24, k<=8: each cost <=n^2/4+48; even the sum of
        // 64 four-term differences has magnitude <2^55, so int64 is exact.
        I64 difference=0;
        for(U32 mask=0;mask<assignments;++mask)if(!(mask&((1U<<first)|(1U<<second))))
            difference+=I64(costs[mask|(1U<<first)|(1U<<second)])-I64(costs[mask|(1U<<first)])-I64(costs[mask|(1U<<second)])+I64(costs[mask]);
        if(!rational_bits({U64(difference<0?-difference:difference),0},{U64(assignments/4),0},bits))return false;
        if(difference<0)bits|=0x80000000U;a.mixed[b*groups+group++]=__uint_as_float(bits);
    }
    U32 forward,reverse;
    if(!rounding_assignment(costs,k,midpoint,false,forward)||!rounding_assignment(costs,k,midpoint,true,reverse))return false;
    rounded[0]=forward;rounded[1]=reverse;const U64 f=costs[forward],r=costs[reverse];U64 best=costs[0],min_start=f,max_start=f;
    for(U32 mask=1;mask<assignments;++mask)if(costs[mask]<best)best=costs[mask];
    for(U32 p=0;p<a.points;++p){
        for(U32 i=0;i<k;++i)point[i]={a.probabilities[(U64(p)*k+i)*2],a.probabilities[(U64(p)*k+i)*2+1]};
        U32 selected;if(!rounding_assignment(costs,k,point,false,selected))return false;rounded[2+p]=selected;
        const U64 c=costs[selected];if(c<min_start)min_start=c;if(c>max_start)max_start=c;
    }
    const U64 offset=U64(a.points)+1;
    globals[offset]=__ull2float_rn(f);globals[offset+1]=__ull2float_rn(best);globals[offset+2]=1;
    globals[offset+3]=__ull2float_rn(f);globals[offset+4]=__ull2float_rn(r);globals[offset+5]=__ull2float_rn(f>r?f-r:r-f);globals[offset+6]=__ull2float_rn(min_start);globals[offset+7]=__ull2float_rn(max_start);
    return true;
}
__global__ void evaluate(Cnot103TemplateResponseArgs a){
    const U64 b=blockIdx.x,width=a.dimension/2,targets=a.dimension-width,cells=width*targets;
    __shared__ U32 run;__shared__ unsigned long long direct;
    clear(a,b);
    if(!threadIdx.x){
        a.status[b]=a.raw_status[b]==0?2:a.raw_status[b]!=1||a.invalid_choices?3:1;
        if(a.status[b]==1){
            for(U32 i=0;i<a.unique_wires;++i)if(a.wires[i]>=width)a.status[b]=3;
            for(U32 i=0;i<a.choices;++i)if(a.choice_data[i*3]>1||a.choice_data[i*3+1]>=a.unique_wires||a.choice_data[i*3+2]>=a.unique_wires||a.choice_data[i*3+1]==a.choice_data[i*3+2])a.status[b]=3;
            for(U64 i=0;i<U64(a.points)*a.choices;++i)if(!a.probabilities[i*2+1]||a.probabilities[i*2]>a.probabilities[i*2+1])a.status[b]=3;
        }
        run=a.status[b]==1;direct=0;
    }
    __syncthreads();if(!run)return;
    const auto* block=a.coefficients+b*cells;U64 local=0;
    for(U64 i=threadIdx.x;i<cells;i+=blockDim.x)local+=block[i];
    atomicAdd(&direct,static_cast<unsigned long long>(local));__syncthreads();
    const U32 assignments=1U<<a.choices;
    for(U32 mask=threadIdx.x;mask<assignments;mask+=blockDim.x){
        U64 preparation=0;I64 change=0;
        for(U32 i=0;i<a.choices;++i)if((mask>>i)&1)preparation+=a.choice_data[i*3]?6:2;
        for(U32 row=0;row<targets;++row){U32 code=0;
            for(U32 i=0;i<a.unique_wires;++i)code|=U32(block[U64(row)*width+a.wires[i]])<<i;
            const U32 before=__popc(code);
            for(U32 i=0;i<a.choices;++i)if((mask>>i)&1){const U32 c=a.choice_data[i*3+1],t=a.choice_data[i*3+2];
                if(a.choice_data[i*3]==0)code^=((code>>t)&1)<<c;
                else {const U32 difference=((code>>c)^(code>>t))&1;code^=(difference<<c)|(difference<<t);}
            }
            change+=I64(__popc(code))-before;
        }
        a.costs[b*assignments+mask]=U64(I64(direct)+change)+preparation;
    }
    __syncthreads();
    if(!threadIdx.x&&!outputs(a,b))a.status[b]=4;
    __syncthreads();
    if(a.status[b]!=1){clear(a,b);return;}
    U32 group=0;const U64 groups=U64(a.choices)*(a.choices-1)/2;
    for(U32 first=0;first<a.choices;++first)for(U32 second=first+1;second<a.choices;++second,++group)if(group%blockDim.x==threadIdx.x){
        a.valid[b*groups+group]=1;
        const U32 physical[4]={a.wires[a.choice_data[first*3+1]],a.wires[a.choice_data[first*3+2]],a.wires[a.choice_data[second*3+1]],a.wires[a.choice_data[second*3+2]]};
        for(U32 role=0;role<4;++role)a.incidence[((b*groups+group)*a.dimension+physical[role])*4+role]=1;
    }
}
bool extent(U64 provided,U64 b,U64 n){return (!n||b<=~U64{0}/n)&&provided>=b*n;}
}
extern "C" std::int32_t cnot103_cuda_launch_template_response(const Cnot103TemplateResponseArgs* a){
    if(!a||!a->coefficients||!a->raw_status||!a->choice_data||!a->wires||!a->probabilities||!a->costs||!a->rounding||!a->globals||!a->mixed||!a->valid||!a->incidence||!a->status)return 1;
    if(!a->batch||a->batch>0x7fffffffU||a->dimension<2||a->dimension>(1U<<24)||a->choices<2||a->choices>8||!a->points||!a->unique_wires||a->unique_wires>16||a->unique_wires>a->dimension/2||a->invalid_choices>1)return 1;
    const U64 w=a->dimension/2,g=U64(a->choices)*(a->choices-1)/2,b=a->batch,p=a->points;
    if(a->batch_elements<b||a->choice_elements<U64(a->choices)*3||a->wire_elements<a->unique_wires||a->probability_elements<p*a->choices*2||
        !extent(a->coefficient_elements,b,w*(a->dimension-w))||!extent(a->cost_elements,b,1U<<a->choices)||!extent(a->rounding_elements,b,p+2)||
        !extent(a->global_elements,b,p+9)||!extent(a->mixed_elements,b,g)||!extent(a->token_elements,b,g)||!extent(a->incidence_elements,b,g*a->dimension*4))return 2;
    evaluate<<<a->batch,lanes,0,static_cast<cudaStream_t>(a->stream)>>>(*a);return static_cast<std::int32_t>(cudaGetLastError());
}
