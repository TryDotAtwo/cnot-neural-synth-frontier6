#include "../src/one_excess_math.cuh"
#include "one_excess_math_cases.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace math = cnot103_one_excess;
using Word = std::uint32_t;
constexpr Word sentinel = 0xa5b6c7d8U;
static void check(bool ok, const char* message) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); } }
static void cuda_check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e)); std::exit(1); } }
template<class T> struct Device {
    T* p; explicit Device(std::size_t n) { cuda_check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cuda_check(cudaFree(p)); }
};
struct Input { unsigned op,limbs,small; int sa,sb; };
struct Output { int ok,sign,comparison; unsigned remainder,used; };

__global__ void arithmetic(Word* a, Word* b, Word* out, const Input* inputs, Output* results,
                           unsigned count, std::size_t stride, unsigned alias) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Input c = inputs[i];
    math::Limbs av{a + i, c.limbs, stride}, bv{b + i, c.limbs, stride};
    math::Limbs dst{out + i, c.limbs, stride};
    if (c.op != 2 && alias == 1) dst = av;
    if (c.op != 2 && alias == 2) dst = bv;
    Output r{1,0,0,0,0};
    switch (c.op) {
        case 0: r.ok = math::add(dst,av,bv); break;
        case 1: r.ok = math::subtract(dst,av,bv); break;
        case 2: r.ok = math::multiply(dst,av,bv); break;
        case 3: r.ok = math::multiply_small(dst,av,c.small); break;
        case 4: r.ok = math::divide_small(dst,av,c.small,r.remainder); break;
        case 5: r.ok = math::signed_add(dst,r.sign,av,c.sa,bv,c.sb); break;
        case 6: r.comparison = math::compare(av,bv); math::copy(dst,av); break;
        case 7: math::set_small(dst,c.small); break;
        case 8: math::clear(dst); break;
        default: r.ok = -1;
    }
    r.used = math::used(dst);
    results[i] = r;
}

__global__ void capacity(const unsigned* n, unsigned* out, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = math::limbs_for_dimension(n[i]);
}

// The exact finite-difference identity sum (-1)^(m-q) C(m,q) q^m = m!
// exercises the same signs, products and cancellation as Stage38. Expected
// factorials and component-interleaving ratios come from Python integers.
__global__ void differences(Word* scratch, unsigned limbs, unsigned m, int* result) {
    math::Limbs sum{scratch,limbs,1}, power{scratch+limbs,limbs,1};
    math::Limbs choose{scratch+2*limbs,limbs,1}, term{scratch+3*limbs,limbs,1};
    math::Limbs multinomial{scratch+4*limbs,limbs,1};
    math::clear(sum); math::set_small(choose,1);
    int sign = 0; unsigned rem = 0; bool ok = true;
    for (unsigned q = 1; q <= m; ++q) {
        ok &= math::multiply_small(choose,choose,m-q+1);
        ok &= math::divide_small(choose,choose,q,rem) && rem == 0;
        math::set_small(power,1);
        for (unsigned i = 0; i < m; ++i) ok &= math::multiply_small(power,power,q);
        ok &= math::multiply(term,choose,power);
        ok &= math::signed_add(sum,sign,sum,sign,term,(m-q)%2 ? -1 : 1);
    }
    math::copy(multinomial,sum);
    const unsigned first=m/3,second=(m-first)/2,parts[3]={first,second,m-first-second};
    for (unsigned part : parts) for (unsigned d=1; d<=part; ++d)
        ok &= math::divide_small(multinomial,multinomial,d,rem) && rem == 0;
    result[0] = ok; result[1] = sign;
}

__global__ void fatal(unsigned kind) {
    Word a[1]={0xffffffffU},b[1]={1},out[1]={0},remainder=0;
    math::Limbs av{a,1,1},bv{b,1,1},dst{out,1,1};
    if(kind==0) math::require_exact(math::add(dst,av,bv));
    if(kind==1) math::require_exact(math::subtract(dst,bv,av));
    if(kind==2) math::require_exact(math::divide_small(dst,av,0,remainder));
}

static std::vector<Word> parse(const char* hex, unsigned limbs) {
    std::vector<Word> out(limbs,0);
    const std::size_t size=std::strlen(hex);
    check(size <= std::size_t(limbs)*8,"fixture capacity");
    for (std::size_t i=0;i<size;++i) {
        char c=hex[size-1-i]; unsigned digit=c<='9'?c-'0':c-'a'+10;
        check(digit<16,"fixture hex"); out[i/8] |= Word(digit) << (4*(i%8));
    }
    return out;
}

int main(int argc,char** argv) {
    if(argc==2) {
        unsigned kind=std::strcmp(argv[1],"overflow")==0?0:std::strcmp(argv[1],"underflow")==0?1:2;
        check(kind!=2 || std::strcmp(argv[1],"division")==0,"fatal mode");
        fatal<<<1,1>>>(kind);
        cudaError_t error=cudaDeviceSynchronize();
        check(error==cudaErrorLaunchFailure,"expected exact arithmetic trap");
        std::printf("PASS expected %s integrity trap: %s\n",argv[1],cudaGetErrorString(error));
        return 0;
    }
    const unsigned count=sizeof(one_excess_math_cases)/sizeof(one_excess_math_cases[0]);
    const std::size_t stride=((count+31)/32)*32;
    unsigned maximum=0; for (const auto& c:one_excess_math_cases) maximum=std::max(maximum,c.limbs);
    const std::size_t size=(std::size_t(maximum)+2)*stride;
    std::vector<Word> original_a(size,sentinel),original_b(size,sentinel),blank(size,sentinel);
    std::vector<Input> input(count); std::vector<Output> result(count);
    for (unsigned i=0;i<count;++i) {
        const auto& c=one_excess_math_cases[i]; input[i]={c.op,c.limbs,c.small,c.sa,c.sb};
        auto a=parse(c.a,c.limbs),b=parse(c.b,c.limbs);
        for(unsigned j=0;j<c.limbs;++j) { original_a[(j+1)*stride+i]=a[j]; original_b[(j+1)*stride+i]=b[j]; }
    }
    Device<Word> da(size),db(size),dout(size); Device<Input> di(count); Device<Output> dr(count);
    cuda_check(cudaMemcpy(di.p,input.data(),count*sizeof(Input),cudaMemcpyHostToDevice));
    for (unsigned alias=0;alias<3;++alias) {
        cuda_check(cudaMemcpy(da.p,original_a.data(),size*sizeof(Word),cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(db.p,original_b.data(),size*sizeof(Word),cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(dout.p,blank.data(),size*sizeof(Word),cudaMemcpyHostToDevice));
        arithmetic<<<(count+127)/128,128>>>(da.p+stride,db.p+stride,dout.p+stride,di.p,dr.p,count,stride,alias);
        cuda_check(cudaDeviceSynchronize());
        cuda_check(cudaMemcpy(result.data(),dr.p,count*sizeof(Output),cudaMemcpyDeviceToHost));
        std::vector<Word> a(size),b(size),out(size);
        cuda_check(cudaMemcpy(a.data(),da.p,size*sizeof(Word),cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(b.data(),db.p,size*sizeof(Word),cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(out.data(),dout.p,size*sizeof(Word),cudaMemcpyDeviceToHost));
        for (unsigned i=0;i<stride;++i) {
            unsigned selected=(i<count && one_excess_math_cases[i].op!=2)?alias:0;
            unsigned limbs=i<count?one_excess_math_cases[i].limbs:0;
            std::vector<Word> expected;
            if (i<count) {
                const auto& c=one_excess_math_cases[i]; const auto& r=result[i]; expected=parse(c.expected,c.limbs);
                if (r.ok!=c.ok || (c.ok && r.sign!=c.sign) || r.remainder!=c.remainder || r.comparison!=c.comparison) {
                    std::fprintf(stderr,"case %u op %u alias %u metadata\n",i,c.op,alias); std::exit(1);
                }
                unsigned used=limbs; while(used && expected[used-1]==0) --used;
                check(r.used==used,"used limbs");
            }
            for (unsigned j=0;j<maximum+2;++j) {
                const std::size_t p=std::size_t(j)*stride+i;
                bool write=i<count && j>0 && j<=limbs;
                Word ea=write && selected==1?expected[j-1]:original_a[p];
                Word eb=write && selected==2?expected[j-1]:original_b[p];
                Word eo=write && selected==0?expected[j-1]:sentinel;
                if(a[p]!=ea || b[p]!=eb || out[p]!=eo) {
                    std::fprintf(stderr,"case %u limb %u alias %u payload/guard\n",i,j,alias); std::exit(1);
                }
            }
        }
    }
    const unsigned nc=sizeof(one_excess_capacity_cases)/sizeof(one_excess_capacity_cases[0]);
    std::vector<unsigned> n(nc),caps(nc); Device<unsigned> dn(nc),dc(nc);
    for(unsigned i=0;i<nc;++i) { n[i]=one_excess_capacity_cases[i].n; check(math::limbs_for_dimension(n[i])==one_excess_capacity_cases[i].limbs,"host capacity"); }
    cuda_check(cudaMemcpy(dn.p,n.data(),nc*sizeof(unsigned),cudaMemcpyHostToDevice));
    capacity<<<(nc+127)/128,128>>>(dn.p,dc.p,nc); cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaMemcpy(caps.data(),dc.p,nc*sizeof(unsigned),cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<nc;++i) check(caps[i]==one_excess_capacity_cases[i].limbs,"device capacity");
    unsigned difference_count=0;
    for(const auto& c:one_excess_difference_cases) {
        Device<Word> space(std::size_t(c.limbs)*5); Device<int> flags(2);
        differences<<<1,1>>>(space.p,c.limbs,c.m,flags.p); cuda_check(cudaDeviceSynchronize());
        int status[2]; cuda_check(cudaMemcpy(status,flags.p,sizeof(status),cudaMemcpyDeviceToHost));
        check(status[0]==1 && status[1]==1,"finite difference arithmetic");
        std::vector<Word> values(std::size_t(c.limbs)*5);
        cuda_check(cudaMemcpy(values.data(),space.p,values.size()*sizeof(Word),cudaMemcpyDeviceToHost));
        auto factorial=parse(c.factorial,c.limbs),multinomial=parse(c.multinomial,c.limbs);
        check(std::equal(factorial.begin(),factorial.end(),values.begin()),"exact finite difference");
        check(std::equal(multinomial.begin(),multinomial.end(),values.begin()+4*c.limbs),"exact component division");
        ++difference_count;
    }
    std::printf("PASS one-excess arithmetic: %u integer cases x 3 alias layouts, %u capacity cases, %u exact finite-difference/interleaving cases\n",count,nc,difference_count);
}
