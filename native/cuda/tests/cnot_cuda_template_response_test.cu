#include "cnot103/cnot_cuda.h"
#include "../src/template_response_math.cuh"
#include "template_response_math_cases.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using U8=std::uint8_t;using U32=std::uint32_t;using U64=std::uint64_t;using Big=unsigned __int128;
using Probability=cnot103_template::Probability;using Gate=std::array<U32,2>;
void expect(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
template<typename T>struct Buffer{
    T* p=nullptr;U64 n;explicit Buffer(U64 count):n(count){ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0xa5,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void put(const std::vector<T>& v){expect(v.size()==n,"upload extent");ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get()const{std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
__global__ void arithmetic(const TemplateRationalCase* input,U64* output,U32 count){
    const U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;using namespace cnot103_template;
    const auto c=input[i];const Wide a{c.nl,c.nh},b{c.dl,c.dh};Wide sum{},product{},q{},r{};U32 bits=0;U64* out=output+U64(i)*13;
    const bool added=add(a,b,sum),multiplied=multiply(a,b.lo,product),divided=divide(a,b,q,r),converted=rational_bits(a,b,bits);
    out[0]=sum.lo;out[1]=sum.hi;out[2]=added;out[3]=product.lo;out[4]=product.hi;out[5]=multiplied;out[6]=q.lo;out[7]=q.hi;out[8]=r.lo;out[9]=r.hi;out[10]=divided;out[11]=bits;out[12]=converted;
}
void arithmetic_checks(){
    const U32 count=sizeof(template_rational_cases)/sizeof(template_rational_cases[0]);Buffer<TemplateRationalCase> input(count);Buffer<U64> output(U64(count)*13);
    input.put(std::vector<TemplateRationalCase>(std::begin(template_rational_cases),std::end(template_rational_cases)));arithmetic<<<(count+127)/128,128>>>(input.p,output.p,count);ck(cudaGetLastError());ck(cudaDeviceSynchronize());const auto actual=output.get();
    const Big maximum=~Big{0};for(U32 i=0;i<count;++i){const auto c=template_rational_cases[i];const Big a=(Big(c.nh)<<64)|c.nl,b=(Big(c.dh)<<64)|c.dl;const U64* x=actual.data()+U64(i)*13;
        auto same=[](U64 lo,U64 hi,Big value){return lo==U64(value)&&hi==U64(value>>64);};
        expect(same(x[0],x[1],a+b)&&bool(x[2])==(b<=maximum-a),"u128 checked addition");
        expect(same(x[3],x[4],a*c.dl)&&bool(x[5])==(!c.dl||a<=maximum/c.dl),"u128 checked multiply by u64");
        expect(bool(x[10])==(b!=0),"u128 division status");if(b)expect(same(x[6],x[7],a/b)&&same(x[8],x[9],a%b),"u128 quotient and remainder");
        if(x[12]!=c.ok||(c.ok&&x[11]!=c.bits)){std::fprintf(stderr,"rational case %u got ok=%llu bits=%llx expected=%u/%x\n",i,(unsigned long long)x[12],(unsigned long long)x[11],c.ok,c.bits);throw std::runtime_error("independent exact Python rational rounding");}
    }
    std::printf("arithmetic PASS: %u exact rational cases plus independent host __int128 arithmetic\n",count);
}
struct Choice{U32 kind,control,target;};
std::vector<U64> identity(U32 n){std::vector<U64> m(U64(n)*((n+63)/64));for(U32 i=0;i<n;++i)m[U64(i)*((n+63)/64)+i/64]=U64{1}<<(i%64);return m;}
void apply(std::vector<U64>& m,U32 n,Gate gate){U32 w=(n+63)/64;for(U32 i=0;i<w;++i)m[U64(gate[1])*w+i]^=m[U64(gate[0])*w+i];}
U64 response_samples=0,assignments_checked=0,physical_words=0;
struct Case{
    U32 n,b,k,p,g,width,targets;std::vector<Choice> choices;std::vector<std::vector<Probability>> points;
    Buffer<U8> coefficients,raw,status,valid;Buffer<U32> descriptors,wires;Buffer<U64> probabilities,costs,rounding;Buffer<float> globals,mixed,incidence;
    Cnot103TemplateResponseArgs a{};
    Case(U32 dim,U32 batch,const std::vector<Choice>& ch,const std::vector<std::vector<Probability>>& pts,const std::vector<U8>& block,const std::vector<U8>& raw_codes):
        n(dim),b(batch),k(ch.size()),p(pts.size()),g(k*(k-1)/2),width(n/2),targets(n-width),choices(ch),points(pts),
        coefficients(U64(b)*width*targets),raw(b),status(b),valid(U64(b)*g),descriptors(k*3),wires(unique(ch).size()),probabilities(U64(p)*k*2),costs(U64(b)*(1U<<k)),rounding(U64(b)*(p+2)),globals(U64(b)*(p+9)),mixed(U64(b)*g),incidence(U64(b)*g*n*4){
        coefficients.put(block);raw.put(raw_codes);auto unique_wires=unique(ch);wires.put(unique_wires);std::vector<U32> packed;for(auto c:ch){packed.push_back(c.kind);packed.push_back(std::find(unique_wires.begin(),unique_wires.end(),c.control)-unique_wires.begin());packed.push_back(std::find(unique_wires.begin(),unique_wires.end(),c.target)-unique_wires.begin());}descriptors.put(packed);
        std::vector<U64> rational;for(const auto& point:pts){expect(point.size()==k,"point width");for(auto value:point){rational.push_back(value.numerator);rational.push_back(value.denominator);}}probabilities.put(rational);
        a.coefficients=coefficients.p;a.raw_status=raw.p;a.choice_data=descriptors.p;a.wires=wires.p;a.probabilities=probabilities.p;a.costs=costs.p;a.rounding=rounding.p;a.globals=globals.p;a.mixed=mixed.p;a.valid=valid.p;a.incidence=incidence.p;a.status=status.p;
        a.coefficient_elements=coefficients.n;a.choice_elements=descriptors.n;a.wire_elements=wires.n;a.probability_elements=probabilities.n;a.cost_elements=costs.n;a.rounding_elements=rounding.n;a.global_elements=globals.n;a.mixed_elements=mixed.n;a.token_elements=valid.n;a.incidence_elements=incidence.n;a.batch_elements=b;
        a.batch=b;a.dimension=n;a.choices=k;a.points=p;a.unique_wires=unique_wires.size();
    }
    static std::vector<U32> unique(const std::vector<Choice>& ch){std::vector<U32> x;for(auto c:ch){x.push_back(c.control);x.push_back(c.target);}std::sort(x.begin(),x.end());x.erase(std::unique(x.begin(),x.end()),x.end());return x;}
    void run(){expect(!cnot103_cuda_launch_template_response(&a),"template launch");ck(cudaDeviceSynchronize());response_samples+=b;}
    void cleared(U32 sample,U8 expected_status){
        expect(status.get()[sample]==expected_status,"source guard priority");
        auto check=[&](const auto& buffer,U64 stride){auto data=buffer.get();for(U64 i=0;i<stride;++i)expect(data[U64(sample)*stride+i]==0,"unavailable sample clears all payloads");};
        check(costs,1U<<k);check(rounding,p+2);check(globals,p+9);check(mixed,g);check(valid,g);check(incidence,U64(g)*n*4);
    }
    void compare(){
        const auto block=coefficients.get(),raw_codes=raw.get(),code=status.get(),tokens=valid.get();const auto actual=costs.get(),round=rounding.get();const auto values=globals.get(),differences=mixed.get(),roles=incidence.get();
        for(U32 sample=0;sample<b;++sample){if(raw_codes[sample]!=1){cleared(sample,raw_codes[sample]==0?2:3);continue;}expect(code[sample]==1,"computed status");
            const U64 cells=U64(width)*targets;std::vector<U64> costs_host(1U<<k);auto endpoint=identity(n);
            for(U32 r=0;r<targets;++r)for(U32 c=0;c<width;++c)if(block[U64(sample)*cells+U64(r)*width+c])apply(endpoint,n,{c,width+r});
            for(U32 mask=0;mask<(1U<<k);++mask){std::vector<U8> transformed(block.begin()+U64(sample)*cells,block.begin()+U64(sample+1)*cells);std::vector<Gate> prep;U64 cost=0;
                for(U32 i=0;i<k;++i)if((mask>>i)&1){auto c=choices[i];cost+=c.kind?6:2;for(U32 r=0;r<targets;++r){if(c.kind)std::swap(transformed[U64(r)*width+c.control],transformed[U64(r)*width+c.target]);else transformed[U64(r)*width+c.control]^=transformed[U64(r)*width+c.target];}prep.push_back({c.control,c.target});if(c.kind){prep.push_back({c.target,c.control});prep.push_back({c.control,c.target});}}
                for(U8 bit:transformed)cost+=bit;costs_host[mask]=cost;expect(actual[U64(sample)*(1U<<k)+mask]==cost,"complete physical assignment costs");++assignments_checked;
                auto replay=identity(n);for(auto gate:prep)apply(replay,n,gate);for(U32 r=0;r<targets;++r)for(U32 c=0;c<width;++c)if(transformed[U64(r)*width+c])apply(replay,n,{c,width+r});for(auto it=prep.rbegin();it!=prep.rend();++it)apply(replay,n,*it);expect(replay==endpoint,"independent full physical template word");++physical_words;
            }
            std::vector<Probability> mid(k,Probability{1,2});std::vector<float> expected(p+9);expected[0]=expectation(costs_host,mid);
            for(U32 i=0;i<p;++i)expected[1+i]=expectation(costs_host,points[i]);
            const U32 forward=rounding_ref(costs_host,mid,false),reverse=rounding_ref(costs_host,mid,true);expect(round[U64(sample)*(p+2)]==forward&&round[U64(sample)*(p+2)+1]==reverse,"rounding order/tie masks");
            U64 lo=costs_host[forward],hi=lo;for(U32 i=0;i<p;++i){auto choice=rounding_ref(costs_host,points[i],false);expect(round[U64(sample)*(p+2)+2+i]==choice,"probe conditional assignment");lo=std::min(lo,costs_host[choice]);hi=std::max(hi,costs_host[choice]);}
            const U64 f=costs_host[forward],r=costs_host[reverse];expected[p+1]=float(f);expected[p+2]=float(*std::min_element(costs_host.begin(),costs_host.end()));expected[p+3]=1;expected[p+4]=float(f);expected[p+5]=float(r);expected[p+6]=float(f>r?f-r:r-f);expected[p+7]=float(lo);expected[p+8]=float(hi);
            expect(std::equal(expected.begin(),expected.end(),values.begin()+U64(sample)*(p+9)),"all global feature channels");
            U32 group=0;for(U32 first=0;first<k;++first)for(U32 second=first+1;second<k;++second,++group){
                std::int64_t difference=0;for(U32 mask=0;mask<(1U<<k);++mask)if(!(mask&((1U<<first)|(1U<<second))))difference+=std::int64_t(costs_host[mask|(1U<<first)|(1U<<second)])-std::int64_t(costs_host[mask|(1U<<first)])-std::int64_t(costs_host[mask|(1U<<second)])+std::int64_t(costs_host[mask]);
                expect(differences[U64(sample)*g+group]==float(difference)/float((1U<<k)/4),"signed mixed differences");expect(tokens[U64(sample)*g+group]==1,"computed pair mask");
                const U32 physical[4]={choices[first].control,choices[first].target,choices[second].control,choices[second].target};for(U32 node=0;node<n;++node)for(U32 role=0;role<4;++role)expect(roles[((U64(sample)*g+group)*n+node)*4+role]==float(node==physical[role]),"node-major physical roles");
            }
        }
    }
    // This integration oracle uses small denominators; its 80-bit conversion
    // cannot straddle an f32 boundary here. General u128 conversion is checked
    // separately against 1143 unbounded-integer fixture results.
    static float expectation(const std::vector<U64>& costs,const std::vector<Probability>& point){Big numerator=0,denominator=1;for(auto p:point)denominator*=p.denominator;for(U32 mask=0;mask<costs.size();++mask){Big weight=1;for(U32 i=0;i<point.size();++i)weight*=((mask>>i)&1)?point[i].numerator:point[i].denominator-point[i].numerator;numerator+=weight*costs[mask];}return float((long double)numerator/(long double)denominator);}
    static U32 rounding_ref(const std::vector<U64>& costs,std::vector<Probability> point,bool reverse){
        const U32 k=point.size();for(U32 step=0;step<k;++step){const U32 fixed=reverse?k-1-step:step;Big off=0,on=0;for(U32 mask=0;mask<costs.size();++mask)if(!((mask>>fixed)&1)){Big weight=1;for(U32 i=0;i<k;++i)if(i!=fixed)weight*=((mask>>i)&1)?point[i].numerator:point[i].denominator-point[i].numerator;off+=weight*costs[mask];on+=weight*costs[mask|(1U<<fixed)];}point[fixed]={U64(on<off),1};}U32 result=0;for(U32 i=0;i<k;++i)result|=U32(point[i].numerator==1)<<i;return result;
    }
};
void response_checks(){
    const std::vector<Choice> choices={{0,1,2},{0,0,1},{1,0,2}};std::vector<U8> block(512*9),raw(512,1);raw[0]=0;
    for(U32 mask=0;mask<512;++mask)for(U32 i=0;i<9;++i)block[mask*9+i]=(mask>>i)&1;
    Case small(6,512,choices,{{{1,3},{2,5},{1,7}},{{0,1},{1,1},{1,2}}},block,raw);small.run();small.compare();
    std::vector<Choice> many;for(U32 i=0;i<8;++i)many.push_back(choices[i%choices.size()]);std::vector<U8> wide(3*32*33);for(U64 i=32*33;i<wide.size();++i)wide[i]=(i*13+(i/32)*5)%7<3;
    Case big(65,3,many,{std::vector<Probability>(8,{1,3}),std::vector<Probability>(8,{1,2})},wide,{0,1,1});big.run();big.compare();
    for(auto member:{&Cnot103TemplateResponseArgs::coefficient_elements,&Cnot103TemplateResponseArgs::choice_elements,&Cnot103TemplateResponseArgs::wire_elements,&Cnot103TemplateResponseArgs::probability_elements,&Cnot103TemplateResponseArgs::cost_elements,&Cnot103TemplateResponseArgs::rounding_elements,&Cnot103TemplateResponseArgs::global_elements,&Cnot103TemplateResponseArgs::mixed_elements,&Cnot103TemplateResponseArgs::token_elements,&Cnot103TemplateResponseArgs::incidence_elements,&Cnot103TemplateResponseArgs::batch_elements}){auto bad=big.a;--(bad.*member);expect(cnot103_cuda_launch_template_response(&bad)==2,"extent guard");}
    Case flags(6,3,choices,{{{1,2},{1,2},{1,2}}},std::vector<U8>(27,1),{0,1,2});flags.a.invalid_choices=1;flags.run();flags.cleared(0,2);flags.cleared(1,3);flags.cleared(2,3);
    Case overflow(6,3,many,{std::vector<Probability>(8,{1,~U64{0}})},std::vector<U8>(27,1),{0,1,2});overflow.run();overflow.cleared(0,2);overflow.cleared(1,4);overflow.cleared(2,3);
    const U64 d=(U64{1}<<63)-1;std::vector<U8> sparse(4);sparse[0]=1;
    Case late(4,1,{{1,0,1},{0,0,1},{0,0,1}},{{{0,1},{1,d},{1,d}}},sparse,{1});late.run();late.cleared(0,4);
}
int main(){try{{arithmetic_checks();response_checks();}ck(cudaDeviceSynchronize());std::printf("PASS template: %llu response rows, %llu assignment costs and %llu independently replayed physical words\n",(unsigned long long)response_samples,(unsigned long long)assignments_checked,(unsigned long long)physical_words);ck(cudaDeviceReset());return 0;}catch(const std::exception& e){std::fprintf(stderr,"FAIL template: %s\n",e.what());return 1;}}
