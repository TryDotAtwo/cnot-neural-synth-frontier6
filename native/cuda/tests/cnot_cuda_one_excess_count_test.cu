#include "cnot103/cnot_cuda.h"
#include "../src/one_excess_count.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <vector>

using U32=std::uint32_t;using U64=std::uint64_t;
using Row=std::vector<U64>;using History=std::vector<Row>;using Histories=std::vector<History>;
using Pair=std::pair<U32,U32>;using Layout=cnot103_one_excess::CountLayout;
struct Case { Histories h;std::set<U32> protected_rows;std::vector<U32> expected;int bag=-1,processed=-1;U32 status=1; };
struct Metrics { U64 rows=0,positive=0,zero=0,wide=0,resource=0,evictions=0; } metrics;
static void check(bool ok,const char* message){if(!ok){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}}
static void cu(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(1);}}
template<class T>struct Buffer {
    T* base;std::vector<T> host;std::size_t count;static constexpr std::size_t guard=32;
    explicit Buffer(std::size_t n):host(n+2*guard,static_cast<T>(0xa55aa55aa55aa55aULL)),count(n){cu(cudaMalloc(&base,host.size()*sizeof(T)));
        cu(cudaMemcpy(base,host.data(),guard*sizeof(T),cudaMemcpyHostToDevice));cu(cudaMemcpy(base+guard+count,host.data()+guard+count,guard*sizeof(T),cudaMemcpyHostToDevice));}
    ~Buffer(){cudaFree(base);}T* data(){return base+guard;}
    void upload(){cu(cudaMemcpy(base,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));}
    void load(const std::vector<T>& values){check(values.size()==count,"input count");std::copy(values.begin(),values.end(),host.begin()+guard);upload();}
    void check_guards(){cu(cudaMemcpy(host.data(),base,guard*sizeof(T),cudaMemcpyDeviceToHost));
        cu(cudaMemcpy(host.data()+guard+count,base+guard+count,guard*sizeof(T),cudaMemcpyDeviceToHost));
        for(std::size_t i=0;i<guard;++i)check(host[i]==static_cast<T>(0xa55aa55aa55aa55aULL)&&host[guard+count+i]==static_cast<T>(0xa55aa55aa55aa55aULL),"allocation guard");
    }
    std::vector<T> read(){check_guards();cu(cudaMemcpy(host.data()+guard,data(),count*sizeof(T),cudaMemcpyDeviceToHost));
        return std::vector<T>(host.begin()+guard,host.begin()+guard+count);}
};
static Row unit(U32 n,U32 r){Row v((n+63)/64);v[r/64]=U64(1)<<(r%64);return v;}
static Histories from_word(U32 n,const std::vector<Pair>& word){Histories h(n);for(U32 r=0;r<n;++r)h[r].push_back(unit(n,r));
    for(auto[c,t]:word){Row v=h[t].back();for(U32 w=0;w<v.size();++w)v[w]^=h[c].back()[w];h[t].push_back(v);}return h;}

// Independent physical oracle: count all legal CNOT orders by the current row
// values. It uses no symbolic factors, graph ordering, lattice tables, finite
// differences or production integer arithmetic. The small cases have <=9 gates.
static std::vector<U32> enumerate(const Histories& h){
    U64 states=1;for(const auto& row:h){check(states<=1000000/row.size(),"small oracle capacity");states*=row.size();}
    std::vector<U64> memo(states,~U64(0));std::vector<U32> pos(h.size());
    std::function<U64()> visit=[&](){U64 key=0,multiplier=1;bool done=true;
        for(U32 r=0;r<h.size();++r){key+=pos[r]*multiplier;multiplier*=h[r].size();done&=pos[r]+1==h[r].size();}
        if(done)return U64(1);if(memo[key]!=~U64(0))return memo[key];U64 total=0;
        for(U32 t=0;t<h.size();++t)if(pos[t]+1<h[t].size()){
            Row difference=h[t][pos[t]];for(U32 w=0;w<difference.size();++w)difference[w]^=h[t][pos[t]+1][w];
            for(U32 c=0;c<h.size();++c)if(c!=t&&h[c][pos[c]]==difference){++pos[t];const U64 child=visit();--pos[t];check(total<=~U64(0)-child,"oracle addition");total+=child;}
        }
        return memo[key]=total;
    };
    const U64 count=visit();return {static_cast<U32>(count),static_cast<U32>(count>>32)};
}

// Large analytic counts use base10 factorial/division, then long division by
// 2^32 to export the result. This is independent of the CUDA limb routines.
static std::vector<U32> factorial(U32 n,U32 divisor=1){
    std::vector<U32> decimal{1};for(U32 f=2;f<=n;++f){U32 carry=0;for(U32& digit:decimal){U32 v=digit*f+carry;digit=v%10;carry=v/10;}while(carry){decimal.push_back(carry%10);carry/=10;}}
    U32 remainder=0;for(std::size_t i=decimal.size();i-->0;){U32 v=remainder*10+decimal[i];decimal[i]=v/divisor;remainder=v%divisor;}check(!remainder,"analytic factorial division");
    std::vector<U32> words;while(!decimal.empty()){U64 carry=0;for(std::size_t i=decimal.size();i-->0;){U64 v=carry*10+decimal[i];decimal[i]=static_cast<U32>(v>>32);carry=static_cast<U32>(v);}
        words.push_back(static_cast<U32>(carry));while(!decimal.empty()&&!decimal.back())decimal.pop_back();}
    return words;
}

static void run(const std::vector<Case>& cases,U64 limit,U32 cache,U32 lanes,U32 workers,const char* fatal=nullptr){
    const U32 n=cases[0].h.size(),batch=cases.size();workers=std::min(workers,batch);const Layout l(n,lanes,cache);const auto& sl=l.structure;
    std::vector<U64> histories(U64(batch)*3*sl.w*n);std::vector<U32> lengths(U64(batch)*n);std::vector<std::uint8_t> protected_rows(U64(batch)*n);
    for(U32 b=0;b<batch;++b)for(U32 r=0;r<n;++r){lengths[U64(b)*n+r]=cases[b].h[r].size();protected_rows[U64(b)*n+r]=cases[b].protected_rows.count(r);
        for(U32 p=0;p<cases[b].h[r].size();++p)for(U32 w=0;w<sl.w;++w)histories[U64(b)*3*sl.w*n+(U64(p)*sl.w+w)*n+r]=cases[b].h[r][p][w];}
    if(fatal){
        if(!std::strcmp(fatal,"length"))lengths[0]=0;
        else if(!std::strcmp(fatal,"zero"))for(U32 w=0;w<sl.w;++w)histories[(U64(sl.w)+w)*n]=0;
        else if(!std::strcmp(fatal,"repeat"))for(U32 w=0;w<sl.w;++w)histories[(U64(sl.w)+w)*n]=histories[U64(w)*n];
        else if(!std::strcmp(fatal,"padding"))histories[U64(sl.w-1)*n]|=U64(1)<<(n%64);
        else if(!std::strcmp(fatal,"singular"))for(U32 w=0;w<sl.w;++w)histories[U64(w)*n]=histories[U64(w)*n+1];
        else if(!std::strcmp(fatal,"final"))for(U32 w=0;w<sl.w;++w)histories[(U64(lengths[0]-1)*sl.w+w)*n]=histories[(U64(lengths[1]-1)*sl.w+w)*n+1];
        else if(!std::strcmp(fatal,"double")){lengths[1]=3;for(U32 w=0;w<sl.w;++w)histories[(U64(2)*sl.w+w)*n+1]=histories[U64(w)*n+1];}
        else if(!std::strcmp(fatal,"protected"))protected_rows[0]=2;
        else if(!std::strcmp(fatal,"unused"))histories[U64(sl.w)*n+2]=1;
        else check(false,"fatal mode");
    }
    Buffer<U64> h(histories.size()),sb(U64(workers)*sl.bit_count),ss(U64(workers)*sl.scratch_count),pg(U64(workers)*l.program_words),out(U64(batch)*4);
    Buffer<U32> len(lengths.size()),sm(U64(workers)*sl.meta_count),ar(U64(workers)*l.arithmetic_words),counts(U64(batch)*l.limbs);Buffer<std::uint8_t> pr(protected_rows.size());
    h.load(histories);len.load(lengths);pr.load(protected_rows);cudaStream_t stream;cu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Cnot103OneExcessCountArgs a{h.data(),len.data(),pr.data(),sm.data(),sb.data(),ss.data(),pg.data(),ar.data(),counts.data(),out.data(),stream,
        h.count,len.count,pr.count,sm.count,sb.count,ss.count,pg.count,ar.count,counts.count,out.count,limit,batch,n,workers,lanes,cache};
    check(cnot103_cuda_launch_one_excess_count(&a)==0,"launch");const cudaError_t error=cudaStreamSynchronize(stream);
    if(fatal){check(error==cudaErrorLaunchFailure,"expected invalid-history trap");std::printf("PASS expected count %s history trap: %s\n",fatal,cudaGetErrorString(error));return;}
    cu(error);cu(cudaStreamDestroy(stream));auto got=counts.read();auto headers=out.read();sm.check_guards();sb.check_guards();ss.check_guards();pg.check_guards();ar.check_guards();
    check(h.read()==histories&&len.read()==lengths&&pr.read()==protected_rows,"input preservation");
    for(U32 b=0;b<batch;++b){const auto& c=cases[b];const auto expected=c.expected.empty()?enumerate(c.h):c.expected;++metrics.rows;
        check(headers[4*b]==c.status,"typed source outcome");if(c.bag>=0)check(headers[4*b+1]==static_cast<U64>(c.bag),"source max_bag");
        if(c.processed>=0)check(headers[4*b+2]==static_cast<U64>(c.processed),"lazy component ordering");
        bool positive=false;for(U32 i=0;i<l.limbs;++i){const U32 want=i<expected.size()?expected[i]:0;
            if(got[U64(b)*l.limbs+i]!=want){std::fprintf(stderr,"n%u row%u cache%u lane%u limb%u got%u expected%u\n",n,b,cache,lanes,i,got[U64(b)*l.limbs+i],want);std::exit(1);}positive|=want!=0;}
        check(expected.size()<=l.limbs,"test magnitude capacity");if(c.status==4){++metrics.resource;check(!positive&&headers[4*b+1]==0,"cleared resource output");}
        else if(positive){++metrics.positive;if(expected.size()>4)++metrics.wide;}else ++metrics.zero;
        if(!cache)check(!headers[4*b+3],"no-cache diagnostic");metrics.evictions+=headers[4*b+3];
    }
    for(U32 field=0;field<10;++field){auto bad=a;U64* extents[]={&bad.history_words,&bad.length_elements,&bad.protected_elements,&bad.meta_elements,&bad.bit_words,&bad.scratch_words,&bad.program_words,&bad.arithmetic_elements,&bad.count_elements,&bad.result_words};--*extents[field];check(cnot103_cuda_launch_one_excess_count(&bad)==2,"short extent");}
    check(cnot103_cuda_launch_one_excess_count(nullptr)==1,"null args");
    for(U32 mode=0;mode<9;++mode){auto bad=a;switch(mode){case 0:bad.batch=0;break;case 1:bad.max_entries=0;break;case 2:bad.dimension=1;break;case 3:bad.dimension=(1U<<24)+1;break;case 4:bad.workers=0;break;case 5:bad.workers=batch+1;break;case 6:bad.lanes=0;break;case 7:bad.lanes=33;break;default:bad.program=nullptr;}
        check(cnot103_cuda_launch_one_excess_count(&bad)==1,"invalid metadata");}
    for(U32 workers:{1U,32U}){auto bad=a;bad.dimension=workers==1?(1U<<24):(1U<<20);bad.workers=bad.batch=workers;bad.lanes=32;bad.cache_slots=~U32(0);
        bad.history_words=bad.length_elements=bad.protected_elements=bad.meta_elements=bad.bit_words=bad.scratch_words=bad.program_words=bad.arithmetic_elements=bad.count_elements=bad.result_words=~U64(0);
        check(cnot103_cuda_launch_one_excess_count(&bad)==2,"individual/aggregate byte extent overflow");}
}

int main(int argc,char** argv){
    if(argc==2){U32 n=!std::strcmp(argv[1],"padding")?65:3;run({{from_word(n,{{1,0},{2,1},{1,0}}),{},{}}},1024,1,3,1,argv[1]);return 0;}
    std::vector<Case> small;
    for(U32 code=0;code<512;++code){U32 rows[3]={code&7,(code>>3)&7,(code>>6)&7};std::set<U32> span;
        for(U32 mask=0;mask<8;++mask){U32 value=0;for(U32 r=0;r<3;++r)if(mask>>r&1)value^=rows[r];span.insert(value);}if(span.size()!=8)continue;
        for(U32 target=0;target<3;++target)if(rows[target]!=(1U<<target))for(U32 middle=1;middle<8;++middle)if(middle!=rows[target]&&middle!=(1U<<target)){
            Histories h(3);for(U32 r=0;r<3;++r){h[r].push_back({rows[r]});if(r==target)h[r].push_back({middle});if(rows[r]!=(1U<<r))h[r].push_back({U64(1)<<r});}small.push_back({h,{},enumerate(h)});}}
    check(small.size()==2160,"complete GL3 population");
    run(small,~U64(0),0,1,1);run(small,~U64(0),1,3,3);run(small,~U64(0),127,32,4);
    // Impossible transitions have count0 before any component budget preflight.
    Histories impossible{{{1},{2}},{{2},{1}},{{4}}};run({{impossible,{},{0},0,0}},1,1,2,1);
    auto late=from_word(8,{{1,0},{3,2},{4,3},{5,4},{6,5}});run({{late,{},{0},0,2,4}},8,17,3,1);
    U64 seed=0xf08c0a17;auto next=[&](){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;};
    for(U32 n:{5U,8U}){std::vector<Case> cases;for(U32 sample=0;sample<12;++sample){const U32 twice=next()%n;std::vector<U32> targets;
            for(U32 r=0;r<n;++r)targets.push_back(r);targets.push_back(twice);for(U32 i=targets.size()-1;i>0;--i)std::swap(targets[i],targets[next()%(i+1)]);
            std::vector<Pair> word;for(U32 t:targets)word.emplace_back((t+1+next()%(n-1))%n,t);auto h=from_word(n,word);cases.push_back({h,{twice},enumerate(h)});}
        run(cases,~U64(0),8192,4,2);}
    for(U32 n:{2U,65U,144U}){std::vector<Case> cases{{from_word(n,{}),{},{1},0,0}};
        if(n>2){std::vector<Pair> word{{1,0},{2,0}};for(U32 t=3;t<n-1;t+=2)word.emplace_back(t+1,t);auto h=from_word(n,word);
            cases.push_back({h,{},factorial(word.size(),2),2,static_cast<int>(word.size()-1)});h[0].erase(h[0].begin());cases.push_back({h,{0},factorial(word.size()-1),1,static_cast<int>(word.size()-1)});}
        run(cases,~U64(0),1,1,1);run(cases,~U64(0),257,4,2);
        if(n>2){auto limited=cases[1];limited.expected={0};limited.bag=0;limited.processed=1;limited.status=4;run({limited},1,1,4,1);}}
    // Connected star: 38! words, with wide lattice values and alternating
    // finite-difference intermediates, not just a wide final k! multiplier.
    std::vector<Pair> star_word;for(U32 t=1;t<39;++t)star_word.emplace_back(0,t);star_word.emplace_back(39,0);
    const auto star=from_word(40,star_word);run({{star,{0},factorial(38),2,1}},1521,7,8,1);
    run({{star,{0},{0},0,1,4}},1520,7,8,1);
    check(metrics.positive&&metrics.zero&&metrics.wide&&metrics.resource&&metrics.evictions,"coverage");
    std::printf("PASS exact history counts: %llu rows, %llu positive/%llu zero/%llu source-resource, %llu wide, %llu cache evictions; independent physical-order and decimal-factorial oracles; guarded buffers/stream/worker reuse\n",
        (unsigned long long)metrics.rows,(unsigned long long)metrics.positive,(unsigned long long)metrics.zero,(unsigned long long)metrics.resource,(unsigned long long)metrics.wide,(unsigned long long)metrics.evictions);
}
