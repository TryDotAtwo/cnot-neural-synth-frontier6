#include "cnot103/cnot_cuda.h"
#include "../src/one_excess_structure.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <tuple>
#include <vector>

using U32=std::uint32_t;using U64=std::uint64_t;
using Row=std::vector<U64>;using History=std::vector<Row>;using Histories=std::vector<History>;
using Pair=std::pair<U32,U32>;using Term=std::vector<Pair>;
using Layout=cnot103_one_excess::StructureLayout;
struct Case{Histories h;std::set<U32> protected_rows;};
struct Factor{std::vector<Term> terms;std::set<U32> scope;};
struct Metrics{U64 rows=0,factors=0,empty=0,cycles=0,duplicates=0,multiple=0,protected_orders=0,fallback_orders=0,admitted=0,limited=0;}metrics;
static void check(bool ok,const char* message){if(!ok){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}}
static void cu(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(1);}}
template<class T>struct Buffer{
    T* base;std::vector<T> host;std::size_t count;static constexpr std::size_t guard=32;
    explicit Buffer(std::size_t n):host(n+2*guard,static_cast<T>(0xa55aa55aa55aa55aULL)),count(n){cu(cudaMalloc(&base,host.size()*sizeof(T)));upload();}
    ~Buffer(){cudaFree(base);}T* data(){return base+guard;}
    void upload(){cu(cudaMemcpy(base,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));}
    void load(const std::vector<T>& values){check(values.size()==count,"input count");std::copy(values.begin(),values.end(),host.begin()+guard);upload();}
    std::vector<T> read(){cu(cudaMemcpy(host.data(),base,host.size()*sizeof(T),cudaMemcpyDeviceToHost));
        for(std::size_t i=0;i<guard;++i)check(host[i]==static_cast<T>(0xa55aa55aa55aa55aULL)&&host[guard+count+i]==static_cast<T>(0xa55aa55aa55aa55aULL),"allocation guard");
        return std::vector<T>(host.begin()+guard,host.begin()+guard+count);}
};
static Row unit(U32 n,U32 r){Row out((n+63)/64);out[r/64]=U64(1)<<(r%64);return out;}
static Histories from_word(U32 n,const std::vector<Pair>& word){Histories h(n);for(U32 r=0;r<n;++r)h[r].push_back(unit(n,r));
    for(auto [c,t]:word){Row next=h[t].back();for(U32 w=0;w<next.size();++w)next[w]^=h[c].back()[w];h[t].push_back(next);}return h;}
static Factor factor(std::vector<Term> terms){Factor f{std::move(terms),{}};for(const auto& t:f.terms)for(auto[a,b]:t){f.scope.insert(a);f.scope.insert(b);}return f;}

// Independent host oracle: ordered sets and explicit neighbor pairs, not GPU
// bitset/popcount elimination. Mandatory-cycle checking uses recursive DFS.
static std::pair<std::vector<U32>,std::vector<U64>> reference(const Case& c,U64 limit){
    const U32 n=c.h.size();const Layout l(n);std::vector<U32> out(l.meta_count);std::vector<U64> bits(l.bit_count);
    std::fill_n(out.begin()+l.ids,l.k,l.k);std::fill_n(out.begin()+l.vertices,l.k,l.k);std::fill_n(out.begin()+l.order,l.k,l.k);
    std::vector<Pair> events;std::vector<U32> offsets(n+1);std::set<U32> retained=c.protected_rows,protected_events;
    out[4]=n;out[6]=1;++metrics.rows;
    for(U32 r=0;r<n;++r){offsets[r]=events.size();if(c.h[r].size()==3){out[4]=r;retained.insert(r);}for(U32 p=0;p+1<c.h[r].size();++p)events.emplace_back(r,p);}
    offsets[n]=events.size();out[0]=events.size();
    for(U32 e=0;e<events.size();++e){out[8+e]=events[e].first;out[8+l.k+e]=events[e].second;if(retained.count(events[e].first))protected_events.insert(e);}
    out[5]=protected_events.size();for(U32 r=0;r<=n;++r)out[8+2*l.k+r]=offsets[r];
    std::vector<Factor> raw,factors;
    for(U32 r=0;r<n;++r)if(c.h[r].size()==3)raw.push_back(factor({{{offsets[r],offsets[r]+1}}}));
    for(U32 e=0;e<events.size();++e){auto[r,p]=events[e];Row difference=c.h[r][p];for(U32 w=0;w<difference.size();++w)difference[w]^=c.h[r][p+1][w];
        std::vector<Term> terms;for(U32 control=0;control<n;++control)if(control!=r)for(U32 pos=0;pos<c.h[control].size();++pos)if(difference==c.h[control][pos]){
            Term t;if(pos)t.emplace_back(offsets[control]+pos-1,e);if(pos+1<c.h[control].size())t.emplace_back(e,offsets[control]+pos);std::sort(t.begin(),t.end());terms.push_back(t);}
        raw.push_back(factor(terms));}
    for(const auto& f:raw){bool duplicate=false;if(f.terms.size()==1)for(const auto& previous:factors)duplicate|=f.terms==previous.terms&&f.scope==previous.scope;
        if(duplicate){++metrics.duplicates;continue;}if(f.terms.size()>1)++metrics.multiple;factors.push_back(f);}
    out[1]=factors.size();metrics.factors+=factors.size();
    for(U32 f=0;f<factors.size();++f){out[8+3*l.k+f]=factors[f].terms.size();for(U32 v:factors[f].scope)bits[U64(v/64)*l.f+f]|=U64(1)<<(v%64);
        for(U32 t=0;t<factors[f].terms.size();++t){out[l.term+U64(t)*l.f+f]=factors[f].terms[t].size();U32 p=0;
            for(auto[a,b]:factors[f].terms[t]){out[l.term+(U64(1+2*p)*l.t+t)*l.f+f]=a;out[l.term+(U64(2+2*p)*l.t+t)*l.f+f]=b;++p;}}}
    if(std::any_of(factors.begin(),factors.end(),[](const Factor& f){return f.terms.empty();})){out[2]=1;++metrics.empty;return{out,bits};}
    const U32 k=events.size();std::vector<std::set<U32>> graph(k),mandatory(k);
    for(const auto& f:factors){for(U32 a:f.scope)for(U32 b:f.scope)if(a!=b)graph[a].insert(b);
        for(auto edge:f.terms[0])if(std::all_of(f.terms.begin(),f.terms.end(),[&](const Term& t){return std::find(t.begin(),t.end(),edge)!=t.end();}))mandatory[edge.first].insert(edge.second);}
    for(U32 a=0;a<k;++a){for(U32 b:graph[a])bits[U64(l.f)*l.ew+U64(b/64)*l.k+a]|=U64(1)<<(b%64);
        for(U32 b:mandatory[a])bits[U64(l.f+l.k)*l.ew+U64(b/64)*l.k+a]|=U64(1)<<(b%64);}
    std::vector<U32> color(k);std::function<bool(U32)> cycle=[&](U32 v){if(color[v]==1)return true;if(color[v]==2)return false;color[v]=1;for(U32 u:mandatory[v])if(cycle(u))return true;color[v]=2;return false;};
    for(U32 v=0;v<k;++v)if(cycle(v)){out[2]=2;++metrics.cycles;return{out,bits};}
    std::vector<bool> seen(k);U32 cc=0,offset=0;
    for(U32 root=0;root<k;++root)if(!seen[root]){
        std::vector<U32> component{root};seen[root]=true;for(std::size_t head=0;head<component.size();++head)for(U32 u:graph[component[head]])if(!seen[u]){seen[u]=true;component.push_back(u);}
        std::sort(component.begin(),component.end());out[l.offsets+cc]=offset;for(U32 v:component){out[l.ids+v]=cc;out[l.vertices+offset++]=v;}out[l.offsets+cc+1]=offset;
        std::set<U32> unprotected;for(U32 v:component)if(!protected_events.count(v))unprotected.insert(v);
        std::set<U32> visited;bool pseudo=true;
        for(U32 start:unprotected)if(!visited.count(start)){std::vector<U32> queue{start};visited.insert(start);U64 degree=0;
            for(std::size_t head=0;head<queue.size();++head)for(U32 u:graph[queue[head]])if(unprotected.count(u)){++degree;if(visited.insert(u).second)queue.push_back(u);}
            if(degree/2>queue.size())pseudo=false;}
        const bool use_protected=!protected_events.empty()&&pseudo;out[l.protected_flags+cc]=use_protected;
        if(use_protected)++metrics.protected_orders;else if(!protected_events.empty())++metrics.fallback_orders;
        std::set<U32> alive(component.begin(),component.end());auto adjacency=graph;U32 position=out[l.offsets+cc],bag=0;
        while(!alive.empty()){
            bool pool=use_protected&&std::any_of(alive.begin(),alive.end(),[&](U32 v){return !protected_events.count(v);});
            U32 selected=k;std::tuple<U64,U64,U64,U32> best;
            for(U32 v:alive)if(!pool||!protected_events.count(v)){U64 fill=0,degree=adjacency[v].size(),free_degree=0;
                std::vector<U32> neighbors(adjacency[v].begin(),adjacency[v].end());for(U32 u:neighbors)free_degree+=!protected_events.count(u);
                for(U32 a=0;a<neighbors.size();++a)for(U32 b=a+1;b<neighbors.size();++b)fill+=!adjacency[neighbors[a]].count(neighbors[b]);
                auto key=std::make_tuple(pool?free_degree:0,fill,degree,v);if(selected==k||key<best){selected=v;best=key;}}
            check(selected<k,"reference selection");if(pool)check(std::get<0>(best)<=2,"reference protected theorem");
            bag=std::max<U32>(bag,adjacency[selected].size()+1);out[l.order+position++]=selected;
            for(U32 a:adjacency[selected])for(U32 b:adjacency[selected])if(a!=b)adjacency[a].insert(b);
            for(U32 u:adjacency[selected])adjacency[u].erase(selected);adjacency[selected].clear();alive.erase(selected);
        }
        out[l.bag+cc]=bag;
        // Division-based budget oracle avoids overflowing a power/product.
        U64 room=limit;bool fits=true;for(U32 i=0;i<bag;++i){room/=component.size();if(room==0){fits=false;break;}}
        out[l.budget+cc]=fits;if(fits)++metrics.admitted;else ++metrics.limited;++cc;
    }
    out[3]=cc;return{out,bits};
}

static void run(const std::vector<Case>& cases,U64 limit,const char* fatal=nullptr){
    check(!cases.empty(),"cases");const U32 n=cases[0].h.size(),batch=cases.size();const Layout l(n);
    std::vector<U64> h(U64(batch)*3*l.w*n),expected_bits;std::vector<U32> lengths(U64(batch)*n),expected_output;std::vector<std::uint8_t> protected_rows(U64(batch)*n);
    for(U32 b=0;b<batch;++b){for(U32 r=0;r<n;++r){lengths[U64(b)*n+r]=cases[b].h[r].size();protected_rows[U64(b)*n+r]=cases[b].protected_rows.count(r);
            for(U32 p=0;p<cases[b].h[r].size();++p)for(U32 w=0;w<l.w;++w)h[U64(b)*3*l.w*n+(U64(p)*l.w+w)*n+r]=cases[b].h[r][p][w];}
        if(!fatal){auto expected=reference(cases[b],limit);expected_output.insert(expected_output.end(),expected.first.begin(),expected.first.end());expected_bits.insert(expected_bits.end(),expected.second.begin(),expected.second.end());}}
    if(fatal){
        if(!std::strcmp(fatal,"length"))lengths[0]=0;
        else if(!std::strcmp(fatal,"zero"))for(U32 w=0;w<l.w;++w)h[(U64(l.w)+w)*n]=0;
        else if(!std::strcmp(fatal,"repeat"))for(U32 w=0;w<l.w;++w)h[(U64(l.w)+w)*n]=h[U64(w)*n];
        else if(!std::strcmp(fatal,"padding"))h[U64(l.w-1)*n]|=U64(1)<<(n%64);
        else if(!std::strcmp(fatal,"singular"))for(U32 w=0;w<l.w;++w)h[U64(w)*n]=h[U64(w)*n+1];
        else if(!std::strcmp(fatal,"final"))for(U32 w=0;w<l.w;++w)h[(U64(lengths[0]-1)*l.w+w)*n]=h[(U64(lengths[1]-1)*l.w+w)*n+1];
        else if(!std::strcmp(fatal,"double")){lengths[1]=3;for(U32 w=0;w<l.w;++w)h[(U64(2)*l.w+w)*n+1]=h[U64(w)*n+1];}
        else if(!std::strcmp(fatal,"protected"))protected_rows[0]=2;
        else if(!std::strcmp(fatal,"unused"))h[U64(l.w)*n+2]=1;
        else check(false,"fatal mode");
    }
    Buffer<U64> dh(h.size()),db(U64(batch)*l.bit_count),scratch(U64(batch)*l.scratch_count);
    Buffer<U32> dl(lengths.size()),out(U64(batch)*l.meta_count);Buffer<std::uint8_t> dp(protected_rows.size());
    dh.load(h);dl.load(lengths);dp.load(protected_rows);
    Cnot103OneExcessStructureArgs a{dh.data(),dl.data(),dp.data(),out.data(),db.data(),scratch.data(),nullptr,h.size(),lengths.size(),protected_rows.size(),out.count,db.count,scratch.count,limit,batch,n};
    check(cnot103_cuda_launch_one_excess_structure(&a)==0,"launch");const cudaError_t error=cudaDeviceSynchronize();
    if(fatal){check(error==cudaErrorLaunchFailure,"expected invalid-history trap");std::printf("PASS expected %s history trap: %s\n",fatal,cudaGetErrorString(error));return;}
    cu(error);auto got=out.read();auto bitsets=db.read();scratch.read();
    for(U64 i=0;i<got.size();++i)if(got[i]!=expected_output[i]){std::fprintf(stderr,"n%u budget%llu row%llu meta%llu got%u expected%u\n",n,(unsigned long long)limit,(unsigned long long)(i/l.meta_count),(unsigned long long)(i%l.meta_count),got[i],expected_output[i]);std::exit(1);}
    check(bitsets==expected_bits,"full source graphs/scopes");check(dh.read()==h&&dl.read()==lengths&&dp.read()==protected_rows,"input preservation");
    for(U32 field=0;field<6;++field){auto short_args=a;U64* extents[]={&short_args.history_words,&short_args.length_elements,&short_args.protected_elements,&short_args.output_elements,&short_args.bitset_words,&short_args.scratch_words};--*extents[field];check(cnot103_cuda_launch_one_excess_structure(&short_args)==2,"short extent");}
    auto bad=a;bad.batch=0;check(cnot103_cuda_launch_one_excess_structure(&bad)==1,"empty batch");bad=a;bad.max_entries=0;check(cnot103_cuda_launch_one_excess_structure(&bad)==1,"zero budget");bad=a;bad.dimension=1;check(cnot103_cuda_launch_one_excess_structure(&bad)==1,"invalid dimension");
}

__global__ void powers(const U64* cases,U32* fits,U32 count){U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){U64 value=0;fits[i]=cnot103_one_excess::checked_power(cases[3*i],static_cast<U32>(cases[3*i+1]),cases[3*i+2],value);}}
static U32 check_powers(){
    std::vector<U64> cases;std::vector<U32> expected;
    for(U64 base:{0ULL,1ULL,2ULL,3ULL,65ULL,144ULL,16777217ULL,~0ULL})for(U32 exponent:{0U,1U,2U,3U,7U,16U,63U,64U,65U,128U})for(U64 limit:{0ULL,1ULL,1024ULL,~0ULL}){
        bool fits=true;if(exponent && base){U64 room=limit;for(U32 i=0;i<exponent;++i){room/=base;if(!room){fits=false;break;}}}
        cases.insert(cases.end(),{base,exponent,limit});expected.push_back(fits);
    }
    Buffer<U64> input(cases.size());Buffer<U32> output(expected.size());input.load(cases);
    powers<<<(expected.size()+127)/128,128>>>(input.data(),output.data(),expected.size());cu(cudaDeviceSynchronize());
    check(output.read()==expected,"usize power/overflow semantics");check(input.read()==cases,"power input preservation");return expected.size();
}

int main(int argc,char** argv){
    if(argc==2){U32 n=!std::strcmp(argv[1],"padding")?65:3;run({{from_word(n,{{1,0},{2,1},{1,0}}),{}}},1024,argv[1]);return 0;}
    const U32 power_cases=check_powers();std::vector<Case> small;
    for(U32 code=0;code<512;++code){U32 rows[3]={code&7,(code>>3)&7,(code>>6)&7};std::set<U32> truth;
        for(U32 mask=0;mask<8;++mask){U32 value=0;for(U32 r=0;r<3;++r)if(mask>>r&1)value^=rows[r];truth.insert(value);}if(truth.size()!=8)continue;
        for(U32 target=0;target<3;++target)if(rows[target]!=(1U<<target))for(U32 middle=1;middle<8;++middle)if(middle!=rows[target]&&middle!=(1U<<target)){
            Histories h(3);for(U32 r=0;r<3;++r){h[r].push_back({rows[r]});if(r==target)h[r].push_back({middle});if(rows[r]!=(1U<<r))h[r].push_back({U64(1)<<r});}small.push_back({h,{}});}}
    for(U64 limit:{1ULL,64ULL,~0ULL})run(small,limit);
    U64 seed=0xf08a615;auto next=[&](){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;};
    for(U32 n:{5U,8U}){std::vector<Case> cases;for(U32 sample=0;sample<256;++sample){U32 twice=next()%n;std::vector<U32> targets;
            for(U32 r=0;r<n;++r)targets.push_back(r);targets.push_back(twice);for(U32 i=targets.size()-1;i>0;--i)std::swap(targets[i],targets[next()%(i+1)]);
            std::vector<Pair> word;for(U32 t:targets)word.emplace_back((t+1+next()%(n-1))%n,t);cases.push_back({from_word(n,word),{twice,U32(next()%n)}});}
        for(U64 limit:{1ULL,64ULL,1024ULL,~0ULL})run(cases,limit);}
    for(U32 n:{2U,65U,144U}){std::vector<Pair> word;for(U32 t=0;t<n-1;++t)word.emplace_back(t+1,t);word.emplace_back(n-1,0);
        auto h=from_word(n,word);std::vector<Case> cases{{from_word(n,{}),{}},{h,{}}};h[0].erase(h[0].begin());cases.push_back({h,{0}});
        if(n>=3){auto x=from_word(n,{{0,n-1},{n/2,0},{0,n-1}});cases.push_back({x,{n-1}});x[n-1].erase(x[n-1].begin());cases.push_back({x,{n-1}});}
        for(U64 limit:{1ULL,64ULL,1024ULL,~0ULL})run(cases,limit);}
    check(metrics.empty&&metrics.cycles&&metrics.duplicates&&metrics.multiple&&metrics.protected_orders&&metrics.fallback_orders&&metrics.admitted&&metrics.limited,"branch coverage");
    std::printf("PASS one-excess structure: %llu rows, %llu factors, %llu empty/%llu cycles, %llu single-term duplicates, %llu multi-term factors, %llu protected/%llu fallback orders, %llu admitted/%llu limited preflights\n",(unsigned long long)metrics.rows,(unsigned long long)metrics.factors,(unsigned long long)metrics.empty,(unsigned long long)metrics.cycles,(unsigned long long)metrics.duplicates,(unsigned long long)metrics.multiple,(unsigned long long)metrics.protected_orders,(unsigned long long)metrics.fallback_orders,(unsigned long long)metrics.admitted,(unsigned long long)metrics.limited);
    std::printf("PASS %u independent usize-power budget cases\n",power_cases);
}
