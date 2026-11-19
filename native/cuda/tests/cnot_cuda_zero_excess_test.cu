#include "cnot103/cnot_cuda.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using U8=std::uint8_t;using U32=std::uint32_t;using U64=std::uint64_t;using I64=std::int64_t;
using Matrix=std::vector<U64>;using Gate=std::array<U32,2>;
void expect(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
template<typename T>struct Buffer{
    T* p=nullptr;U64 n;explicit Buffer(U64 count):n(count){ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0xa5,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void put(const std::vector<T>& v){expect(v.size()==n,"upload extent");ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get()const{std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
Matrix identity(U32 n){Matrix m(U64(n)*((n+63)/64));for(U32 i=0;i<n;++i)m[U64(i)*((n+63)/64)+i/64]=U64{1}<<(i%64);return m;}
void apply(Matrix& m,U32 n,Gate g){U32 w=(n+63)/64;for(U32 i=0;i<w;++i)m[U64(g[1])*w+i]^=m[U64(g[0])*w+i];}
bool bit(const Matrix& m,U32 n,U32 row,U32 col){return (m[U64(row)*((n+63)/64)+col/64]>>(col%64))&1;}
std::vector<U32> changed(const Matrix& m,U32 n){const auto id=identity(n);const U32 w=(n+63)/64;std::vector<U32> x;
    for(U32 i=0;i<n;++i)if(!std::equal(m.begin()+U64(i)*w,m.begin()+U64(i+1)*w,id.begin()+U64(i)*w))x.push_back(i);return x;
}
// Complete group BFS: these distances use every physical CNOT, no pair theorem.
std::map<Matrix,U32> distances(U32 n){std::map<Matrix,U32> out;std::queue<Matrix> todo;auto id=identity(n);out[id]=0;todo.push(id);
    while(!todo.empty()){auto m=todo.front();todo.pop();const U32 next=out.at(m)+1;
        for(U32 c=0;c<n;++c)for(U32 t=0;t<n;++t)if(c!=t){auto child=m;apply(child,n,{c,t});if(out.emplace(child,next).second)todo.push(child);}}
    return out;
}
struct Reference {
    std::vector<U32> changed,missing,order,parents;
    bool exact;
};
Reference reference(const Matrix& m,U32 n){
    const U32 w=(n+63)/64;const auto id=identity(n);Reference ref;ref.changed=changed(m,n);ref.parents.resize(n*2);
    std::map<Matrix,U32> initial,final_rows;
    for(U32 r=0;r<n;++r){initial[Matrix(m.begin()+U64(r)*w,m.begin()+U64(r+1)*w)]=r;final_rows[Matrix(id.begin()+U64(r)*w,id.begin()+U64(r+1)*w)]=r;ref.parents[r*2]=n;ref.parents[r*2+1]=2;}
    std::vector<std::set<U32>> edges(n);std::vector<U32> indegree(n);
    for(auto t:ref.changed){Matrix diff(m.begin()+U64(t)*w,m.begin()+U64(t+1)*w);diff[t/64]^=U64{1}<<(t%64);U32 control=n,mode=2;
        auto first=initial.find(diff),last=final_rows.find(diff);
        if(first!=initial.end()&&first->second!=t){control=first->second;mode=0;}
        else if(last!=final_rows.end()&&last->second!=t){control=last->second;mode=1;}
        ref.parents[t*2]=control;ref.parents[t*2+1]=mode;if(control==n){ref.missing.push_back(t);continue;}
        if(std::find(ref.changed.begin(),ref.changed.end(),control)!=ref.changed.end()){
            U32 from=mode?control:t,to=mode?t:control;if(edges[from].insert(to).second)++indegree[to];
        }
    }
    ref.exact=ref.missing.empty();if(!ref.exact)return ref;
    std::queue<U32> fifo;for(auto r:ref.changed)if(!indegree[r])fifo.push(r);
    while(!fifo.empty()){const U32 r=fifo.front();fifo.pop();ref.order.push_back(r);for(auto next:edges[r])if(!--indegree[next])fifo.push(next);}
    expect(ref.order.size()==ref.changed.size(),"reference acyclicity");return ref;
}
U64 rows_checked=0,exact_words=0,negative_witnesses=0,bfs_labels=0,mutations=0;
struct Case{
    U32 n,b,w;std::vector<Matrix> matrices;
    Buffer<U64> states,scratch,headers;Buffer<U32> graph,parents,changed_rows,missing_rows,order,gates;
    Buffer<I64> lengths;Buffer<U8> verified,status;Buffer<float> feature;Cnot103ZeroExcessArgs a{};
    Case(U32 dimension,const std::vector<Matrix>& input):n(dimension),b(input.size()),w((n+63)/64),matrices(input),
        states(U64(b)*n*w),scratch(states.n),headers(U64(b)*3),graph(U64(b)*n*3),parents(U64(b)*n*2),changed_rows(U64(b)*n),missing_rows(U64(b)*n),order(U64(b)*n),gates(U64(b)*n*2),lengths(b),verified(b),status(b),feature(U64(b)*3){
        Matrix packed;for(const auto& m:matrices)packed.insert(packed.end(),m.begin(),m.end());states.put(packed);
        a.states=states.p;a.scratch=scratch.p;a.graph=graph.p;a.parents=parents.p;a.changed_rows=changed_rows.p;a.missing_rows=missing_rows.p;a.update_order=order.p;a.gates=gates.p;a.headers=headers.p;a.lengths=lengths.p;a.verified=verified.p;a.status=status.p;a.feature=feature.p;
        a.state_words=states.n;a.scratch_words=scratch.n;a.graph_elements=graph.n;a.parent_elements=parents.n;a.changed_elements=changed_rows.n;a.missing_elements=missing_rows.n;a.order_elements=order.n;a.gate_elements=gates.n;a.header_elements=headers.n;a.feature_elements=feature.n;a.batch_elements=b;a.batch=b;a.dimension=n;a.words_per_row=w;
    }
    void run(){expect(!cnot103_cuda_launch_zero_excess(&a),"zero-excess launch");ck(cudaDeviceSynchronize());rows_checked+=b;}
    void compare(const std::map<Matrix,U32>* bfs=nullptr){
        const auto codes=status.get(),proof=verified.get();const auto head=headers.get();const auto len=lengths.get();const auto values=feature.get();
        const auto par=parents.get(),chg=changed_rows.get(),miss=missing_rows.get(),ord=order.get(),word=gates.get();
        for(U32 sample=0;sample<b;++sample){const auto& m=matrices[sample];const auto ref=reference(m,n);const U64 base=U64(sample)*n;
            expect(codes[sample]==1&&proof[sample]==1,"exact and negative are computed verified results");
            expect(head[U64(sample)*3]==(ref.exact?1:2)&&head[U64(sample)*3+1]==ref.changed.size()&&head[U64(sample)*3+2]==ref.missing.size(),"header class and full counts");
            expect(values[U64(sample)*3]==float(ref.exact)&&values[U64(sample)*3+1]==float(!ref.exact)&&!values[U64(sample)*3+2],"source three-channel meaning");
            for(U32 row=0;row<n;++row){expect(chg[base+row]==(row<ref.changed.size()?ref.changed[row]:n),"complete changed list and padding");expect(miss[base+row]==(row<ref.missing.size()?ref.missing[row]:n),"complete negative witness and padding");expect(ord[base+row]==(row<ref.order.size()?ref.order[row]:n),"source FIFO order and padding");for(U32 j=0;j<2;++j)expect(par[(base+row)*2+j]==ref.parents[row*2+j],"source parent priority and mode");}
            expect(len[sample]==I64(ref.exact?ref.changed.size():0),"encoder length");
            if(ref.exact){auto replay=identity(n);for(U32 i=0;i<ref.order.size();++i){const U32 t=ref.order[ref.order.size()-1-i];const Gate g{word[(base+i)*2],word[(base+i)*2+1]};expect(g[0]==ref.parents[t*2]&&g[1]==t,"complete canonical encoder word");apply(replay,n,g);}expect(replay==m,"full independently replayed endpoint");++exact_words;}
            else{expect(!ref.missing.empty(),"negative witness nonempty");++negative_witnesses;}
            for(U32 i=ref.exact?ref.order.size():0;i<n;++i)expect(!word[(base+i)*2]&&!word[(base+i)*2+1],"zero unused encoder");
            if(bfs){const U32 distance=bfs->at(m);expect(ref.exact?distance==ref.changed.size():distance>ref.changed.size(),"independent complete group BFS classification");++bfs_labels;}
        }
    }
};
void corruptions(const std::string& fault){
    Case c(4,{Matrix{2,3,6,12},Matrix{2,1,4,8},identity(4),Matrix{1,3,4,8}});c.run();c.compare();
    const auto h=c.headers.get();const auto par=c.parents.get(),chg=c.changed_rows.get(),miss=c.missing_rows.get(),ord=c.order.get(),word=c.gates.get();const auto len=c.lengths.get();const auto codes=c.status.get();
    expect(std::vector<U32>(ord.begin(),ord.begin()+4)==std::vector<U32>({2,0,3,1}),"literal FIFO differs from global sorting");
    expect(par[4]==0&&par[5]==0,"literal initial row takes priority over final row1");
    auto restore=[&](){c.headers.put(h);c.parents.put(par);c.changed_rows.put(chg);c.missing_rows.put(miss);c.order.put(ord);c.gates.put(word);c.lengths.put(len);c.status.put(codes);};
    auto check_bad=[&](U32 target){expect(!cnot103_cuda_zero_excess_check(&c.a),"corruption checker launch");ck(cudaDeviceSynchronize());const auto result=c.verified.get();for(U32 sample=0;sample<c.b;++sample)expect(result[sample]==U8(sample!=target),"corruption admission");++mutations;};
    for(U32 sample=0;sample<4;++sample){
        for(U32 j=0;j<3;++j){auto x=h;++x[sample*3+j];c.headers.put(x);check_bad(sample);restore();}
        for(U32 j=0;j<8;++j){auto x=par;x[sample*8+j]^=1;c.parents.put(x);check_bad(sample);restore();}
        for(U32 j=0;j<4;++j){auto x=chg;x[sample*4+j]^=1;c.changed_rows.put(x);check_bad(sample);restore();x=miss;x[sample*4+j]^=1;c.missing_rows.put(x);check_bad(sample);restore();x=ord;x[sample*4+j]^=1;c.order.put(x);check_bad(sample);restore();}
    }
    if(fault=="exact"||fault=="negative"){
        const U32 target=fault=="exact"?0:1;auto x=h;x[target*3]=3-x[target*3];c.headers.put(x);check_bad(target);expect(!cnot103_cuda_zero_excess_publish(&c.a),"fatal publisher launch");
        const auto e=cudaDeviceSynchronize();expect(e==cudaErrorIllegalInstruction||e==cudaErrorLaunchFailure,"rejected witness did not trap");std::printf("PASS expected %s witness trap: %s\n",fault.c_str(),cudaGetErrorString(e));return;
    }
    // A valid alternative topological word has the same endpoint, but source
    // FIFO order is part of the certificate contract and must be rejected.
    auto alternative=word;std::swap(alternative[0],alternative[2]);std::swap(alternative[1],alternative[3]);auto alternative_order=ord;std::swap(alternative_order[2],alternative_order[3]);
    auto replay=identity(4);for(U32 i=0;i<4;++i)apply(replay,4,{alternative[i*2],alternative[i*2+1]});expect(replay==c.matrices[0],"alternative endpoint equality");
    c.gates.put(alternative);c.order.put(alternative_order);check_bad(0);restore();
    for(auto bad:std::vector<Gate>{{4,0},{0,0},{3,2}}){auto x=word;x[0]=bad[0];x[1]=bad[1];c.gates.put(x);check_bad(0);restore();}
    auto dirty=word;dirty[16]=1;c.gates.put(dirty);check_bad(2);restore();
    for(I64 bad:{I64(-1),I64(0),I64(5)}){auto x=len;x[0]=bad;c.lengths.put(x);check_bad(0);restore();}
    auto wrong=codes;wrong[1]=3;c.status.put(wrong);check_bad(1);restore();
    expect(!cnot103_cuda_zero_excess_check(&c.a),"restored checker");expect(!cnot103_cuda_zero_excess_publish(&c.a),"restored publisher");ck(cudaDeviceSynchronize());c.compare();
    expect(cnot103_cuda_launch_zero_excess(nullptr)==1,"null args");using Member=U64 Cnot103ZeroExcessArgs::*;
    for(Member member:{&Cnot103ZeroExcessArgs::state_words,&Cnot103ZeroExcessArgs::scratch_words,&Cnot103ZeroExcessArgs::graph_elements,&Cnot103ZeroExcessArgs::parent_elements,&Cnot103ZeroExcessArgs::changed_elements,&Cnot103ZeroExcessArgs::missing_elements,&Cnot103ZeroExcessArgs::order_elements,&Cnot103ZeroExcessArgs::gate_elements,&Cnot103ZeroExcessArgs::header_elements,&Cnot103ZeroExcessArgs::feature_elements,&Cnot103ZeroExcessArgs::batch_elements}){auto a=c.a;--(a.*member);expect(cnot103_cuda_launch_zero_excess(&a)==2,"short extent admitted");}
    auto a=c.a;a.words_per_row=0;expect(cnot103_cuda_launch_zero_excess(&a)==1,"wrong row width");a=c.a;a.batch=0;expect(cnot103_cuda_launch_zero_excess(&a)==1,"empty batch");
}
int main(int argc,char** argv){try{
    const std::string fault=argc>1?argv[1]:"";if(fault=="exact"||fault=="negative"){corruptions(fault);return 0;}
    if(fault=="cycle"||fault=="padding"){
        const U32 n=fault=="cycle"?2:65;auto m=fault=="cycle"?Matrix{3,3}:identity(n);if(fault=="padding")m[1]|=U64{1}<<63;
        Case c(n,{m});expect(!cnot103_cuda_launch_zero_excess(&c.a),"invalid-state launch");const auto e=cudaDeviceSynchronize();expect(e==cudaErrorIllegalInstruction||e==cudaErrorLaunchFailure,"invalid state did not trap");std::printf("PASS expected %s trap: %s\n",fault.c_str(),cudaGetErrorString(e));return 0;
    }
    for(U32 n:{2,3,4}){auto bfs=distances(n);expect(bfs.size()==(n==2?6:n==3?168:20160),"complete group cardinality");std::vector<Matrix> matrices;for(const auto& entry:bfs)matrices.push_back(entry.first);Case c(n,matrices);c.run();c.compare(&bfs);}
    for(U32 n:{65,144}){auto base=identity(n),chain=base,reverse=base,swap=base;for(U32 t=1;t<n;++t)apply(chain,n,{(t*17+3)%t,t});for(U32 t=n-1;t>0;--t)apply(reverse,n,{t,t-1});for(auto g:std::vector<Gate>{{0,n-1},{n-1,0},{0,n-1}})apply(swap,n,g);Case c(n,{base,chain,reverse,swap});c.run();c.compare();}
    corruptions("");ck(cudaDeviceReset());std::printf("PASS zero excess: %llu rows, %llu exact physical words, %llu complete negative witnesses, %llu independent BFS labels, %llu corruption rejections\n",(unsigned long long)rows_checked,(unsigned long long)exact_words,(unsigned long long)negative_witnesses,(unsigned long long)bfs_labels,(unsigned long long)mutations);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
