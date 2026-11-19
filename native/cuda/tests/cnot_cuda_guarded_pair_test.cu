#include "cnot103/cnot_cuda.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <queue>
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
U64 rows_checked=0,words_checked=0,bfs_labels=0,mutations_checked=0;
struct Case {
    U32 n,b,w;U64 q;std::vector<Matrix> matrices;
    Buffer<U64> states,scratch,graph,certificate;Buffer<U32> gates;Buffer<I64> lengths;
    Buffer<U8> status,verified,valid;Buffer<float> f02,f06,incidence;Cnot103GuardedPairArgs a{};
    Case(U32 dimension,const std::vector<Matrix>& inputs):n(dimension),b(inputs.size()),w((n+63)/64),q(n+1),matrices(inputs),
        states(U64(b)*n*w),scratch(states.n),graph(U64(b)*n*2),certificate(U64(b)*16),gates(U64(b)*q*2),lengths(b),status(b),verified(b),valid(b),f02(b),f06(U64(b)*6),incidence(U64(b)*n*3){
        std::vector<U64> packed;for(const auto& m:matrices)packed.insert(packed.end(),m.begin(),m.end());states.put(packed);
        a.states=states.p;a.scratch=scratch.p;a.graph=graph.p;a.gates=gates.p;a.lengths=lengths.p;a.certificate=certificate.p;a.status=status.p;a.verified=verified.p;a.f02=f02.p;a.f06=f06.p;a.valid=valid.p;a.incidence=incidence.p;
        a.word_capacity=q;a.state_words=states.n;a.scratch_words=scratch.n;a.graph_words=graph.n;a.gate_elements=gates.n;a.certificate_words=certificate.n;a.f06_elements=f06.n;a.incidence_elements=incidence.n;a.batch_elements=b;
        a.batch=b;a.dimension=n;a.words_per_row=w;
    }
    void run(){expect(!cnot103_cuda_launch_guarded_pair(&a),"pair launch");ck(cudaDeviceSynchronize());rows_checked+=b;}
    void compare(const std::map<Matrix,U32>* bfs=nullptr){
        const auto codes=status.get(),proofs=verified.get(),tokens=valid.get();const auto meta=certificate.get();const auto word=gates.get();const auto len=lengths.get();const auto low=f02.get(),pair=f06.get(),roles=incidence.get();
        for(U32 sample=0;sample<b;++sample){const auto& m=matrices[sample];const auto rows=changed(m,n);const bool exact=rows.size()<=2;
            expect(codes[sample]==(exact?1:3)&&proofs[sample]==1,"scope status and independent proof");
            if(!exact){expect(!len[sample]&&!tokens[sample]&&!low[sample],"unavailable scalar clearing");
                for(U32 j=0;j<16;++j)expect(!meta[U64(sample)*16+j],"unavailable certificate clearing");
                for(U64 j=0;j<q*2;++j)expect(!word[U64(sample)*q*2+j],"unavailable word clearing");
                for(U32 j=0;j<6;++j)expect(!pair[U64(sample)*6+j],"unavailable feature clearing");
                for(U64 j=0;j<U64(n)*3;++j)expect(!roles[U64(sample)*n*3+j],"unavailable incidence clearing");continue;
            }
            auto active=rows;for(U32 row=0;active.size()<2;++row)if(std::find(active.begin(),active.end(),row)==active.end())active.push_back(row);
            U32 f[2]={0,0};for(U32 r=0;r<2;++r)for(U32 c=0;c<2;++c)f[r]|=U32(bit(m,n,active[r],active[c]))<<c;
            U32 nonzero=0,shared=0;std::vector<U32> shared_sources;
            for(U32 c=0;c<n;++c)if(c!=active[0]&&c!=active[1]){
                // Solve two scalar equations rather than the CUDA inverse.
                bool found=false;for(U32 v=0;v<4;++v){bool good=true;for(U32 r=0;r<2;++r)good&=(bool(__builtin_popcount(f[r]&v)%2)==bit(m,n,active[r],c));
                    if(good){found=true;nonzero+=v!=0;if(v==3){++shared;shared_sources.push_back(c);}break;}}
                expect(found,"reference frame solution");
            }
            // Independent transitive closure of the complete support graph.
            std::vector<std::vector<bool>> reach(n,std::vector<bool>(n));
            for(U32 i=0;i<n;++i)for(U32 j=0;j<n;++j)reach[i][j]=i==j||bit(m,n,i,j)||bit(m,n,j,i);
            for(U32 k=0;k<n;++k)for(U32 i=0;i<n;++i)if(reach[i][k])for(U32 j=0;j<n;++j)reach[i][j]=reach[i][j]||reach[k][j];
            U32 components=0,zeros=0;for(U32 i=0;i<n;++i){bool first=true;for(U32 j=0;j<i;++j)first&=!reach[i][j];components+=first;zeros+=!bit(m,n,i,i);}
            const U32 connectivity=n-components;const bool id=f[0]==1&&f[1]==2,motion=!id||shared>2;
            const U32 kind=!id?0:shared>=2?1:2;const U64 lower=U64(connectivity)+(kind==0?zeros:kind==1?1:0);
            const U64 expected[16]={rows.size(),active[0],active[1],f[0],f[1],nonzero,shared,U64(motion),connectivity,zeros,lower,kind,
                shared_sources.size()?shared_sources[0]:n,shared_sources.size()>1?shared_sources[1]:n,rows.size()?rows[0]:n,rows.size()>1?rows[1]:n};
            for(U32 j=0;j<16;++j)expect(meta[U64(sample)*16+j]==expected[j],"all certificate fields");
            expect(len[sample]>=0&&U64(len[sample])==lower&&lower<=q,"exact distance and bound");
            auto replay=identity(n);for(U64 i=0;i<lower;++i){Gate g{word[(U64(sample)*q+i)*2],word[(U64(sample)*q+i)*2+1]};expect(g[0]<n&&g[1]<n&&g[0]!=g[1],"physical gate validity");apply(replay,n,g);}expect(replay==m,"full encoder endpoint");
            for(U64 i=lower;i>0;--i)apply(replay,n,{word[(U64(sample)*q+i-1)*2],word[(U64(sample)*q+i-1)*2+1]});expect(replay==identity(n),"full decoder endpoint");++words_checked;
            if(bfs){expect(bfs->at(m)==lower,"complete group BFS optimal distance");++bfs_labels;}
            for(U64 i=lower*2;i<q*2;++i)expect(!word[U64(sample)*q*2+i],"zero word padding");
            const float values[6]={float(f[0]&1),float(f[0]>>1),float(f[1]&1),float(f[1]>>1),float(nonzero),float(shared)};
            expect(low[sample]==float(connectivity+zeros)&&tokens[sample]==1,"F02 and token");
            for(U32 i=0;i<6;++i)expect(pair[U64(sample)*6+i]==values[i],"F06 six channels");
            for(U32 i=0;i<n;++i)for(U32 r=0;r<3;++r)expect(roles[(U64(sample)*n+i)*3+r]==float(r==0?i==active[0]:r==1?i==active[1]:i!=active[0]&&i!=active[1]),"all physical roles including zero external columns");
        }
    }
};
Matrix fixture(U32 n,std::array<U32,2> active,std::array<U32,2> f,U32 mask){
    auto m=identity(n);const U32 w=(n+63)/64;for(U32 r=0;r<2;++r){std::fill(m.begin()+U64(active[r])*w,m.begin()+U64(active[r]+1)*w,0);
        for(U32 c=0;c<2;++c)if((f[r]>>c)&1)m[U64(active[r])*w+active[c]/64]|=U64{1}<<(active[c]%64);
        U32 offset=0;for(U32 col=0;col<n;++col)if(col!=active[0]&&col!=active[1]){const U32 v=n<=6?(mask>>(offset*2))&3:(col*7+mask)%4;++offset;
            if(__builtin_popcount(f[r]&v)%2)m[U64(active[r])*w+col/64]|=U64{1}<<(col%64);}}
    return m;
}
void corruption(const std::string& fault){
    Case c(3,{Matrix{2,1,4},Matrix{2,4,1}});c.run();c.compare();
    const auto cert=c.certificate.get();const auto word=c.gates.get();const auto len=c.lengths.get();const auto codes=c.status.get();
    auto check_bad=[&](U32 sample){expect(!cnot103_cuda_guarded_pair_check(&c.a),"corruption check launch");ck(cudaDeviceSynchronize());const auto got=c.verified.get();expect(got[sample]==0&&got[1-sample]==1,"corruption must be rejected independently");++mutations_checked;};
    auto restore=[&](){c.certificate.put(cert);c.gates.put(word);c.lengths.put(len);c.status.put(codes);};
    for(U32 j=0;j<16;++j){auto x=cert;x[j]^=1;c.certificate.put(x);check_bad(0);restore();}
    if(fault=="certificate"){
        auto x=cert;++x[10];c.certificate.put(x);check_bad(0);expect(!cnot103_cuda_guarded_pair_publish(&c.a),"fatal publish launch");
        const auto e=cudaDeviceSynchronize();expect(e==cudaErrorIllegalInstruction||e==cudaErrorLaunchFailure,"rejected proof did not trap");std::printf("PASS expected certificate trap: %s\n",cudaGetErrorString(e));return;
    }
    for(auto bad:std::vector<Gate>{{3,0},{0,0},{2,1}}){auto x=word;x[0]=bad[0];x[1]=bad[1];c.gates.put(x);check_bad(0);restore();}
    // A different valid shortest word for the swap still violates source order.
    auto alternate=word;for(U32 i=0;i<3;++i){alternate[2*i]=1-(i%2);alternate[2*i+1]=i%2;}c.gates.put(alternate);check_bad(0);restore();
    auto padding=word;padding[6]=1;c.gates.put(padding);check_bad(0);restore();
    for(I64 bad:{I64(-1),I64(0),I64(5)}){auto x=len;x[0]=bad;c.lengths.put(x);check_bad(0);restore();}
    auto wrong_status=codes;wrong_status[0]=3;c.status.put(wrong_status);check_bad(0);restore();
    wrong_status=codes;wrong_status[1]=1;c.status.put(wrong_status);check_bad(1);restore();
    auto dirty=cert;dirty[16]=1;c.certificate.put(dirty);check_bad(1);restore();
    expect(!cnot103_cuda_guarded_pair_check(&c.a),"restored check");expect(!cnot103_cuda_guarded_pair_publish(&c.a),"restored publish");ck(cudaDeviceSynchronize());c.compare();
    expect(cnot103_cuda_launch_guarded_pair(nullptr)==1,"null args");
    using Member=U64 Cnot103GuardedPairArgs::*;
    for(Member member:{&Cnot103GuardedPairArgs::state_words,&Cnot103GuardedPairArgs::scratch_words,&Cnot103GuardedPairArgs::graph_words,&Cnot103GuardedPairArgs::gate_elements,&Cnot103GuardedPairArgs::certificate_words,&Cnot103GuardedPairArgs::f06_elements,&Cnot103GuardedPairArgs::incidence_elements,&Cnot103GuardedPairArgs::batch_elements,&Cnot103GuardedPairArgs::word_capacity}){auto a=c.a;--(a.*member);expect(cnot103_cuda_launch_guarded_pair(&a)==2,"short extent accepted");}
    auto a=c.a;a.word_capacity=~U64{0};expect(cnot103_cuda_launch_guarded_pair(&a)==2,"overflow capacity");
    a=c.a;a.words_per_row=0;expect(cnot103_cuda_launch_guarded_pair(&a)==1,"invalid packed width");
}
int main(int argc,char** argv){try{
    const std::string fault=argc>1?argv[1]:"";
    if(fault=="certificate"){corruption(fault);return 0;}
    if(fault=="padding"||fault=="singular"){
        auto m=identity(65);if(fault=="padding")m[1]|=U64{1}<<63;else m[0]=0;
        Case c(65,{m});expect(!cnot103_cuda_launch_guarded_pair(&c.a),"malformed-state launch");const auto e=cudaDeviceSynchronize();
        expect(e==cudaErrorIllegalInstruction||e==cudaErrorLaunchFailure,"invalid state did not trap");std::printf("PASS expected %s trap: %s\n",fault.c_str(),cudaGetErrorString(e));return 0;
    }
    for(U32 n:{2,3,4}){auto all=distances(n);expect(all.size()==(n==2?6:n==3?168:20160),"complete GL BFS cardinality");std::vector<Matrix> states;U32 outside=0;
        for(const auto& entry:all)if(changed(entry.first,n).size()<=2||outside++<16)states.push_back(entry.first);
        Case c(n,states);c.run();c.compare(&all);
    }
    std::vector<Matrix> small;for(auto f:std::vector<std::array<U32,2>>{{1,2},{1,3},{3,2},{2,3},{3,1},{2,1}})for(U32 mask=0;mask<256;++mask)small.push_back(fixture(6,{0,1},f,mask));
    {Case c(6,small);c.run();c.compare();}
    for(U32 n:{65,144}){std::vector<Matrix> moved{identity(n)};for(auto f:std::vector<std::array<U32,2>>{{1,2},{1,3},{3,2},{2,3},{3,1},{2,1}})moved.push_back(fixture(n,{n-1,0},f,7));Case c(n,moved);c.run();c.compare();}
    corruption("");ck(cudaDeviceReset());std::printf("PASS guarded pair: %llu rows, %llu full physical words, %llu independent BFS distance labels, %llu corruption rejections\n",(unsigned long long)rows_checked,(unsigned long long)words_checked,(unsigned long long)bfs_labels,(unsigned long long)mutations_checked);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
