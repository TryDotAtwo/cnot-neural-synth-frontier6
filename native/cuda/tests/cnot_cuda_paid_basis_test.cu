// Standalone CUDA acceptance without Torch or the Rust source implementation.
// Golden constants come from paid_shear_reference_v9.json. Tiny exhaustive
// expected words use independent truth-map BFS; all endpoints use CPU replay.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
using U8=std::uint8_t;using U32=std::uint32_t;using U64=std::uint64_t;using I64=std::int64_t;
using Gate=std::array<U32,2>;
void expect(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
template<typename T>struct Buffer{
    T* p=nullptr;U64 n;explicit Buffer(U64 count):n(count){ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0xa5,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void put(const std::vector<T>& v){expect(v.size()==n,"upload extent");ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get()const{std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
template<typename T>void zero(const Buffer<T>& b){for(auto v:b.get())expect(v==0,"failure/unused payload must be zero");}
std::vector<U64> identity(U32 n,U32 b=1){U32 w=(n+63)/64;std::vector<U64> x(U64(b)*n*w);for(U32 k=0;k<b;++k)for(U32 r=0;r<n;++r)x[(U64(k)*n+r)*w+r/64]=U64{1}<<(r%64);return x;}
void gate(std::vector<U64>& x,U32 n,Gate g){U32 w=(n+63)/64;expect(g[0]<n&&g[1]<n&&g[0]!=g[1],"physical gate");for(U32 k=0;k<w;++k)x[U64(g[1])*w+k]^=x[U64(g[0])*w+k];}
std::vector<Gate> slice(const std::vector<U32>& x,U64 offset,U64 count){std::vector<Gate> out;for(U64 i=0;i<count;++i)out.push_back({x[offset+2*i],x[offset+2*i+1]});return out;}
struct Atlases{
    std::array<std::unique_ptr<Buffer<U32>>,6> tables;
    Atlases(){for(U32 s=0;s<2;++s)for(U32 w=2;w<=4;++w){U32 i=s*3+w-2;tables[i]=std::make_unique<Buffer<U32>>((U64{1}<<(w*w))*(6+(1U<<w))+1);Cnot103PaidAtlasPrepareArgs a{tables[i]->p,nullptr,tables[i]->n,w,s};expect(!cnot103_cuda_prepare_paid_atlas(&a),"atlas prepare");}}
};
struct Shear{
    U32 n,b,s,t,w;U64 c,p,l;
    Buffer<U64> input,endpoint,replay,scratch,stats,moves;
    Buffer<U32> sources,targets,prep,restore,core,word,cancels;
    Buffer<I64> lengths;Buffer<U8> block,sbasis,tbasis,status,verified;
    Cnot103PaidShearArgs a{};
    Shear(Atlases& atlas,U32 dim,U32 batch,const std::vector<U32>& src,const std::vector<U32>& tgt,const std::vector<U64>& states,U32 local=4,U32 sweeps=2):
        n(dim),b(batch),s(src.size()),t(tgt.size()),w((n+63)/64),c(U64(s)*t),p(std::max<U64>(1,c/2)),l(std::max(s,t)),
        input(U64(b)*n*w),endpoint(input.n),replay(input.n),scratch(U64(b)*(15*l+p+c)),stats(U64(b)*16),moves(U64(b)*p*11),
        sources(s),targets(t),prep(U64(b)*p*2),restore(prep.n),core(U64(b)*c*2),word(core.n),cancels(prep.n),
        lengths(b),block(U64(b)*c),sbasis(U64(b)*s*s),tbasis(U64(b)*t*t),status(b),verified(b){
        input.put(states);sources.put(src);targets.put(tgt);
        a.states=input.p;a.sources=sources.p;a.targets=targets.p;for(U32 i=0;i<6;++i){a.atlases[i]=atlas.tables[i]->p;a.atlas_words[i]=atlas.tables[i]->n;}
        a.block=block.p;a.source_basis=sbasis.p;a.target_basis=tbasis.p;a.preparation=prep.p;a.restoration=restore.p;a.core=core.p;a.word=word.p;
        a.cancellations=cancels.p;a.moves=moves.p;a.statistics=stats.p;a.lengths=lengths.p;a.endpoint=endpoint.p;a.replay=replay.p;a.scratch=scratch.p;a.status=status.p;a.verified=verified.p;
        a.state_words=input.n;a.block_elements=block.n;a.source_basis_elements=sbasis.n;a.target_basis_elements=tbasis.n;a.preparation_elements=prep.n;
        a.core_elements=core.n;a.move_elements=moves.n;a.stat_elements=stats.n;a.scratch_words=scratch.n;a.batch_elements=b;a.source_elements=s;a.target_elements=t;
        a.batch=b;a.dimension=n;a.words_per_row=w;a.source_count=s;a.target_count=t;a.local_width=local;a.sweeps=sweeps;
    }
    void run(){expect(!cnot103_cuda_launch_paid_shear(&a),"shear launch");ck(cudaDeviceSynchronize());}
    void check(){
        auto st=stats.get(),in=input.get(),end=endpoint.get();auto src=sources.get(),tgt=targets.get(),out=word.get(),pre=prep.get();
        auto lens=lengths.get();auto code=status.get(),actual_block=block.get(),sb=sbasis.get(),tb=tbasis.get();
        for(U32 batch=0;batch<b;++batch){
            expect(code[batch]==1,"verified compiler status");const U64* k=st.data()+batch*16;
            expect(U64(lens[batch])==k[1]&&k[1]<=k[0]&&k[15]==2*k[2]+k[5]&&k[15]<=k[0],"cost invariants");
            auto target=identity(n),replayed=identity(n);std::vector<U8> expected_block(c),source_basis(U64(s)*s),target_basis(U64(t)*t);
            for(U32 i=0;i<s;++i)source_basis[U64(i)*s+i]=1;for(U32 i=0;i<t;++i)target_basis[U64(i)*t+i]=1;
            for(U32 r=0;r<t;++r)for(U32 col=0;col<s;++col){U8 bit=(in[(U64(batch)*n+tgt[r])*w+src[col]/64]>>(src[col]%64))&1;expected_block[U64(r)*s+col]=bit;if(bit)gate(target,n,{src[col],tgt[r]});}
            for(auto g:slice(out,U64(batch)*c*2,k[1]))gate(replayed,n,g);
            expect(target==replayed,"independent full endpoint replay");expect(std::equal(target.begin(),target.end(),end.begin()+U64(batch)*n*w),"published endpoint");
            for(auto g:slice(pre,U64(batch)*p*2,k[2])){
                auto ci=std::find(src.begin(),src.end(),g[0]),ti=std::find(src.begin(),src.end(),g[1]);
                if(ci!=src.end()&&ti!=src.end()){U32 c0=ci-src.begin(),t0=ti-src.begin();for(U32 r=0;r<t;++r)expected_block[U64(r)*s+c0]^=expected_block[U64(r)*s+t0];for(U32 col=0;col<s;++col)source_basis[U64(t0)*s+col]^=source_basis[U64(c0)*s+col];}
                else {auto ct=std::find(tgt.begin(),tgt.end(),g[0]),tt=std::find(tgt.begin(),tgt.end(),g[1]);expect(ct!=tgt.end()&&tt!=tgt.end(),"basis gate stays within one side");U32 c0=ct-tgt.begin(),t0=tt-tgt.begin();for(U32 col=0;col<s;++col)expected_block[U64(t0)*s+col]^=expected_block[U64(c0)*s+col];for(U32 col=0;col<t;++col)target_basis[U64(t0)*t+col]^=target_basis[U64(c0)*t+col];}
            }
            expect(std::equal(expected_block.begin(),expected_block.end(),actual_block.begin()+U64(batch)*c),"transformed block independently reconstructed");
            expect(std::equal(source_basis.begin(),source_basis.end(),sb.begin()+U64(batch)*s*s),"source basis replay");expect(std::equal(target_basis.begin(),target_basis.end(),tb.begin()+U64(batch)*t*t),"target basis replay");
            for(U64 i=k[1]*2;i<c*2;++i)expect(out[U64(batch)*c*2+i]==0,"unused word slots");
        }
    }
};
U64 samples=0;
void golden(Atlases& atlas,U32 n,const std::vector<U32>& src,const std::vector<U32>& tgt){
    const U8 bits[24]={0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,0,0,1,1,1,0,1,0};
    auto in=identity(n);for(U32 r=0;r<6;++r)for(U32 c=0;c<4;++c)if(bits[r*4+c])gate(in,n,{src[c],tgt[r]});
    Shear x(atlas,n,1,src,tgt,in);x.run();x.check();
    std::vector<U32> wires=src;wires.insert(wires.end(),tgt.begin(),tgt.end());
    const std::vector<Gate> expected={{2,0},{2,4},{0,5},{0,6},{1,6},{3,6},{0,7},{1,7},{2,8},{3,8},{0,9},{2,0}};
    auto remap=expected;for(auto& g:remap){g[0]=wires[g[0]];g[1]=wires[g[1]];}
    expect(slice(x.word.get(),0,12)==remap,"pinned original Python full word");
    expect(x.stats.get()==std::vector<U64>({14,12,1,1,0,10,1,10,1,1,0,1,0,0,0,12}),"pinned golden all stats");
    expect(x.moves.get()[7]==14&&x.moves.get()[8]==12,"pinned accepted move costs");++samples;
    for(auto member:{&Cnot103PaidShearArgs::state_words,&Cnot103PaidShearArgs::block_elements,&Cnot103PaidShearArgs::source_basis_elements,&Cnot103PaidShearArgs::target_basis_elements,&Cnot103PaidShearArgs::preparation_elements,&Cnot103PaidShearArgs::core_elements,&Cnot103PaidShearArgs::move_elements,&Cnot103PaidShearArgs::stat_elements,&Cnot103PaidShearArgs::scratch_words,&Cnot103PaidShearArgs::batch_elements,&Cnot103PaidShearArgs::source_elements,&Cnot103PaidShearArgs::target_elements}){auto bad=x.a;--(bad.*member);expect(cnot103_cuda_launch_paid_shear(&bad)==2,"compiler extent guard");}
}
// All six GL(2,2) maps in queue order, represented by four output values.
struct Tiny{std::array<U8,4> map;std::vector<Gate> path;};
std::vector<Tiny> tiny_atlas(){
    std::vector<Tiny> nodes{{{0,1,2,3},{}}};
    for(U32 i=0;i<nodes.size();++i)for(Gate g:std::vector<Gate>{{0,1},{1,0}}){auto next=nodes[i];for(auto& code:next.map)code^=((code>>g[1])&1)<<g[0];bool found=false;for(const auto& prior:nodes)found|=prior.map==next.map;if(!found){next.path.push_back(g);nodes.push_back(next);}}
    expect(nodes.size()==6,"independent GL2 cardinality");return nodes;
}
void exhaustive(Atlases& atlas){
    const std::vector<U32> src={0,2},tgt={1,3,4};auto in=identity(5,64);const auto tiny=tiny_atlas();
    for(U32 bits=0;bits<64;++bits)for(U32 r=0;r<3;++r)for(U32 c=0;c<2;++c)if((bits>>(r*2+c))&1)in[bits*5+tgt[r]]^=U64{1}<<src[c];
    Shear x(atlas,5,64,src,tgt,in,2,2);x.run();x.check();auto out=x.word.get();auto lens=x.lengths.get();
    for(U32 bits=0;bits<64;++bits){U64 best=~U64{0};U32 chosen=0;for(U32 k=0;k<tiny.size();++k){U64 cost=2*tiny[k].path.size();for(U32 r=0;r<3;++r){U8 code=tiny[k].map[(bits>>(r*2))&3];cost+=(code&1)+((code>>1)&1);}if(cost<best){best=cost;chosen=k;}}
        std::vector<Gate> expected;for(auto g:tiny[chosen].path)expected.push_back({src[g[0]],src[g[1]]});
        for(U32 r=0;r<3;++r)for(U32 c=0;c<2;++c)if((tiny[chosen].map[(bits>>(r*2))&3]>>c)&1)expected.push_back({src[c],tgt[r]});
        for(auto i=tiny[chosen].path.rbegin();i!=tiny[chosen].path.rend();++i)expected.push_back({src[(*i)[0]],src[(*i)[1]]});
        expect(U64(lens[bits])==best&&slice(out,U64(bits)*12,best)==expected,"all 64 exact tiny full words");++samples;
    }
}
void wide(Atlases& atlas){
    std::vector<U32> many(129);for(U32 i=0;i<129;++i)many[i]=i;std::vector<U32> few={130,131,143};
    for(U32 side=0;side<2;++side){const auto& src=side?few:many;const auto& tgt=side?many:few;auto in=identity(144,2);
        for(U32 r:tgt)for(U32 c:src)in[(144ULL+r)*3+c/64]^=U64{1}<<(c%64);
        Shear x(atlas,144,2,src,tgt,in,2,2);x.run();x.check();samples+=2;
    }
}
struct Combined{
    Buffer<U32> word,overflow;Buffer<U64> labels,endpoint,replay,counts;Buffer<I64> lengths;
    Buffer<float> groups,incidence,pairs;Buffer<U8> valid,status,verified;Cnot103PaidBasisArgs a{};
    Combined(U32 n,U32 b,U32 g,U64 q):word(U64(b)*q*2),overflow(b),labels(U64(b)*q),endpoint(U64(b)*n*((n+63)/64)),replay(endpoint.n),counts(U64(b)*n*n*3),lengths(b),groups(U64(b)*g*18),incidence(U64(b)*g*n*2),pairs(U64(b)*n*n*4),valid(U64(b)*g),status(b),verified(b){
        a.word=word.p;a.labels=labels.p;a.lengths=lengths.p;a.endpoint=endpoint.p;a.replay=replay.p;a.counts=counts.p;a.groups=groups.p;a.incidence=incidence.p;a.valid=valid.p;a.pairs=pairs.p;a.status=status.p;a.verified=verified.p;a.overflow=overflow.p;
        a.capacity=q;a.state_words=endpoint.n;a.gate_elements=word.n;a.label_words=labels.n;a.count_elements=counts.n;a.group_elements=groups.n;a.incidence_elements=incidence.n;a.token_elements=valid.n;a.pair_elements=pairs.n;a.batch_elements=b;a.batch=b;a.dimension=n;a.words_per_row=(n+63)/64;a.context_count=g;
    }
    void init(){expect(!cnot103_cuda_paid_basis_init(&a),"combine init");}
    void add(const Shear& x,const Shear* previous,U32 index){
        Cnot103PaidBasisAppendArgs in{};in.output=a;in.states=x.input.p;in.sources=x.sources.p;in.targets=x.targets.p;in.word=x.word.p;in.lengths=x.lengths.p;in.statistics=x.stats.p;in.status=x.status.p;
        in.capacity=x.c;in.gate_elements=x.word.n;in.stat_elements=x.stats.n;in.source_elements=x.s;in.target_elements=x.t;in.source_count=x.s;in.target_count=x.t;in.context_index=index;
        if(previous){in.previous_word=previous->word.p;in.previous_lengths=previous->lengths.p;in.previous_capacity=previous->c;in.previous_gate_elements=previous->word.n;}
        expect(!cnot103_cuda_paid_basis_append(&in),"combine append");
    }
    void finish(){expect(!cnot103_cuda_paid_basis_finish(&a),"combine finish");ck(cudaDeviceSynchronize());}
    void failure(U8 code){expect(status.get()==std::vector<U8>(a.batch,code),"typed failure priority");zero(word);zero(lengths);zero(endpoint);zero(counts);zero(groups);zero(incidence);zero(valid);zero(pairs);}
};
void combine(Atlases& atlas){
    for(bool empty:{false,true}){
        auto in=identity(8);for(Gate g:std::vector<Gate>{{0,1},{2,3},{4,5}})gate(in,8,g);
        std::vector<std::unique_ptr<Shear>> ctx;U64 q=0;
        auto add=[&](std::vector<U32> s,std::vector<U32> t){ctx.push_back(std::make_unique<Shear>(atlas,8,1,s,t,in));ctx.back()->run();q+=ctx.back()->c;};
        add({0},{1});add({2},{3});add({4},{5});if(empty)add({6},{7});add({4,6},{5});add({2,6},{3});add({0,6},{1});
        Combined out(8,1,ctx.size(),q);out.init();for(U32 i=0;i<ctx.size();++i)out.add(*ctx[i],i?ctx[i-1].get():nullptr,i);out.finish();
        expect(out.status.get()==std::vector<U8>{1},"combined verified");zero(out.word);zero(out.lengths);expect(out.endpoint.get()==identity(8),"cascading identity endpoint");
        auto pairs=out.pairs.get();float compatible=0,canceled=0;for(U32 c=0;c<8;++c)for(U32 t=0;t<8;++t){U64 i=(c*8+t)*4;expect(pairs[i]==float(c==t)&&pairs[i+1]==0,"D14 endpoint/reduced counts");compatible+=pairs[i+2];canceled+=pairs[i+3];}
        expect(canceled==3&&compatible==float(!empty),"empty original neighbor and cascade labels");
        for(auto member:{&Cnot103PaidBasisArgs::state_words,&Cnot103PaidBasisArgs::gate_elements,&Cnot103PaidBasisArgs::label_words,&Cnot103PaidBasisArgs::count_elements,&Cnot103PaidBasisArgs::group_elements,&Cnot103PaidBasisArgs::incidence_elements,&Cnot103PaidBasisArgs::token_elements,&Cnot103PaidBasisArgs::pair_elements,&Cnot103PaidBasisArgs::batch_elements}){auto bad=out.a;--(bad.*member);expect(cnot103_cuda_paid_basis_init(&bad)==2,"combine extent guard");}
    }
    auto in=identity(4);gate(in,4,{0,1});gate(in,4,{1,2});Shear x(atlas,4,1,{0},{1},in),y(atlas,4,1,{1},{2},in);x.run();y.run();
    for(bool reverse:{false,true}){Combined out(4,1,2,2);out.init();out.add(reverse?y:x,nullptr,0);out.add(reverse?x:y,reverse?&y:&x,1);out.finish();expect(out.status.get()[0]==1,"noncommuting combined status");expect(out.pairs.get()[(0*4+2)*4]==float(!reverse),"preserve context order");}
}
void guards(Atlases& atlas){
    auto in=identity(4);Shear x(atlas,4,1,{0},{1},in),y(atlas,4,1,{2},{3},in);x.run();y.run();
    for(U64 count:{U64{1}<<24,(U64{1}<<24)+1}){
        auto stats=x.stats.get();stats[3]=count;x.stats.put(stats);
        Combined out(4,1,2,2);out.init();out.add(x,nullptr,0);out.add(y,&x,1);out.finish();
        if(count>(U64{1}<<24))out.failure(4);else {expect(out.status.get()[0]==1,"exact f32 inclusive boundary");expect(out.groups.get()[0]==float(count),"exact f32 count");}
    }
    // First compiler failure dominates earlier group-count overflow. These
    // injected guard fixtures are not claims of huge-matrix compiler coverage.
    y.status.put({3});Combined bad(4,1,2,2);bad.init();bad.add(x,nullptr,0);bad.add(y,&x,1);bad.finish();bad.failure(3);y.status.put({1});x.stats.put(std::vector<U64>(16));
    for(U64 count:{U64{1}<<24,(U64{1}<<24)+1}){
        Combined out(4,1,1,1);out.init();out.add(x,nullptr,0);auto counts=out.counts.get();counts[2]=count;out.counts.put(counts);out.finish();
        if(count>(U64{1}<<24))out.failure(4);else expect(out.status.get()[0]==1&&out.pairs.get()[3]==float(count),"pair exact f32 inclusive boundary");
    }
    Combined corrupt(4,1,1,1);corrupt.init();corrupt.add(x,nullptr,0);auto endpoint=corrupt.endpoint.get();endpoint[1]^=1;corrupt.endpoint.put(endpoint);corrupt.finish();corrupt.failure(3);
    x.lengths.put({2});Combined too_long(4,1,1,1);too_long.init();too_long.add(x,nullptr,0);too_long.finish();too_long.failure(3);
    x.lengths.put({1});x.word.put({9,0});Combined invalid_gate(4,1,1,1);invalid_gate.init();invalid_gate.add(x,nullptr,0);invalid_gate.finish();invalid_gate.failure(3);
}
int main(){try{{Atlases atlas;golden(atlas,10,{0,1,2,3},{4,5,6,7,8,9});golden(atlas,65,{0,1,31,64},{2,3,30,32,62,63});golden(atlas,144,{0,63,64,129},{1,62,65,127,128,143});exhaustive(atlas);wide(atlas);combine(atlas);guards(atlas);}ck(cudaDeviceSynchronize());std::printf("PASS paid basis: %llu compiler samples, exact GL2 words, full basis/endpoint replay, ordered cancellation, float/status/capacity guards\n",static_cast<unsigned long long>(samples));ck(cudaDeviceReset());return 0;}catch(const std::exception& e){std::fprintf(stderr,"FAIL paid basis: %s\n",e.what());return 1;}}
