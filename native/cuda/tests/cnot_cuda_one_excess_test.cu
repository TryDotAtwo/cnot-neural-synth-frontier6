#include "cnot103/cnot_cuda.h"
#include "../src/one_excess.cuh"
#include "one_excess_reference.hpp"
#include <cstring>

using namespace one_excess_reference;
using Layout=cnot103_one_excess::ClassifierLayout;
static void cu(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(1);}}
template<class T>struct Buffer {
    T* base;std::size_t count;std::vector<T> host;static constexpr std::size_t guard=32;
    explicit Buffer(std::size_t n):count(n),host(n+2*guard,static_cast<T>(0xa55aa55aa55aa55aULL)){cu(cudaMalloc(&base,host.size()*sizeof(T)));
        cu(cudaMemcpy(base,host.data(),guard*sizeof(T),cudaMemcpyHostToDevice));cu(cudaMemcpy(base+guard+count,host.data()+guard+count,guard*sizeof(T),cudaMemcpyHostToDevice));}
    ~Buffer(){cudaFree(base);}T* data(){return base+guard;}
    void load(const std::vector<T>& x){check(x.size()==count,"input extent");cu(cudaMemcpy(data(),x.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    void guards(){cu(cudaMemcpy(host.data(),base,guard*sizeof(T),cudaMemcpyDeviceToHost));cu(cudaMemcpy(host.data()+guard+count,base+guard+count,guard*sizeof(T),cudaMemcpyDeviceToHost));
        for(std::size_t i=0;i<guard;++i)check(host[i]==static_cast<T>(0xa55aa55aa55aa55aULL)&&host[guard+count+i]==static_cast<T>(0xa55aa55aa55aa55aULL),"allocation guard");}
    std::vector<T> read(){guards();std::vector<T> x(count);cu(cudaMemcpy(x.data(),data(),count*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
struct Metrics {U64 rows=0,exact=0,negative=0,unsupported=0,resource=0,gates=0,wide=0,bfs_labels=0;}metrics;
__global__ void corrupt(Cnot103OneExcessArgs a,U32 mode){
    if(mode<6){a.headers[mode]^=1;return;}
    if(mode==6)a.counts[0]^=1;
    else if(mode==7)a.counts[cnot103_one_excess::limbs_for_dimension(a.dimension)-1]^=1;
    else if(mode==8)a.gates[0]=a.dimension;
    else if(mode==9)a.gates[2*(a.dimension)]=1;
    else if(mode==10)--a.lengths[0];
    else if(mode==11)const_cast<U64*>(a.zero_headers)[0]^=3;
    else if(mode==12)++const_cast<U64*>(a.zero_headers)[1];
    else if(mode==13)const_cast<std::uint8_t*>(a.zero_status)[0]=0;
    else if(mode==14)const_cast<U64*>(a.states)[a.words_per_row-1]|=U64(1)<<(a.dimension%64);
    else if(mode==15)for(U32 w=0;w<a.words_per_row;++w)const_cast<U64*>(a.states)[w]=a.states[a.words_per_row+w];
}
static int mode_index(const char* mode){const char* names[]={"class","changed","templates","bag","transport","distance","count","high_count","gate","unused","length","prerequisite","prereq_count","prereq_status","padding","singular","publish"};
    for(U32 i=0;i<17;++i)if(!std::strcmp(mode,names[i]))return i;check(false,"fatal mode");return -1;}
static void run(const std::vector<Case>& cases,U64 limit,U32 cache,U32 lanes,U32 workers,const char* fatal=nullptr){
    const U32 batch=cases.size(),n=cases[0].state.size(),w=(n+63)/64;workers=std::min(workers,batch);const Layout l(n,lanes,cache);const auto& c=l.count;const auto& s=c.structure;
    std::vector<U64> states;for(const auto& row:cases)for(const auto& value:row.state)states.insert(states.end(),value.begin(),value.end());Buffer<U64> input(states.size());input.load(states);
    cudaStream_t stream;cu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Buffer<U64> zscratch(states.size()),zh(3*U64(batch));Buffer<U32> zg(3*U64(batch)*n),zp(2*U64(batch)*n),zc(U64(batch)*n),zm(U64(batch)*n),zo(U64(batch)*n),zword(2*U64(batch)*n);
    Buffer<std::int64_t> zl(batch);Buffer<std::uint8_t> zv(batch),zs(batch);Buffer<float> zf(3*U64(batch));
    Cnot103ZeroExcessArgs za{input.data(),zscratch.data(),zg.data(),zp.data(),zc.data(),zm.data(),zo.data(),zword.data(),zh.data(),zl.data(),zv.data(),zs.data(),zf.data(),stream,
        input.count,zscratch.count,zg.count,zp.count,zc.count,zm.count,zo.count,zword.count,zh.count,zf.count,batch,batch,n,w};
    check(cnot103_cuda_launch_zero_excess(&za)==0,"Stage37 prerequisite launch");cu(cudaStreamSynchronize(stream));const auto zheaders=zh.read();const auto zstatus=zs.read();
    Buffer<U64> hist(U64(workers)*l.history_words),bits(U64(workers)*s.bit_count),scratch(U64(workers)*s.scratch_count),pg(U64(workers)*c.program_words),classifier(U64(workers)*l.classifier_words),headers(6*U64(batch)),word_scratch(states.size());
    Buffer<U32> lengths(U64(workers)*n),meta(U64(workers)*s.meta_count),ar(U64(workers)*l.arithmetic_words),wg(U64(workers)*l.gate_elements),counts(U64(batch)*c.limbs),gates(U64(batch)*l.gate_elements);
    Buffer<std::uint8_t> protected_rows(U64(workers)*n),verified(batch),status(batch);Buffer<std::int64_t> output_lengths(batch);Buffer<float> feature(3*U64(batch));
    Cnot103OneExcessArgs a{input.data(),zh.data(),zs.data(),hist.data(),lengths.data(),protected_rows.data(),meta.data(),bits.data(),scratch.data(),pg.data(),ar.data(),classifier.data(),wg.data(),headers.data(),counts.data(),gates.data(),output_lengths.data(),verified.data(),status.data(),feature.data(),word_scratch.data(),stream,
        input.count,zh.count,zs.count,hist.count,lengths.count,protected_rows.count,meta.count,bits.count,scratch.count,pg.count,ar.count,classifier.count,wg.count,headers.count,counts.count,gates.count,batch,feature.count,word_scratch.count,limit,batch,n,w,workers,lanes,cache};
    if(fatal){const int mode=mode_index(fatal);check(cnot103_cuda_one_excess_construct(&a)==0,"fatal construct");cu(cudaStreamSynchronize(stream));
        if(mode==16)check(cnot103_cuda_one_excess_publish(&a)==0,"unverified publish");
        else{corrupt<<<1,1,0,stream>>>(a,mode);check(cnot103_cuda_one_excess_check(&a)==0,"fatal check launch");}
        const cudaError_t error=cudaStreamSynchronize(stream);check(error==cudaErrorLaunchFailure,"expected certificate trap");std::printf("PASS expected one-excess %s integrity trap: %s\n",fatal,cudaGetErrorString(error));return;}
    check(cnot103_cuda_launch_one_excess(&a)==0,"recognize/check/publish launch");cu(cudaStreamSynchronize(stream));
    const auto h=headers.read();const auto cnt=counts.read();const auto word=gates.read();const auto len=output_lengths.read();const auto st=status.read();const auto f=feature.read();const auto valid=verified.read();
    if(batch==1&&cases[0].resource_phase){const auto remaining=lengths.read();const auto private_data=classifier.read();U64 events=0;for(U32 r=0;r<n;++r)events+=remaining[r]-1;
        check(private_data[6]<n,"resource after first positive template");
        check(cases[0].resource_phase==1?events==h[1]+1:events<h[1]+1,"resource phase: full enumeration history versus shortened recovery history");}
    check(input.read()==states&&zh.read()==zheaders&&zs.read()==zstatus,"input/prerequisite preservation");
    for(U32 b=0;b<batch;++b){const Expected& e=cases[b].expected;const U32 cls=e.header[0];++metrics.rows;
        for(U32 i=0;i<6;++i)if(e.header[i]>=0&&h[6*b+i]!=static_cast<U64>(e.header[i])){std::fprintf(stderr,"n%u row%u header%u got%llu want%lld\n",n,b,i,(unsigned long long)h[6*b+i],(long long)e.header[i]);std::exit(1);}
        check(h[6*b+3]<=n+1,"bag domain");check(st[b]==(cls==4?4:1)&&valid[b]==1,"verified typed result");
        for(U32 i=0;i<3;++i)check(f[3*b+i]==float(cls==i+1),"feature class");
        for(U32 i=0;i<c.limbs;++i)check(cnt[U64(b)*c.limbs+i]==(i<e.count.size()?e.count[i]:0),"complete exact count/canonical high limbs");
        check(len[b]==static_cast<std::int64_t>(e.encoder.size()),"encoder length");Matrix replay(n);for(U32 r=0;r<n;++r)replay[r]=unit(n,r);
        for(U32 i=0;i<c.k;++i){const U32 control=word[U64(b)*l.gate_elements+2*i],target=word[U64(b)*l.gate_elements+2*i+1];
            if(i<e.encoder.size()){check(control==e.encoder[i].first&&target==e.encoder[i].second,"source deterministic complete word");check(control<n&&target<n&&control!=target,"physical word index");replay[target]=sum(replay[target],replay[control]);++metrics.gates;}
            else check(!control&&!target,"canonical unused gate padding");}
        if(cls==1){++metrics.exact;check(replay==cases[b].state,"independent complete encoder replay");if(e.count.size()>4)++metrics.wide;}
        else if(cls==2)++metrics.negative;else if(cls==3)++metrics.unsupported;else ++metrics.resource;
    }
    hist.guards();lengths.guards();protected_rows.guards();meta.guards();bits.guards();scratch.guards();pg.guards();ar.guards();classifier.guards();wg.guards();word_scratch.guards();
    zscratch.guards();zg.guards();zp.guards();zc.guards();zm.guards();zo.guards();zword.guards();zl.guards();zv.guards();zf.guards();
    for(U32 field=0;field<19;++field){auto bad=a;U64* extents[]={&bad.state_words,&bad.zero_header_words,&bad.zero_status_elements,&bad.history_words,&bad.length_elements,&bad.protected_elements,&bad.meta_elements,&bad.bit_words,&bad.scratch_words,&bad.program_words,&bad.arithmetic_elements,&bad.classifier_words,&bad.workspace_gate_elements,&bad.header_elements,&bad.count_elements,&bad.gate_elements,&bad.batch_elements,&bad.feature_elements,&bad.word_scratch_words};
        --*extents[field];check(cnot103_cuda_launch_one_excess(&bad)==2,"short extent");}
    check(cnot103_cuda_launch_one_excess(nullptr)==1,"null args");for(U32 field=0;field<10;++field){auto bad=a;
        switch(field){case 0:bad.batch=0;break;case 1:bad.dimension=1;break;case 2:bad.dimension=(1U<<24)+1;break;case 3:bad.words_per_row=0;break;case 4:bad.workers=0;break;case 5:bad.workers=batch+1;break;case 6:bad.lanes=0;break;case 7:bad.lanes=33;break;case 8:bad.max_entries=0;break;default:bad.classifier=nullptr;}
        check(cnot103_cuda_launch_one_excess(&bad)==1,"invalid metadata");}
    for(U32 workers:{1U,32U}){auto bad=a;bad.dimension=workers==1?(1U<<24):(1U<<20);bad.words_per_row=(bad.dimension+63)/64;bad.workers=bad.batch=workers;bad.lanes=32;bad.cache_slots=~U32(0);
        bad.state_words=bad.zero_header_words=bad.zero_status_elements=bad.history_words=bad.length_elements=bad.protected_elements=bad.meta_elements=bad.bit_words=bad.scratch_words=bad.program_words=bad.arithmetic_elements=bad.classifier_words=bad.workspace_gate_elements=bad.header_elements=bad.count_elements=bad.gate_elements=bad.batch_elements=bad.feature_elements=bad.word_scratch_words=~U64(0);
        check(cnot103_cuda_launch_one_excess(&bad)==2,"individual/aggregate byte extent overflow");}
    cu(cudaStreamDestroy(stream));
}
int main(int argc,char** argv){
    if(argc==2){const U32 n=!std::strcmp(argv[1],"padding")?65:3;run({swap_and_shears(n,false)},100000,17,3,1,argv[1]);return 0;}
    for(U32 n:{2U,3U,4U}){BFS bfs(n);metrics.bfs_labels+=bfs.queue.size();std::vector<Case> cases;
        for(std::size_t i=0;i<bfs.queue.size();++i){if(n==4&&i%211!=0)continue;const U32 code=bfs.queue[i];auto m=bfs.matrix(code);cases.push_back({m,reference(m,bfs.distance[code],bfs.ways[code])});}
        run(cases,100000,0,1,1);run(cases,100000,1,3,3);run(cases,100000,127,8,4);
    }
    BFS small(3);
    for(U32 n:{65U,144U}){std::vector<Case> cases;for(U32 i=0;i<small.queue.size();i+=19){const U32 code=small.queue[i];auto m=embed(small.matrix(code),n,{n-1,0,n/2});cases.push_back({m,reference(m,small.distance[code],small.ways[code])});}
        run(cases,100000,257,4,2);auto wide=swap_and_shears(n);run({wide},27,257,4,1);run({resource(wide)},26,1,3,1);}
    run({resource(swap_and_shears(2,false))},1,0,1,1);
    Matrix late{{3},{5},{1}};U32 late_code=3|(5<<3)|(1<<6);
    run({resource({late,reference(late,small.distance[late_code],small.ways[late_code])},1)},16,1,3,1);
    Matrix recovery{{17},{130},{128},{40},{88},{160},{104},{132}};Histories base(8);
    for(U32 r=0;r<8;++r)base[r]={recovery[r],unit(8,r)};
    check(orders(base)==0,"independent n8 zero-excess exclusion");
    // Every row changes, so eight updates are a lower bound. The independent
    // physical-history DP excludes eight and certifies/counts all nine-update
    // words without using a Stage38 result or a BFS table of size 2^64.
    Case recovered{recovery,reference(recovery,9,~U64(0))};
    run({resource(recovered,2)},729,1,3,1);run({recovered},100000,127,4,1);
    check(metrics.exact&&metrics.negative&&metrics.unsupported&&metrics.resource&&metrics.wide,"classification coverage");
    std::printf("PASS complete one-excess: %llu rows, %llu exact/%llu negative/%llu unsupported/%llu resource, %llu gate replays, %llu wide; %llu complete independent BFS labels, physical history/recovery and decimal-factorial oracles\n",
        (unsigned long long)metrics.rows,(unsigned long long)metrics.exact,(unsigned long long)metrics.negative,(unsigned long long)metrics.unsupported,(unsigned long long)metrics.resource,(unsigned long long)metrics.gates,(unsigned long long)metrics.wide,(unsigned long long)metrics.bfs_labels);
}
