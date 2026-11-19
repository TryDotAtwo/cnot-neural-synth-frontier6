#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using U32=std::uint32_t;using U64=std::uint64_t;using U8=std::uint8_t;
void expect(bool value,const char* what){if(!value)throw std::runtime_error(what);}
void ck(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
template<typename T> struct Buffer {
    T* p=nullptr;std::size_t n;
    explicit Buffer(std::size_t count):n(count){ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0xa5,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<T>& v){expect(v.size()==n,"upload extent");ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get()const{std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
using Mapping=std::array<U8,16>;
struct Node{Mapping map;U32 parent,control,target,distance;};
unsigned pop(unsigned value){unsigned n=0;while(value){n+=value&1;value>>=1;}return n;}
// Independent oracle uses complete truth mappings, not the kernel's packed
// matrix-row representation or its direct-index seen table.
U64 key(const Mapping& mapping,U32 width){U64 k=0;for(U32 i=0;i<(1U<<width);++i)k|=U64(mapping[i])<<(i*width);return k;}
std::vector<Node> reference(U32 width,U32 side) {
    Mapping identity{};for(U32 i=0;i<(1U<<width);++i)identity[i]=i;
    const U32 none=std::numeric_limits<U32>::max();
    std::vector<Node> nodes{{identity,none,none,none,0}};
    std::unordered_map<U64,U32> visited{{key(identity,width),0}};
    for(U32 index=0;index<nodes.size();++index) {
        const auto current=nodes[index];
        for(U32 control=0;control<width;++control)for(U32 target=0;target<width;++target) {
            if(control==target)continue;
            Mapping next=current.map;
            for(U32 code=0;code<(1U<<width);++code) {
                std::array<U8,4> bits{};for(U32 bit=0;bit<width;++bit)bits[bit]=(next[code]>>bit)&1;
                if(side==0)bits[control]^=bits[target];else bits[target]^=bits[control];
                next[code]=0;for(U32 bit=0;bit<width;++bit)next[code]|=bits[bit]<<bit;
            }
            const U64 encoded=key(next,width);
            if(visited.find(encoded)!=visited.end())continue;
            visited.emplace(encoded,static_cast<U32>(nodes.size()));
            nodes.push_back({next,index,control,target,current.distance+1});
        }
    }
    return nodes;
}
U32 matrix_key(const Mapping& map,U32 width) {
    U32 result=0;
    for(U32 column=0;column<width;++column)for(U32 row=0;row<width;++row)
        result|=U32((map[1U<<column]>>row)&1)<<(row*width+column);
    return result;
}
std::vector<std::pair<U32,U32>> operations(const std::vector<Node>& nodes,U32 selected) {
    std::vector<std::pair<U32,U32>> result;
    while(selected){const auto& node=nodes[selected];result.emplace_back(node.control,node.target);selected=node.parent;}
    std::reverse(result.begin(),result.end());return result;
}
U64 cases=0,states=0;
void verify_atlas(const std::vector<U32>& storage,const std::vector<Node>& nodes,U32 width) {
    const U32 size=1U<<(width*width),codes=1U<<width;
    expect(storage.back()==nodes.size(),"atlas cardinality");
    for(U32 i=0;i<nodes.size();++i) {
        const auto& node=nodes[i];const U32 matrix=matrix_key(node.map,width);
        expect(storage[i]==matrix,"ordered complete map");
        expect(storage[size+i]==node.parent,"BFS parent");
        expect(storage[2*size+i]==node.control,"ordered parent control");
        expect(storage[3*size+i]==node.target,"ordered parent target");
        expect(storage[4*size+i]==node.distance,"BFS distance");
        expect(storage[5*size+matrix]==i+1,"direct table exact identity");
        for(U32 code=0;code<codes;++code)
            expect(storage[U64(6)*size+U64(i)*codes+code]==pop(node.map[code]),"full truth population");
    }
    states+=nodes.size();
}
void check_batch(Buffer<U32>& storage,const std::vector<Node>& nodes,U32 width,U32 side,U32 batch,U64 rows,const std::vector<U8>& input) {
    const U64 capacity=std::max(1U,(width+3)*(width-1));
    Buffer<U8> vectors(input.size()),transformed(input.size()),status(batch);
    Buffer<std::int32_t> gates(batch*capacity*2);
    Buffer<U64> lengths(batch),direct(batch),paid(batch);
    vectors.put(input);
    Cnot103PaidAtlasArgs a{};
    a.storage=storage.p;a.vectors=vectors.p;a.transformed=transformed.p;a.operations=gates.p;
    a.lengths=lengths.p;a.direct_cost=direct.p;a.paid_cost=paid.p;a.status=status.p;
    a.storage_words=storage.n;a.vector_elements=input.size();a.gate_elements=gates.n;
    a.batch_elements=batch;a.word_capacity=capacity;a.rows=rows;a.batch=batch;a.width=width;a.side=side;
    expect(cnot103_cuda_optimize_paid_atlas(&a)==0,"optimizer launch");ck(cudaDeviceSynchronize());
    const auto out=transformed.get(),codes=status.get();const auto word=gates.get();
    const auto len=lengths.get(),d=direct.get(),p=paid.get();
    expect(vectors.get()==input,"input remains immutable");
    for(U32 sample=0;sample<batch;++sample) {
        ++cases;std::array<U64,16> histogram{};U64 direct_count=0;bool valid=true;
        for(U64 row=0;row<rows;++row) {
            U32 code=0;for(U32 bit=0;bit<width;++bit){const U8 value=input[(sample*rows+row)*width+bit];valid&=value<=1;direct_count+=value;code|=(value&1)<<bit;}
            ++histogram[code];
        }
        if(!valid) {
            expect(codes[sample]==3 && len[sample]==0 && d[sample]==0 && p[sample]==0,"invalid sample status and counts");
            for(U64 i=0;i<capacity*2;++i)expect(word[sample*capacity*2+i]==0,"invalid word cleared");
            for(U64 i=0;i<rows*width;++i)expect(out[sample*rows*width+i]==0,"invalid vectors cleared");
            continue;
        }
        U64 best=std::numeric_limits<U64>::max();U32 selected=0;
        for(U32 index=0;index<nodes.size();++index) {
            U64 cost=2*nodes[index].distance;
            for(U32 code=0;code<(1U<<width);++code)cost+=histogram[code]*pop(nodes[index].map[code]);
            if(cost<best){best=cost;selected=index;}
        }
        const auto ops=operations(nodes,selected);
        expect(codes[sample]==1 && len[sample]==ops.size() && d[sample]==direct_count && p[sample]==best,"exact minimizing cost and length");
        for(U64 i=0;i<capacity;++i)for(U32 role=0;role<2;++role) {
            const U32 expected=i<ops.size()?(role?ops[i].second:ops[i].first):0;
            expect(word[(sample*capacity+i)*2+role]==static_cast<std::int32_t>(expected),"full ordered word and padding");
        }
        U64 replay_cost=2*ops.size();
        for(U64 row=0;row<rows;++row) {
            std::array<U8,4> bits{};
            for(U32 bit=0;bit<width;++bit)bits[bit]=input[(sample*rows+row)*width+bit];
            for(auto op:ops) {if(side==0)bits[op.first]^=bits[op.second];else bits[op.second]^=bits[op.first];}
            for(U32 bit=0;bit<width;++bit){expect(out[(sample*rows+row)*width+bit]==bits[bit],"independent vector path replay");replay_cost+=bits[bit];}
        }
        expect(replay_cost==best && best<=direct_count,"verified paid cost bound");
    }
    auto bad=a;bad.storage_words--;expect(cnot103_cuda_optimize_paid_atlas(&bad)==2,"atlas storage guard");
    bad=a;bad.vector_elements--;expect(cnot103_cuda_optimize_paid_atlas(&bad)==2,"vector extent guard");
    bad=a;bad.gate_elements--;expect(cnot103_cuda_optimize_paid_atlas(&bad)==2,"word extent guard");
    bad=a;bad.batch_elements--;expect(cnot103_cuda_optimize_paid_atlas(&bad)==2,"batch extent guard");
    bad=a;bad.word_capacity--;expect(cnot103_cuda_optimize_paid_atlas(&bad)==1,"proved path capacity guard");
    bad=a;bad.rows=~U64{0};expect(cnot103_cuda_optimize_paid_atlas(&bad)==1,"row arithmetic overflow guard");
    bad=a;bad.side=2;expect(cnot103_cuda_optimize_paid_atlas(&bad)==1,"side guard");
    bad=a;bad.batch=0;expect(cnot103_cuda_optimize_paid_atlas(&bad)==1,"zero batch guard");
}
int main() {
    try {
        ck(cudaSetDevice(0));
        expect(cnot103_cuda_prepare_paid_atlas(nullptr)==1,"null prepare");
        expect(cnot103_cuda_optimize_paid_atlas(nullptr)==1,"null optimize");
        for(U32 width=1;width<=4;++width)for(U32 side=0;side<2;++side) {
            const U32 size=1U<<(width*width),codes=1U<<width;
            Buffer<U32> storage(U64(size)*(6+codes)+1);
            Cnot103PaidAtlasPrepareArgs a{storage.p,nullptr,storage.n,width,side};
            expect(cnot103_cuda_prepare_paid_atlas(&a)==0,"atlas launch");ck(cudaDeviceSynchronize());
            const auto snapshot=storage.get();const auto nodes=reference(width,side);
            const U32 cardinality[5]={0,1,6,168,20160};expect(nodes.size()==cardinality[width],"oracle GL cardinality");
            verify_atlas(snapshot,nodes,width);
            auto bad=a;bad.storage_words--;expect(cnot103_cuda_prepare_paid_atlas(&bad)==2,"prepare capacity guard");
            bad=a;bad.width=0;expect(cnot103_cuda_prepare_paid_atlas(&bad)==1,"prepare width guard");
            bad=a;bad.width=5;expect(cnot103_cuda_prepare_paid_atlas(&bad)==1,"prepare upper width guard");
            bad=a;bad.side=2;expect(cnot103_cuda_prepare_paid_atlas(&bad)==1,"prepare side guard");
            if(width==2) {
                std::vector<U8> input(256*12*2,0);
                for(U32 sample=0;sample<256;++sample) {U32 row=0;for(U32 code=0;code<4;++code)for(U32 k=0;k<((sample>>(2*code))&3);++k,++row)
                    for(U32 bit=0;bit<2;++bit)input[(sample*12+row)*2+bit]=(code>>bit)&1;}
                check_batch(storage,nodes,width,side,256,12,input);
            } else {
                const U32 batch=width==1?2:48;const U64 rows=129;
                std::vector<U8> input(batch*rows*width,0);
                for(U32 sample=0;sample<batch;++sample)for(U64 row=0;row<rows;++row) {
                    const U32 code=sample==0?0:sample==1?codes-1:(sample*97+row*31+(row/7)*sample)%codes;
                    for(U32 bit=0;bit<width;++bit)input[(sample*rows+row)*width+bit]=(code>>bit)&1;
                }
                check_batch(storage,nodes,width,side,batch,rows,input);
            }
            std::vector<U8> invalid(2*3*width,1);invalid[width]=255;
            check_batch(storage,nodes,width,side,2,3,invalid);
            expect(storage.get()==snapshot,"prepared atlas remains immutable");
        }
        std::printf("Paid atlas PASS: %llu ordered maps/parents/gates/distances and %llu exact local optimization samples\n",
            static_cast<unsigned long long>(states),static_cast<unsigned long long>(cases));
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
