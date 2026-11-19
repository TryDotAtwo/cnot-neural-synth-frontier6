// Independent subset expansion of dense coefficient matrices for small packets,
// plus literal long-packet ranks and an overflow-boundary counting argument.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>
using Word=std::uint64_t;
using Index=std::uint32_t;
using Gate=std::pair<Index,Index>;
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<typename T>struct Buffer{
    T* data=nullptr;std::size_t size;
    explicit Buffer(std::size_t n):size(n){checked(cudaMalloc(reinterpret_cast<void**>(&data),std::max(n,std::size_t{1})*sizeof(T)));checked(cudaMemset(data,0xa5,std::max(n,std::size_t{1})*sizeof(T)));}
    ~Buffer(){cudaFree(data);}
    void upload(const std::vector<T>& values){require(values.size()==size,"upload extent");if(size)checked(cudaMemcpy(data,values.data(),size*sizeof(T),cudaMemcpyHostToDevice));}
    void equals(const std::vector<T>& expected,const char* message){std::vector<T> actual(size);if(size)checked(cudaMemcpy(actual.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));require(actual==expected,message);}
};
struct Expected{std::vector<float> values,incidence;std::vector<std::uint8_t> valid;std::uint8_t status;};
Expected zero(Index n,Index slots,std::uint8_t status){return {std::vector<float>(Word{slots+1}*4,0),std::vector<float>(Word{slots+1}*n*2,0),std::vector<std::uint8_t>(slots+1,status==1?1:0),status};}
Index rank(std::vector<Word> columns){
    Index count=0;
    for(Index bit=0;bit<63;++bit){
        Index found=count;while(found<columns.size()&&!(columns[found]&(Word{1}<<bit)))++found;
        if(found==columns.size())continue;
        std::swap(columns[count],columns[found]);
        for(Index row=count+1;row<columns.size();++row)if(columns[row]&(Word{1}<<bit))columns[row]^=columns[count];
        ++count;
    }return count;
}
void incidence(Expected& e,Index n,const std::vector<Gate>& gates,const std::vector<std::vector<Index>>& scopes){
    for(Index cut=0;cut<=gates.size();++cut){
        Index live=0;
        for(Index slot=0;slot<cut;++slot){
            bool used=false;
            for(const auto& scope:scopes)
                if(!scope.empty()&&*std::max_element(scope.begin(),scope.end())>=cut&&std::find(scope.begin(),scope.end(),slot)!=scope.end())used=true;
            if(used){++live;e.incidence[(Word{cut}*n+gates[slot].first)*2]=1;e.incidence[(Word{cut}*n+gates[slot].second)*2+1]=1;}
        }
        e.values[Word{cut}*4]=cut;e.values[Word{cut}*4+2]=live;e.values[Word{cut}*4+3]=e.values[Word{cut}*4+1]+live;
    }
}
Expected enumerate(Index n,const std::vector<Gate>& gates,const std::vector<std::vector<Index>>& scopes,Word budget){
    require(n*n<63&&gates.size()<20,"small oracle bounds");
    Word terms=0;bool nonlinear=false;std::vector<Word> columns(gates.size(),0);
    for(Word subset=0;subset<(Word{1}<<gates.size());++subset){
        std::vector<unsigned> coefficient(n*n,0);
        for(Index i=0;i<n;++i)coefficient[i*n+i]=1;
        Index degree=0,label=0;
        for(Index slot=0;slot<gates.size();++slot)if(subset&(Word{1}<<slot)){
            ++degree;label=slot;std::vector<unsigned> elementary(n*n,0),product(n*n,0);
            elementary[gates[slot].second*n+gates[slot].first]=1;
            for(Index row=0;row<n;++row)for(Index col=0;col<n;++col)for(Index inner=0;inner<n;++inner)
                product[row*n+col]^=elementary[row*n+inner]&coefficient[inner*n+col];
            coefficient=std::move(product);
        }
        Word flat=0;for(Index i=0;i<n*n;++i)if(coefficient[i])flat|=Word{1}<<i;
        if(!flat)continue;
        ++terms;nonlinear|=degree>1;if(degree==1)columns[label]=flat;
    }
    const std::uint8_t status=terms>budget?4:nonlinear?3:1;
    auto result=zero(n,gates.size(),status);if(status!=1)return result;
    const Index total=rank(columns);
    for(Index cut=0;cut<=gates.size();++cut){
        std::vector<Word> left(columns.begin(),columns.begin()+cut),right(columns.begin()+cut,columns.end());
        result.values[Word{cut}*4+1]=rank(left)+rank(right)-total;
    }
    incidence(result,n,gates,scopes);return result;
}
void run(Index n,const std::vector<Gate>& gates,const std::vector<std::vector<Index>>& scopes,Word budget,const Expected& expected){
    const Index slots=gates.size(),factors=scopes.size();const Word groups=Word{slots}+1;
    std::vector<Index> flat_gates,flat_scopes;std::vector<Word> offsets{0};
    for(const auto& gate:gates){flat_gates.push_back(gate.first);flat_gates.push_back(gate.second);}
    for(const auto& scope:scopes){flat_scopes.insert(flat_scopes.end(),scope.begin(),scope.end());offsets.push_back(flat_scopes.size());}
    Buffer<Index> dgates(flat_gates.size()),dscopes(flat_scopes.size()),indices(std::max(Word{1},Word{factors}+Word{slots}*3));
    Buffer<Word> doffsets(offsets.size()),counts(n);Buffer<float> values(groups*4),roles(groups*n*2);Buffer<std::uint8_t>valid(groups),status(1);
    dgates.upload(flat_gates);dscopes.upload(flat_scopes);doffsets.upload(offsets);
    Cnot103PositivePacketArgs a{};
    a.gates=slots?dgates.data:nullptr;a.factor_offsets=doffsets.data;a.factor_scopes=flat_scopes.empty()?nullptr:dscopes.data;
    a.counts=counts.data;a.indices=indices.data;a.values=values.data;a.valid=valid.data;a.incidence=roles.data;a.status=status.data;
    a.max_monomials=budget;a.scope_elements=flat_scopes.size();a.count_words=counts.size;a.index_elements=indices.size;
    a.value_elements=values.size;a.group_elements=valid.size;a.incidence_elements=roles.size;a.dimension=n;a.slots=slots;a.factors=factors;
    for(Index field=0;field<5;++field){
        auto short_buffer=a;
        switch(field){case 0:--short_buffer.count_words;break;case 1:if(!factors&&!slots)continue;short_buffer.index_elements=Word{factors}+Word{slots}*3-1;break;case 2:--short_buffer.value_elements;break;case 3:--short_buffer.group_elements;break;case 4:--short_buffer.incidence_elements;break;}
        require(cnot103_cuda_prepare_positive_packet(&short_buffer)==2,"short I08 capacity accepted");
    }
    auto invalid=a;invalid.max_monomials=0;require(cnot103_cuda_prepare_positive_packet(&invalid)==1,"zero I08 monomial budget accepted");
    require(cnot103_cuda_prepare_positive_packet(&a)==0,"I08 setup launch failed");checked(cudaDeviceSynchronize());
    status.equals({expected.status},"I08 classification/priority differs");values.equals(expected.values,"I08 cut-width/live values differ");
    valid.equals(expected.valid,"I08 group mask differs");roles.equals(expected.incidence,"I08 physical roles differ");
    dgates.equals(flat_gates,"I08 changed gates");dscopes.equals(flat_scopes,"I08 changed scopes");doffsets.equals(offsets,"I08 changed factor offsets");
}
void exhaustive(){
    const std::vector<Gate> choices{{0,1},{0,2},{1,0},{1,2},{2,0},{2,1}};
    Index power=1,checked_cases=0;
    for(Index size=0;size<=3;++size){
        for(Index code=0;code<power;++code){
            Index remaining=code;std::vector<Gate> gates;
            for(Index i=0;i<size;++i){gates.push_back(choices[remaining%6]);remaining/=6;}
            std::vector<std::vector<Index>> scopes;if(size>1)scopes={{size-1,0},{}};
            for(Word budget:std::vector<Word>{Word{1}<<size,std::max(Word{1},Word{size})}){
                run(3,gates,scopes,budget,enumerate(3,gates,scopes,budget));++checked_cases;
            }
        }
        std::printf("I08 exhaustive length<=%u: %u dense subset-reference checks PASS\n",size,checked_cases);power*=6;
    }
}
void wide(Index n){
    const Index slots=130;std::vector<Gate> gates;std::vector<std::vector<Index>> scopes(1);
    for(Index j=0;j<slots;++j){gates.push_back({j%2,n-1});scopes[0].push_back(slots-1-j);}
    auto expected=zero(n,slots,1);
    for(Index cut=0;cut<=slots;++cut)expected.values[Word{cut}*4+1]=std::min(2u,cut)+std::min(2u,slots-cut)-2;
    incidence(expected,n,gates,scopes);run(n,gates,scopes,131,expected);
    run(n,gates,scopes,130,zero(n,slots,4));
    std::printf("I08 n=%u long labelled scope and exact monomial boundary PASS\n",n);
}
int main(){try{
    exhaustive();wide(65);wide(144);
    std::vector<Gate> cycles;for(Index j=0;j<128;++j)cycles.push_back(j%2?Gate{1,0}:Gate{0,1});
    // Choosing any subset of the64 complete two-gate cycles produces a
    // nonzero coefficient, hence at least2^64 terms. This exceeds UINT64_MAX.
    run(2,cycles,{},~Word{0},zero(2,cycles.size(),4));
    std::puts("I08 UINT64_MAX capacity arithmetic PASS");checked(cudaDeviceReset());return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
