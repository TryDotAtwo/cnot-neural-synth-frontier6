// Bounded physical-word enumeration and an independent dense six-state walk
// recurrence check the sparse CUDA DP. Host unsigned __int128 is the oracle
// for carry/overflow and correctly rounded conversion tests.
#include "cnot103/cnot_cuda.h"
#include "../src/endpoint_response_math.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
using Word=std::uint64_t;
using Index=std::uint32_t;
using Bit=std::uint8_t;
using Wide=unsigned __int128;
using Key=std::vector<Word>;
using Layer=std::map<Key,Wide>;
using Gate=std::pair<Index,Index>;
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
template<typename T>struct Buffer{
    T* data=nullptr;std::size_t size;
    explicit Buffer(std::size_t n):size(n){checked(cudaMalloc(reinterpret_cast<void**>(&data),std::max(n,std::size_t{1})*sizeof(T)));checked(cudaMemset(data,0xa5,std::max(n,std::size_t{1})*sizeof(T)));}
    ~Buffer(){cudaFree(data);}
    void upload(const std::vector<T>& v){require(v.size()==size,"upload extent");if(size)checked(cudaMemcpy(data,v.data(),size*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read()const{std::vector<T> v(size);if(size)checked(cudaMemcpy(v.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));return v;}
    void equals(const std::vector<T>& expected,const char* message){require(read()==expected,message);}
};
std::vector<Gate> gates(Index n){std::vector<Gate> out;for(Index c=0;c<n;++c)for(Index t=0;t<n;++t)if(c!=t)out.push_back({c,t});return out;}
Key identity(Index n){const Index w=(n+63)/64;Key key(Word{n}*w,0);for(Index i=0;i<n;++i)key[Word{i}*w+i/64]|=Word{1}<<(i%64);return key;}
Key moved(Key key,Index n,Gate gate){const Index w=(n+63)/64;for(Index i=0;i<w;++i)key[Word{gate.second}*w+i]^=key[Word{gate.first}*w+i];return key;}
Layer enumerate(Index n,Word degree){
    const auto alphabet=gates(n);Word count=1;for(Word i=0;i<degree;++i){require(count<=1000000/alphabet.size(),"enumeration test too large");count*=alphabet.size();}
    Layer result;
    for(Word encoded=0;encoded<count;++encoded){Word cursor=encoded;Key key=identity(n);
        for(Word i=0;i<degree;++i){key=moved(std::move(key),n,alphabet[cursor%alphabet.size()]);cursor/=alphabet.size();}
        ++result[key];
    }return result;
}
std::vector<Key> gl2(){return {{1,2},{1,3},{2,1},{2,3},{3,1},{3,2}};}
bool dense_next(const Layer& previous,Layer& next){
    const auto vertices=gl2();Index adjacency[6][6]{};
    for(Index from=0;from<6;++from)for(auto gate:gates(2)){
        const auto key=moved(vertices[from],2,gate);auto it=std::find(vertices.begin(),vertices.end(),key);
        require(it!=vertices.end(),"literal GL2 closure");++adjacency[from][it-vertices.begin()];
    }
    for(Index to=0;to<6;++to){Wide coefficient=0;
        for(Index from=0;from<6;++from)if(adjacency[from][to]){
            auto it=previous.find(vertices[from]);if(it!=previous.end()){
                require(adjacency[from][to]==1,"literal simple GL2 alphabet");
                if(coefficient>~Wide{0}-it->second)return false;coefficient+=it->second;
            }
        }if(coefficient)next[vertices[to]]=coefficient;
    }return true;
}
struct Config{Index n;Word degree,states,transitions;};
struct Oracle{Bit status=1;Word evaluations=0;Layer terminal;std::map<Key,std::pair<Word,Wide>> first;};
Oracle oracle(Config c){
    Oracle out;Layer previous{{identity(c.n),1}};out.first[identity(c.n)]={0,1};const Word edges=Word{c.n}*(c.n-1);
    for(Word degree=1;degree<=c.degree;++degree){
        if(previous.size()>c.states || previous.size()>~Word{0}/edges || previous.size()*edges>c.transitions-out.evaluations){out.status=4;break;}
        Layer next;
        if(c.n==2){if(!dense_next(previous,next)){out.status=4;break;}}
        else next=enumerate(c.n,degree);
        out.evaluations+=previous.size()*edges;
        if(next.size()>c.states){out.status=4;break;}
        for(const auto& item:next)if(!out.first.count(item.first))out.first[item.first]={degree,item.second};
        previous=std::move(next);if(degree==c.degree)break;
    }
    if(out.status==1)out.terminal=std::move(previous);else{out.evaluations=0;out.first.clear();}
    return out;
}
std::vector<Key> targets(Index n){
    if(n==2)return gl2();
    if(n==3){std::vector<Key> result;
        for(Index bits=0;bits<512;++bits){Key rows{bits&7,(bits>>3)&7,bits>>6};
            // The six permutation terms of the 3x3 determinant over GF(2).
            Index det=0;for(Index a=0;a<3;++a)for(Index b=0;b<3;++b)if(a!=b){Index c=3-a-b;
                det^=((rows[0]>>a)&1)&((rows[1]>>b)&1)&((rows[2]>>c)&1);}
            if(det)result.push_back(rows);
        }return result;
    }
    Key root=identity(n);return {root,moved(root,n,{n-1,0}),moved(moved(root,n,{0,1}),n,{n-2,n-1})};
}
Word power_two(Word n){Word result=1;while(result<n)result*=2;return result;}
void run(Config c){
    const auto reference=oracle(c);const auto input=targets(c.n);const Word b=input.size(),words=Word{c.n}*((c.n+63)/64);
    Word capacity=c.degree?std::max(Word{1},std::min(c.states,c.transitions)):1;
    if(Word{c.n}*c.n<64)capacity=std::min(capacity,Word{1}<<(c.n*c.n));
    const Word buckets=power_two(2*capacity+1),history_buckets=power_two(4*capacity+1);
    const Word storage_words=4*capacity*words+10*capacity+2*buckets+history_buckets+128*words+387;
    Buffer<Word> storage(storage_words);Buffer<Bit> prepared(1);
    Cnot103EndpointPrepareArgs p{};p.storage=storage.data;p.status=prepared.data;p.storage_words=storage_words;
    p.capacity=capacity;p.buckets=buckets;p.history_buckets=history_buckets;p.max_degree=c.degree;
    p.max_layer_states=c.states;p.max_transitions=c.transitions;p.dimension=c.n;p.words_per_row=(c.n+63)/64;
    auto invalid=p;--invalid.storage_words;require(cnot103_cuda_prepare_endpoint_response(&invalid)==2,"short I10 preparation buffer accepted");
    invalid=p;invalid.buckets=capacity;require(cnot103_cuda_prepare_endpoint_response(&invalid)==1,"invalid I10 hash extent accepted");
    invalid=p;invalid.words_per_row=0;require(cnot103_cuda_prepare_endpoint_response(&invalid)==1,"invalid I10 width accepted");
    if(capacity>1){invalid=p;invalid.capacity=capacity-1;require(cnot103_cuda_prepare_endpoint_response(&invalid)==1,"hidden narrower I10 state cap accepted");}
    require(cnot103_cuda_prepare_endpoint_response(&p)==0,"I10 preparation launch");checked(cudaDeviceSynchronize());
    prepared.equals({reference.status},"I10 whole-horizon classification differs");const auto immutable=storage.read();
    std::vector<Word> flat_input;for(const auto& key:input)flat_input.insert(flat_input.end(),key.begin(),key.end());
    Buffer<Word> states(flat_input.size()),degrees(b),leading(2*b),terminal(2*b),evaluations(b);
    Buffer<float> values(4*b);Buffer<Bit> status(b),oracle_status(b);states.upload(flat_input);
    Cnot103EndpointArgs a{};a.states=states.data;a.storage=storage.data;a.prepared_status=prepared.data;
    a.values=values.data;a.feature_status=status.data;a.oracle_status=oracle_status.data;a.leading_degrees=degrees.data;
    a.leading_counts=leading.data;a.terminal_counts=terminal.data;a.evaluations=evaluations.data;
    a.state_words=states.size;a.storage_words=storage_words;a.value_elements=values.size;a.count_elements=leading.size;a.batch_elements=b;
    a.capacity=capacity;a.buckets=buckets;a.history_buckets=history_buckets;a.max_degree=c.degree;
    a.batch=b;a.dimension=c.n;a.words_per_row=p.words_per_row;
    for(Index field=0;field<5;++field){auto short_buffer=a;switch(field){case 0:--short_buffer.state_words;break;case 1:--short_buffer.storage_words;break;
        case 2:--short_buffer.value_elements;break;case 3:--short_buffer.count_elements;break;case 4:--short_buffer.batch_elements;break;}
        require(cnot103_cuda_launch_endpoint_response(&short_buffer)==2,"short I10 result buffer accepted");}
    require(cnot103_cuda_launch_endpoint_response(&a)==0,"I10 lookup launch");checked(cudaDeviceSynchronize());
    std::vector<float> expected_values(4*b,0);std::vector<Word> expected_degrees(b,0),expected_leading(2*b,0),expected_terminal(2*b,0),expected_evaluations(b,reference.evaluations);
    std::vector<Bit> expected_status(b,reference.status),expected_oracle(b,reference.status);
    for(Word i=0;i<b;++i)if(reference.status==1){
        auto first=reference.first.find(input[i]);if(first==reference.first.end())expected_status[i]=3;
        else{const Word degree=first->second.first;const Wide count=first->second.second;
            expected_degrees[i]=degree;expected_leading[2*i]=count;expected_leading[2*i+1]=count>>64;
            expected_values[4*i]=degree;expected_values[4*i+1]=std::log1p(static_cast<double>(count));expected_values[4*i+2]=1;}
        auto last=reference.terminal.find(input[i]);if(last!=reference.terminal.end()){expected_terminal[2*i]=last->second;expected_terminal[2*i+1]=last->second>>64;}
    }
    values.equals(expected_values,"I10 source projection differs");degrees.equals(expected_degrees,"I10 first degrees differ");
    leading.equals(expected_leading,"I10 first multiplicities differ");terminal.equals(expected_terminal,"I10 terminal coefficients differ");
    evaluations.equals(expected_evaluations,"I10 exact transition audit differs");status.equals(expected_status,"I10 feature statuses differ");
    oracle_status.equals(expected_oracle,"I10 oracle statuses differ");states.equals(flat_input,"I10 changed states");storage.equals(immutable,"I10 lookup mutated prepared tables");
}
__global__ void math_probe(const Word* input,Word* sums,Bit* fits,double* converted,float* logs,Index count){
    Index i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const Word low=input[4ULL*i],high=input[4ULL*i+1];Word a=low,b=high;
    fits[i]=cnot103_i10::add(a,b,input[4ULL*i+2],input[4ULL*i+3]);sums[2ULL*i]=a;sums[2ULL*i+1]=b;
    converted[i]=cnot103_i10::as_double(low,high);logs[i]=log1p(converted[i]);
}
void arithmetic(){
    std::vector<std::pair<Wide,Wide>> cases{{0,0},{~Wide{0},1},{Wide{1}<<64,~Wide{0}},{~Wide{0},~Wide{0}}};
    for(Index bit=0;bit<128;++bit){const Wide value=Wide{1}<<bit;cases.push_back({value-1,1});cases.push_back({value,value});}
    for(Index bit=64;bit<128;++bit){const Index shift=bit-52;const Wide half=Wide{1}<<(shift-1);
        for(Index parity=0;parity<2;++parity){const Wide base=(Wide{1}<<bit)+(Wide{2+parity}<<shift);
            for(int offset=-1;offset<=1;++offset)cases.push_back({base+half+offset,half});}
    }
    Word seed=0x39b75acf1658ULL;auto random=[&](){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;};
    for(Index i=0;i<1024;++i){Wide x=Wide{random()}<<64;x|=random();Wide y=Wide{random()}<<64;y|=random();cases.push_back({x,y});}
    std::vector<Word> input,expected_sums;std::vector<Bit> expected_fits;std::vector<double> expected_double;std::vector<float> expected_logs;
    for(auto pair:cases){input.push_back(pair.first);input.push_back(pair.first>>64);input.push_back(pair.second);input.push_back(pair.second>>64);
        const Wide sum=pair.first+pair.second;expected_sums.push_back(sum);expected_sums.push_back(sum>>64);expected_fits.push_back(pair.first<=~Wide{0}-pair.second);
        expected_double.push_back(static_cast<double>(pair.first));expected_logs.push_back(std::log1p(expected_double.back()));}
    Buffer<Word> x(input.size()),sums(expected_sums.size());Buffer<Bit> fits(cases.size());Buffer<double> converted(cases.size());Buffer<float> logs(cases.size());x.upload(input);
    math_probe<<<(cases.size()+127)/128,128>>>(x.data,sums.data,fits.data,converted.data,logs.data,cases.size());checked(cudaDeviceSynchronize());
    sums.equals(expected_sums,"I10 u128 addition differs");fits.equals(expected_fits,"I10 u128 overflow differs");converted.equals(expected_double,"I10 u128 double rounding differs");
    const auto actual_logs=logs.read();for(Index i=0;i<actual_logs.size();++i)require(std::fabs(actual_logs[i]-expected_logs[i])<=1e-5f,"I10 log projection differs");
    std::printf("I10 %zu independent u128 carry/overflow/rounding/log probes PASS\n",cases.size());
}
int main(){try{
    arithmetic();Index cases=0;
    for(Word degree:std::vector<Word>{0,1,2,3,4,5,63,64,65,66,127,128,129,130,131,~Word{0}}){run({2,degree,3,~Word{0}});++cases;}
    for(Word degree=0;degree<=4;++degree)for(Word states:std::vector<Word>{0,1,2,3})for(Word transitions:std::vector<Word>{0,2,5,6,11,12,17,18}){run({2,degree,states,transitions});++cases;}
    for(Word degree=0;degree<=5;++degree){run({3,degree,500,10000});++cases;}
    for(Word transitions:std::vector<Word>{1475,1476}){run({3,5,500,transitions});++cases;}
    for(Word states:std::vector<Word>{1,6,20,500}){run({3,3,states,10000});++cases;}
    for(Word degree=0;degree<=3;++degree){run({4,degree,5000,10000});++cases;}
    run({65,1,4160,4160});run({65,0,0,0});run({144,0,0,0});run({144,1,10,1});cases+=4;
    std::printf("I10 %u native fixture batches: direct words, dense GL2 recurrence, exact coefficients, guards and wide keys PASS\n",cases);
    checked(cudaDeviceReset());return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
