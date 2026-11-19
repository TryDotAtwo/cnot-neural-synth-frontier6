// Literal full words and intervals, independent of the Rust compiler.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

void checked(cudaError_t result){if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));}
template<typename T>struct DeviceBuffer{
    T* data=nullptr;std::size_t size;
    explicit DeviceBuffer(std::size_t count):size(count){if(size)checked(cudaMalloc(reinterpret_cast<void**>(&data),size*sizeof(T)));}
    ~DeviceBuffer(){if(data)cudaFree(data);}
    void upload(const std::vector<T>& values){
        if(values.size()!=size)throw std::runtime_error("fixture input extent mismatch");
        if(size)checked(cudaMemcpy(data,values.data(),size*sizeof(T),cudaMemcpyHostToDevice));
    }
    void equals(const std::vector<T>& expected,const char* label){
        std::vector<T> actual(size);
        if(size)checked(cudaMemcpy(actual.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));
        if(actual!=expected)throw std::runtime_error(label);
    }
};

void fixture(std::uint32_t n,std::uint64_t capacity){
    const auto words=(n+63)/64;
    const auto cells=static_cast<std::uint64_t>(n)*words;
    std::vector<std::uint64_t> input(4*cells,0);
    const auto set=[&](std::uint32_t b,std::uint32_t row,std::uint32_t column){input[b*cells+static_cast<std::uint64_t>(row)*words+column/64]|=std::uint64_t{1}<<(column%64);};
    for(std::uint32_t b=0;b<4;++b)
        for(std::uint32_t row=0;row<n;++row){
            const auto column=b==1?(row==0?n-1:(row==n-1?0:row)):row;
            set(b,row,column);
        }
    set(2,0,n-1);set(3,1,0);set(3,2,0);set(3,2,1);
    DeviceBuffer<std::uint64_t> states(input.size()),scratch(input.size());states.upload(input);
    DeviceBuffer<std::uint32_t> gates(4*capacity*2);
    DeviceBuffer<std::int64_t> lengths(4),lower(4);
    DeviceBuffer<std::uint8_t> status(4),verified(4);
    DeviceBuffer<float> f09(16),f10(20);
    Cnot103GeneralCompilerArgs a{};
    a.states=states.data;a.scratch=scratch.data;a.gates=gates.data;a.lengths=lengths.data;
    a.status=status.data;a.verified=verified.data;a.lower_bounds=lower.data;a.f09=f09.data;a.f10=f10.data;
    a.word_capacity=capacity;a.workspace_words=scratch.size;a.gate_elements=gates.size;
    a.batch=4;a.dimension=n;a.words_per_row=words;
    if(cnot103_cuda_launch_general_compiler(&a)!=0)throw std::runtime_error("general compiler launch rejected");
    checked(cudaDeviceSynchronize());
    const std::vector<std::vector<std::uint32_t>> literal={
        {},{n-1,0,0,n-1,n-1,0},{n-1,0},{1,2,0,2,0,1}};
    const std::int64_t lower_literal[4]={0,2,1,2};
    std::vector<std::uint32_t> expected_gates(gates.size,0);
    std::vector<std::int64_t> expected_lengths(4,0),expected_lower(4,0);
    std::vector<std::uint8_t> expected_status(4,4),expected_verified(4,0);
    std::vector<float> expected_f09(16,0),expected_f10(20,0);
    for(std::uint32_t b=0;b<4;++b){
        const auto count=literal[b].size()/2;
        if(count>capacity)continue;
        expected_status[b]=1;expected_verified[b]=1;expected_lengths[b]=count;expected_lower[b]=lower_literal[b];
        for(std::size_t i=0;i<literal[b].size();++i)expected_gates[b*capacity*2+i]=literal[b][i];
        const auto tight=count==static_cast<std::size_t>(lower_literal[b])?1.0f:0.0f;
        expected_f09[b*4]=count;expected_f09[b*4+1]=lower_literal[b];
        expected_f09[b*4+2]=count-lower_literal[b];expected_f09[b*4+3]=tight;
        expected_f10[b*5]=1;expected_f10[b*5+1]=1;expected_f10[b*5+2]=tight;expected_f10[b*5+4]=1;
    }
    gates.equals(expected_gates,"literal gate order mismatch");lengths.equals(expected_lengths,"literal length mismatch");
    lower.equals(expected_lower,"literal lower bound mismatch");status.equals(expected_status,"literal status mismatch");
    verified.equals(expected_verified,"literal replay mismatch");f09.equals(expected_f09,"F09 mismatch");f10.equals(expected_f10,"F10 mismatch");
    if(capacity==4){
        auto replay=[&](){
            if(cnot103_cuda_verify_cnot_words(states.data,gates.data,lengths.data,scratch.data,verified.data,
                4,n,words,capacity,scratch.size,gates.size,nullptr)!=0)throw std::runtime_error("verifier launch rejected");
            checked(cudaDeviceSynchronize());verified.equals({1,0,1,1},"corrupt word accepted");
        };
        auto corrupt=expected_gates;corrupt[capacity*2]=0;corrupt[capacity*2+1]=n-1;
        gates.upload(corrupt);replay(); // valid gates, incorrect endpoint
        corrupt[capacity*2]=n;gates.upload(corrupt);replay(); // out of range
        corrupt[capacity*2]=0;corrupt[capacity*2+1]=0;gates.upload(corrupt);replay(); // self gate
        gates.upload(expected_gates);
        for(const auto bad_length:{std::int64_t{-1},std::int64_t{5},std::int64_t{0}}){
            auto wrong=expected_lengths;wrong[1]=bad_length;lengths.upload(wrong);replay();
        }
    }
    --a.workspace_words;
    if(cnot103_cuda_launch_general_compiler(&a)!=1)throw std::runtime_error("short scratch accepted");
    std::printf("compiler fixture PASS n=%u Q=%llu\n",n,static_cast<unsigned long long>(capacity));std::fflush(stdout);
}
int main(){try{
    for(auto n:{3u,65u,144u,1024u})for(auto capacity:{0ull,2ull,4ull})fixture(n,capacity);
    checked(cudaDeviceReset());std::puts("general compiler literal fixtures PASS");return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}}
