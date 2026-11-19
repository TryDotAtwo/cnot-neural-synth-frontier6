// Invert a known nonidentity source basis and check its exact coefficients.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

void checked(cudaError_t result) {
    if (result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template <typename T> struct DeviceBuffer {
    T* data=nullptr;
    std::size_t size;
    explicit DeviceBuffer(std::size_t count):size(count) {
        if(size) checked(cudaMalloc(reinterpret_cast<void**>(&data),size*sizeof(T)));
    }
    ~DeviceBuffer(){if(data) cudaFree(data);}
    void upload(const std::vector<T>& values){
        if(values.size()!=size) throw std::runtime_error("fixture input size mismatch");
        if(size) checked(cudaMemcpy(data,values.data(),size*sizeof(T),cudaMemcpyHostToDevice));
    }
    void equals(const std::vector<T>& expected){
        std::vector<T> values(size);
        if(size) checked(cudaMemcpy(values.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));
        if(values!=expected) throw std::runtime_error("literal partial-shear mismatch");
    }
};

void fixture(std::uint32_t n,bool shift_columns,bool emit) {
    const auto width=n/2,targets=n-width,words=(n+63)/64,coefficient_words=(width+63)/64;
    const auto cells=static_cast<std::uint64_t>(n)*words;
    std::vector<std::uint64_t> states(3*cells,0);
    const auto column=[&](std::uint32_t wire){return (wire+(shift_columns?targets:0))%n;};
    const auto set=[&](std::uint32_t b,std::uint32_t row,std::uint32_t wire){
        const auto col=column(wire);
        states[b*cells+static_cast<std::uint64_t>(row)*words+col/64]|=std::uint64_t{1}<<(col%64);
    };
    for(std::uint32_t b=0;b<3;++b)
        for(std::uint32_t row=0;row<n;++row)
            if(b!=2||row!=0) set(b,row,row);
    // R1=e1+e0. At wide size R66=e66+R1. Thus raw e1 gives
    // coefficients {0,1}, and raw e66 gives coefficients {1,66}.
    set(1,1,0);
    if(n==144){set(1,66,0);set(1,66,1);}
    for(std::uint32_t target=width;target<n;++target) set(1,target,n==144?66:1);
    DeviceBuffer<std::uint64_t> matrices(states.size()),scratch(static_cast<std::size_t>(3)*width*(words+coefficient_words+1));
    DeviceBuffer<std::uint8_t> coefficients(static_cast<std::size_t>(3)*targets*width),status(3);
    DeviceBuffer<std::int64_t> gradients(emit?static_cast<std::size_t>(3)*width*width:0);
    matrices.upload(states);
    const auto result=cnot103_cuda_launch_basis_gradient(matrices.data,scratch.data,coefficients.data,status.data,gradients.data,
        3,n,words,scratch.size,coefficients.size,gradients.size,emit?1:0,nullptr);
    if(result!=0) throw std::runtime_error("partial-shear launch rejected");
    checked(cudaDeviceSynchronize());
    status.equals({0,1,2});
    const auto present=[&](std::uint32_t source){return n==144?(source==1||source==66):(source==0||source==1);};
    std::vector<std::uint8_t> expected(coefficients.size,0);
    for(std::uint32_t row=0;row<targets;++row)
        for(std::uint32_t source=0;source<width;++source)
            expected[(targets+row)*width+source]=present(source)?1:0;
    coefficients.equals(expected);
    std::vector<std::int64_t> expected_gradients(gradients.size,0);
    if(emit){
        for(std::uint32_t source=0;source<width;++source)
            for(std::uint32_t target=0;target<width;++target){
                if(source==target) continue;
                // With all target rows equal, each pair is one of three literal cases.
                const std::int64_t value=!present(target)?2:(present(source)?2-static_cast<std::int64_t>(targets):2+targets);
                expected_gradients[(static_cast<std::uint64_t>(width)+source)*width+target]=value;
            }
    }
    gradients.equals(expected_gradients);
    if(cnot103_cuda_launch_basis_gradient(matrices.data,scratch.data,coefficients.data,status.data,gradients.data,
        3,n,words,scratch.size-1,coefficients.size,gradients.size,emit?1:0,nullptr)!=1)
        throw std::runtime_error("short scratch accepted");
    std::printf("basis-gradient fixture PASS n=%u shifted=%d gradients=%d\n",n,shift_columns,emit);std::fflush(stdout);
}
int main(){
    try{
        for(auto n:{4u,65u,144u})
            for(bool shifted:{false,true})
                for(bool gradients:{false,true}) fixture(n,shifted,gradients);
        checked(cudaDeviceReset());
        std::puts("basis-gradient literal fixtures PASS");return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
