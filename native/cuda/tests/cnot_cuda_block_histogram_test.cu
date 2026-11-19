// Literal patterns, independent of the Rust producer and model adapter.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    std::size_t size;
    explicit DeviceBuffer(std::size_t count) : size(count) {
        if (size) checked(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void upload(const std::vector<T>& values) {
        if (values.size() != size) throw std::runtime_error("fixture upload extent mismatch");
        if (size) checked(cudaMemcpy(data, values.data(), size*sizeof(T), cudaMemcpyHostToDevice));
    }
    void equals(const std::vector<T>& expected) {
        std::vector<T> values(size);
        if (size) checked(cudaMemcpy(values.data(), data, size*sizeof(T), cudaMemcpyDeviceToHost));
        if (values != expected) throw std::runtime_error("literal histogram mismatch");
    }
};

void fixture(std::uint32_t n, std::uint32_t p, std::uint64_t q, bool emit) {
    const auto words=(n+63)/64;
    const std::uint32_t sources[3]={0,1,n==144?66u:2u};
    const std::uint32_t codes[3][4]={{0,0,0,0},{1,1,3,5},{0,2,2,4}};
    const std::uint32_t patterns[3][4]={{0,0,0,0},{0,1,3,5},{0,2,4,0}};
    const std::uint32_t counts[3][4]={{4,0,0,0},{0,2,1,1},{1,2,1,0}};
    const std::uint32_t sizes[3]={1,4,3};
    const std::uint64_t pairs[3]={0,3,1};
    const std::uint32_t representatives[3][4]={{0,0,0,0},{0,n-3,n-1,n},{0,n-2,n,0}};
    const std::uint32_t pair_left[3]={1,1,2}, pair_right[3]={2,3,3};
    std::vector<std::uint64_t> state(3*n*words,0), source_mask(words,0), target_mask(words,0);
    for (auto wire:sources) source_mask[wire/64] |= std::uint64_t{1}<<(wire%64);
    for (std::uint32_t target=n-4;target<n;++target) target_mask[target/64] |= std::uint64_t{1}<<(target%64);
    for (std::uint32_t b=0;b<3;++b) {
        for (std::uint32_t row=0;row<n;++row) state[(b*n+row)*words+row/64] |= std::uint64_t{1}<<(row%64);
        for (std::uint32_t t=0;t<4;++t)
            for (std::uint32_t bit=0;bit<3;++bit)
                if (codes[b][t]&(1u<<bit)) state[(b*n+n-4+t)*words+sources[bit]/64] |= std::uint64_t{1}<<(sources[bit]%64);
    }
    DeviceBuffer<std::uint64_t> matrices(state.size()), sm(words), tm(words), statistics(12);
    matrices.upload(state); sm.upload(source_mask); tm.upload(target_mask);
    DeviceBuffer<std::uint32_t> frequencies(3*(n+1)), ids(emit||q?3*p:0);
    DeviceBuffer<float> values(emit?3*p*3:0), incidence(emit?3*p*n*3:0), pair_values(3*q), pair_incidence(3*q*n*3);
    DeviceBuffer<std::uint8_t> valid(emit?3*p:0), pair_valid(3*q);
    Cnot103BlockHistogramArgs a{};
    a.states=matrices.data; a.source_masks=sm.data; a.target_masks=tm.data;
    a.frequencies=frequencies.data; a.statistics=statistics.data; a.ordered_ids=ids.data;
    a.pattern_values=values.data; a.pattern_valid=valid.data; a.pattern_incidence=incidence.data;
    a.pair_values=pair_values.data; a.pair_valid=pair_valid.data; a.pair_incidence=pair_incidence.data;
    a.pair_capacity=q; a.frequency_elements=frequencies.size; a.statistic_elements=statistics.size;
    a.pattern_groups=ids.size; a.pattern_incidence_elements=incidence.size;
    a.pair_groups=pair_valid.size; a.pair_incidence_elements=pair_incidence.size;
    a.batch=3; a.contexts=1; a.dimension=n; a.words_per_row=words; a.pattern_capacity=p; a.emit_patterns=emit?1:0;
    if (cnot103_cuda_launch_block_histograms(&a)!=0) throw std::runtime_error("histogram launch rejected");
    checked(cudaDeviceSynchronize());
    statistics.equals({1,0,0,0, 4,4,2,3, 3,3,2,1});
    std::vector<std::uint32_t> expected_ids(ids.size,0xffffffffu);
    std::vector<float> expected_values(values.size,0), expected_incidence(incidence.size,0), expected_pairs(pair_values.size,0), expected_pair_incidence(pair_incidence.size,0);
    std::vector<std::uint8_t> expected_valid(valid.size,0), expected_pair_valid(pair_valid.size,0);
    for (std::uint32_t b=0;b<3;++b) {
        if (sizes[b]>p) continue;
        for (std::uint32_t group=0;group<sizes[b];++group) {
            if (emit||q) expected_ids[b*p+group]=representatives[b][group];
            if (!emit) continue;
            expected_valid[b*p+group]=1;
            expected_values[(b*p+group)*3]=counts[b][group];
            expected_values[(b*p+group)*3+1]=1;
            expected_values[(b*p+group)*3+2]=group?counts[b][group]:0;
            for (std::uint32_t bit=0;bit<3;++bit) {
                expected_incidence[((b*p+group)*n+sources[bit])*3]=1;
                expected_incidence[((b*p+group)*n+sources[bit])*3+1]=(patterns[b][group]>>bit)&1;
            }
            for (std::uint32_t t=0;t<4;++t)
                expected_incidence[((b*p+group)*n+n-4+t)*3+2]=codes[b][t]==patterns[b][group]?1:0;
        }
        if (pairs[b]>q) continue;
        for (std::uint64_t pair=0;pair<pairs[b];++pair) {
            expected_pair_valid[b*q+pair]=1;
            expected_pairs[b*q+pair]=b==1?1.0f:0.0f;
            for (std::uint32_t bit=0;bit<3;++bit) {
                expected_pair_incidence[((b*q+pair)*n+sources[bit])*3]=1;
                expected_pair_incidence[((b*q+pair)*n+sources[bit])*3+1]=(patterns[b][pair_left[pair]]>>bit)&1;
                expected_pair_incidence[((b*q+pair)*n+sources[bit])*3+2]=(patterns[b][pair_right[pair]]>>bit)&1;
            }
        }
    }
    ids.equals(expected_ids); values.equals(expected_values); valid.equals(expected_valid); incidence.equals(expected_incidence);
    pair_values.equals(expected_pairs); pair_valid.equals(expected_pair_valid); pair_incidence.equals(expected_pair_incidence);
    // Capacity failures must be rejected before launching any work.
    --a.statistic_elements;
    if (cnot103_cuda_launch_block_histograms(&a)!=1) throw std::runtime_error("short output capacity accepted");
    std::printf("histogram fixture PASS n=%u P=%u Q=%llu emit=%d\n",n,p,static_cast<unsigned long long>(q),emit); std::fflush(stdout);
}
int main() {
    try {
        for (auto n:{8u,65u,144u}) {
            fixture(n,5,4,true); fixture(n,2,4,true); fixture(n,5,1,true);
            fixture(n,5,4,false); fixture(n,5,0,false);
        }
        checked(cudaDeviceReset());
        std::puts("block histogram literal fixtures PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"%s\n",error.what()); return 1;
    }
}
