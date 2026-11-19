#pragma once
#include "cnot103/cnot_cuda.h"

struct Cnot103PairDiffusionConfig {
    double taus[4], floor;
    std::uint64_t max_pair_states;
    std::uint32_t dimension;
};

struct Cnot103PairDiffusionLayout {
    std::uint64_t classes;
    std::uint32_t weight_stride;
};

struct Cnot103PairDiffusionPrepare {
    Cnot103PairDiffusionConfig config;
    double* work;               // [30,classes]: catalog, 16 edges, steps, mass
    std::int64_t* predecessors; // [16,classes] in (source,edge) order
    double* log_factorial;      // [dimension+1]
    double* weights;            // [4,weight_stride]
    std::uint8_t* phase_status; // catalog followed by four horizons
    float* descriptors;         // [classes,28]
    std::uint8_t* status;
    void* stream;
    std::uint64_t work_len, predecessor_len, factorial_len, weight_len, phase_len, descriptor_len,
        status_len;
};

struct Cnot103PairDiffusionProject {
    const float* column_weights;       // A06 [batch,n,1]
    const float* column_intersections; // A14 [batch,n,n,1]
    const float* descriptors;
    const std::uint8_t* profile_status;
    std::int64_t* class_ids; // [batch,n,n] when there is a catalog
    float* values;           // [batch,n,n,28]
    std::uint8_t* status;    // [batch]
    void* stream;
    std::uint64_t max_pair_states, weight_batch_stride, intersection_batch_stride;
    std::uint64_t weight_len, intersection_len, descriptor_len, profile_status_len, class_len,
        value_len, status_len;
    std::uint32_t batch, dimension;
};

extern "C" {
CNOT103_CUDA_API std::int32_t
cnot103_cuda_pair_diffusion_layout(const Cnot103PairDiffusionConfig* config,
                                   Cnot103PairDiffusionLayout* layout);
CNOT103_CUDA_API std::int32_t
cnot103_cuda_prepare_pair_diffusion(const Cnot103PairDiffusionPrepare* args);
CNOT103_CUDA_API std::int32_t
cnot103_cuda_project_pair_diffusion(const Cnot103PairDiffusionProject* args);
}
