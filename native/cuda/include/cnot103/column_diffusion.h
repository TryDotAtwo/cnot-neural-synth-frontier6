#pragma once

#include "cnot103/cnot_cuda.h"

// Preparation produces one immutable class table. All element counts refer to
// individual entries of the pointed-to type. No device allocation is made here.
struct Cnot103ColumnConfig {
    double taus[4], floor;
    std::uint32_t dimension, max_n;
};

struct Cnot103ColumnLayout {
    std::uint32_t classes, weight_stride;
};

struct Cnot103ColumnPrepare {
    Cnot103ColumnConfig config;
    double* work;               // [18,classes], see ProfileWork
    std::int32_t* predecessors; // [4,classes], in ascending source order
    double* weights;            // [4,weight_stride]
    std::uint8_t* phase_status; // class catalog followed by four horizons
    float* descriptors;         // [classes,47], shared G01/G02, G03, G04
    std::uint8_t* status;       // one status for the entire profile
    void* stream;
    std::uint64_t work_len, predecessor_len, weight_len, phase_len, descriptor_len, status_len;
};

struct Cnot103ColumnProject {
    const float* weight;           // A05 channel 0 (row) or A06 (column)
    const float* missing_diagonal; // A11, equal to 1 - M[i,i]
    const float* descriptors;
    const std::uint8_t* profile_status;
    float* values;
    std::uint8_t* status;
    void* stream;
    std::uint64_t weight_batch_stride, weight_wire_stride, diagonal_batch_stride;
    std::uint64_t weight_len, diagonal_len, descriptor_len, profile_status_len, value_len,
        status_len;
    std::uint32_t batch, dimension, max_n, ordinal;
};

extern "C" {
CNOT103_CUDA_API std::int32_t cnot103_cuda_column_layout(const Cnot103ColumnConfig* config,
                                                         Cnot103ColumnLayout* layout);
CNOT103_CUDA_API std::int32_t
cnot103_cuda_prepare_column_diffusion(const Cnot103ColumnPrepare* args);
CNOT103_CUDA_API std::int32_t
cnot103_cuda_project_column_diffusion(const Cnot103ColumnProject* args);
}
