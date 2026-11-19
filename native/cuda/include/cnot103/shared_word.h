#pragma once
#include "cnot103/cnot_cuda.h"

struct Cnot103SharedWordConfig {
    double tau;
    std::uint64_t seed, samples;
    std::uint32_t dimension, max_n;
};
struct Cnot103SharedWordPrepare {
    Cnot103SharedWordConfig config;
    std::uint64_t* transforms; // [sample,input_word64,output_row]
    std::uint64_t* total_gates;
    std::uint8_t* status;
    void* stream;
    std::uint64_t transform_len, total_len, status_len;
};
struct Cnot103SharedWordEval {
    Cnot103SharedWordConfig config;
    const std::uint64_t* states;
    const std::uint64_t* transforms;
    const std::uint64_t* total_gates;
    const std::uint8_t* profile_status;
    std::uint32_t* input_columns;  // [worker,column,word32]
    std::uint32_t* sample_columns; // [worker,column,word32,sample]
    std::uint32_t* weights;        // [worker,column,sample]
    float* values[4];              // selected G07..G10, with 1/5/2/7 channels
    std::uint8_t* statuses[4];     // [batch] for each selected feature
    void* stream;
    std::uint64_t state_len, transform_len, total_len, profile_status_len;
    std::uint64_t input_len, sample_len, weight_len, value_len[4], status_len[4];
    std::uint32_t batch, workers, selected;
};
extern "C" {
// Static capacity decision only. The returned count is the number of u64 words
// in the transform buffer, or zero when the source width/working-set guard fires.
CNOT103_CUDA_API std::int32_t cnot103_cuda_shared_word_layout(const Cnot103SharedWordConfig* config,
                                                              std::uint64_t* transform_len);
CNOT103_CUDA_API std::int32_t
cnot103_cuda_prepare_shared_word(const Cnot103SharedWordPrepare* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_eval_shared_word(const Cnot103SharedWordEval* args);
}
