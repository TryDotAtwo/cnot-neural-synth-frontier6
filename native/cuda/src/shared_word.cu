#include "cnot103/shared_word.h"
#include "shared_word_rng.cuh"
#include "shared_word_moments.cuh"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
using namespace cnot103_shared_word;
using Byte = std::uint8_t;
constexpr U64 byte_limit = std::numeric_limits<std::int64_t>::max();

__global__ void prepare(Cnot103SharedWordPrepare a, bool admitted) {
    const U32 lane = threadIdx.x, n = a.config.dimension, words = (n + 63) / 64;
    const double lambda = __dmul_rn(n, a.config.tau);
    bool valid = admitted && isfinite(lambda);
    U64 total = 0;
    NumpyRng rng(a.config.seed);
    for (U64 s = 0; valid && s < a.config.samples; ++s) {
        U64 steps = 0;
        if (lane == 0) {
            if (n > 1)
                valid = rng.poisson(lambda, steps);
            if (steps > ~U64(0) - total)
                valid = false;
            if (valid)
                total += steps;
        }
        valid = __shfl_sync(full_warp, int(valid), 0) != 0;
        if (!valid)
            break;
        steps = __shfl_sync(full_warp, static_cast<unsigned long long>(steps), 0);
        U64* transform = a.transforms + s * words * n;
        for (U32 index = lane; index < words * n; index += 32) {
            const U32 row = index % n, word = index / n;
            transform[index] = word == row / 64 ? U64(1) << (row % 64) : 0;
        }
        __syncwarp();
        for (U64 step = 0; step < steps; ++step) {
            U32 control = 0, target = 0;
            if (lane == 0) {
                control = rng.integer(n);
                target = rng.integer(n - 1);
                target += target >= control;
            }
            control = __shfl_sync(full_warp, control, 0);
            target = __shfl_sync(full_warp, target, 0);
            // A fixed lane owns each input word for the entire sampled program.
            for (U32 word = lane; word < words; word += 32)
                transform[word * n + target] ^= transform[word * n + control];
        }
    }
    if (lane == 0) {
        *a.total_gates = valid ? total : 0;
        *a.status = valid ? 1 : 4;
    }
}

__global__ void statuses(Eval a) {
    for (U64 sample = U64(blockIdx.x) * blockDim.x + threadIdx.x; sample < a.batch;
         sample += U64(gridDim.x) * blockDim.x) {
        for (int f = 0; f < 4; ++f)
            if (a.selected & (1u << f)) {
                Byte code = *a.profile_status;
                if (code != 1 && code != 4)
                    asm volatile("trap;");
                // Every raw moment is exact and bounded by n²*samples <= 2²².
                // The remaining source nonfinite case is G10's explicit f32 tau.
                if (code == 1 && f == 3 && !isfinite(__double2float_rn(a.config.tau)))
                    code = 3;
                a.statuses[f][sample] = code;
            }
    }
}

__global__ void pack_input(Eval a, U32 start, U64 groups) {
    if (*a.profile_status != 1)
        return;
    const U32 n = a.config.dimension, w32 = (n + 31) / 32, w64 = (n + 63) / 64,
              lane = threadIdx.x & 31;
    for (U64 i = (U64(blockIdx.x) * blockDim.x + threadIdx.x) / 32; i < groups;
         i += U64(gridDim.x) * blockDim.x / 32) {
        const U32 column = (i / w32) % n, row = 32 * (i % w32) + lane;
        const U64 sample = start + i / (U64(n) * w32);
        const bool bit =
            row < n && ((a.states[(sample * n + row) * w64 + column / 64] >> (column % 64)) & 1);
        const U32 packed = __ballot_sync(full_warp, bit);
        if (!lane)
            a.input_columns[i] = packed;
    }
}

__global__ void transform_columns(Eval a, U64 groups) {
    if (*a.profile_status != 1)
        return;
    const U32 n = a.config.dimension, w32 = (n + 31) / 32, w64 = (n + 63) / 64,
              lane = threadIdx.x & 31;
    const U32 samples = a.config.samples;
    for (U64 i = (U64(blockIdx.x) * blockDim.x + threadIdx.x) / 32; i < groups;
         i += U64(gridDim.x) * blockDim.x / 32) {
        const U32 s = i % samples;
        const U64 group = i / samples, column = group / w32;
        const U32 row = 32 * (group % w32) + lane;
        U64 dot = 0;
        if (row < n) {
            for (U32 word = 0; word < w64; ++word) {
                U64 input = a.input_columns[column * w32 + 2 * word];
                if (2 * word + 1 < w32)
                    input |= U64(a.input_columns[column * w32 + 2 * word + 1]) << 32;
                dot ^= input & a.transforms[(U64(s) * w64 + word) * n + row];
            }
        }
        const U32 packed = __ballot_sync(full_warp, __popcll(dot) & 1);
        if (!lane)
            a.sample_columns[i] = packed;
    }
}

__global__ void column_weights(Eval a, U64 count) {
    if (*a.profile_status != 1)
        return;
    const U32 samples = a.config.samples, words = (a.config.dimension + 31) / 32;
    for (U64 i = U64(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += U64(blockDim.x) * gridDim.x) {
        U32 sum = 0;
        for (U32 word = 0; word < words; ++word)
            sum += __popc(a.sample_columns[((i / samples) * words + word) * samples + i % samples]);
        a.weights[i] = sum;
    }
}

struct Capacity {
    U64 bytes = 0;
    bool add(U64 rows, U64 per_row, U64 supplied, U64 size) {
        if (per_row && rows > (byte_limit / size) / per_row)
            return false;
        if (rows * per_row > supplied || supplied > (byte_limit - bytes) / size)
            return false;
        bytes += supplied * size;
        return true;
    }
};
unsigned blocks(U64 count, U32 items = 1) {
    return unsigned(std::min<U64>(65535, (count + items - 1) / items));
}
int launch_status() { return cudaGetLastError() == cudaSuccess ? 0 : 3; }
} // namespace

extern "C" std::int32_t cnot103_cuda_shared_word_layout(const Cnot103SharedWordConfig* cfg,
                                                        U64* words) {
    if (!cfg || !words || !cfg->dimension || cfg->dimension > (1u << 24) || !cfg->max_n ||
        cfg->samples < 2 || !std::isfinite(cfg->tau) || cfg->tau < 0.)
        return 1;
    const U64 n = cfg->dimension;
    constexpr U64 max_cells = (256ULL << 20) / (8 * 8);
    *words = n > cfg->max_n || cfg->samples > max_cells / (n * n)
                 ? 0
                 : cfg->samples * n * ((n + 63) / 64);
    return 0;
}

extern "C" std::int32_t cnot103_cuda_prepare_shared_word(const Cnot103SharedWordPrepare* a) {
    U64 words;
    if (!a || cnot103_cuda_shared_word_layout(&a->config, &words) || !a->total_gates ||
        !a->status || (words && !a->transforms))
        return 1;
    Capacity cap;
    if (!cap.add(1, words, a->transform_len, 8) || !cap.add(1, 1, a->total_len, 8) ||
        !cap.add(1, 1, a->status_len, 1))
        return 2;
    prepare<<<1, 32, 0, static_cast<cudaStream_t>(a->stream)>>>(*a, words != 0);
    return launch_status();
}

extern "C" std::int32_t cnot103_cuda_eval_shared_word(const Eval* a) {
    U64 transform_len;
    if (!a || cnot103_cuda_shared_word_layout(&a->config, &transform_len) || !a->states ||
        !a->profile_status || !a->total_gates || !a->batch || a->batch > 0x7fffffffU ||
        !a->workers || a->workers > a->batch || !a->selected || a->selected >= 16)
        return 1;
    if (transform_len && (!a->transforms || !a->input_columns || !a->sample_columns || !a->weights))
        return 1;
    const U64 n = a->config.dimension, w32 = (n + 31) / 32, w64 = (n + 63) / 64,
              samples = a->config.samples;
    const U64 input_per_worker = transform_len ? n * w32 : 0;
    const U64 weights_per_worker = transform_len ? n * samples : 0;
    Capacity cap;
    if (!cap.add(a->batch, n * w64, a->state_len, 8) ||
        !cap.add(1, transform_len, a->transform_len, 8) || !cap.add(1, 1, a->total_len, 8) ||
        !cap.add(1, 1, a->profile_status_len, 1) ||
        !cap.add(a->workers, input_per_worker, a->input_len, 4) ||
        !cap.add(a->workers, input_per_worker * samples, a->sample_len, 4) ||
        !cap.add(a->workers, weights_per_worker, a->weight_len, 4))
        return 2;
    constexpr U32 widths[] = {1, 5, 2, 7};
    for (int f = 0; f < 4; ++f) {
        const bool selected = a->selected & (1u << f);
        if (selected && (!a->values[f] || !a->statuses[f]))
            return 1;
        const U64 per_row = selected ? (f == 0 ? n : n * n) * widths[f] : 0;
        if (!cap.add(a->batch, per_row, a->value_len[f], 4) ||
            !cap.add(a->batch, selected ? 1 : 0, a->status_len[f], 1))
            return 2;
    }
    const auto stream = static_cast<cudaStream_t>(a->stream);
    statuses<<<blocks(a->batch, 128), 128, 0, stream>>>(*a);
    if (launch_status())
        return 3;
    if (!transform_len) {
        for (int f = 0; f < 4; ++f)
            if (a->selected & (1u << f)) {
                const U64 bytes = U64(a->batch) * (f == 0 ? n : n * n) * widths[f] * sizeof(float);
                if (cudaMemsetAsync(a->values[f], 0, bytes, stream) != cudaSuccess)
                    return 3;
            }
        return 0;
    }
    for (U32 start = 0; start < a->batch;) {
        const U32 count = std::min(a->workers, a->batch - start);
        const U64 columns = U64(count) * n, input_groups = columns * w32,
                  sample_groups = input_groups * samples;
        pack_input<<<blocks(input_groups, 4), 128, 0, stream>>>(*a, start, input_groups);
        if (launch_status())
            return 3;
        transform_columns<<<blocks(sample_groups, 4), 128, 0, stream>>>(*a, sample_groups);
        if (launch_status())
            return 3;
        column_weights<<<blocks(columns * samples, 128), 128, 0, stream>>>(*a, columns * samples);
        if (launch_status())
            return 3;
        if (a->selected & 1) {
            node_means<<<blocks(columns), workers_per_block, 0, stream>>>(*a, start, count);
            if (launch_status())
                return 3;
        }
        if (a->selected & 14) {
            pair_moments<<<blocks(columns * n), workers_per_block, 0, stream>>>(*a, start, count);
            if (launch_status())
                return 3;
        }
        start += count;
    }
    return 0;
}
