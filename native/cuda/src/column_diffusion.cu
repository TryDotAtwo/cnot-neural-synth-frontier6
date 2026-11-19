#include "cnot103/column_diffusion.h"
#include "diffusion_chain.cuh"

namespace {
using namespace cnot103_diffusion;
using ColumnWork = ProfileWork<4, int>;
constexpr int channels = 47;

U32 class_count(U32 n, U32 max_n) {
    // For n >= 2049, the first 1024 factors in the central binomial
    // recurrence are >= 2. The Rust f64 size necessarily overflows.
    return n > max_n || n >= 2049 ? 0 : 2 * n - 1;
}

__device__ void add_edge(Cnot103ColumnPrepare& a, const ColumnWork& w, int classes, int dest,
                         int& slot, int source, int count, int total) {
    if (count == 0)
        return;
    const int offset = slot++ * classes + dest;
    a.predecessors[offset] = source;
    w.probability[offset] = __ddiv_rn(count, total);
}

__global__ void build_catalog(Cnot103ColumnPrepare a, int classes) {
    const int n = a.config.dimension;
    ColumnWork w(a.work, classes);
    for (int dest = threadIdx.x; dest < classes; dest += blockDim.x) {
        const int bit = (dest + 1) / n;
        const int k = (dest + 1) % n;
        const int count = min(k, n - 1 - k);
        double size = 1.0;
        for (int i = 1; i <= count; ++i) {
            size = __dmul_rn(size, __ddiv_rn(n - 1 - count + i, i));
        }
        w.log_size[dest] = size;
        int slot = 0;
        if (n == 1) {
            add_edge(a, w, classes, dest, slot, dest, 1, 1);
        } else {
            const int total = n * (n - 1);
            // Source-ordered incoming edges reproduce the Rust scatter sum.
            if (bit == 1)
                add_edge(a, w, classes, dest, slot, k - 1, k, total);
            if (k > 0)
                add_edge(a, w, classes, dest, slot, dest - 1, (bit + k - 1) * (n - k), total);
            add_edge(a, w, classes, dest, slot, dest, (n - bit - k) * (n - 1), total);
            if (k + 1 < n)
                add_edge(a, w, classes, dest, slot, dest + 1, (k + 1) * (bit + k), total);
            if (bit == 0)
                add_edge(a, w, classes, dest, slot, n + k - 1, k, total);
        }
        for (; slot < 4; ++slot) {
            a.predecessors[slot * classes + dest] = -1;
            w.probability[slot * classes + dest] = 0.0;
        }
    }
    __syncthreads();
    __shared__ double total;
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < classes; ++i)
            sum = __dadd_rn(sum, w.log_size[i]);
        total = sum;
        a.phase_status[0] = isfinite(sum) && sum > 0 ? computed : limited;
    }
    __syncthreads();
    if (a.phase_status[0] != computed)
        return;
    for (int i = threadIdx.x; i < classes; i += blockDim.x) {
        const double size = w.log_size[i];
        w.stationary[i] = __ddiv_rn(size, total);
        w.log_size[i] = log(size);
    }
}

__global__ void finish_profile(Cnot103ColumnPrepare a, int classes) {
    bool available = classes != 0;
    for (int t = 0; available && t < 5; ++t)
        available = a.phase_status[t] == computed;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
        *a.status = available ? computed : limited;
    if (!available || i >= classes)
        return;
    ColumnWork w(a.work, classes);
    const double stationary = w.stationary[i], log_stationary = log(stationary);
    float* descriptor = a.descriptors + i * channels;
    for (int t = 0; t < 4; ++t) {
        const double raw = w.mass[t * classes + i];
        const bool exact_zero = a.config.taus[t] == 0 && i != a.config.dimension - 1;
        const bool underflow = (raw == 0 && !exact_zero) ||
                               (raw > 0 && __double2float_rn(raw) == 0) ||
                               (stationary > 0 && __double2float_rn(stationary) == 0);
        const double log_mass = log(fmax(raw, a.config.floor));
        float* node = descriptor + 7 * t;
        node[0] = __double2float_rn(raw);
        node[1] = __double2float_rn(w.log_size[i]);
        node[2] = __double2float_rn(stationary);
        node[3] = __double2float_rn(__dsub_rn(log_mass, log_stationary));
        node[4] = exact_zero;
        node[5] = underflow;
        node[6] = raw < a.config.floor;
        float* logs = descriptor + 28 + 4 * t;
        logs[0] = __double2float_rn(log_mass);
        logs[1] = node[4];
        logs[2] = node[5];
        logs[3] = node[6];
        if (t < 3)
            descriptor[44 + t] = __double2float_rn(__dsub_rn(w.mass[(t + 1) * classes + i], raw));
    }
}

__global__ void project(Cnot103ColumnProject a, int classes, U32 width, int offset) {
    extern __shared__ int selected[];
    const U64 sample = blockIdx.x;
    Byte status = *a.profile_status;
    if (status == computed) {
        if (classes == 0)
            asm volatile("trap;");
        bool invalid = false;
        for (U32 i = threadIdx.x; i < a.dimension; i += blockDim.x) {
            const float missing = a.missing_diagonal[sample * a.diagonal_batch_stride + i];
            const float weight =
                a.weight[sample * a.weight_batch_stride + i * a.weight_wire_stride];
            if ((missing != 0.f && missing != 1.f) || !(weight >= 1.f && weight <= a.dimension) ||
                truncf(weight) != weight) {
                invalid = true;
                continue;
            }
            const int bit = missing == 0.f, other = static_cast<int>(weight) - bit;
            if (other < 0 || other >= static_cast<int>(a.dimension))
                invalid = true;
            else
                selected[i] = bit * a.dimension + other - 1;
        }
        if (__syncthreads_or(invalid))
            status = inapplicable;
    } else if (status != limited) {
        asm volatile("trap;");
    }
    if (threadIdx.x == 0)
        a.status[sample] = status;
    const U64 elements = U64(a.dimension) * width;
    for (U64 i = threadIdx.x; i < elements; i += blockDim.x) {
        float value = 0.f;
        if (status == computed) {
            value = a.descriptors[selected[i / width] * channels + offset + i % width];
            // A valid zero floor may give -inf in an unused class or channel.
            // Match packet validation only after the entire feature is available.
            if (!isfinite(value))
                asm volatile("trap;");
        }
        a.values[sample * elements + i] = value;
    }
}

} // namespace

extern "C" std::int32_t cnot103_cuda_column_layout(const Cnot103ColumnConfig* config,
                                                   Cnot103ColumnLayout* layout) {
    if (!config || !layout || !config->dimension || config->dimension > (1u << 24) ||
        !config->max_n || !(config->floor >= 0 && config->floor < 1.))
        return 1;
    const U32 classes = class_count(config->dimension, config->max_n);
    U32 stride;
    if (!weight_capacity(config->taus, config->dimension, classes != 0, stride))
        return 1;
    *layout = {classes, stride};
    return 0;
}

extern "C" std::int32_t cnot103_cuda_prepare_column_diffusion(const Cnot103ColumnPrepare* a) {
    Cnot103ColumnLayout layout{};
    if (!a || cnot103_cuda_column_layout(&a->config, &layout) || !a->status)
        return 1;
    const U64 c = layout.classes;
    if (c && (!a->work || !a->predecessors || !a->weights || !a->phase_status || !a->descriptors))
        return 1;
    if (!extent(18 * c, a->work_len, 8) || !extent(4 * c, a->predecessor_len, 4) ||
        !extent(4ULL * layout.weight_stride, a->weight_len, 8) ||
        !extent(c ? 5 : 0, a->phase_len, 1) || !extent(channels * c, a->descriptor_len, 4) ||
        !extent(1, a->status_len, 1) ||
        !total_fits({8 * a->work_len, 4 * a->predecessor_len, 8 * a->weight_len, a->phase_len,
                     4 * a->descriptor_len, a->status_len}))
        return 2;
    const auto stream = static_cast<cudaStream_t>(a->stream);
    if (c) {
        build_catalog<<<1, threads, 0, stream>>>(*a, c);
        if (cudaGetLastError() != cudaSuccess)
            return 3;
        propagate<4><<<4, threads, 0, stream>>>(*a, int(c), int(a->config.dimension) - 1,
                                                layout.weight_stride);
        if (cudaGetLastError() != cudaSuccess)
            return 3;
    }
    finish_profile<<<std::max(1u, (layout.classes + threads - 1) / threads), threads, 0, stream>>>(
        *a, c);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" std::int32_t cnot103_cuda_project_column_diffusion(const Cnot103ColumnProject* a) {
    if (!a || !a->weight || !a->missing_diagonal || !a->profile_status || !a->values ||
        !a->status || !a->batch || a->batch > 0x7fffffffu || !a->dimension ||
        a->dimension > (1u << 24) || !a->max_n || a->ordinal < 1 || a->ordinal > 4)
        return 1;
    const U32 c = class_count(a->dimension, a->max_n);
    if (c && !a->descriptors)
        return 1;
    const U32 width = a->ordinal <= 2 ? 28 : (a->ordinal == 3 ? 16 : 3);
    const int offset = a->ordinal <= 2 ? 0 : (a->ordinal == 3 ? 28 : 44);
    const U64 nodes = U64(a->batch) * a->dimension;
    if (!readable_view(a->batch, a->dimension, a->weight_batch_stride, a->weight_wire_stride,
                       a->weight_len) ||
        !readable_view(a->batch, a->dimension, a->diagonal_batch_stride, 1, a->diagonal_len) ||
        !extent(U64(c) * channels, a->descriptor_len, 4) || !extent(1, a->profile_status_len, 1) ||
        !extent(nodes * width, a->value_len, 4) || !extent(a->batch, a->status_len, 1) ||
        !total_fits({4 * a->weight_len, 4 * a->diagonal_len, 4 * a->descriptor_len,
                     a->profile_status_len, 4 * a->value_len, a->status_len}))
        return 2;
    const auto stream = static_cast<cudaStream_t>(a->stream);
    project<<<a->batch, threads, c ? a->dimension * sizeof(int) : 0, stream>>>(*a, c, width,
                                                                               offset);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
