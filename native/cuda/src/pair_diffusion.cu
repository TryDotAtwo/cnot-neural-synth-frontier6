#include "cnot103/pair_diffusion.h"
#include "diffusion_chain.cuh"

namespace {
using namespace cnot103_diffusion;
using Index = std::int64_t;
using PairWork = ProfileWork<16, Index>;

U64 class_count(U32 n, U64 limit) {
    if (n < 2)
        return 0;
    const U64 top = U64(n) + 3;
    const U64 first_two = top * (top - 1);
    // The source checks the multiplication before division, even when the
    // binomial coefficient itself would fit in a machine word.
    if (first_two > std::numeric_limits<U64>::max() / (top - 2))
        return 0;
    const U64 count = first_two * (top - 2) / 6 - 1 - 3ULL * n;
    constexpr U64 bytes_per_class = 30 * sizeof(double) + 16 * sizeof(Index) + 28 * sizeof(float);
    if (count > limit || count > byte_limit / bytes_per_class)
        return 0;
    return count;
}

struct Histogram {
    U32 h[4];
};

__device__ bool rank_two(Histogram h) {
    return int(h.h[1] != 0) + int(h.h[2] != 0) + int(h.h[3] != 0) >= 2;
}

__device__ U64 all_histograms(U64 r) { return (r + 3) * (r + 2) * (r + 1) / 6; }
__device__ U64 before_h0(U32 n, U32 h0) {
    // Each preceding slice omits exactly three rank-one histograms.
    return all_histograms(n) - all_histograms(n - h0) - 3ULL * h0;
}
__device__ U64 before_h1(U32 r, U32 h1) { return U64(h1) * (2ULL * r - h1 + 3) / 2; }

__device__ Index rank_class(U32 n, Histogram h) {
    const U32 h0 = h.h[0], h1 = h.h[1], h2 = h.h[2];
    // In lexicographic order the omitted triples in this slice are at 0, r
    // and the final position. A valid triple never occupies these positions.
    return before_h0(n, h0) + before_h1(n - h0, h1) + h2 - 1 - (h1 != 0);
}

__device__ Histogram unrank_class(U32 n, Index id) {
    U32 lo = 0, hi = n - 1;
    while (lo + 1 < hi) {
        const U32 mid = lo + (hi - lo) / 2;
        if (before_h0(n, mid) <= U64(id))
            lo = mid;
        else
            hi = mid;
    }
    const U32 h0 = lo, r = n - h0;
    U64 full = U64(id) - before_h0(n, h0) + 1;
    if (full >= r)
        ++full;
    lo = 0;
    hi = r;
    while (lo + 1 < hi) {
        const U32 mid = lo + (hi - lo) / 2;
        if (before_h1(r, mid) <= full)
            lo = mid;
        else
            hi = mid;
    }
    const U32 h1 = lo, h2 = full - before_h1(r, h1);
    return {{h0, h1, h2, r - h1 - h2}};
}

struct Incoming {
    Index source;
    int order;
    double probability;
};

__device__ void incoming_edges(Cnot103PairDiffusionPrepare a, const PairWork& work, Index classes,
                               Index destination, Histogram h) {
    Incoming edges[16];
    int count = 0;
    const U32 n = a.config.dimension;
    const double rate = double(U64(n) * (n - 1));
    for (int control = 0; control < 4; ++control) {
        for (int target = 0; target < 4; ++target) {
            Histogram source = h;
            const int result = control ^ target;
            if (result != target) {
                if (!source.h[result])
                    continue;
                --source.h[result];
                ++source.h[target];
            }
            if (!source.h[control] || source.h[target] <= U32(control == target))
                continue;
            if (!rank_two(source))
                asm volatile("trap;");
            const U64 ways = U64(source.h[control]) * (source.h[target] - (control == target));
            const Incoming edge{rank_class(n, source), control * 4 + target,
                                __ddiv_rn(double(ways), rate)};
            // Preserve the separate parallel edges of each source and their
            // original control/target order, including all self transitions.
            int pos = count++;
            while (pos > 0 &&
                   (edges[pos - 1].source > edge.source ||
                    (edges[pos - 1].source == edge.source && edges[pos - 1].order > edge.order))) {
                edges[pos] = edges[pos - 1];
                --pos;
            }
            edges[pos] = edge;
        }
    }
    for (int i = 0; i < 16; ++i) {
        const Index offset = i * classes + destination;
        a.predecessors[offset] = i < count ? edges[i].source : -1;
        work.probability[offset] = i < count ? edges[i].probability : 0.;
    }
}

__global__ void build_catalog(Cnot103PairDiffusionPrepare a, Index classes) {
    const U32 n = a.config.dimension;
    PairWork w(a.work, classes);
    if (threadIdx.x == 0) {
        a.log_factorial[0] = 0.;
        for (U32 i = 1; i <= n; ++i)
            a.log_factorial[i] = __dadd_rn(a.log_factorial[i - 1], log(double(i)));
    }
    __syncthreads();
    for (Index id = threadIdx.x; id < classes; id += blockDim.x) {
        const Histogram h = unrank_class(n, id);
        if (!rank_two(h) || rank_class(n, h) != id)
            asm volatile("trap;");
        double sum = 0.;
        for (int j = 0; j < 4; ++j)
            sum = __dadd_rn(sum, a.log_factorial[h.h[j]]);
        w.log_size[id] = __dsub_rn(a.log_factorial[n], sum);
        incoming_edges(a, w, classes, id, h);
    }
    __syncthreads();
    __shared__ double log_total;
    if (threadIdx.x == 0) {
        double maximum = -INFINITY;
        for (Index i = 0; i < classes; ++i)
            maximum = fmax(maximum, w.log_size[i]);
        double sum = 0.;
        for (Index i = 0; i < classes; ++i)
            sum = __dadd_rn(sum, exp(__dsub_rn(w.log_size[i], maximum)));
        a.phase_status[0] = isfinite(sum) && sum > 0 ? computed : limited;
        log_total = __dadd_rn(maximum, log(sum));
    }
    __syncthreads();
    if (a.phase_status[0] == computed)
        for (Index i = threadIdx.x; i < classes; i += blockDim.x)
            w.stationary[i] = exp(__dsub_rn(w.log_size[i], log_total));
}

__global__ void finish_profile(Cnot103PairDiffusionPrepare a, Index classes) {
    bool available = classes != 0;
    for (int i = 0; available && i < 5; ++i)
        available = a.phase_status[i] == computed;
    if (!blockIdx.x && !threadIdx.x)
        *a.status = available ? computed : (a.config.dimension == 1 ? unsupported : limited);
    if (!available)
        return;
    PairWork w(a.work, classes);
    for (Index id = Index(blockIdx.x) * blockDim.x + threadIdx.x; id < classes;
         id += Index(gridDim.x) * blockDim.x) {
        const double stationary = w.stationary[id];
        const double stationary_log = log(fmax(stationary, a.config.floor));
        for (int t = 0; t < 4; ++t) {
            const double raw = w.mass[t * classes + id];
            const bool exact = a.config.taus[t] == 0 && id != classes - 1;
            float* values = a.descriptors + id * 28 + t * 7;
            values[0] = __double2float_rn(raw);
            values[1] = __double2float_rn(w.log_size[id]);
            values[2] = __double2float_rn(stationary);
            values[3] =
                __double2float_rn(__dsub_rn(log(fmax(raw, a.config.floor)), stationary_log));
            values[4] = exact;
            values[5] = (raw == 0 && !exact) || (raw > 0 && values[0] == 0);
            values[6] = raw < a.config.floor;
        }
    }
}

__device__ bool valid_count(float value, U32 n) {
    return value >= 0 && value <= n && value == truncf(value);
}

__global__ void classify_pairs(Cnot103PairDiffusionProject a, Index classes) {
    const U64 sample = blockIdx.x, n = a.dimension, pairs = n * n;
    Byte status = *a.profile_status;
    if (status == computed) {
        if (!classes)
            asm volatile("trap;");
        const float* weights = a.column_weights + sample * a.weight_batch_stride;
        const float* intersections = a.column_intersections + sample * a.intersection_batch_stride;
        bool invalid = false;
        for (U64 pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
            const U32 first = pair / n, second = pair % n;
            Index id = -1;
            if (first != second) {
                const float x = weights[first], y = weights[second], both = intersections[pair];
                if (valid_count(x, n) && valid_count(y, n) && valid_count(both, n) && both <= x &&
                    both <= y && U64(x) + U64(y) - U64(both) <= n) {
                    const U32 i = U32(both), first_only = U32(x) - i, second_only = U32(y) - i;
                    const Histogram h{
                        {U32(n) - first_only - second_only - i, first_only, second_only, i}};
                    if (rank_two(h))
                        id = rank_class(n, h);
                }
                if (id < 0 || id >= classes)
                    invalid = true;
            }
            a.class_ids[sample * pairs + pair] = id;
        }
        if (__syncthreads_or(invalid))
            status = inapplicable;
    } else if (status != limited && status != unsupported)
        asm volatile("trap;");
    if (threadIdx.x == 0)
        a.status[sample] = status;
}

__global__ void publish_pairs(Cnot103PairDiffusionProject a, U64 elements) {
    const U64 cells = U64(a.dimension) * a.dimension;
    for (U64 i = U64(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
         i += U64(blockDim.x) * gridDim.x) {
        const U64 pair = i / 28;
        float value = 0.f;
        if (a.status[pair / cells] == computed) {
            const Index id = a.class_ids[pair];
            if (id >= 0) {
                value = a.descriptors[id * 28 + i % 28];
                if (!isfinite(value))
                    asm volatile("trap;");
            }
        }
        a.values[i] = value;
    }
}

unsigned blocks(U64 count) {
    return unsigned(std::max<U64>(1, std::min<U64>(65535, (count + threads - 1) / threads)));
}
} // namespace

extern "C" std::int32_t cnot103_cuda_pair_diffusion_layout(const Cnot103PairDiffusionConfig* cfg,
                                                           Cnot103PairDiffusionLayout* layout) {
    if (!cfg || !layout || !cfg->dimension || cfg->dimension > (1u << 24) ||
        !cfg->max_pair_states || !(cfg->floor >= 0 && cfg->floor < 1.))
        return 1;
    const U64 c = class_count(cfg->dimension, cfg->max_pair_states);
    U32 stride;
    if (!weight_capacity(cfg->taus, cfg->dimension, c != 0, stride))
        return 1;
    *layout = {c, stride};
    return 0;
}

extern "C" std::int32_t cnot103_cuda_prepare_pair_diffusion(const Cnot103PairDiffusionPrepare* a) {
    Cnot103PairDiffusionLayout layout{};
    if (!a || !a->status || cnot103_cuda_pair_diffusion_layout(&a->config, &layout))
        return 1;
    const U64 c = layout.classes;
    if (c && (!a->work || !a->predecessors || !a->log_factorial || !a->weights ||
              !a->phase_status || !a->descriptors))
        return 1;
    if (!extent(30 * c, a->work_len, 8) || !extent(16 * c, a->predecessor_len, 8) ||
        !extent(c ? a->config.dimension + 1 : 0, a->factorial_len, 8) ||
        !extent(4ULL * layout.weight_stride, a->weight_len, 8) ||
        !extent(c ? 5 : 0, a->phase_len, 1) || !extent(28 * c, a->descriptor_len, 4) ||
        !extent(1, a->status_len, 1) ||
        !total_fits({8 * a->work_len, 8 * a->predecessor_len, 8 * a->factorial_len,
                     8 * a->weight_len, a->phase_len, 4 * a->descriptor_len, a->status_len}))
        return 2;
    const auto stream = static_cast<cudaStream_t>(a->stream);
    if (c) {
        build_catalog<<<1, threads, 0, stream>>>(*a, Index(c));
        if (cudaGetLastError() != cudaSuccess)
            return 3;
        // [n-2,1,1,0] is the final rank-two histogram in source catalog order.
        propagate<16><<<4, threads, 0, stream>>>(*a, Index(c), Index(c - 1), layout.weight_stride);
        if (cudaGetLastError() != cudaSuccess)
            return 3;
    }
    finish_profile<<<blocks(c), threads, 0, stream>>>(*a, Index(c));
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" std::int32_t cnot103_cuda_project_pair_diffusion(const Cnot103PairDiffusionProject* a) {
    if (!a || !a->column_weights || !a->column_intersections || !a->profile_status || !a->values ||
        !a->status || !a->batch || a->batch > 0x7fffffffU || !a->max_pair_states || !a->dimension ||
        a->dimension > (1u << 24))
        return 1;
    const U64 c = class_count(a->dimension, a->max_pair_states);
    if (c && (!a->descriptors || !a->class_ids))
        return 1;
    const U64 cells = U64(a->dimension) * a->dimension;
    if (cells > byte_limit / (28 * sizeof(float)) / a->batch)
        return 2;
    const U64 pairs = cells * a->batch, elements = pairs * 28;
    if (!readable_view(a->batch, a->dimension, a->weight_batch_stride, 1, a->weight_len) ||
        !readable_view(a->batch, cells, a->intersection_batch_stride, 1, a->intersection_len) ||
        !extent(28 * c, a->descriptor_len, 4) || !extent(1, a->profile_status_len, 1) ||
        !extent(c ? pairs : 0, a->class_len, 8) || !extent(elements, a->value_len, 4) ||
        !extent(a->batch, a->status_len, 1) ||
        !total_fits({4 * a->weight_len, 4 * a->intersection_len, 4 * a->descriptor_len,
                     a->profile_status_len, 8 * a->class_len, 4 * a->value_len, a->status_len}))
        return 2;
    const auto stream = static_cast<cudaStream_t>(a->stream);
    classify_pairs<<<a->batch, threads, 0, stream>>>(*a, Index(c));
    if (cudaGetLastError() != cudaSuccess)
        return 3;
    publish_pairs<<<blocks(elements), threads, 0, stream>>>(*a, elements);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
