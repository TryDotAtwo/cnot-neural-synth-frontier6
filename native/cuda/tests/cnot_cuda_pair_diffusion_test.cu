#include "cnot103/pair_diffusion.h"
#include "diffusion_test_support.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>

namespace {
using namespace diffusion_test;
using Histogram = std::array<int, 4>;
using Index = std::int64_t;

Cnot103PairDiffusionLayout layout_for(const Cnot103PairDiffusionConfig& cfg) {
    Cnot103PairDiffusionLayout layout{};
    require(cnot103_cuda_pair_diffusion_layout(&cfg, &layout) == 0, "pair layout");
    return layout;
}

struct Profile {
    Cnot103PairDiffusionLayout layout;
    Buffer<double> work, factorial, weights;
    Buffer<Index> pred;
    Buffer<float> descriptors;
    Buffer<Byte> phases, status;
    Cnot103PairDiffusionPrepare args;
    Profile(Cnot103PairDiffusionConfig cfg, cudaStream_t stream)
        : layout(layout_for(cfg)), work(30 * layout.classes),
          factorial(layout.classes ? cfg.dimension + 1 : 0), weights(4 * layout.weight_stride),
          pred(16 * layout.classes), descriptors(28 * layout.classes),
          phases(layout.classes ? 5 : 0), status(1),
          args{cfg,          work.data,   pred.data,        factorial.data,
               weights.data, phases.data, descriptors.data, status.data,
               stream,       work.size,   pred.size,        factorial.size,
               weights.size, phases.size, descriptors.size, 1} {
        require(cnot103_cuda_prepare_pair_diffusion(&args) == 0, "prepare");
        check(cudaStreamSynchronize(stream));
    }
    Byte code() const { return status.read(0, 1)[0]; }
};

Histogram histogram(int first, int second, int n) {
    Histogram h{};
    for (int i = 0; i < n; ++i)
        ++h[((first >> i) & 1) + 2 * ((second >> i) & 1)];
    return h;
}

struct PhysicalChain {
    int n, masks, gates;
    std::map<Histogram, int> classes;
    std::vector<int> ids, sizes;
    std::vector<std::vector<int>> next;
    explicit PhysicalChain(int width)
        : n(width), masks(1 << n), gates(n * (n - 1)), ids(masks * masks, -1), next(masks * masks) {
        for (int x = 1; x < masks; ++x)
            for (int y = 1; y < masks; ++y)
                if (x != y)
                    classes.emplace(histogram(x, y, n), 0);
        int index = 0;
        for (auto& item : classes)
            item.second = index++;
        sizes.assign(classes.size(), 0);
        for (int x = 1; x < masks; ++x)
            for (int y = 1; y < masks; ++y) {
                if (x == y)
                    continue;
                const int state = x * masks + y;
                ids[state] = classes.at(histogram(x, y, n));
                ++sizes[ids[state]];
                for (int control = 0; control < n; ++control)
                    for (int target = 0; target < n; ++target) {
                        if (control == target)
                            continue;
                        const int a = x ^ (((x >> control) & 1) << target);
                        const int b = y ^ (((y >> control) & 1) << target);
                        next[state].push_back(a * masks + b);
                    }
            }
    }
    std::vector<long double> class_mass(long double time) const {
        std::vector<long double> state(ids.size()), mass(classes.size());
        state[masks + 2] = 1.;
        long double weight = std::exp(-time);
        for (int step = 0; step <= 192; ++step) {
            for (size_t i = 0; i < state.size(); ++i)
                if (ids[i] >= 0)
                    mass[ids[i]] += weight * state[i];
            if (time == 0)
                break;
            std::vector<long double> following(ids.size());
            for (size_t i = 0; i < state.size(); ++i)
                for (int j : next[i])
                    following[j] += state[i] / gates;
            state.swap(following);
            weight *= time / (step + 1);
        }
        return mass;
    }
};

void compare_labeled_pairs(int n, cudaStream_t stream) {
    PhysicalChain ref(n);
    const int c = ref.classes.size();
    Profile gpu({{0., 0.125, 0.75, 4.}, 0.2, U64(c), U32(n)}, stream);
    require(gpu.layout.classes == U64(c) && gpu.code() == 1, "exact catalog bound");
    const auto source = gpu.pred.read(0, 16 * c);
    const auto probability = gpu.work.read(2 * c, 16 * c);
    const auto log_size = gpu.work.read(0, c);
    const auto stationary = gpu.work.read(c, c);
    const auto masses = gpu.work.read(26 * c, 4 * c);
    const auto channels = gpu.descriptors.read(0, 28 * c);
    std::vector<int> histories(c * c);
    for (size_t i = 0; i < ref.next.size(); ++i)
        for (int j : ref.next[i])
            ++histories[ref.ids[i] * c + ref.ids[j]];
    for (int dest = 0; dest < c; ++dest) {
        std::vector<double> incoming(c);
        int previous = -1;
        for (int e = 0; e < 16; ++e) {
            const Index src = source[e * c + dest];
            if (src < 0)
                break;
            require(src >= previous && src < c, "incoming source order");
            previous = src;
            incoming[src] += probability[e * c + dest];
        }
        for (int src = 0; src < c; ++src) {
            const double exact = double(histories[src * c + dest]) / (ref.gates * ref.sizes[src]);
            require(std::abs(exact - incoming[src]) < 1e-15, "labeled CNOT transition");
        }
        require(std::abs(log_size[dest] - std::log(double(ref.sizes[dest]))) < 3e-15,
                "class log size");
        const double expected = double(ref.sizes[dest]) / ((ref.masks - 1) * (ref.masks - 2));
        require(std::abs(expected - stationary[dest]) < 2e-15, "stationary mass");
    }
    for (int t = 0; t < 4; ++t) {
        auto wanted = ref.class_mass(n * gpu.args.config.taus[t]);
        double total = 0.;
        for (int id = 0; id < c; ++id) {
            const double raw = masses[t * c + id];
            total += raw;
            require(std::abs(raw - wanted[id]) < 2e-14L, "labeled pair probability");
            const float* ch = channels.data() + id * 28 + 7 * t;
            require(ch[0] == float(raw) && ch[1] == float(log_size[id]) &&
                        ch[2] == float(stationary[id]),
                    "published class");
            const double floor = gpu.args.config.floor;
            const float ratio =
                std::log(std::max(raw, floor)) - std::log(std::max(stationary[id], floor));
            require(std::abs(ch[3] - ratio) < 2e-7f, "floored stationary ratio");
            require(ch[4] == float(t == 0 && id != ref.ids[ref.masks + 2]), "exact zero");
            require(ch[5] == 0 && ch[6] == float(raw < floor), "underflow/floor flags");
        }
        require(std::abs(total - 1.) < 1e-14, "normalized pair masses");
    }
}

struct Projection {
    Buffer<float> weights, intersection, values;
    Buffer<Index> classes;
    Buffer<Byte> status;
    Cnot103PairDiffusionProject args;
    Projection(Profile& p, U32 batch)
        : weights(batch * p.args.config.dimension),
          intersection(weights.size * p.args.config.dimension), values(28 * intersection.size),
          classes(p.layout.classes ? intersection.size : 0), status(batch),
          args{weights.data,
               intersection.data,
               p.descriptors.data,
               p.status.data,
               classes.data,
               values.data,
               status.data,
               p.args.stream,
               p.args.config.max_pair_states,
               p.args.config.dimension,
               U64(p.args.config.dimension) * p.args.config.dimension,
               weights.size,
               intersection.size,
               p.descriptors.size,
               1,
               classes.size,
               values.size,
               status.size,
               batch,
               p.args.config.dimension} {}
    void launch() { require(cnot103_cuda_project_pair_diffusion(&args) == 0, "project"); }
};

void projection_and_limits(cudaStream_t stream, bool fatal) {
    PhysicalChain ref(3);
    Profile p({{0., 1., 2., 4.}, fatal ? 0. : 1e-30, 10, 3}, stream);
    Projection out(p, 2);
    // First matrix has distinct columns 001, 011, 111. The second contains a duplicate.
    out.weights.upload({1, 2, 3, 1, 2, 2});
    out.intersection.upload({1, 1, 1, 1, 2, 2, 1, 2, 3, 1, 1, 1, 1, 2, 2, 1, 2, 2});
    out.launch();
    const auto code = cudaStreamSynchronize(stream);
    if (fatal) {
        require(code == cudaErrorLaunchFailure || code == cudaErrorIllegalInstruction,
                "expected selected zero floor to trap");
        std::printf("PASS G06 zero-floor trap %d\n", int(code));
        return;
    }
    check(code);
    require(out.status.read(0, 2) == std::vector<Byte>{1, 2}, "rank-two projection status");
    const auto ids = out.classes.read(0, 9);
    const auto values = out.values.read(0, out.values.size);
    const auto table = p.descriptors.read(0, p.descriptors.size);
    const int cols[] = {1, 3, 7};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const int pair = 3 * i + j;
            const int want = i == j ? -1 : ref.classes.at(histogram(cols[i], cols[j], 3));
            require(ids[pair] == want, "physical class orientation or diagonal");
            for (int k = 0; k < 28; ++k)
                require(values[pair * 28 + k] == (want < 0 ? 0.f : table[want * 28 + k]),
                        "projection channel");
        }
    for (size_t i = 9 * 28; i < values.size(); ++i)
        require(values[i] == 0.f, "inapplicable sample payload");
    for (auto field :
         {&Cnot103PairDiffusionProject::weight_len, &Cnot103PairDiffusionProject::intersection_len,
          &Cnot103PairDiffusionProject::descriptor_len,
          &Cnot103PairDiffusionProject::profile_status_len, &Cnot103PairDiffusionProject::class_len,
          &Cnot103PairDiffusionProject::value_len, &Cnot103PairDiffusionProject::status_len}) {
        auto a = out.args;
        --(a.*field);
        require(cnot103_cuda_project_pair_diffusion(&a) == 2, "short projection extent");
    }
    for (auto field :
         {&Cnot103PairDiffusionPrepare::work_len, &Cnot103PairDiffusionPrepare::predecessor_len,
          &Cnot103PairDiffusionPrepare::factorial_len, &Cnot103PairDiffusionPrepare::weight_len,
          &Cnot103PairDiffusionPrepare::phase_len, &Cnot103PairDiffusionPrepare::descriptor_len,
          &Cnot103PairDiffusionPrepare::status_len}) {
        auto a = p.args;
        --(a.*field);
        require(cnot103_cuda_prepare_pair_diffusion(&a) == 2, "short preparation extent");
    }
    auto a = out.args;
    a.batch = 0x7fffffff;
    a.dimension = 1 << 24;
    require(cnot103_cuda_project_pair_diffusion(&a) == 2, "output extent overflow");
    a = out.args;
    a.value_len = std::numeric_limits<Index>::max() / 4;
    require(cnot103_cuda_project_pair_diffusion(&a) == 2, "aggregate output overflow");
    auto prep = p.args;
    prep.work_len = std::numeric_limits<Index>::max() / 8;
    require(cnot103_cuda_prepare_pair_diffusion(&prep) == 2, "aggregate workspace overflow");

    // A nonfinite class in a sample with a later duplicate pair is never published.
    Profile zero({{0., 0.25, 2., 4.}, 0., 10, 3}, stream);
    Projection duplicate(zero, 1);
    duplicate.weights.upload({1, 2, 2});
    duplicate.intersection.upload({1, 1, 1, 1, 2, 2, 1, 2, 2});
    duplicate.launch();
    check(cudaStreamSynchronize(stream));
    require(duplicate.status.read(0, 1)[0] == 2, "inapplicable precedes selected finite check");
    for (U32 n : {1u, 3u}) {
        Profile unavailable({{0., 1., 2., 3.}, 0., 9, n}, stream);
        require(unavailable.code() == (n == 1 ? 3 : 4) && unavailable.layout.classes == 0,
                "singleton/budget guard");
        Projection other(unavailable, 1);
        other.launch();
        check(cudaStreamSynchronize(stream));
        require(other.status.read(0, 1)[0] == unavailable.code(), "unavailable pair status");
        for (float v : other.values.read(0, other.values.size))
            require(v == 0, "unavailable pair payload");
    }
    const auto huge =
        layout_for({{0., 0., 0., 0.}, 1e-30, std::numeric_limits<U64>::max(), 1u << 24});
    require(huge.classes == 0 && huge.weight_stride == 0, "source count multiplication overflow");
}
} // namespace

int main(int argc, char** argv) {
    try {
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (argc > 1) {
            require(std::string(argv[1]) == "zero-floor", "unknown mode");
            projection_and_limits(stream, true);
            return 0;
        }
        for (int n = 2; n <= 5; ++n)
            compare_labeled_pairs(n, stream);
        projection_and_limits(stream, false);
        check(cudaStreamDestroy(stream));
        std::puts("PASS independent labeled rank-two CNOT chains n2..5, ordered projection, "
                  "statuses and extents");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
