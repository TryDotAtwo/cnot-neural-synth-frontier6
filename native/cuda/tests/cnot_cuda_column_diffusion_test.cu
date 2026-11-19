#include "cnot103/column_diffusion.h"
#include "diffusion_test_support.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace diffusion_test;

Cnot103ColumnLayout layout_for(const Cnot103ColumnConfig& config) {
    Cnot103ColumnLayout l{};
    require(cnot103_cuda_column_layout(&config, &l) == 0, "layout");
    return l;
}

struct Profile {
    const Cnot103ColumnLayout layout;
    Buffer<double> work, weights;
    Buffer<int> predecessors;
    Buffer<Byte> phases, status;
    Buffer<float> descriptors;
    Cnot103ColumnPrepare args;

    Profile(Cnot103ColumnConfig config, cudaStream_t stream)
        : layout(layout_for(config)), work(18 * layout.classes), weights(4 * layout.weight_stride),
          predecessors(4 * layout.classes), phases(layout.classes ? 5 : 0), status(1),
          descriptors(47 * layout.classes),
          args{config,           work.data,         predecessors.data, weights.data,
               phases.data,      descriptors.data,  status.data,       stream,
               work.size,        predecessors.size, weights.size,      phases.size,
               descriptors.size, status.size} {
        require(cnot103_cuda_prepare_column_diffusion(&args) == 0, "prepare");
        check(cudaStreamSynchronize(stream));
    }
    Byte code() const { return status.read(0, 1)[0]; }
};

int physical_class(int mask, int n) {
    int count = 0;
    for (int i = 1; i < n; ++i)
        count += (mask >> i) & 1;
    return (mask & 1) * n + count - 1;
}

// Enumerate transitions between individual nonzero bit strings, independently
// of the lumped four-edge chain. Poisson terms start at exp(-time) in long
// double. The 256-term sum is sufficient for these fixtures (time <= 24).
void labeled_column_reference(int n, cudaStream_t stream) {
    Cnot103ColumnConfig config{{0., 0.25, 1., 4.}, 1e-30, U32(n), U32(n)};
    Profile gpu(config, stream);
    require(gpu.code() == 1, "small profile unavailable");
    const int c = 2 * n - 1, masks = 1 << n, gates = std::max(1, n * (n - 1));
    std::vector<int> sizes(c), transitions(c * c);
    std::vector<std::vector<int>> next(masks);
    for (int mask = 1; mask < masks; ++mask) {
        const int source = physical_class(mask, n);
        ++sizes[source];
        for (int control = 0; control < n; ++control)
            for (int target = 0; target < n; ++target)
                if (control != target)
                    next[mask].push_back(mask ^ (((mask >> control) & 1) << target));
        if (n == 1)
            next[mask].push_back(mask);
        for (const int child : next[mask])
            ++transitions[source * c + physical_class(child, n)];
    }
    auto sources = gpu.predecessors.read(0, 4 * c);
    auto probabilities = gpu.work.read(2 * c, 4 * c);
    for (int dest = 0; dest < c; ++dest) {
        int slot = 0;
        for (int source = 0; source < c; ++source) {
            const int histories = transitions[source * c + dest];
            if (!histories)
                continue;
            require(slot < 4 && sources[slot * c + dest] == source, "predecessor order");
            require(probabilities[slot * c + dest] == double(histories) / (sizes[source] * gates),
                    "transition mass");
            ++slot;
        }
        for (; slot < 4; ++slot)
            require(sources[slot * c + dest] == -1, "unused predecessor");
    }
    auto log_size = gpu.work.read(0, c);
    auto stationary = gpu.work.read(c, c);
    auto masses = gpu.work.read(14 * c, 4 * c);
    for (int t = 0; t < 4; ++t) {
        std::vector<long double> current(masks), mass(c);
        const long double time = n * config.taus[t];
        long double weight = std::exp(-time);
        current[1] = 1;
        for (int step = 0; step <= 256; ++step) {
            for (int mask = 1; mask < masks; ++mask)
                mass[physical_class(mask, n)] += weight * current[mask];
            if (time == 0)
                break;
            std::vector<long double> following(masks);
            for (int mask = 1; mask < masks; ++mask)
                for (int child : next[mask])
                    following[child] += current[mask] / gates;
            current.swap(following);
            weight *= time / (step + 1);
        }
        double total = 0;
        for (int k = 0; k < c; ++k) {
            total += masses[t * c + k];
            require(std::abs(masses[t * c + k] - mass[k]) < 2e-14L,
                    "independent physical probability");
            require(std::abs(stationary[k] - double(sizes[k]) / (masks - 1)) < 2e-15,
                    "stationary class");
            require(std::abs(log_size[k] - std::log(double(sizes[k]))) < 2e-15,
                    "class multiplicity");
        }
        require(std::abs(total - 1.) < 1e-14, "mass normalization");
    }
}

struct Projection {
    Buffer<float> weight, missing, values;
    Buffer<Byte> status;
    Cnot103ColumnProject args;
    Projection(Profile& p, U32 batch, U32 ordinal)
        : weight(batch * p.args.config.dimension), missing(weight.size),
          values(weight.size * (ordinal <= 2 ? 28 : (ordinal == 3 ? 16 : 3))), status(batch),
          args{weight.data,
               missing.data,
               p.descriptors.data,
               p.status.data,
               values.data,
               status.data,
               p.args.stream,
               p.args.config.dimension,
               1,
               p.args.config.dimension,
               weight.size,
               missing.size,
               p.descriptors.size,
               1,
               values.size,
               batch,
               batch,
               p.args.config.dimension,
               p.args.config.max_n,
               ordinal} {}
    void launch() { require(cnot103_cuda_project_column_diffusion(&args) == 0, "projection"); }
};

void status_and_extents(cudaStream_t stream) {
    Profile p({{0., 0.25, 1., 4.}, 0., 4, 4}, stream);
    // A zero weight in the last wire takes priority over a nonfinite log mass
    // in the first wire. The second sample consists solely of the identity class.
    for (U32 ordinal = 1; ordinal <= 4; ++ordinal) {
        Projection out(p, 2, ordinal);
        out.weight.upload({2, 1, 1, 0, 1, 1, 1, 1});
        out.missing.upload({0, 0, 0, 0, 0, 0, 0, 0});
        out.launch();
        check(cudaStreamSynchronize(stream));
        require(out.status.read(0, 2) == std::vector<Byte>({2, 1}), "whole feature status");
        auto payload = out.values.read(0, out.values.size);
        for (size_t i = 0; i < payload.size() / 2; ++i)
            require(payload[i] == 0.f, "unavailable payload");
        for (float value : payload)
            require(std::isfinite(value), "computed output finite");

        for (auto field :
             {&Cnot103ColumnProject::weight_len, &Cnot103ColumnProject::diagonal_len,
              &Cnot103ColumnProject::descriptor_len, &Cnot103ColumnProject::profile_status_len,
              &Cnot103ColumnProject::value_len, &Cnot103ColumnProject::status_len}) {
            auto a = out.args;
            --(a.*field);
            require(cnot103_cuda_project_column_diffusion(&a) == 2, "short projection extent");
        }
        auto a = out.args;
        a.weight_batch_stride = std::numeric_limits<U64>::max();
        require(cnot103_cuda_project_column_diffusion(&a) == 2, "input span overflow");
        a = out.args;
        a.value_len = std::numeric_limits<std::int64_t>::max() / 4;
        require(cnot103_cuda_project_column_diffusion(&a) == 2, "projection aggregate overflow");
        a = out.args;
        a.ordinal = 5;
        require(cnot103_cuda_project_column_diffusion(&a) == 1, "invalid ordinal");
    }
    for (auto field : {&Cnot103ColumnPrepare::work_len, &Cnot103ColumnPrepare::predecessor_len,
                       &Cnot103ColumnPrepare::weight_len, &Cnot103ColumnPrepare::phase_len,
                       &Cnot103ColumnPrepare::descriptor_len, &Cnot103ColumnPrepare::status_len}) {
        auto a = p.args;
        --(a.*field);
        require(cnot103_cuda_prepare_column_diffusion(&a) == 2, "short preparation extent");
    }
    auto a = p.args;
    a.work_len = std::numeric_limits<U64>::max();
    require(cnot103_cuda_prepare_column_diffusion(&a) == 2, "workspace byte overflow");
    a = p.args;
    a.weight_len = std::numeric_limits<std::int64_t>::max() / 8;
    require(cnot103_cuda_prepare_column_diffusion(&a) == 2, "workspace aggregate overflow");

    for (U32 n : {1023u, 1024u, 1025u, 2048u, 2049u}) {
        Profile wide({{0., 0., 0., 0.}, 1e-30, n, n}, stream);
        require(wide.code() == (n <= 1024 ? 1 : 4), "finite binomial boundary");
        if (n >= 2049)
            require(wide.work.size == 0, "static overflow scratch");
    }
    Profile limited({{0., 1., 2., 3.}, 0., 4, 3}, stream);
    require(limited.code() == 4 && limited.work.size == 0, "width guard");
    Projection out(limited, 1, 1);
    out.launch(); // width guard precedes any read of the two input views
    check(cudaStreamSynchronize(stream));
    require(out.status.read(0, 1)[0] == 4, "resource status");
    for (float value : out.values.read(0, out.values.size))
        require(value == 0.f, "resource payload");
}

void term_boundary(cudaStream_t stream) {
    // last = ceil(time + 12*sqrt(time) + 32). Choose consecutive integer
    // times straddling the cap without relying on a transcendental equality.
    int last = 1'000'000;
    while (std::ceil(last + 12. * std::sqrt(double(last)) + 32.) > 1'000'000)
        --last;
    for (int time : {last, last + 1}) {
        Profile p({{0., 0., 0., double(time)}, 1e-30, 1, 1}, stream);
        require(p.code() == (time == last ? 1 : 4), "million-term boundary");
        if (time == last)
            require(p.work.read(17, 1)[0] == 1., "singleton boundary mass");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (argc > 1 && std::string(argv[1]) != "term-boundary") {
            const std::string mode(argv[1]);
            require(mode == "zero-floor-g01" || mode == "zero-floor-g02" ||
                        mode == "zero-floor-g03",
                    "unknown mode");
            Profile p({{0., 1., 2., 3.}, 0., 2, 2}, stream);
            Projection out(p, 1, mode.back() - '0');
            require(p.code() == 1, "zero-floor profile itself must be available");
            const auto log_value =
                p.descriptors.read(2 * 47 + (out.args.ordinal == 3 ? 28 : 3), 1)[0];
            require(std::isinf(log_value) && log_value < 0,
                    "fixture must select negative infinity");
            out.weight.upload({2, 1});
            out.missing.upload({0, 0});
            out.launch();
            const auto error = cudaStreamSynchronize(stream);
            std::printf("selected nonfinite %s: CUDA %d (%s)\n", argv[1], int(error),
                        cudaGetErrorString(error));
            require(error == cudaErrorIllegalInstruction || error == cudaErrorLaunchFailure,
                    "selected nonfinite payload must trap");
            std::printf("PASS Release rejected %s\n", argv[1]);
            return 0;
        }
        if (argc > 1)
            term_boundary(stream);
        else {
            for (int n = 1; n <= 6; ++n)
                labeled_column_reference(n, stream);
            status_and_extents(stream);
        }
        check(cudaStreamDestroy(stream));
        std::puts(argc > 1 ? "PASS million-term boundary"
                           : "PASS physical column chains n1..6, profile and projection bounds, "
                             "statuses and f64 width boundary");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
