#include "cnot103/shared_word.h"
#include "../src/shared_word_rng.cuh"
#include "diffusion_test_support.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

namespace {
using namespace diffusion_test;
using cnot103_shared_word::NumpyRng;
constexpr int widths[4] = {1, 5, 2, 7};

template <class T> T read_one(std::istream& input) {
    T value{};
    require(bool(input.read(reinterpret_cast<char*>(&value), sizeof(T))), "truncated fixture");
    return value;
}
template <class T> std::vector<T> read_many(std::istream& input, size_t n) {
    std::vector<T> data(n);
    if (n)
        require(bool(input.read(reinterpret_cast<char*>(data.data()), n * sizeof(T))),
                "truncated fixture array");
    return data;
}
struct Fixture {
    Cnot103SharedWordConfig config{};
    U64 count;
    std::vector<U64> lengths, transforms;
    std::vector<U32> gates;
    explicit Fixture(std::istream& input) {
        const U32 n = read_one<U32>(input), samples = read_one<U32>(input);
        require(n >= 1 && n <= 144 && samples >= 2 && samples <= 129, "unknown fixture extent");
        const double tau = read_one<double>(input);
        config = {tau, read_one<U64>(input), samples, n, n};
        count = read_one<U64>(input);
        require(count < 10000, "fixture gate count");
        lengths = read_many<U64>(input, samples);
        gates = read_many<U32>(input, count);
        transforms = read_many<U64>(input, U64(samples) * n * ((n + 63) / 64));
    }
};

__global__ void trace_words(Cnot103SharedWordConfig cfg, U64* lengths, U32* gates, U64 capacity) {
    NumpyRng rng(cfg.seed);
    U64 cursor = 0;
    for (U64 s = 0; s < cfg.samples; ++s) {
        U64 steps = 0;
        if (cfg.dimension > 1 && !rng.poisson(__dmul_rn(cfg.dimension, cfg.tau), steps))
            asm volatile("trap;");
        lengths[s] = steps;
        for (U64 k = 0; k < steps; ++k) {
            const U32 c = rng.integer(cfg.dimension);
            U32 t = rng.integer(cfg.dimension - 1);
            t += t >= c;
            if (cursor >= capacity)
                asm volatile("trap;");
            gates[cursor++] = c * cfg.dimension + t;
        }
    }
    if (cursor != capacity)
        asm volatile("trap;");
}

U64 word_capacity(const Cnot103SharedWordConfig& cfg) {
    U64 words = 0;
    require(cnot103_cuda_shared_word_layout(&cfg, &words) == 0, "shared word layout");
    return words;
}
struct Profile {
    Buffer<U64> transforms, total;
    Buffer<Byte> status;
    Cnot103SharedWordPrepare args;
    Profile(Cnot103SharedWordConfig cfg, cudaStream_t stream)
        : transforms(word_capacity(cfg)), total(1), status(1),
          args{cfg, transforms.data, total.data, status.data, stream, transforms.size, 1, 1} {
        require(cnot103_cuda_prepare_shared_word(&args) == 0, "prepare");
        check(cudaStreamSynchronize(stream));
    }
};

std::vector<U64> input_states(U32 n, U32 batch) {
    const U32 w = (n + 63) / 64;
    std::vector<U64> rows(U64(batch) * n * w);
    for (U32 b = 0; b < batch; ++b) {
        U64* m = rows.data() + b * n * w;
        for (U32 i = 0; i < n; ++i)
            m[i * w + i / 64] |= U64(1) << (i % 64);
        if (b % 2 == 1) {
            for (U32 i = 1; i < n; ++i)
                for (U32 k = 0; k < w; ++k)
                    m[i * w + k] ^= m[(i - 1) * w + k];
        }
        if (b % 3 == 2 && n > 1) {
            for (U32 i = n - 1; i > 0; --i)
                for (U32 k = 0; k < w; ++k)
                    m[(i - 1) * w + k] ^= m[i * w + k];
        }
    }
    return rows;
}

// Replay each recorded physical word directly on the input matrix. Use
// centered observations for covariance/standard error, independently of the
// integer raw-moment reduction in the production kernel.
std::array<std::vector<double>, 4> reference(const Fixture& f, const U64* matrix) {
    const U32 n = f.config.dimension, w = (n + 63) / 64, samples = f.config.samples;
    std::vector<U64> columns(U64(samples) * n * w);
    std::vector<int> weights(U64(samples) * n);
    size_t gate = 0;
    for (U32 s = 0; s < samples; ++s) {
        std::vector<U64> m(matrix, matrix + n * w);
        for (U64 i = 0; i < f.lengths[s]; ++i) {
            const U32 code = f.gates.at(gate++), c = code / n, t = code % n;
            for (U32 k = 0; k < w; ++k)
                m[t * w + k] ^= m[c * w + k];
        }
        for (U32 r = 0; r < n; ++r)
            for (U32 c = 0; c < n; ++c) {
                if ((m[r * w + c / 64] >> (c % 64)) & 1) {
                    columns[(U64(s) * n + c) * w + r / 64] |= U64(1) << (r % 64);
                    ++weights[s * n + c];
                }
            }
    }
    require(gate == f.count, "physical gate count");
    std::array<std::vector<double>, 4> values;
    for (int feature = 0; feature < 4; ++feature)
        values[feature].resize((feature ? n * n : n) * widths[feature]);
    for (U32 i = 0; i < n; ++i)
        for (U32 j = 0; j < n; ++j) {
            std::vector<std::array<double, 7>> obs(samples);
            std::array<double, 7> mean{};
            for (U32 s = 0; s < samples; ++s) {
                const int x = weights[s * n + i], y = weights[s * n + j], u = x % 2 ? -1 : 1,
                          v = y % 2 ? -1 : 1;
                int inter = 0;
                for (U32 k = 0; k < w; ++k) {
                    U64 bits = columns[(s * n + i) * w + k] & columns[(s * n + j) * w + k];
                    while (bits) {
                        bits &= bits - 1;
                        ++inter;
                    }
                }
                obs[s] = {double(x), double(y), double(inter), double(x + y - 2 * inter),
                          double(u), double(v), double(u * v)};
                for (int k = 0; k < 7; ++k)
                    mean[k] += obs[s][k];
            }
            for (double& x : mean)
                x /= samples;
            auto cov = [&](int l, int r) {
                double sum = 0.;
                for (const auto& row : obs)
                    sum += (row[l] - mean[l]) * (row[r] - mean[r]);
                return sum / (samples - 1);
            };
            const int pair = i * n + j;
            values[0][i] = mean[0];
            double* g = values[1].data() + 5 * pair;
            g[0] = cov(0, 1);
            g[1] = mean[2];
            g[2] = mean[3];
            g[3] = cov(0, 2);
            g[4] = cov(0, 3);
            g = values[2].data() + 2 * pair;
            g[0] = mean[6];
            g[1] = mean[6] - mean[4] * mean[5];
            g = values[3].data() + 7 * pair;
            if (std::isfinite(float(f.config.tau))) {
                g[0] = std::sqrt(cov(2, 2) / samples);
                g[1] = std::sqrt(cov(3, 3) / samples);
                g[2] = std::sqrt(cov(6, 6) / samples);
                g[3] = std::abs(mean[6] - (i == j ? 1. : -1. / (std::ldexp(1., n) - 1.)));
                g[4] = samples;
                g[5] = double(f.count) / samples;
                g[6] = f.config.tau;
            }
        }
    return values;
}

void evaluate(const Fixture& f, Profile& profile, U32 batch, U32 workers) {
    const U32 n = f.config.dimension, w32 = (n + 31) / 32, w64 = (n + 63) / 64;
    const auto matrices = input_states(n, batch);
    Buffer<U64> states(matrices.size());
    states.upload(matrices);
    Buffer<U32> input(workers * n * w32), sample(input.size * f.config.samples),
        weights(workers * n * f.config.samples);
    const size_t pairs = U64(batch) * n * n;
    Buffer<float> values[] = {Buffer<float>(U64(batch) * n), Buffer<float>(5 * pairs),
                              Buffer<float>(2 * pairs), Buffer<float>(7 * pairs)};
    Buffer<Byte> statuses[] = {Buffer<Byte>(batch), Buffer<Byte>(batch), Buffer<Byte>(batch),
                               Buffer<Byte>(batch)};
    Cnot103SharedWordEval a{};
    a.config = f.config;
    a.states = states.data;
    a.transforms = profile.transforms.data;
    a.total_gates = profile.total.data;
    a.profile_status = profile.status.data;
    a.stream = profile.args.stream;
    a.input_columns = input.data;
    a.sample_columns = sample.data;
    a.weights = weights.data;
    a.state_len = states.size;
    a.transform_len = profile.transforms.size;
    a.total_len = 1;
    a.profile_status_len = 1;
    a.input_len = input.size;
    a.sample_len = sample.size;
    a.weight_len = weights.size;
    a.batch = batch;
    a.workers = workers;
    a.selected = 15;
    for (int k = 0; k < 4; ++k) {
        a.values[k] = values[k].data;
        a.value_len[k] = values[k].size;
        a.statuses[k] = statuses[k].data;
        a.status_len[k] = batch;
    }
    require(!cnot103_cuda_eval_shared_word(&a), "eval");
    check(cudaStreamSynchronize(static_cast<cudaStream_t>(a.stream)));
    for (U32 b = 0; b < batch; ++b) {
        auto expected = reference(f, matrices.data() + b * n * w64);
        for (int k = 0; k < 4; ++k) {
            const Byte want = k == 3 && !std::isfinite(float(f.config.tau)) ? 3 : 1;
            require(statuses[k].read(b, 1)[0] == want, "per-feature numeric status");
            const auto actual = values[k].read(b * expected[k].size(), expected[k].size());
            for (size_t i = 0; i < actual.size(); ++i)
                require(std::abs(actual[i] - expected[k][i]) <=
                            2e-6 * (1. + std::abs(expected[k][i])),
                        "physical replay moment");
        }
    }
    for (auto field :
         {&Cnot103SharedWordEval::state_len, &Cnot103SharedWordEval::transform_len,
          &Cnot103SharedWordEval::total_len, &Cnot103SharedWordEval::profile_status_len,
          &Cnot103SharedWordEval::input_len, &Cnot103SharedWordEval::sample_len,
          &Cnot103SharedWordEval::weight_len}) {
        auto short_arg = a;
        --(short_arg.*field);
        require(cnot103_cuda_eval_shared_word(&short_arg) == 2, "short scratch capacity");
    }
    for (int k = 0; k < 4; ++k) {
        auto x = a;
        --x.status_len[k];
        require(cnot103_cuda_eval_shared_word(&x) == 2, "short status");
        x = a;
        --x.value_len[k];
        require(cnot103_cuda_eval_shared_word(&x) == 2, "short output");
    }
    auto x = a;
    x.state_len = std::numeric_limits<std::int64_t>::max() / 8;
    require(cnot103_cuda_eval_shared_word(&x) == 2, "aggregate overflow");
    x = a;
    x.workers = batch + 1;
    require(cnot103_cuda_eval_shared_word(&x) == 1, "worker range");
}

void run(std::istream& file, cudaStream_t stream) {
    require(read_many<char>(file, 8) == std::vector<char>({'C', 'N', 'O', 'T', 'M', 'C', '0', '1'}),
            "fixture identity");
    const U32 cases = read_one<U32>(file);
    require(cases == 7, "fixture count");
    U64 word_count = 0, gate_count = 0, transform_words = 0;
    for (U32 i = 0; i < cases; ++i) {
        Fixture f(file);
        Profile p(f.config, stream);
        require(p.status.read(0, 1)[0] == 1 && p.total.read(0, 1)[0] == f.count,
                "seeded profile status/total");
        require(p.transforms.read(0, p.transforms.size) == f.transforms,
                "complete packed transforms");
        Buffer<U64> lengths(f.config.samples);
        Buffer<U32> gates(f.count);
        trace_words<<<1, 1, 0, stream>>>(f.config, lengths.data, gates.data, f.count);
        check(cudaStreamSynchronize(stream));
        require(gates.read(0, gates.size) == f.gates && lengths.read(0, lengths.size) == f.lengths,
                "individual NumPy word lengths and ordered gates");
        word_count += f.config.samples;
        gate_count += f.count;
        transform_words += p.transforms.size;
        if (f.config.dimension <= 4)
            evaluate(f, p, 3, 2);
        if (f.config.dimension == 65)
            evaluate(f, p, 1, 1);
        for (auto field :
             {&Cnot103SharedWordPrepare::transform_len, &Cnot103SharedWordPrepare::total_len,
              &Cnot103SharedWordPrepare::status_len}) {
            auto x = p.args;
            --(x.*field);
            require(cnot103_cuda_prepare_shared_word(&x) == 2, "short plan extent");
        }
    }
    require(file.peek() == std::istream::traits_type::eof(), "unexpected fixture tail");
    Profile guarded({1., 0, 1025, 64, 64}, stream);
    require(guarded.transforms.size == 0 && guarded.status.read(0, 1)[0] == 4,
            "working set boundary");
    Profile numeric({1e100, 0, 2, 2, 2}, stream);
    require(numeric.status.read(0, 1)[0] == 4 && numeric.total.read(0, 1)[0] == 0,
            "Poisson numeric guard");
    std::printf("PASS shared-word NumPy 2.2.6: %llu words, %llu gates, %llu transform words; "
                "physical moments and bounds\n",
                (unsigned long long)word_count, (unsigned long long)gate_count,
                (unsigned long long)transform_words);
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected NumPy fixture path");
        std::ifstream file(argv[1], std::ios::binary);
        require(bool(file), "cannot open NumPy fixture");
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        run(file, stream);
        check(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
