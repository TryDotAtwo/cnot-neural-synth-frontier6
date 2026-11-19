#include "cnot103/cnot_cuda.h"
#include "packed_block_scratch.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace {

// All threads enter each barrier. One thread owns an entire non-pivot row
// during elimination; the pivot row remains read-only until the next barrier.
__device__ void canonical_rows(std::uint64_t* rows, std::uint32_t row_count,
                               std::uint32_t bits, std::uint32_t words,
                               std::int32_t* rank, std::int32_t* pivot) {
    if (threadIdx.x == 0) *rank = 0;
    __syncthreads();
    for (std::uint32_t reverse = bits; reverse > 0; --reverse) {
        const auto bit = reverse - 1;
        if (threadIdx.x == 0) *pivot = static_cast<std::int32_t>(row_count);
        __syncthreads();
        for (std::uint32_t row = static_cast<std::uint32_t>(*rank) + threadIdx.x;
             row < row_count; row += blockDim.x) {
            if (((rows[static_cast<std::uint64_t>(row) * words + bit / 64] >> (bit % 64)) & 1) != 0)
                atomicMin(pivot, static_cast<std::int32_t>(row));
        }
        __syncthreads();
        if (*pivot < static_cast<std::int32_t>(row_count)) {
            const auto active = static_cast<std::uint32_t>(*rank);
            const auto selected = static_cast<std::uint32_t>(*pivot);
            for (std::uint32_t word = threadIdx.x; word < words; word += blockDim.x) {
                const auto a = static_cast<std::uint64_t>(active) * words + word;
                const auto b = static_cast<std::uint64_t>(selected) * words + word;
                const auto saved = rows[a]; rows[a] = rows[b]; rows[b] = saved;
            }
            __syncthreads();
            for (std::uint32_t row = threadIdx.x; row < row_count; row += blockDim.x) {
                if (row == active) continue;
                const auto offset = static_cast<std::uint64_t>(row) * words;
                if (((rows[offset + bit / 64] >> (bit % 64)) & 1) != 0) {
                    for (std::uint32_t word = 0; word < words; ++word)
                        rows[offset + word] ^= rows[static_cast<std::uint64_t>(active) * words + word];
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) ++*rank;
            __syncthreads();
        }
        // Also synchronize the no-pivot path. Otherwise thread 0 may reset
        // pivot for the next bit while another warp still reads this condition.
        __syncthreads();
    }
}

__global__ void boundary_relations_kernel(
    const std::uint64_t* __restrict__ states,
    const std::uint32_t* __restrict__ port_wires,
    const std::uint8_t* __restrict__ protected_ports,
    const std::uint32_t* __restrict__ equation_outputs,
    std::uint32_t dimension, std::uint32_t words_per_row,
    std::uint32_t input_count, std::uint32_t output_count, std::uint32_t equations,
    float* __restrict__ values, std::uint8_t* __restrict__ valid,
    float* __restrict__ incidence, float* __restrict__ ranks, cnot103::PackedBlockScratch scratch) {
    extern __shared__ std::uint64_t shared[];
    auto* rows = scratch.rows(shared);
    __shared__ std::int32_t rank;
    __shared__ std::int32_t pivot;
    const auto ports = input_count + output_count;
    const auto words = (ports + 63u) / 64u;
    const auto state_base = scratch.item() * dimension * words_per_row;
    const auto group_base = scratch.item() * equations;
    const auto packed_count = static_cast<std::uint64_t>(equations) * words;
    for (std::uint64_t i = threadIdx.x; i < packed_count; i += blockDim.x) {
        const auto row = static_cast<std::uint32_t>(i / words);
        const auto word = static_cast<std::uint32_t>(i % words);
        const auto output = equation_outputs[row];
        const auto physical_row = port_wires[input_count + output];
        std::uint64_t value = 0;
        for (std::uint32_t k = 0; k < 64; ++k) {
            const auto port = word * 64u + k;
            if (port >= ports) break;
            bool set = port == input_count + output;
            if (row < output_count && port < input_count) {
                const auto wire = port_wires[port];
                set = ((states[state_base + static_cast<std::uint64_t>(physical_row) * words_per_row + wire / 64u] >> (wire % 64u)) & 1) != 0;
            }
            if (set) value |= std::uint64_t{1} << k;
        }
        rows[i] = value;
    }
    __syncthreads();
    canonical_rows(rows, equations, ports, words, &rank, &pivot);
    const auto relation_rank = static_cast<std::uint32_t>(rank);
    for (std::uint32_t row = threadIdx.x; row < equations; row += blockDim.x) {
        const auto group = group_base + row;
        const bool present = row < relation_rank;
        valid[group] = present ? 1 : 0;
        values[group * 3] = present ? static_cast<float>(relation_rank) : 0.0f;
        values[group * 3 + 1] = present ? static_cast<float>(ports - relation_rank) : 0.0f;
        values[group * 3 + 2] = 0.0f;
    }
    const auto incidence_count = static_cast<std::uint64_t>(equations) * dimension * 3;
    const auto incidence_base = group_base * dimension * 3;
    for (std::uint64_t i = threadIdx.x; i < incidence_count; i += blockDim.x)
        incidence[incidence_base + i] = 0.0f;
    __syncthreads();
    // Static input and output wire lists are individually unique. A port
    // writes a distinct (group,wire,role); protected input uses a third role.
    for (std::uint64_t i = threadIdx.x; i < static_cast<std::uint64_t>(relation_rank) * ports; i += blockDim.x) {
        const auto row = static_cast<std::uint32_t>(i / ports);
        const auto port = static_cast<std::uint32_t>(i % ports);
        if (((rows[static_cast<std::uint64_t>(row) * words + port / 64u] >> (port % 64u)) & 1) != 0) {
            const auto offset = ((group_base + row) * dimension + port_wires[port]) * 3;
            incidence[offset + (port < input_count ? 0 : 1)] = 1.0f;
            if (protected_ports[port] != 0) incidence[offset + 2] = 1.0f;
        }
    }
    __syncthreads();
    // dim(R intersect protected-coordinate subspace) = rank(R) -
    // rank(projection of R onto all nonprotected coordinates).
    for (std::uint64_t i = threadIdx.x; i < packed_count; i += blockDim.x) {
        const auto word = static_cast<std::uint32_t>(i % words);
        auto value = rows[i];
        for (std::uint32_t k = 0; k < 64; ++k) {
            const auto port = word * 64u + k;
            if (port >= ports) break;
            if (protected_ports[port] != 0) value &= ~(std::uint64_t{1} << k);
        }
        rows[i] = value;
    }
    __syncthreads();
    canonical_rows(rows, equations, ports, words, &rank, &pivot);
    if (threadIdx.x == 0) {
        ranks[scratch.item() * 2] = static_cast<float>(relation_rank);
        ranks[scratch.item() * 2 + 1] = static_cast<float>(relation_rank - rank);
    }
}
} // namespace

extern "C" std::int32_t cnot103_cuda_launch_boundary_relations(
    const std::uint64_t* states, const std::uint32_t* port_wires,
    const std::uint8_t* protected_ports, const std::uint32_t* equation_outputs,
    std::uint32_t batch, std::uint32_t dimension, std::uint32_t words_per_row,
    std::uint32_t inputs, std::uint32_t outputs, std::uint32_t equations,
    float* values, std::uint8_t* valid, float* incidence, float* ranks,
    std::uint64_t group_capacity, std::uint64_t incidence_capacity, void* stream) {
    const auto ports = static_cast<std::uint64_t>(inputs) + outputs;
    if (!states || !port_wires || !protected_ports || !equation_outputs || !values || !valid || !incidence || !ranks ||
        batch == 0 || dimension < 2 || inputs == 0 || outputs == 0 || inputs > dimension || outputs > dimension ||
        equations < outputs || static_cast<std::uint64_t>(equations) > static_cast<std::uint64_t>(outputs) * 2 ||
        ports > (1u << 24) || equations > (1u << 24) ||
        words_per_row != (dimension / 64u + (dimension % 64u != 0))) return 1;
    const auto groups = static_cast<std::uint64_t>(batch) * equations;
    if (groups > std::numeric_limits<std::uint64_t>::max() / dimension / 3) return 2;
    if (group_capacity < groups || incidence_capacity < groups * dimension * 3) return 1;
    const auto words = static_cast<std::uint64_t>(equations) * ((ports + 63) / 64);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return cnot103::launch_packed_blocks(batch, words, cuda_stream,
        [&](unsigned blocks, std::size_t shared_bytes, cnot103::PackedBlockScratch scratch) {
            boundary_relations_kernel<<<blocks, 256, shared_bytes, cuda_stream>>>(
                states, port_wires, protected_ports, equation_outputs, dimension, words_per_row,
                inputs, outputs, equations, values, valid, incidence, ranks, scratch);
        });
}
