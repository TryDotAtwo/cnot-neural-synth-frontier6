#include "cnot103/cnot_cuda.h"
#include "packed_block_scratch.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr std::int32_t kOk = 0;
constexpr std::int32_t kInvalidArgument = 1;
constexpr std::int32_t kSizeOverflow = 2;
constexpr std::int32_t kLaunchFailure = 3;
constexpr std::int32_t kSyncFailure = 4;

__device__ inline std::uint64_t matrix_bit(
    const std::uint64_t* input,
    std::uint64_t state_base,
    std::uint32_t row,
    std::uint32_t column,
    std::uint32_t words_per_row) {
    const auto word = input[state_base +
        static_cast<std::uint64_t>(row) * words_per_row + column / 64u];
    return (word >> (column % 64u)) & 1u;
}

__global__ void expand_children_kernel(
    const std::uint64_t* __restrict__ input,
    std::uint64_t* __restrict__ output,
    std::uint64_t total_words,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint64_t actions_per_state) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_words) return;

    std::uint64_t cursor = index;
    const std::uint32_t word = static_cast<std::uint32_t>(cursor % words_per_row);
    cursor /= words_per_row;
    const std::uint32_t row = static_cast<std::uint32_t>(cursor % dimension);
    cursor /= dimension;
    const std::uint64_t action = cursor % actions_per_state;
    const std::uint64_t batch_index = cursor / actions_per_state;

    const std::uint32_t control = static_cast<std::uint32_t>(action / (dimension - 1));
    const std::uint32_t target_offset = static_cast<std::uint32_t>(action % (dimension - 1));
    const std::uint32_t target = target_offset < control ? target_offset : target_offset + 1;

    const std::uint64_t state_base = batch_index * dimension * words_per_row;
    const std::uint64_t source_index = state_base + static_cast<std::uint64_t>(control) * words_per_row + word;
    const std::uint64_t row_index = state_base + static_cast<std::uint64_t>(row) * words_per_row + word;
    std::uint64_t value = input[row_index];
    if (row == target) value ^= input[source_index];
    output[index] = value;
}

__global__ void family_a_pair_kernel(
    const std::uint64_t* __restrict__ input,
    float* __restrict__ output,
    std::uint64_t total_pairs,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint64_t cells,
    std::uint64_t feature_width) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_pairs) return;

    const std::uint64_t state = index / cells;
    const std::uint64_t pair = index % cells;
    const std::uint32_t row = static_cast<std::uint32_t>(pair / dimension);
    const std::uint32_t column = static_cast<std::uint32_t>(pair % dimension);
    const std::uint64_t state_base =
        state * static_cast<std::uint64_t>(dimension) * words_per_row;
    const std::uint64_t output_base = state * feature_width;

    std::int64_t row_weight_i = 0;
    std::int64_t row_weight_j = 0;
    std::int64_t col_weight_i = 0;
    std::int64_t col_weight_j = 0;
    std::int64_t row_intersection = 0;
    std::int64_t col_intersection = 0;
    std::int64_t residual_row_j_dot_row_i = 0;
    for (std::uint32_t k = 0; k < dimension; ++k) {
        const auto m_ik = matrix_bit(input, state_base, row, k, words_per_row);
        const auto m_jk = matrix_bit(input, state_base, column, k, words_per_row);
        const auto m_ki = matrix_bit(input, state_base, k, row, words_per_row);
        const auto m_kj = matrix_bit(input, state_base, k, column, words_per_row);
        row_weight_i += static_cast<std::int64_t>(m_ik);
        row_weight_j += static_cast<std::int64_t>(m_jk);
        col_weight_i += static_cast<std::int64_t>(m_ki);
        col_weight_j += static_cast<std::int64_t>(m_kj);
        row_intersection += static_cast<std::int64_t>(m_ik * m_jk);
        col_intersection += static_cast<std::int64_t>(m_ki * m_kj);
        const auto residual_jk = m_jk ^ static_cast<std::uint64_t>(column == k);
        residual_row_j_dot_row_i += static_cast<std::int64_t>(residual_jk * m_ik);
    }

    const auto m_ij = matrix_bit(input, state_base, row, column, words_per_row);
    const auto m_ji = matrix_bit(input, state_base, column, row, words_per_row);
    const auto pair_base = output_base;

    output[pair_base + pair] = static_cast<float>(m_ij);                         // A01
    output[pair_base + cells + pair] = static_cast<float>(m_ji);                  // A02
    output[pair_base + 2 * cells + 2 * pair] = row == column ? 1.0f : 0.0f;      // A03.0
    output[pair_base + 2 * cells + 2 * pair + 1] = 1.0f;                          // A03.1
    output[pair_base + 4 * cells + pair] =
        static_cast<float>(m_ij ^ static_cast<std::uint64_t>(row == column));     // A04
    output[pair_base + 5 * cells + pair] = static_cast<float>(row_intersection); // A13
    output[pair_base + 6 * cells + pair] = static_cast<float>(col_intersection); // A14
    output[pair_base + 7 * cells + pair] = static_cast<float>(
        row_weight_i + row_weight_j - 2 * row_intersection);                     // A15
    output[pair_base + 8 * cells + pair] = static_cast<float>(
        col_weight_i + col_weight_j - 2 * col_intersection);                     // A16

    const auto row_diff_ij = row_weight_i - row_intersection;
    const auto row_diff_ji = row_weight_j - row_intersection;
    const auto col_diff_ij = col_weight_i - col_intersection;
    const auto col_diff_ji = col_weight_j - col_intersection;
    output[pair_base + 9 * cells + 2 * pair] = static_cast<float>(row_diff_ij); // A17.0
    output[pair_base + 9 * cells + 2 * pair + 1] = static_cast<float>(row_diff_ji);
    output[pair_base + 11 * cells + 2 * pair] = static_cast<float>(col_diff_ij); // A18.0
    output[pair_base + 11 * cells + 2 * pair + 1] = static_cast<float>(col_diff_ji);
    output[pair_base + 13 * cells + 2 * pair] = row_diff_ij == 0 ? 1.0f : 0.0f; // A19.0
    output[pair_base + 13 * cells + 2 * pair + 1] = col_diff_ij == 0 ? 1.0f : 0.0f;
    if (row == column) {
        output[pair_base + 15 * cells + 2 * pair] = 0.0f;                       // A20
        output[pair_base + 15 * cells + 2 * pair + 1] = 0.0f;
    } else {
        output[pair_base + 15 * cells + 2 * pair] =
            static_cast<float>(row_weight_i - 2 * row_intersection);
        output[pair_base + 15 * cells + 2 * pair + 1] =
            static_cast<float>(row_weight_i - 2 * residual_row_j_dot_row_i);
    }
}

__global__ void family_a_node_kernel(
    const std::uint64_t* __restrict__ input,
    float* __restrict__ output,
    std::uint64_t total_nodes,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint64_t cells,
    std::uint64_t feature_width) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_nodes) return;

    const std::uint64_t state = index / dimension;
    const std::uint32_t node = static_cast<std::uint32_t>(index % dimension);
    const std::uint64_t state_base =
        state * static_cast<std::uint64_t>(dimension) * words_per_row;
    const std::uint64_t node_base = state * feature_width + 17 * cells;
    std::int64_t row_weight = 0;
    std::int64_t col_weight = 0;
    std::int64_t residual_row_weight = 0;
    std::int64_t residual_col_weight = 0;
    for (std::uint32_t k = 0; k < dimension; ++k) {
        const auto row_bit = matrix_bit(input, state_base, node, k, words_per_row);
        const auto col_bit = matrix_bit(input, state_base, k, node, words_per_row);
        const auto diagonal = static_cast<std::uint64_t>(node == k);
        row_weight += static_cast<std::int64_t>(row_bit);
        col_weight += static_cast<std::int64_t>(col_bit);
        residual_row_weight += static_cast<std::int64_t>(row_bit ^ diagonal);
        residual_col_weight += static_cast<std::int64_t>(col_bit ^ diagonal);
    }
    const auto diagonal = matrix_bit(input, state_base, node, node, words_per_row);
    output[node_base + 2 * node] = static_cast<float>(row_weight);                // A05.0
    output[node_base + 2 * node + 1] =
        static_cast<float>(row_weight) / static_cast<float>(dimension);           // A05.1
    output[node_base + 2 * dimension + node] = static_cast<float>(col_weight);    // A06
    output[node_base + 3 * dimension + node] =
        static_cast<float>(residual_row_weight);                                  // A07
    output[node_base + 4 * dimension + node] =
        static_cast<float>(residual_col_weight);                                  // A08
    output[node_base + 5 * dimension + node] =
        residual_row_weight > 0 ? 1.0f : 0.0f;                                   // A09
    output[node_base + 6 * dimension + node] =
        residual_col_weight > 0 ? 1.0f : 0.0f;                                   // A10
    output[node_base + 7 * dimension + node] =
        static_cast<float>(1u - static_cast<std::uint32_t>(diagonal));            // A11
    output[node_base + 8 * dimension + 2 * node] =
        static_cast<float>(row_weight & 1);                                       // A12.0
    output[node_base + 8 * dimension + 2 * node + 1] =
        static_cast<float>(col_weight & 1);                                       // A12.1
}


__global__ void residual_rank_kernel(
    const std::uint64_t* __restrict__ input,
    float* __restrict__ output,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    cnot103::PackedBlockScratch scratch) {
    extern __shared__ std::uint64_t shared[];
    auto* rows = scratch.rows(shared);
    __shared__ std::int32_t rank;
    __shared__ std::int32_t pivot;

    const std::uint64_t state_base = scratch.item() * dimension * words_per_row;
    const std::uint64_t packed_words =
        static_cast<std::uint64_t>(dimension) * words_per_row;
    for (std::uint64_t index = threadIdx.x; index < packed_words; index += blockDim.x) {
        const auto row = static_cast<std::uint32_t>(index / words_per_row);
        const auto word = static_cast<std::uint32_t>(index % words_per_row);
        std::uint64_t value = input[state_base + index];
        if (word == row / 64u) value ^= std::uint64_t{1} << (row % 64u);
        rows[index] = value;
    }
    if (threadIdx.x == 0) rank = 0;
    __syncthreads();

    for (std::uint32_t column = 0; column < dimension; ++column) {
        if (threadIdx.x == 0) pivot = static_cast<std::int32_t>(dimension);
        __syncthreads();

        const auto current_rank = static_cast<std::uint32_t>(rank);
        for (std::uint32_t row = current_rank + threadIdx.x;
             row < dimension;
             row += blockDim.x) {
            const auto bit = (rows[static_cast<std::uint64_t>(row) * words_per_row +
                                   column / 64u] >> (column % 64u)) & 1u;
            if (bit != 0u) atomicMin(&pivot, static_cast<std::int32_t>(row));
        }
        __syncthreads();

        const auto selected_pivot = pivot;
        if (selected_pivot < static_cast<std::int32_t>(dimension)) {
            const auto selected_row = static_cast<std::uint32_t>(selected_pivot);
            const auto pivot_row = static_cast<std::uint32_t>(rank);
            for (std::uint32_t word = threadIdx.x;
                 word < words_per_row;
                 word += blockDim.x) {
                const auto pivot_index =
                    static_cast<std::uint64_t>(pivot_row) * words_per_row + word;
                const auto selected_index =
                    static_cast<std::uint64_t>(selected_row) * words_per_row + word;
                const auto saved = rows[pivot_index];
                rows[pivot_index] = rows[selected_index];
                rows[selected_index] = saved;
            }
            __syncthreads();

            const auto active_pivot = static_cast<std::uint32_t>(rank);
            for (std::uint32_t row = threadIdx.x; row < dimension; row += blockDim.x) {
                if (row == active_pivot) continue;
                const auto bit = (rows[static_cast<std::uint64_t>(row) * words_per_row +
                                       column / 64u] >> (column % 64u)) & 1u;
                if (bit != 0u) {
                    for (std::uint32_t word = 0; word < words_per_row; ++word) {
                        rows[static_cast<std::uint64_t>(row) * words_per_row + word] ^=
                            rows[static_cast<std::uint64_t>(active_pivot) * words_per_row + word];
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) ++rank;
            __syncthreads();
        }
        // The no-pivot path must finish reading pivot before its next reset.
        __syncthreads();
    }

    if (threadIdx.x == 0) output[scratch.item()] = static_cast<float>(rank);
}

__global__ void projected_rank_kernel(
    const std::uint64_t* __restrict__ input,
    const std::uint64_t* __restrict__ row_masks,
    const std::uint64_t* __restrict__ column_masks,
    const std::uint64_t* __restrict__ candidate_masks,
    const std::uint32_t* __restrict__ gate_controls,
    const std::uint32_t* __restrict__ gate_targets,
    float* __restrict__ output,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint32_t cut_count,
    std::uint32_t subtract_identity,
    std::uint32_t append_candidate,
    std::uint32_t apply_gate,
    cnot103::PackedBlockScratch scratch) {
    extern __shared__ std::uint64_t shared[];
    auto* rows = scratch.rows(shared);
    __shared__ std::int32_t rank;
    __shared__ std::int32_t pivot;

    const std::uint64_t work_item = scratch.item();
    const std::uint32_t cut = static_cast<std::uint32_t>(work_item % cut_count);
    const std::uint64_t state = work_item / cut_count;
    const std::uint64_t state_base =
        state * static_cast<std::uint64_t>(dimension) * words_per_row;
    const std::uint64_t mask_base = static_cast<std::uint64_t>(cut) * words_per_row;
    const std::uint32_t row_count = dimension + append_candidate;
    const std::uint32_t gate_control = apply_gate != 0u ? gate_controls[cut] : 0u;
    const std::uint32_t gate_target = apply_gate != 0u ? gate_targets[cut] : 0u;
    const std::uint64_t packed_words =
        static_cast<std::uint64_t>(row_count) * words_per_row;

    for (std::uint64_t index = threadIdx.x; index < packed_words; index += blockDim.x) {
        const auto row = static_cast<std::uint32_t>(index / words_per_row);
        const auto word = static_cast<std::uint32_t>(index % words_per_row);
        const auto column_mask = column_masks[mask_base + word];
        std::uint64_t value = 0;
        if (row == dimension) {
            value = candidate_masks[mask_base + word] & column_mask;
        } else {
            const auto row_bit = (row_masks[mask_base + row / 64u] >> (row % 64u)) & 1u;
            if (row_bit != 0u) {
                value = input[state_base + static_cast<std::uint64_t>(row) * words_per_row + word]
                    & column_mask;
                if (apply_gate != 0u && row == gate_target) {
                    value ^= input[state_base + static_cast<std::uint64_t>(gate_control) * words_per_row + word]
                        & column_mask;
                }
                if (subtract_identity != 0u && word == row / 64u) {
                    const std::uint64_t diagonal_bit = std::uint64_t{1} << (row % 64u);
                    if ((column_mask & diagonal_bit) != 0u) value ^= diagonal_bit;
                }
            }
        }
        rows[index] = value;
    }
    if (threadIdx.x == 0) rank = 0;
    __syncthreads();

    for (std::uint32_t column = 0; column < dimension; ++column) {
        if (threadIdx.x == 0) pivot = static_cast<std::int32_t>(row_count);
        __syncthreads();

        const auto current_rank = static_cast<std::uint32_t>(rank);
        for (std::uint32_t row = current_rank + threadIdx.x;
             row < row_count;
             row += blockDim.x) {
            const auto bit = (rows[static_cast<std::uint64_t>(row) * words_per_row +
                                   column / 64u] >> (column % 64u)) & 1u;
            if (bit != 0u) atomicMin(&pivot, static_cast<std::int32_t>(row));
        }
        __syncthreads();

        const auto selected_pivot = pivot;
        if (selected_pivot < static_cast<std::int32_t>(row_count)) {
            const auto selected_row = static_cast<std::uint32_t>(selected_pivot);
            const auto pivot_row = static_cast<std::uint32_t>(rank);
            for (std::uint32_t word = threadIdx.x;
                 word < words_per_row;
                 word += blockDim.x) {
                const auto pivot_index =
                    static_cast<std::uint64_t>(pivot_row) * words_per_row + word;
                const auto selected_index =
                    static_cast<std::uint64_t>(selected_row) * words_per_row + word;
                const auto saved = rows[pivot_index];
                rows[pivot_index] = rows[selected_index];
                rows[selected_index] = saved;
            }
            __syncthreads();

            const auto active_pivot = static_cast<std::uint32_t>(rank);
            for (std::uint32_t row = threadIdx.x; row < row_count; row += blockDim.x) {
                if (row == active_pivot) continue;
                const auto bit = (rows[static_cast<std::uint64_t>(row) * words_per_row +
                                       column / 64u] >> (column % 64u)) & 1u;
                if (bit != 0u) {
                    for (std::uint32_t word = 0; word < words_per_row; ++word) {
                        rows[static_cast<std::uint64_t>(row) * words_per_row + word] ^=
                            rows[static_cast<std::uint64_t>(active_pivot) * words_per_row + word];
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) ++rank;
            __syncthreads();
        }
        // Includes iterations with no pivot; all warps must finish their reads.
        __syncthreads();
    }

    if (threadIdx.x == 0) output[work_item] = static_cast<float>(rank);
}

__global__ void partial_shear_rank_kernel(
    const std::uint64_t* __restrict__ input,
    float* __restrict__ output,
    std::uint8_t* __restrict__ status,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint32_t source_width,
    std::uint32_t target_count,
    std::uint32_t target_words,
    cnot103::PackedBlockScratch scratch) {
    extern __shared__ std::uint64_t shared[];
    auto* workspace = scratch.rows(shared);
    __shared__ std::int32_t pivot_row;
    __shared__ std::int32_t selected;
    __shared__ std::int32_t rank;
    __shared__ std::int32_t pivot;
    __shared__ std::int32_t nonzero;

    const std::uint64_t state_base = scratch.item() * dimension * words_per_row;
    const std::uint64_t source_storage =
        static_cast<std::uint64_t>(source_width) * words_per_row;
    const std::uint64_t target_storage =
        static_cast<std::uint64_t>(target_count) * target_words;
    std::uint64_t* source = workspace;
    std::uint64_t* target = source + source_storage;
    std::uint64_t* pivot_columns = target + target_storage;

    for (std::uint64_t index = threadIdx.x; index < source_storage; index += blockDim.x) {
        source[index] = input[state_base + index];
    }
    if (threadIdx.x == 0) pivot_row = 0;
    __syncthreads();

    for (std::uint32_t column = 0; column < dimension; ++column) {
        const auto active_row = static_cast<std::uint32_t>(pivot_row);
        if (active_row == source_width) break;
        if (threadIdx.x == 0) selected = static_cast<std::int32_t>(source_width);
        __syncthreads();
        const auto column_word = column / 64u;
        const auto column_bit = std::uint64_t{1} << (column % 64u);
        for (std::uint32_t row = active_row + threadIdx.x; row < source_width; row += blockDim.x) {
            if ((source[static_cast<std::uint64_t>(row) * words_per_row + column_word] & column_bit) != 0u) {
                atomicMin(&selected, static_cast<std::int32_t>(row));
            }
        }
        __syncthreads();
        if (selected < static_cast<std::int32_t>(source_width)) {
            const auto found = static_cast<std::uint32_t>(selected);
            for (std::uint32_t word = threadIdx.x; word < words_per_row; word += blockDim.x) {
                const auto pivot_index = static_cast<std::uint64_t>(active_row) * words_per_row + word;
                const auto found_index = static_cast<std::uint64_t>(found) * words_per_row + word;
                const auto saved = source[pivot_index];
                source[pivot_index] = source[found_index];
                source[found_index] = saved;
            }
            __syncthreads();
            for (std::uint32_t row = active_row + 1u + threadIdx.x; row < source_width; row += blockDim.x) {
                const auto row_index = static_cast<std::uint64_t>(row) * words_per_row + column_word;
                if ((source[row_index] & column_bit) != 0u) {
                    const auto pivot_base = static_cast<std::uint64_t>(active_row) * words_per_row;
                    const auto row_base = static_cast<std::uint64_t>(row) * words_per_row;
                    for (std::uint32_t word = 0; word < words_per_row; ++word) {
                        source[row_base + word] ^= source[pivot_base + word];
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                pivot_columns[active_row] = column;
                ++pivot_row;
            }
            __syncthreads();
        }
        // Do not reset selected while another warp tests the no-pivot branch.
        __syncthreads();
    }
    if (pivot_row != static_cast<std::int32_t>(source_width)) {
        if (threadIdx.x == 0) {
            output[scratch.item()] = 0.0f;
            status[scratch.item()] = 2u;
        }
        return;
    }

    for (std::uint64_t index = threadIdx.x; index < target_storage; index += blockDim.x) {
        const auto target_row = static_cast<std::uint32_t>(index / target_words);
        const auto target_word = static_cast<std::uint32_t>(index % target_words);
        std::uint64_t value = 0;
        const auto first_pivot = target_word * 64u;
        const auto candidate_end = first_pivot + 64u;
        const auto end_pivot = candidate_end < source_width ? candidate_end : source_width;
        for (std::uint32_t source = first_pivot; source < end_pivot; ++source) {
            const auto column = static_cast<std::uint32_t>(pivot_columns[source]);
            const auto bit = (input[state_base + static_cast<std::uint64_t>(source_width + target_row) * words_per_row + column / 64u]
                              >> (column % 64u)) & 1u;
            if (bit != 0u) value |= std::uint64_t{1} << (source % 64u);
        }
        target[index] = value;
    }
    if (threadIdx.x == 0) nonzero = 0;
    __syncthreads();
    for (std::uint64_t index = threadIdx.x; index < target_storage; index += blockDim.x) {
        if (target[index] != 0u) atomicExch(&nonzero, 1);
    }
    __syncthreads();
    if (nonzero == 0) {
        if (threadIdx.x == 0) {
            output[scratch.item()] = 0.0f;
            status[scratch.item()] = 0u;
        }
        return;
    }

    if (threadIdx.x == 0) rank = 0;
    __syncthreads();
    for (std::uint32_t column = 0; column < source_width; ++column) {
        if (threadIdx.x == 0) pivot = static_cast<std::int32_t>(target_count);
        __syncthreads();
        const auto active_rank = static_cast<std::uint32_t>(rank);
        const auto column_word = column / 64u;
        const auto column_bit = std::uint64_t{1} << (column % 64u);
        for (std::uint32_t row = active_rank + threadIdx.x; row < target_count; row += blockDim.x) {
            if ((target[static_cast<std::uint64_t>(row) * target_words + column_word] & column_bit) != 0u) {
                atomicMin(&pivot, static_cast<std::int32_t>(row));
            }
        }
        __syncthreads();
        if (pivot < static_cast<std::int32_t>(target_count)) {
            const auto found = static_cast<std::uint32_t>(pivot);
            for (std::uint32_t word = threadIdx.x; word < target_words; word += blockDim.x) {
                const auto rank_index = static_cast<std::uint64_t>(active_rank) * target_words + word;
                const auto found_index = static_cast<std::uint64_t>(found) * target_words + word;
                const auto saved = target[rank_index];
                target[rank_index] = target[found_index];
                target[found_index] = saved;
            }
            __syncthreads();
            for (std::uint32_t row = threadIdx.x; row < target_count; row += blockDim.x) {
                if (row == active_rank) continue;
                const auto row_base = static_cast<std::uint64_t>(row) * target_words;
                if ((target[row_base + column_word] & column_bit) != 0u) {
                    const auto pivot_base = static_cast<std::uint64_t>(active_rank) * target_words;
                    for (std::uint32_t word = 0; word < target_words; ++word) {
                        target[row_base + word] ^= target[pivot_base + word];
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) ++rank;
            __syncthreads();
        }
        __syncthreads(); // complete no-pivot reads before the next pivot reset
    }
    if (threadIdx.x == 0) {
        output[scratch.item()] = static_cast<float>(rank);
        status[scratch.item()] = 1u;
    }
}

constexpr std::uint64_t kMaxExactF32Integer = 1ull << 24;

__device__ bool block_pattern_budget_exceeded(
    const std::uint64_t* input,
    const std::uint64_t* source_mask,
    const std::uint64_t* target_mask,
    std::uint64_t matrix_base,
    std::uint64_t max_patterns,
    std::uint32_t dimension,
    std::uint32_t words_per_row) {
    std::uint64_t target_count = 0;
    for (std::uint32_t word = 0; word < words_per_row; ++word) {
        target_count += __popcll(target_mask[word]);
    }
    if (max_patterns >= target_count + 1u) return false;

    std::uint64_t distinct_patterns = 1u;
    for (std::uint32_t row = 0; row < dimension; ++row) {
        if (((target_mask[row / 64u] >> (row % 64u)) & 1ull) == 0ull) continue;
        bool nonzero = false;
        for (std::uint32_t column = 0; column < dimension; ++column) {
            if (((source_mask[column / 64u] >> (column % 64u)) & 1ull) != 0ull &&
                matrix_bit(input, matrix_base, row, column, words_per_row) != 0ull) {
                nonzero = true;
                break;
            }
        }
        if (!nonzero) continue;

        bool duplicate = false;
        for (std::uint32_t previous = 0; previous < row && !duplicate; ++previous) {
            if (((target_mask[previous / 64u] >> (previous % 64u)) & 1ull) == 0ull) continue;
            bool equal = true;
            for (std::uint32_t column = 0; column < dimension; ++column) {
                if (((source_mask[column / 64u] >> (column % 64u)) & 1ull) == 0ull) continue;
                const auto current_bit = matrix_bit(input, matrix_base, row, column, words_per_row);
                const auto previous_bit = matrix_bit(input, matrix_base, previous, column, words_per_row);
                if (current_bit != previous_bit) {
                    equal = false;
                    break;
                }
            }
            duplicate = equal;
        }
        if (!duplicate) {
            ++distinct_patterns;
            if (distinct_patterns > max_patterns) return true;
        }
    }
    return false;
}

__global__ void block_parity_weight_kernel(
    const std::uint64_t* __restrict__ input,
    const std::uint64_t* __restrict__ source_masks,
    const std::uint64_t* __restrict__ target_masks,
    float* __restrict__ output,
    std::uint8_t* __restrict__ status,
    std::uint8_t* __restrict__ token_valid,
    std::uint64_t cells,
    std::uint32_t batch,
    std::uint32_t context_count,
    std::uint64_t max_patterns,
    std::uint32_t dimension,
    std::uint32_t words_per_row) {
    const std::uint64_t task = blockIdx.x;
    const std::uint32_t context = static_cast<std::uint32_t>(task % context_count);
    const std::uint32_t state = static_cast<std::uint32_t>(task / context_count);
    if (state >= batch) return;

    const std::uint64_t matrix_base =
        static_cast<std::uint64_t>(state) * dimension * words_per_row;
    const std::uint64_t mask_base =
        static_cast<std::uint64_t>(context) * words_per_row;
    const auto* source_mask = source_masks + mask_base;
    const auto* target_mask = target_masks + mask_base;

    std::uint64_t local_count = 0;
    for (std::uint64_t cell = threadIdx.x; cell < cells; cell += blockDim.x) {
        const auto row = static_cast<std::uint32_t>(cell / dimension);
        const auto column = static_cast<std::uint32_t>(cell % dimension);
        const bool selected_target =
            ((target_mask[row / 64u] >> (row % 64u)) & 1ull) != 0ull;
        const bool selected_source =
            ((source_mask[column / 64u] >> (column % 64u)) & 1ull) != 0ull;
        if (selected_target && selected_source &&
            matrix_bit(input, matrix_base, row, column, words_per_row) != 0ull) {
            ++local_count;
        }
    }

    __shared__ std::uint64_t partial[128];
    partial[threadIdx.x] = local_count;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        const bool pattern_budget_exceeded = block_pattern_budget_exceeded(
            input, source_mask, target_mask, matrix_base, max_patterns, dimension, words_per_row);
        if (!pattern_budget_exceeded && partial[0] <= kMaxExactF32Integer) {
            output[task] = static_cast<float>(partial[0]);
            status[task] = 1u;
            token_valid[task] = 1u;
        } else {
            output[task] = 0.0f;
            status[task] = 2u;
            token_valid[task] = 0u;
        }
    }
}

__global__ void block_node_frequency_kernel(
    const std::uint64_t* __restrict__ input,
    const std::uint64_t* __restrict__ source_masks,
    const std::uint64_t* __restrict__ target_masks,
    float* __restrict__ output,
    std::uint8_t* __restrict__ status,
    std::uint64_t batch,
    std::uint32_t context_count,
    std::uint64_t max_patterns,
    std::uint32_t dimension,
    std::uint32_t words_per_row) {
    const std::uint64_t task = blockIdx.x;
    const std::uint32_t context = static_cast<std::uint32_t>(task % context_count);
    const std::uint64_t state = task / context_count;
    if (state >= batch) return;

    const std::uint64_t matrix_base = state * dimension * words_per_row;
    const std::uint64_t mask_base = static_cast<std::uint64_t>(context) * words_per_row;
    const auto* source_mask = source_masks + mask_base;
    const auto* target_mask = target_masks + mask_base;
    const std::uint64_t feature_width = static_cast<std::uint64_t>(dimension) * 2u;
    const std::uint64_t output_base = task * feature_width;

    __shared__ std::uint32_t resource_limited;
    if (threadIdx.x == 0u) {
        resource_limited = block_pattern_budget_exceeded(
            input, source_mask, target_mask, matrix_base, max_patterns, dimension, words_per_row)
            ? 1u
            : 0u;
    }
    __syncthreads();

    for (std::uint64_t item = threadIdx.x; item < feature_width; item += blockDim.x) {
        float value = 0.0f;
        const std::uint32_t wire = static_cast<std::uint32_t>(item / 2u);
        const std::uint32_t channel = static_cast<std::uint32_t>(item % 2u);
        if (resource_limited == 0u && channel == 0u &&
            ((target_mask[wire / 64u] >> (wire % 64u)) & 1ull) != 0ull) {
            std::uint64_t frequency = 0;
            for (std::uint32_t candidate = 0; candidate < dimension; ++candidate) {
                if (((target_mask[candidate / 64u] >> (candidate % 64u)) & 1ull) == 0ull) continue;
                bool equal = true;
                for (std::uint32_t column = 0; column < dimension; ++column) {
                    if (((source_mask[column / 64u] >> (column % 64u)) & 1ull) == 0ull) continue;
                    if (matrix_bit(input, matrix_base, wire, column, words_per_row) !=
                        matrix_bit(input, matrix_base, candidate, column, words_per_row)) {
                        equal = false;
                        break;
                    }
                }
                frequency += static_cast<std::uint64_t>(equal);
            }
            if (frequency > kMaxExactF32Integer) {
                atomicExch(&resource_limited, 1u);
            } else {
                value = static_cast<float>(frequency);
            }
        } else if (resource_limited == 0u && channel == 1u &&
                   ((source_mask[wire / 64u] >> (wire % 64u)) & 1ull) != 0ull) {
            std::uint64_t frequency = 0;
            for (std::uint32_t candidate = 0; candidate < dimension; ++candidate) {
                if (((source_mask[candidate / 64u] >> (candidate % 64u)) & 1ull) == 0ull) continue;
                bool equal = true;
                for (std::uint32_t row = 0; row < dimension; ++row) {
                    if (((target_mask[row / 64u] >> (row % 64u)) & 1ull) == 0ull) continue;
                    if (matrix_bit(input, matrix_base, row, wire, words_per_row) !=
                        matrix_bit(input, matrix_base, row, candidate, words_per_row)) {
                        equal = false;
                        break;
                    }
                }
                frequency += static_cast<std::uint64_t>(equal);
            }
            if (frequency > kMaxExactF32Integer) {
                atomicExch(&resource_limited, 1u);
            } else {
                value = static_cast<float>(frequency);
            }
        }
        output[output_base + item] = value;
    }
    __syncthreads();
    if (resource_limited != 0u) {
        for (std::uint64_t item = threadIdx.x; item < feature_width; item += blockDim.x) {
            output[output_base + item] = 0.0f;
        }
    }
    if (threadIdx.x == 0u) status[task] = resource_limited == 0u ? 1u : 2u;
}

bool checked_mul(std::uint64_t left, std::uint64_t right, std::uint64_t* output) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) return false;
    *output = left * right;
    return true;
}

bool checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t* output) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) return false;
    *output = left + right;
    return true;
}

bool blocks_for(std::uint64_t work_items, std::uint32_t threads, unsigned int* blocks) {
    const std::uint64_t count =
        work_items / threads + static_cast<std::uint64_t>(work_items % threads != 0u);
    if (count == 0 || count > std::numeric_limits<unsigned int>::max()) return false;
    *blocks = static_cast<unsigned int>(count);
    return true;
}

}  // namespace

extern "C" std::int32_t cnot103_cuda_launch_children(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint64_t* output_device,
    std::uint64_t output_capacity_words,
    void* stream) {
    if (input_device == nullptr || output_device == nullptr || batch == 0 || dimension < 2 ||
        words_per_row != dimension / 64u + static_cast<std::uint32_t>(dimension % 64u != 0u)) {
        return kInvalidArgument;
    }

    const std::uint64_t actions = static_cast<std::uint64_t>(dimension) * (dimension - 1u);
    std::uint64_t total_words = batch;
    if (!checked_mul(total_words, actions, &total_words) ||
        !checked_mul(total_words, dimension, &total_words) ||
        !checked_mul(total_words, words_per_row, &total_words)) {
        return kSizeOverflow;
    }
    if (output_capacity_words < total_words) return kInvalidArgument;

    constexpr std::uint32_t kThreads = 256;
    unsigned int blocks = 0;
    if (!blocks_for(total_words, kThreads, &blocks)) return kSizeOverflow;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    expand_children_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
        input_device, output_device, total_words, dimension, words_per_row, actions);
    return cudaGetLastError() == cudaSuccess ? kOk : kLaunchFailure;
}

extern "C" std::int32_t cnot103_cuda_launch_family_a(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    float* output_device,
    std::uint64_t output_capacity_elements,
    void* stream) {
    if (input_device == nullptr || output_device == nullptr || batch == 0 || dimension < 2 ||
        words_per_row != dimension / 64u + static_cast<std::uint32_t>(dimension % 64u != 0u)) {
        return kInvalidArgument;
    }

    std::uint64_t cells = 0;
    if (!checked_mul(dimension, dimension, &cells)) return kSizeOverflow;
    std::uint64_t pair_elements = 0;
    std::uint64_t node_elements = 0;
    if (!checked_mul(17, cells, &pair_elements) ||
        !checked_mul(10, dimension, &node_elements) ||
        pair_elements > std::numeric_limits<std::uint64_t>::max() - node_elements) {
        return kSizeOverflow;
    }
    const std::uint64_t feature_width = pair_elements + node_elements;
    std::uint64_t required = 0;
    if (!checked_mul(batch, feature_width, &required)) return kSizeOverflow;
    if (output_capacity_elements < required) return kInvalidArgument;

    std::uint64_t total_pairs = 0;
    std::uint64_t total_nodes = 0;
    if (!checked_mul(batch, cells, &total_pairs) ||
        !checked_mul(batch, dimension, &total_nodes)) {
        return kSizeOverflow;
    }

    constexpr std::uint32_t kThreads = 256;
    unsigned int pair_blocks = 0;
    unsigned int node_blocks = 0;
    if (!blocks_for(total_pairs, kThreads, &pair_blocks) ||
        !blocks_for(total_nodes, kThreads, &node_blocks)) {
        return kSizeOverflow;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    family_a_pair_kernel<<<pair_blocks, kThreads, 0, cuda_stream>>>(
        input_device, output_device, total_pairs, dimension, words_per_row, cells, feature_width);
    if (cudaGetLastError() != cudaSuccess) return kLaunchFailure;
    family_a_node_kernel<<<node_blocks, kThreads, 0, cuda_stream>>>(
        input_device, output_device, total_nodes, dimension, words_per_row, cells, feature_width);
    return cudaGetLastError() == cudaSuccess ? kOk : kLaunchFailure;
}


extern "C" std::int32_t cnot103_cuda_launch_residual_rank(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    float* output_device,
    std::uint64_t output_capacity_elements,
    void* stream) {
    if (input_device == nullptr || output_device == nullptr || batch == 0 || dimension < 2 ||
        words_per_row != dimension / 64u + static_cast<std::uint32_t>(dimension % 64u != 0u)) {
        return kInvalidArgument;
    }
    if (output_capacity_elements < batch) return kInvalidArgument;

    const auto words = static_cast<std::uint64_t>(dimension) * words_per_row;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return cnot103::launch_packed_blocks(batch, words, cuda_stream,
        [&](unsigned blocks, std::size_t shared_bytes, cnot103::PackedBlockScratch scratch) {
            residual_rank_kernel<<<blocks, 256, shared_bytes, cuda_stream>>>(
                input_device, output_device, dimension, words_per_row, scratch);
        });
}

extern "C" std::int32_t cnot103_cuda_launch_block_node_frequencies(
    const std::uint64_t* input_device,
    const std::uint64_t* source_masks_device,
    const std::uint64_t* target_masks_device,
    std::uint32_t batch,
    std::uint32_t context_count,
    std::uint64_t max_patterns,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    float* output_device,
    std::uint8_t* status_device,
    std::uint64_t output_capacity_elements,
    void* stream) {
    if (input_device == nullptr || source_masks_device == nullptr ||
        target_masks_device == nullptr || output_device == nullptr || status_device == nullptr ||
        batch == 0 || context_count == 0 || max_patterns == 0 || dimension < 2 ||
        words_per_row != dimension / 64u +
            static_cast<std::uint32_t>(dimension % 64u != 0u)) {
        return kInvalidArgument;
    }

    std::uint64_t tasks = 0;
    std::uint64_t output_per_task = 0;
    std::uint64_t required = 0;
    if (!checked_mul(batch, context_count, &tasks) ||
        !checked_mul(dimension, 2u, &output_per_task) ||
        !checked_mul(tasks, output_per_task, &required)) {
        return kSizeOverflow;
    }
    if (output_capacity_elements < required) return kInvalidArgument;
    if (tasks > 2147483647ull) return kSizeOverflow;

    constexpr std::uint32_t kThreads = 256;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    block_node_frequency_kernel<<<static_cast<unsigned int>(tasks), kThreads, 0, cuda_stream>>>(
        input_device, source_masks_device, target_masks_device, output_device, status_device,
        batch, context_count, max_patterns, dimension, words_per_row);
    return cudaGetLastError() == cudaSuccess ? kOk : kLaunchFailure;
}

extern "C" std::int32_t cnot103_cuda_launch_block_parity_weight(
    const std::uint64_t* input_device,
    const std::uint64_t* source_masks_device,
    const std::uint64_t* target_masks_device,
    std::uint32_t batch,
    std::uint32_t context_count,
    std::uint64_t max_patterns,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    float* output_device,
    std::uint8_t* status_device,
    std::uint8_t* token_valid_device,
    std::uint64_t output_capacity_elements,
    void* stream) {
    if (input_device == nullptr || source_masks_device == nullptr ||
        target_masks_device == nullptr || output_device == nullptr ||
        status_device == nullptr || token_valid_device == nullptr || batch == 0 ||
        context_count == 0 || dimension < 2 ||
        words_per_row != dimension / 64u +
            static_cast<std::uint32_t>(dimension % 64u != 0u)) {
        return kInvalidArgument;
    }

    std::uint64_t required = 0;
    std::uint64_t cells = 0;
    if (!checked_mul(batch, context_count, &required) ||
        !checked_mul(dimension, dimension, &cells)) {
        return kSizeOverflow;
    }
    if (output_capacity_elements < required) return kInvalidArgument;
    if (required > 2147483647ull) return kSizeOverflow;

    constexpr std::uint32_t kThreads = 128;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    block_parity_weight_kernel<<<static_cast<unsigned int>(required), kThreads, 0, cuda_stream>>>(
        input_device, source_masks_device, target_masks_device, output_device,
        status_device, token_valid_device, cells, batch, context_count, max_patterns,
        dimension, words_per_row);
    return cudaGetLastError() == cudaSuccess ? kOk : kLaunchFailure;
}

extern "C" std::int32_t cnot103_cuda_launch_partial_shear_ranks(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    float* output_device,
    std::uint8_t* status_device,
    std::uint64_t output_capacity_elements,
    void* stream) {
    if (input_device == nullptr || output_device == nullptr || status_device == nullptr ||
        batch == 0 || dimension < 2 ||
        words_per_row != dimension / 64u + static_cast<std::uint32_t>(dimension % 64u != 0u)) {
        return kInvalidArgument;
    }
    if (output_capacity_elements < batch) return kInvalidArgument;
    const std::uint32_t source_width = dimension / 2u;
    const std::uint32_t target_count = dimension - source_width;
    const std::uint32_t target_words = source_width / 64u +
        static_cast<std::uint32_t>(source_width % 64u != 0u);
    std::uint64_t source_storage = 0;
    std::uint64_t target_storage = 0;
    std::uint64_t shared_words = 0;
    if (!checked_mul(source_width, words_per_row, &source_storage) ||
        !checked_mul(target_count, target_words, &target_storage) ||
        !checked_add(source_storage, target_storage, &shared_words) ||
        !checked_add(shared_words, source_width, &shared_words) ||
        shared_words > std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint64_t)) {
        return kSizeOverflow;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return cnot103::launch_packed_blocks(batch, shared_words, cuda_stream,
        [&](unsigned blocks, std::size_t shared_bytes, cnot103::PackedBlockScratch scratch) {
            partial_shear_rank_kernel<<<blocks, 256, shared_bytes, cuda_stream>>>(
                input_device, output_device, status_device, dimension, words_per_row, source_width,
                target_count, target_words, scratch);
        });
}

extern "C" std::int32_t cnot103_cuda_launch_projected_ranks(
    const std::uint64_t* input_device,
    const std::uint64_t* row_masks_device,
    const std::uint64_t* column_masks_device,
    const std::uint64_t* candidate_masks_device,
    const std::uint32_t* gate_controls_device,
    const std::uint32_t* gate_targets_device,
    std::uint32_t batch,
    std::uint32_t cut_count,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint32_t subtract_identity,
    std::uint32_t append_candidate,
    std::uint32_t apply_gate,
    float* output_device,
    std::uint64_t output_capacity_elements,
    void* stream) {
    if (input_device == nullptr || row_masks_device == nullptr || column_masks_device == nullptr ||
        output_device == nullptr || batch == 0 || cut_count == 0 || dimension < 2 ||
        words_per_row != dimension / 64u + static_cast<std::uint32_t>(dimension % 64u != 0u) ||
        subtract_identity > 1u || append_candidate > 1u || apply_gate > 1u ||
        (append_candidate != 0u && candidate_masks_device == nullptr) ||
        (apply_gate != 0u &&
         (gate_controls_device == nullptr || gate_targets_device == nullptr))) {
        return kInvalidArgument;
    }

    std::uint64_t work_items = 0;
    std::uint64_t shared_words = 0;
    if (!checked_mul(batch, cut_count, &work_items) ||
        work_items > std::numeric_limits<unsigned int>::max() ||
        !checked_mul(static_cast<std::uint64_t>(dimension) + append_candidate,
                     words_per_row, &shared_words) ||
        shared_words > std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint64_t)) {
        return kSizeOverflow;
    }
    if (output_capacity_elements < work_items) return kInvalidArgument;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return cnot103::launch_packed_blocks(work_items, shared_words, cuda_stream,
        [&](unsigned blocks, std::size_t shared_bytes, cnot103::PackedBlockScratch scratch) {
            projected_rank_kernel<<<blocks, 256, shared_bytes, cuda_stream>>>(
                input_device, row_masks_device, column_masks_device, candidate_masks_device,
                gate_controls_device, gate_targets_device, output_device, dimension, words_per_row,
                cut_count, subtract_identity, append_candidate, apply_gate, scratch);
        });
}

extern "C" std::int32_t cnot103_cuda_synchronize(void* stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return cudaStreamSynchronize(cuda_stream) == cudaSuccess ? kOk : kSyncFailure;
}
