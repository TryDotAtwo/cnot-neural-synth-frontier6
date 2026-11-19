#pragma once

#include <cstdint>

#if defined(_WIN32)
#  define CNOT103_CUDA_API __declspec(dllexport)
#else
#  define CNOT103_CUDA_API __attribute__((visibility("default")))
#endif

extern "C" {

CNOT103_CUDA_API std::int32_t cnot103_cuda_current_stream(
    std::int32_t device_index,
    void** stream);

CNOT103_CUDA_API std::int32_t cnot103_cuda_exchange_device(
    std::int32_t device_index,
    std::int32_t* previous);

CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_children(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    std::uint64_t* output_device,
    std::uint64_t output_capacity_words,
    void* stream);

CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_family_a(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    float* output_device,
    std::uint64_t output_capacity_elements,
    void* stream);

CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_residual_rank(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    float* output_device,
    std::uint64_t output_capacity_elements,
    void* stream);

CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_block_node_frequencies(
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
    void* stream);

CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_block_parity_weight(
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
    void* stream);

CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_partial_shear_ranks(
    const std::uint64_t* input_device,
    std::uint32_t batch,
    std::uint32_t dimension,
    std::uint32_t words_per_row,
    float* output_device,
    std::uint8_t* status_device,
    std::uint64_t output_capacity_elements,
    void* stream);

CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_projected_ranks(
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
    void* stream);

CNOT103_CUDA_API std::int32_t cnot103_cuda_synchronize(void* stream);

// Static arrays: ports are ordered inputs followed by outputs; equation_outputs
// gives the output-port index for each original or pinned-zero equation.
// Outputs: values [batch,equations,3], valid [batch,equations],
// incidence [batch,equations,dimension,3], ranks [batch,2].
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_boundary_relations(
    const std::uint64_t* states, const std::uint32_t* port_wires,
    const std::uint8_t* protected_ports, const std::uint32_t* equation_outputs,
    std::uint32_t batch, std::uint32_t dimension, std::uint32_t words_per_row,
    std::uint32_t inputs, std::uint32_t outputs, std::uint32_t equations,
    float* values, std::uint8_t* valid, float* incidence, float* ranks,
    std::uint64_t group_capacity, std::uint64_t incidence_capacity, void* stream);

struct Cnot103BlockHistogramArgs {
    const std::uint64_t* states;
    const std::uint64_t* source_masks;
    const std::uint64_t* target_masks;
    std::uint32_t* frequencies; // [batch,contexts,n+1], zero pattern then row representatives
    std::uint64_t* statistics; // [batch,contexts,4]: patterns,total_nonzero,max_frequency,pairs
    std::uint32_t* ordered_ids; // physical row+1, zero=0, padding=UINT32_MAX
    float* pattern_values; // [batch,contexts,P,3]: D01,D02,D04
    std::uint8_t* pattern_valid;
    float* pattern_incidence; // [batch,contexts,P,n,3]
    float* pair_values; // [batch,contexts,Q,1]: D06
    std::uint8_t* pair_valid;
    float* pair_incidence; // [batch,contexts,Q,n,3]
    void* stream;
    std::uint64_t pair_capacity;
    std::uint64_t frequency_elements;
    std::uint64_t statistic_elements;
    std::uint64_t pattern_groups;
    std::uint64_t pattern_incidence_elements;
    std::uint64_t pair_groups;
    std::uint64_t pair_incidence_elements;
    std::uint32_t batch;
    std::uint32_t contexts;
    std::uint32_t dimension;
    std::uint32_t words_per_row;
    std::uint32_t pattern_capacity;
    std::uint32_t emit_patterns;
};

CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_block_histograms(
    const Cnot103BlockHistogramArgs* args);

// Exact deterministic partial-shear coefficients; status 0 empty,1 computed,2 dependent.
// Scratch per state: w*words_per_row + w*ceil(w/64) + w u64, where w=floor(n/2).
// Coefficients [batch,n-w,w] u8; optional integer gradients [batch,w,w] i64.
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_basis_gradient(
    const std::uint64_t* states, std::uint64_t* scratch, std::uint8_t* coefficients,
    std::uint8_t* status, std::int64_t* gradients, std::uint32_t batch,
    std::uint32_t dimension, std::uint32_t words_per_row,
    std::uint64_t scratch_words, std::uint64_t coefficient_elements,
    std::uint64_t gradient_elements, std::uint32_t emit_gradients, void* stream);

struct Cnot103GeneralCompilerArgs {
    const std::uint64_t* states;
    std::uint64_t* scratch;
    std::uint32_t* gates; // [batch,word_capacity,2], synthesis order, zero padding
    std::int64_t* lengths;
    std::uint8_t* status; // DeviceFeatureStatus ABI: 1 computed,3 unsupported,4 limit
    std::uint8_t* verified;
    std::int64_t* lower_bounds;
    float* f09; // [batch,4]
    float* f10; // [batch,5]
    void* stream;
    std::uint64_t word_capacity;
    std::uint64_t workspace_words;
    std::uint64_t gate_elements;
    std::uint32_t batch;
    std::uint32_t dimension;
    std::uint32_t words_per_row;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_general_compiler(const Cnot103GeneralCompilerArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_verify_cnot_words(
    const std::uint64_t* states, const std::uint32_t* gates, const std::int64_t* lengths,
    std::uint64_t* scratch, std::uint8_t* verified, std::uint32_t batch,
    std::uint32_t dimension, std::uint32_t words_per_row, std::uint64_t word_capacity,
    std::uint64_t workspace_words, std::uint64_t gate_elements, void* stream);

// Stage26 certificate: six counts, n witnesses (UINT64_MAX absent), retained
// and deleted masks (2W), source/projected/H/targets (4nW), W=ceil(n/64).
// H is indexed by high pivot. Targets are ascending, with zero padding.
// Scratch per state: (7n+4)W+n+1 words; see stage26.cu for the layout.
struct Cnot103Stage26Args {
    const std::uint64_t* states;
    std::uint64_t* certificate;
    std::uint64_t* scratch;
    std::uint8_t* verified;
    std::uint8_t* status;
    float* f03; // [batch,5]
    float* f04; // [batch,n,n,3]
    float* f05; // [batch,2]
    void* stream;
    std::uint64_t certificate_words;
    std::uint64_t workspace_words;
    std::uint64_t pair_elements;
    std::uint32_t batch;
    std::uint32_t dimension;
    std::uint32_t words_per_row;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_stage26(const Cnot103Stage26Args* args);
// Separate phases also allow independent corruption tests. Publish traps on a
// rejected certificate, even in Release builds; a proof failure is never masked.
CNOT103_CUDA_API std::int32_t cnot103_cuda_stage26_construct(const Cnot103Stage26Args* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_stage26_check(const Cnot103Stage26Args* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_stage26_publish(const Cnot103Stage26Args* args);

// I07 setup: fixed central-frequency rank and exponent on CUDA.
// statistics has two u64s. Scratch has rows*ceil(columns/64) words.
CNOT103_CUDA_API std::int32_t cnot103_cuda_prepare_frequency(
    const std::uint64_t* central, std::uint64_t* scratch, std::uint64_t* statistics,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t internal_count,
    std::uint64_t scratch_words, void* stream);
struct Cnot103FrequencyArgs {
    const std::uint64_t* states;
    const std::uint8_t* roles; // [n]: source0, internal1, target2
    const std::uint32_t* internal_wires; // caller order, [internal_count]
    const std::uint64_t* statistics; // [kappa, internal_count*kappa], CUDA
    std::uint64_t* scratch; // [batch,internal_count,ceil(internal_count/64)]
    std::uint8_t* status;
    float* values; // [batch,2]
    void* stream;
    std::uint64_t scratch_words;
    std::uint32_t batch;
    std::uint32_t dimension;
    std::uint32_t words_per_row;
    std::uint32_t internal_count;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_frequency(const Cnot103FrequencyArgs* args);

struct Cnot103SharedRelationArgs {
    const std::uint64_t* states;
    const std::uint32_t* row_wires;
    const std::uint32_t* coordinate_wires;
    const std::uint64_t* covectors; // [ports,words_per_row], physical masks
    std::uint64_t* scratch; // per state: (coordinates+vectors)*ceil(vectors/64)+vectors
    float* values; // [batch,vectors+ports,max_vectors+2]
    std::uint8_t* valid; // [batch,vectors+ports]
    float* incidence; // [batch,vectors+ports,n,2]
    std::uint8_t* status;
    void* stream;
    std::uint64_t scratch_words;
    std::uint64_t value_elements;
    std::uint64_t group_elements;
    std::uint64_t incidence_elements;
    std::uint32_t batch;
    std::uint32_t dimension;
    std::uint32_t words_per_row;
    std::uint32_t vectors;
    std::uint32_t coordinates;
    std::uint32_t ports;
    std::uint32_t max_vectors;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_shared_relation(const Cnot103SharedRelationArgs* args);

// I05: prepared template [[UV,A0],[D0,0]], with four device statistics
// [rank(UV),rank([UV,A0]),rank([UV;D0]),t+rank(UV)-p-q].
struct Cnot103BilinearPrepareArgs {
    const std::uint64_t* u; // [p,ceil(t/64)]
    const std::uint64_t* v; // [t,ceil(q/64)]
    const std::uint64_t* a0; // [p,ceil(r/64)]
    const std::uint64_t* d0; // [m,ceil(q/64)]
    std::uint64_t* matrix_template; // [p+m,ceil((q+r)/64)]
    std::uint64_t* scratch; // same shape as matrix_template
    std::uint64_t* statistics; // [4]
    void* stream;
    std::uint64_t template_words;
    std::uint64_t scratch_words;
    std::uint32_t m, r, p, t, q;
};
struct Cnot103BilinearArgs {
    const std::uint64_t* states;
    const std::uint32_t* row_wires; // [m]
    const std::uint32_t* column_wires; // [r]
    const std::uint64_t* matrix_template;
    const std::uint64_t* statistics;
    std::uint64_t* scratch; // [batch,p+m,ceil((q+r)/64)]
    float* values; // [batch,2]
    std::uint8_t* status; // [batch]
    void* stream;
    std::uint64_t template_words;
    std::uint64_t scratch_words;
    std::uint32_t batch, dimension, words_per_row;
    std::uint32_t m, r, p, q;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_prepare_bilinear(const Cnot103BilinearPrepareArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_bilinear(const Cnot103BilinearArgs* args);

// I08 values are independent of the state RHS. Prepare one resident template;
// callers broadcast it to the batch only after source alphabet/budget guards.
struct Cnot103PositivePacketArgs {
    const std::uint32_t* gates; // [slots,2], control then target
    const std::uint64_t* factor_offsets; // [factors+1], validated CSR
    const std::uint32_t* factor_scopes; // [scope_elements]
    std::uint64_t* counts; // [dimension], monomial endpoint counts
    std::uint32_t* indices; // [factors+3*slots], allocate at least one word
    float* values; // [slots+1,4]
    std::uint8_t* valid; // [slots+1]
    float* incidence; // [slots+1,dimension,2]
    std::uint8_t* status; // [1]
    void* stream;
    std::uint64_t max_monomials;
    std::uint64_t scope_elements;
    std::uint64_t count_words;
    std::uint64_t index_elements;
    std::uint64_t value_elements;
    std::uint64_t group_elements;
    std::uint64_t incidence_elements;
    std::uint32_t dimension, slots, factors;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_prepare_positive_packet(const Cnot103PositivePacketArgs* args);

// I09 exact Stage60 compiler. Context arrays are immutable and validated:
// sorted unique directed allowed gates in range; cut binary prefix bits.
// Terms [B,G,cap,3] contain (kind,left,right), kind=0 constant,1 linear,2 product;
// sorted canonical terms occupy the first term_counts[B,G] slots, rest zero.
// cap=(horizon ? dimension+2 : 1); G is the exact equation count.
// Scratch [B,future*ceil(G/64)+cut*ceil(G*future/64)], at least one word.
// All buffers are disjoint. Zero-length context pointers may be null. Output
// group_elements covers both counts and valid; status has at least B bytes.
struct Cnot103QuadraticArgs {
    const std::uint64_t* states;
    const std::uint32_t* allowed_gates; // [allowed_count,2]
    const std::uint8_t* prefix; // [cut]
    std::uint32_t* terms;
    std::uint32_t* term_counts;
    std::uint64_t* scratch;
    float* values; // [B,G,max_variables+4]
    std::uint8_t* valid; // [B,G]
    float* incidence; // [B,G,dimension,1]
    std::uint32_t* ranks; // [B,2] = rank_B,mixed_rank
    std::uint8_t* status; // [B], DeviceFeatureStatus
    void* stream;
    std::uint64_t state_words, term_elements, group_elements, scratch_words;
    std::uint64_t value_elements, incidence_elements, rank_elements;
    std::uint32_t batch, dimension, words_per_row, horizon, cut, allowed_count, equations, max_variables;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_quadratic(const Cnot103QuadraticArgs* args);

// I10 target-independent exact finite-horizon word DP. Storage words:
// 4*S*K + 10*S + 2*H + HH + 128*K + 387, K=n*words_per_row.
// Two layers (keys/counts/hash), history(keys/counts/first degree/hash),
// 128 transition staging records, three statistics. H/HH are powers of two
// with H>=2*S+1, HH>=4*S+1. Counts have low/high uint64 limbs.
// S covers the source state bound or a proved transition/universe bound.
// status is 1 exact through max_degree or 4 resource; failed statistics zero.
struct Cnot103EndpointPrepareArgs {
    std::uint64_t* storage;
    std::uint8_t* status;
    void* stream;
    std::uint64_t storage_words, capacity, buckets, history_buckets;
    std::uint64_t max_degree, max_layer_states, max_transitions;
    std::uint32_t dimension, words_per_row;
};
// Read the prepared tables only on the same or a correctly synchronized stream.
// Resource failure zeros all payloads. An exact horizon miss has oracle_status1
// and feature_status3; evaluations remains the exact completed transition count.
struct Cnot103EndpointArgs {
    const std::uint64_t* states;
    const std::uint64_t* storage;
    const std::uint8_t* prepared_status;
    float* values; // [B,4]
    std::uint8_t* feature_status; // [B]
    std::uint8_t* oracle_status; // [B]
    std::uint64_t* leading_degrees; // [B]
    std::uint64_t* leading_counts; // [B,2] low/high
    std::uint64_t* terminal_counts; // [B,2] coefficient at max_degree
    std::uint64_t* evaluations; // [B], zero on failed preparation
    void* stream;
    std::uint64_t state_words, storage_words, value_elements, count_elements, batch_elements;
    std::uint64_t capacity, buckets, history_buckets, max_degree;
    std::uint32_t batch, dimension, words_per_row;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_prepare_endpoint_response(const Cnot103EndpointPrepareArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_endpoint_response(const Cnot103EndpointArgs* args);

// Complete GL(w,2) BFS atlas for w=1..4; side0=Source,1=Target. Storage has
// 2^(w*w)*(6+2^w)+1 uint32 words: maps,parents,controls,targets,distances,seen,
// populations,then count. Maps store all w*w row-major bits. Queue/gate order
// matches the source BFS exactly. Prepare/consume on the same ordered stream.
struct Cnot103PaidAtlasPrepareArgs {
    std::uint32_t* storage;
    void* stream;
    std::uint64_t storage_words;
    std::uint32_t width, side;
};
// storage must be the immutable prepared atlas for exactly width/side.
// Inputs [B,rows,width] are binary bytes. Outputs are disjoint, fully written;
// invalid bits yield status3 and zero payloads, valid inputs status1.
// Word capacity >= max(1,(width+3)*(width-1)); all unused gate slots zero.
struct Cnot103PaidAtlasArgs {
    const std::uint32_t* storage;
    const std::uint8_t* vectors;
    std::uint8_t* transformed;
    std::int32_t* operations;
    std::uint64_t* lengths;
    std::uint64_t* direct_cost;
    std::uint64_t* paid_cost;
    std::uint8_t* status;
    void* stream;
    std::uint64_t storage_words, vector_elements, gate_elements, batch_elements, word_capacity, rows;
    std::uint32_t batch, width, side;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_prepare_paid_atlas(const Cnot103PaidAtlasPrepareArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_optimize_paid_atlas(const Cnot103PaidAtlasArgs* args);

// Exact source-order paid-shear compiler for one physical context. Source and
// target arrays are immutable, sorted, nonempty, disjoint and in [0,n).
// Atlases[side*3+width-2] match side0=Source,1=Target and every width2..local_width.
// C=S*T, P=max(1,C/2), L=max(S,T). All outputs are disjoint, batch-major:
// block [B,T,S], bases [B,S,S]/[B,T,T], preparation/restoration/cancellations
// [B,P,2], core/word [B,C,2], moves [B,P,11], statistics [B,16].
// Move fields: sweep,side,width,indices[4],cost_before,cost_after,prep_start,end.
// Statistics: direct,paid,prep,source_prep,target_prep,core,restoration,tested,
// moves,source_moves,target_moves,source_changed,target_changed,swaps,cancels,
// unshortened. Cancellation events retain traversal order, one gate per pair.
// Endpoint/replay [B,n,ceil(n/64)]. Scratch [B,15*L+P+C] uint64 words.
// Payload publication follows independent full-word replay; status1=verified,
// status3=invalid certificate with zero payloads. Internal invariants trap.
struct Cnot103PaidShearArgs {
    const std::uint64_t* states;
    const std::uint32_t* sources;
    const std::uint32_t* targets;
    const std::uint32_t* atlases[6];
    std::uint8_t* block;
    std::uint8_t* source_basis;
    std::uint8_t* target_basis;
    std::uint32_t* preparation;
    std::uint32_t* restoration;
    std::uint32_t* core;
    std::uint32_t* word;
    std::uint32_t* cancellations;
    std::uint64_t* moves;
    std::uint64_t* statistics;
    std::int64_t* lengths;
    std::uint64_t* endpoint;
    std::uint64_t* replay;
    std::uint64_t* scratch;
    std::uint8_t* status;
    std::uint8_t* verified;
    void* stream;
    std::uint64_t state_words, block_elements, source_basis_elements, target_basis_elements;
    std::uint64_t preparation_elements, core_elements, move_elements, stat_elements, scratch_words;
    std::uint64_t batch_elements, source_elements, target_elements, atlas_words[6];
    std::uint32_t batch, dimension, words_per_row, source_count, target_count, local_width, sweeps;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_paid_shear(const Cnot103PaidShearArgs* args);

// Ordered D09-D14 aggregation. Q=sum(source_count*target_count) over the
// matching contexts, G=number of contexts. Init -> append(context0..G-1) ->
// finish, all on one stream, with disjoint buffers retained through finish.
// word[B,Q,2], labels[B,Q], endpoint/replay[B,n,W], counts[B,n,n,3],
// groups[B,G,18], incidence[B,G,n,2], valid[B,G], pairs[B,n,n,4].
// Counts channels: reduced gates, original adjacent compatibility, canceled
// cross-context pairs. Labels retain original context indices through pops.
// Status1=verified,3=unsupported,4=inexact f32 count; failure zeros payloads.
struct Cnot103PaidBasisArgs {
    std::uint32_t* word;
    std::uint64_t* labels;
    std::int64_t* lengths;
    std::uint64_t* endpoint;
    std::uint64_t* replay;
    std::uint64_t* counts;
    float* groups;
    float* incidence;
    std::uint8_t* valid;
    float* pairs;
    std::uint8_t* status;
    std::uint8_t* verified;
    std::uint32_t* overflow;
    void* stream;
    std::uint64_t capacity, state_words, gate_elements, label_words, count_elements;
    std::uint64_t group_elements, incidence_elements, token_elements, pair_elements, batch_elements;
    std::uint32_t batch, dimension, words_per_row, context_count;
};
// Sources/targets are sorted, unique, disjoint in-range static metadata.
// Current/previous words are original individual candidates, never the
// combined stack; empty candidates remain as actual neighbors.
struct Cnot103PaidBasisAppendArgs {
    Cnot103PaidBasisArgs output;
    const std::uint64_t* states;
    const std::uint32_t* sources;
    const std::uint32_t* targets;
    const std::uint32_t* word;
    const std::int64_t* lengths;
    const std::uint64_t* statistics;
    const std::uint8_t* status;
    const std::uint32_t* previous_word;
    const std::int64_t* previous_lengths;
    std::uint64_t capacity, gate_elements, stat_elements, source_elements, target_elements;
    std::uint64_t previous_capacity, previous_gate_elements;
    std::uint32_t source_count, target_count, context_index;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_paid_basis_init(const Cnot103PaidBasisArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_paid_basis_append(const Cnot103PaidBasisAppendArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_paid_basis_finish(const Cnot103PaidBasisArgs* args);

// Exact finite-template responses. w=floor(n/2), k=choices, P=points,
// A=2^k, G=choose(k,2). Coefficients[B,n-w,w] and raw_status[B] come from
// basis_gradient emit_gradients=false:0=empty,1=nonempty,2=dependent sources.
// choice_data[k,3]=(kind0Add/1Swap,control_position,target_position),
// wires[unique<=16] maps positions to source-block coordinates; immutable.
// probabilities[P,k,2] are normalized u64 numerator/denominator metadata.
// Output costs[B,A] and rounding[B,P+2] (midpoint forward/reverse,probes),
// globals[B,P+9] (E02,E03[P],E07,E08[2],E09[5]), mixed[B,G,1],
// valid[B,G], incidence[B,G,n,4], status[B]. All allocations disjoint/live
// on stream; failures2/3/4 clear all output payloads. Status1=computed.
struct Cnot103TemplateResponseArgs {
    const std::uint8_t* coefficients;
    const std::uint8_t* raw_status;
    const std::uint32_t* choice_data;
    const std::uint32_t* wires;
    const std::uint64_t* probabilities;
    std::uint64_t* costs;
    std::uint64_t* rounding;
    float* globals;
    float* mixed;
    std::uint8_t* valid;
    float* incidence;
    std::uint8_t* status;
    void* stream;
    std::uint64_t coefficient_elements, choice_elements, wire_elements, probability_elements;
    std::uint64_t cost_elements, rounding_elements, global_elements, mixed_elements, token_elements, incidence_elements, batch_elements;
    std::uint32_t batch, dimension, choices, points, unique_wires, invalid_choices;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_template_response(const Cnot103TemplateResponseArgs* args);

// Stage35 exact pair certificate. Resident states/scratch[B,n,ceil(n/64)],
// graph[B,2n], encoder gates[B,Q,2] (Q>=n+1), lengths[B], certificate[B,16].
// Certificate slots: changed_count,active0,active1,frame0,frame1,nonzero,shared,
// motion,connectivity,zero_diagonal,lower_bound,lower_kind,shared0,shared1,
// changed0,changed1. Absent row slots=n. status1=computed,3=outside scope.
// f02[B,1],f06[B,1,6],valid[B,1],incidence[B,1,n,3]. Unavailable payloads=0.
// All buffers disjoint/live on stream. Host has checked static width/alphabet
// policy. Publish traps on a rejected proof, including in release builds.
struct Cnot103GuardedPairArgs {
    const std::uint64_t* states;
    std::uint64_t* scratch;
    std::uint64_t* graph;
    std::uint32_t* gates;
    std::int64_t* lengths;
    std::uint64_t* certificate;
    std::uint8_t* status;
    std::uint8_t* verified;
    float* f02;
    float* f06;
    std::uint8_t* valid;
    float* incidence;
    void* stream;
    std::uint64_t word_capacity,state_words,scratch_words,graph_words,gate_elements,certificate_words,f06_elements,incidence_elements,batch_elements;
    std::uint32_t batch,dimension,words_per_row;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_guarded_pair_construct(const Cnot103GuardedPairArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_guarded_pair_check(const Cnot103GuardedPairArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_guarded_pair_publish(const Cnot103GuardedPairArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_guarded_pair(const Cnot103GuardedPairArgs* args);

// Stage37 exact/negative witness. states/scratch[B,n,ceil(n/64)]u64;
// graph[B,3n]u32; parents[B,n,2]u32=(control,mode0Initial/1Final/2Absent);
// changed/missing/order[B,n]u32 padded with n, encoder[B,n,2]u32 padded0.
// headers[B,3]u64=(class1Exact/2NoZeroExcess,changed_count,missing_count);
// lengths[B]i64 counts encoder gates (0 for negative); verified/status[B]u8;
// feature[B,3]f32: exact[1,0,0], negative[0,1,0]. Both have feature status1.
// Host enforces immutable width/alphabet/solver policy. Disjoint allocations
// stay live on stream; publication traps on any rejected complete witness.
struct Cnot103ZeroExcessArgs {
    const std::uint64_t* states;
    std::uint64_t* scratch;
    std::uint32_t* graph;
    std::uint32_t* parents;
    std::uint32_t* changed_rows;
    std::uint32_t* missing_rows;
    std::uint32_t* update_order;
    std::uint32_t* gates;
    std::uint64_t* headers;
    std::int64_t* lengths;
    std::uint8_t* verified;
    std::uint8_t* status;
    float* feature;
    void* stream;
    std::uint64_t state_words,scratch_words,graph_elements,parent_elements,changed_elements,missing_elements,order_elements,gate_elements,header_elements,feature_elements,batch_elements;
    std::uint32_t batch,dimension,words_per_row;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_zero_excess_construct(const Cnot103ZeroExcessArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_zero_excess_check(const Cnot103ZeroExcessArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_zero_excess_publish(const Cnot103ZeroExcessArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_zero_excess(const Cnot103ZeroExcessArgs* args);

// Stage38 structure dependency, not the F08 classifier. Histories are packed
// [B,3,ceil(n/64),n], inactive slots zero; lengths[B,n] in1..3 with <=one3;
// protected_rows[B,n] boolean includes retained recovery rows. Disjoint buffers
// use StructureLayout(n) from one_excess_structure.cuh. Header: event_count,
// factor_count, zero_reason(0none/1empty/2mandatorycycle),component_count,
// double_row(n=none),protected_event_count,valid1,reserved0. Full ordered terms,
// scopes, graphs, components, orders/bags and per-component preflights follow.
// This audit orders all components but does not combine their budget flags.
// Full contraction must call the device primitives lazily in component order.
// All allocations remain live on stream. Invalid histories/invariants trap.
struct Cnot103OneExcessStructureArgs {
    const std::uint64_t* histories;
    const std::uint32_t* lengths;
    const std::uint8_t* protected_rows;
    std::uint32_t* output;
    std::uint64_t* bitsets;
    std::uint64_t* scratch;
    void* stream;
    std::uint64_t history_words,length_elements,protected_elements,output_elements,bitset_words,scratch_words,max_entries;
    std::uint32_t batch,dimension;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_one_excess_structure(const Cnot103OneExcessStructureArgs* args);

// Exact one-excess history count, not template enumeration/F08 admission.
// Histories match the structure ABI. Flat CountLayout(n,lanes,cache_slots)
// workspaces belong to workers that stride through batch rows ondevice.
// counts[B,limbs]u32 is canonical nonnegative little endian. results[B,4]u64:
// status1Computed/4SourceResourceLimit, source max_bag(0 on resource), processed
// components, saturating cache evictions. Last two fields are diagnostics, not
// FeaturePolicy evaluation accounting. Cache0/1 are valid exact configurations.
// Disjoint allocations must stay live on stream; integrity failures trap.
struct Cnot103OneExcessCountArgs {
    const std::uint64_t* histories;
    const std::uint32_t* lengths;
    const std::uint8_t* protected_rows;
    std::uint32_t* structure_meta;
    std::uint64_t* structure_bits;
    std::uint64_t* structure_scratch;
    std::uint64_t* program;
    std::uint32_t* arithmetic;
    std::uint32_t* counts;
    std::uint64_t* results;
    void* stream;
    std::uint64_t history_words,length_elements,protected_elements,meta_elements,bit_words,scratch_words,program_words,arithmetic_elements,count_elements,result_words,max_entries;
    std::uint32_t batch,dimension,workers,lanes,cache_slots;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_one_excess_count(const Cnot103OneExcessCountArgs* args);

// Complete Stage38 recognition. Prerequisite Stage37 headers/status are resident
// and checked against independently derived changed/missing-parent counts.
// headers[B,6]: class1Exact/2NoOneExcess/3Stage37Exact/4Resource, changed,
// templates, max_bag, transport_free(exact only), distance/excluded_length.
// counts[B,limbs] is the exact total; gates[B,n+1,2] encode I->M, unused=0.
// Full recomputation and independent word replay precede feature publication.
// Shape-only workspaces use ClassifierLayout; all allocations are disjoint/live
// on stream. Metadata errors return1/2, integrity failures trap in Release.
struct Cnot103OneExcessArgs {
    const std::uint64_t* states;
    const std::uint64_t* zero_headers;
    const std::uint8_t* zero_status;
    std::uint64_t* histories;
    std::uint32_t* history_lengths;
    std::uint8_t* protected_rows;
    std::uint32_t* structure_meta;
    std::uint64_t* structure_bits;
    std::uint64_t* structure_scratch;
    std::uint64_t* program;
    std::uint32_t* arithmetic;
    std::uint64_t* classifier;
    std::uint32_t* workspace_gates;
    std::uint64_t* headers;
    std::uint32_t* counts;
    std::uint32_t* gates;
    std::int64_t* lengths;
    std::uint8_t* verified;
    std::uint8_t* status;
    float* feature;
    std::uint64_t* word_scratch;
    void* stream;
    std::uint64_t state_words,zero_header_words,zero_status_elements,history_words,length_elements,protected_elements,meta_elements,bit_words,scratch_words,program_words,arithmetic_elements,classifier_words,workspace_gate_elements,header_elements,count_elements,gate_elements,batch_elements,feature_elements,word_scratch_words,max_entries;
    std::uint32_t batch,dimension,words_per_row,workers,lanes,cache_slots,max_changed_rows,max_component_events,max_history_counts;
};
CNOT103_CUDA_API std::int32_t cnot103_cuda_one_excess_construct(const Cnot103OneExcessArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_one_excess_check(const Cnot103OneExcessArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_one_excess_publish(const Cnot103OneExcessArgs* args);
CNOT103_CUDA_API std::int32_t cnot103_cuda_launch_one_excess(const Cnot103OneExcessArgs* args);

}
