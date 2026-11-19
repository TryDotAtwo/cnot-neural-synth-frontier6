#include "cnot103/cnot_cuda.h"
#include "one_excess_structure.cuh"

namespace {
using namespace cnot103_one_excess;
__global__ void structure_audit(Cnot103OneExcessStructureArgs a) {
    const U64 b=blockIdx.x;const StructureLayout l(a.dimension);
    HistoryView h{a.histories+b*3*l.n*l.w,a.lengths+b*l.n,a.protected_rows+b*l.n,l.n,l.w};
    StructureView s{l,a.output+b*l.meta_count,a.bitsets+b*l.bit_count,a.scratch+b*l.scratch_count};
    require_exact(prepare_structure(h,s));
    if(s.meta[2])return;
    for(U32 c=0;c<s.meta[3];++c)require_exact(order_component(h,s,c,a.max_entries));
}
bool extent(U64 batch,U64 stride,U64 available){return batch<=available/stride;}
}
extern "C" std::int32_t cnot103_cuda_launch_one_excess_structure(const Cnot103OneExcessStructureArgs* a) {
    if(!a||!a->histories||!a->lengths||!a->protected_rows||!a->output||!a->bitsets||!a->scratch||!a->batch||a->batch>0x7fffffffU||a->dimension<2||a->dimension>(1U<<24)||!a->max_entries)return 1;
    const StructureLayout l(a->dimension);const U64 b=a->batch;
    if(!extent(b,3*U64(l.n)*l.w,a->history_words)||!extent(b,l.n,a->length_elements)||!extent(b,l.n,a->protected_elements)||
        !extent(b,l.meta_count,a->output_elements)||!extent(b,l.bit_count,a->bitset_words)||!extent(b,l.scratch_count,a->scratch_words))return 2;
    structure_audit<<<a->batch,128,0,static_cast<cudaStream_t>(a->stream)>>>(*a);
    return cudaGetLastError()==cudaSuccess?0:3;
}
