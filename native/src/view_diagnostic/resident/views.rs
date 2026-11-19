//! Versioned probe schedule for the complete unit-cost row-XOR graph.
//! Inverting reverses a reduction word; transposing reverses its order and
//! swaps control and target. Simultaneous row/column permutation maps both wires.
use super::super::H07ProbeKind;
use crate::{CnotAlphabet, MatrixState};
use serde_json::{json, Value};
use tch::{Device, Kind, Tensor};

pub(super) const PROBES: [(&str, H07ProbeKind); 5] = [
    ("canonical", H07ProbeKind::Canonical),
    ("inverse", H07ProbeKind::Inverse),
    ("inverse_transpose", H07ProbeKind::InverseTranspose),
    ("transpose", H07ProbeKind::Transpose),
    ("wire_reversal", H07ProbeKind::Relabel),
];

pub(crate) fn problem_scope(root: &MatrixState, alphabet: &CnotAlphabet) -> Result<Value, String> {
    let n = root.dimension();
    if n < 2
        || root.alphabet_id() != alphabet.id()
        || *alphabet != CnotAlphabet::all_to_all_unit(n).map_err(|e| e.to_string())?
    {
        return Err("H07 physical probe schedule requires the exact ordered all-to-all unit alphabet of the root".into());
    }
    let actions: Vec<_> = alphabet
        .gates()
        .iter()
        .enumerate()
        .map(|(i, g)| {
            [
                g.control,
                g.target,
                alphabet.cost_at(i).expect("validated gate"),
            ]
        })
        .collect();
    Ok(
        json!({"schema":"cnot103_h07_complete_unit_views_v1","task":"left_row_xor_reduction_to_identity",
        "dimension":n,"root_packed_rows":root.packed_rows(),"alphabet_id":alphabet.id(),
        "ordered_control_target_cost":actions,"wire_reversal_old_to_new":(0..n).rev().collect::<Vec<_>>(),
        "probe_ids":PROBES.map(|(id,_)|id),"physical_feature_contexts":"same_trained_physical_providers_per_view"}),
    )
}

pub(super) fn packed_views(root: &MatrixState, device: Device) -> [Tensor; 5] {
    let n = i64::from(root.dimension());
    let words = root.words_per_row() as i64;
    let raw: Vec<_> = root.packed_rows().iter().map(|&v| v as i64).collect();
    let canonical = Tensor::from_slice(&raw)
        .view([1, n, words])
        .to_device(device);
    let dense = crate::device_state::unpack_matrix(&canonical, n);
    let inv = crate::device_state::gf2_inverse(&dense, 1, n, device).to_kind(Kind::Int64);
    let bit = Tensor::arange(64, (Kind::Int64, device));
    let pack = |view: Tensor| {
        view.constant_pad_nd([0, 64 * words - n])
            .reshape([1, n, words, 64])
            .bitwise_left_shift(&bit)
            .sum_dim_intlist([3i64].as_slice(), false, Kind::Int64)
            .contiguous()
    };
    [
        canonical,
        pack(inv.shallow_clone()),
        pack(inv.transpose(1, 2)),
        pack(dense.transpose(1, 2)),
        pack(dense.flip([1, 2])),
    ]
}
