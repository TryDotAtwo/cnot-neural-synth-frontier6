use crate::alphabet::CnotAlphabet;
use crate::block_parities::BlockContext;
use crate::matrix_boundary::MatrixBoundarySpec;
use crate::rank_features::{LabeledCut, RankIncrementProbe, RestrictedIntersection};
use crate::state::CnotGate;
use tch::{Device, Tensor};

fn upload_masks(sets: &[Vec<u32>], dimension: u32, device_index: usize) -> Result<Tensor, String> {
    if dimension < 2 || sets.is_empty() {
        return Err("device context needs a positive width and at least one mask".into());
    }
    let words = usize::try_from(dimension.div_ceil(64))
        .map_err(|_| "device context word count overflow".to_owned())?;
    let count = sets
        .len()
        .checked_mul(words)
        .ok_or_else(|| "device context mask extent overflow".to_owned())?;
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(count)
        .map_err(|_| "device context mask allocation failed".to_owned())?;
    for set in sets {
        let mut row = vec![0u64; words];
        for &index in set {
            if index >= dimension {
                return Err("device context wire is outside its width".into());
            }
            let word = usize::try_from(index / 64)
                .map_err(|_| "device context wire conversion failed".to_owned())?;
            row[word] |= 1u64 << (index % 64);
        }
        packed.extend(row.into_iter().map(|word| word as i64));
    }
    let rows = i64::try_from(sets.len())
        .map_err(|_| "device context count exceeds tensor range".to_owned())?;
    let columns = i64::try_from(words)
        .map_err(|_| "device context word count exceeds tensor range".to_owned())?;
    Ok(Tensor::from_slice(&packed)
        .view([rows, columns])
        .to_device(Device::Cuda(device_index)))
}

fn upload_incidence(
    contexts: &[Vec<Vec<u32>>],
    dimension: u32,
    device_index: usize,
) -> Result<Tensor, String> {
    if dimension == 0 || contexts.is_empty() {
        return Err("device incidence needs at least one context".into());
    }
    let roles = contexts[0].len();
    if roles == 0 || contexts.iter().any(|context| context.len() != roles) {
        return Err("device incidence roles are inconsistent".into());
    }
    let dimension = usize::try_from(dimension)
        .map_err(|_| "device incidence width overflow".to_owned())?;
    let extent = contexts
        .len()
        .checked_mul(dimension)
        .and_then(|value| value.checked_mul(roles))
        .ok_or_else(|| "device incidence extent overflow".to_owned())?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(extent)
        .map_err(|_| "device incidence allocation failed".to_owned())?;
    for context in contexts {
        for wire in 0..dimension {
            for indices in context {
                values.push(f32::from(indices.binary_search(&(wire as u32)).is_ok()));
            }
        }
    }
    let context_count = i64::try_from(contexts.len())
        .map_err(|_| "device incidence context count overflow".to_owned())?;
    let dimension = i64::try_from(dimension)
        .map_err(|_| "device incidence width exceeds tensor range".to_owned())?;
    let roles = i64::try_from(roles)
        .map_err(|_| "device incidence role count overflow".to_owned())?;
    Ok(Tensor::from_slice(&values)
        .view([context_count, dimension, roles])
        .to_device(Device::Cuda(device_index)))
}

fn upload_gate_indices(values: &[u32], device_index: usize) -> Result<Tensor, String> {
    let indices = values
        .iter()
        .map(|&value| i32::try_from(value).map_err(|_| "gate index exceeds int32".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor::from_slice(&indices).to_device(Device::Cuda(device_index)))
}

fn union(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(left.len().saturating_add(right.len()));
    result.extend_from_slice(left);
    result.extend_from_slice(right);
    result.sort_unstable();
    result.dedup();
    result
}

pub(crate) struct DeviceLabeledCutContext {
    pub(crate) dimension: u32,
    pub(crate) device_index: usize,
    pub(crate) cut_count: i64,
    pub(crate) side_a: Tensor,
    pub(crate) side_b: Tensor,
    pub(crate) incidence: Tensor,
}

impl DeviceLabeledCutContext {
    pub(crate) fn new(
        dimension: u32,
        cuts: &[LabeledCut],
        device_index: usize,
    ) -> Result<Self, String> {
        let relevant = cuts
            .iter()
            .filter(|cut| cut.dimension() == dimension)
            .collect::<Vec<_>>();
        if relevant.is_empty() {
            return Err("labeled-cut device context is empty".into());
        }
        let side_a = relevant.iter().map(|cut| cut.side_a().to_vec()).collect::<Vec<_>>();
        let side_b = relevant.iter().map(|cut| cut.side_b()).collect::<Vec<_>>();
        let incidence = relevant
            .iter()
            .map(|cut| vec![cut.side_a().to_vec(), cut.side_b()])
            .collect::<Vec<_>>();
        Ok(Self {
            dimension,
            device_index,
            cut_count: i64::try_from(relevant.len())
                .map_err(|_| "labeled-cut count exceeds tensor range".to_owned())?,
            side_a: upload_masks(&side_a, dimension, device_index)?,
            side_b: upload_masks(&side_b, dimension, device_index)?,
            incidence: upload_incidence(&incidence, dimension, device_index)?,
        })
    }
}

pub(crate) struct DeviceRestrictedIntersectionContext {
    pub(crate) dimension: u32,
    pub(crate) device_index: usize,
    pub(crate) count: i64,
    pub(crate) row_oriented: Tensor,
    pub(crate) row_columns: Tensor,
    pub(crate) transposed_rows: Tensor,
    pub(crate) transposed_columns: Tensor,
    pub(crate) incidence: Tensor,
}

impl DeviceRestrictedIntersectionContext {
    pub(crate) fn new(
        dimension: u32,
        probes: &[RestrictedIntersection],
        device_index: usize,
    ) -> Result<Self, String> {
        let relevant = probes
            .iter()
            .filter(|probe| probe.dimension() == dimension)
            .collect::<Vec<_>>();
        if relevant.is_empty() {
            return Err("restricted-intersection device context is empty".into());
        }
        let mut row_oriented = Vec::with_capacity(relevant.len() * 3);
        let mut row_columns = Vec::with_capacity(relevant.len() * 3);
        let mut transposed_rows = Vec::with_capacity(relevant.len() * 3);
        let mut transposed_columns = Vec::with_capacity(relevant.len() * 3);
        let mut incidence = Vec::with_capacity(relevant.len());
        for probe in &relevant {
            let left = probe.left().to_vec();
            let right = probe.right().to_vec();
            let joint = union(&left, &right);
            let coordinates = probe.coordinates().to_vec();
            row_oriented.extend([left.clone(), right.clone(), joint.clone()]);
            row_columns.extend([coordinates.clone(), coordinates.clone(), coordinates.clone()]);
            transposed_rows.extend([coordinates.clone(), coordinates.clone(), coordinates.clone()]);
            transposed_columns.extend([left.clone(), right.clone(), joint]);
            incidence.push(vec![left, right, coordinates]);
        }
        Ok(Self {
            dimension,
            device_index,
            count: i64::try_from(relevant.len())
                .map_err(|_| "restricted-intersection count overflow".to_owned())?,
            row_oriented: upload_masks(&row_oriented, dimension, device_index)?,
            row_columns: upload_masks(&row_columns, dimension, device_index)?,
            transposed_rows: upload_masks(&transposed_rows, dimension, device_index)?,
            transposed_columns: upload_masks(&transposed_columns, dimension, device_index)?,
            incidence: upload_incidence(&incidence, dimension, device_index)?,
        })
    }
}

pub(crate) struct DeviceRankIncrementContext {
    pub(crate) dimension: u32,
    pub(crate) device_index: usize,
    pub(crate) probe_count: i64,
    pub(crate) row_masks: Tensor,
    pub(crate) column_masks: Tensor,
    pub(crate) candidate_masks: Tensor,
    pub(crate) incidence: Tensor,
}

impl DeviceRankIncrementContext {
    pub(crate) fn new(
        dimension: u32,
        probes: &[RankIncrementProbe],
        device_index: usize,
    ) -> Result<Self, String> {
        let relevant = probes
            .iter()
            .filter(|probe| probe.dimension() == dimension)
            .collect::<Vec<_>>();
        if relevant.is_empty() {
            return Err("rank-increment device context is empty".into());
        }
        let rows = relevant.iter().map(|probe| probe.span_rows().to_vec()).collect::<Vec<_>>();
        let columns = relevant.iter().map(|probe| probe.coordinates().to_vec()).collect::<Vec<_>>();
        let candidates = relevant
            .iter()
            .map(|probe| probe.candidate_support().to_vec())
            .collect::<Vec<_>>();
        let incidence = relevant
            .iter()
            .map(|probe| vec![
                probe.span_rows().to_vec(),
                probe.coordinates().to_vec(),
                probe.candidate_support().to_vec(),
            ])
            .collect::<Vec<_>>();
        Ok(Self {
            dimension,
            device_index,
            probe_count: i64::try_from(relevant.len())
                .map_err(|_| "rank-increment count overflow".to_owned())?,
            row_masks: upload_masks(&rows, dimension, device_index)?,
            column_masks: upload_masks(&columns, dimension, device_index)?,
            candidate_masks: upload_masks(&candidates, dimension, device_index)?,
            incidence: upload_incidence(&incidence, dimension, device_index)?,
        })
    }
}

pub(crate) struct DeviceSelectedResidualContext {
    pub(crate) dimension: u32,
    pub(crate) device_index: usize,
    pub(crate) alphabet_id: String,
    pub(crate) probe_count: i64,
    pub(crate) row_masks: Tensor,
    pub(crate) column_masks: Tensor,
    pub(crate) gate_controls: Tensor,
    pub(crate) gate_targets: Tensor,
    pub(crate) incidence: Tensor,
}

impl DeviceSelectedResidualContext {
    pub(crate) fn new(
        alphabet: &CnotAlphabet,
        gates: &[CnotGate],
        device_index: usize,
    ) -> Result<Self, String> {
        let dimension = alphabet.dimension();
        if dimension < 2 || gates.is_empty() {
            return Err("selected-residual device context is empty".into());
        }
        let wires = (0..dimension).collect::<Vec<_>>();
        let mut controls = Vec::with_capacity(gates.len());
        let mut targets = Vec::with_capacity(gates.len());
        let mut row_sets = Vec::with_capacity(gates.len());
        let mut column_sets = Vec::with_capacity(gates.len());
        let mut incidence = Vec::with_capacity(gates.len());
        for &gate in gates {
            if alphabet.cost(gate).is_none() {
                return Err("selected-residual gate is not in its alphabet".into());
            }
            controls.push(gate.control);
            targets.push(gate.target);
            row_sets.push(wires.clone());
            column_sets.push(wires.clone());
            incidence.push(vec![vec![gate.control], vec![gate.target]]);
        }
        Ok(Self {
            dimension,
            device_index,
            alphabet_id: alphabet.id().to_owned(),
            probe_count: i64::try_from(gates.len())
                .map_err(|_| "selected-residual gate count overflow".to_owned())?,
            row_masks: upload_masks(&row_sets, dimension, device_index)?,
            column_masks: upload_masks(&column_sets, dimension, device_index)?,
            gate_controls: upload_gate_indices(&controls, device_index)?,
            gate_targets: upload_gate_indices(&targets, device_index)?,
            incidence: upload_incidence(&incidence, dimension, device_index)?,
        })
    }
}

pub(crate) struct DeviceBlockParityContext {
    pub(crate) dimension: u32,
    pub(crate) device_index: usize,
    pub(crate) context_count: i64,
    pub(crate) max_patterns: u64,
    pub(crate) source_masks: Tensor,
    pub(crate) target_masks: Tensor,
    pub(crate) incidence: Tensor,
}

impl DeviceBlockParityContext {
    pub(crate) fn new(
        dimension: u32,
        contexts: &[BlockContext],
        device_index: usize,
        max_patterns: usize,
    ) -> Result<Self, String> {
        let relevant = contexts
            .iter()
            .filter(|context| context.dimension() == dimension)
            .collect::<Vec<_>>();
        if relevant.is_empty() || max_patterns == 0 {
            return Err("block-parity device context is empty or unbounded".into());
        }
        let sources = relevant.iter().map(|context| context.sources().to_vec()).collect::<Vec<_>>();
        let targets = relevant.iter().map(|context| context.targets().to_vec()).collect::<Vec<_>>();
        let mut incidence = Vec::with_capacity(relevant.len());
        for context in &relevant {
            let other = (0..dimension)
                .filter(|wire| {
                    context.sources().binary_search(wire).is_err()
                        && context.targets().binary_search(wire).is_err()
                })
                .collect::<Vec<_>>();
            incidence.push(vec![
                context.sources().to_vec(),
                context.targets().to_vec(),
                other,
            ]);
        }
        Ok(Self {
            dimension,
            device_index,
            context_count: i64::try_from(relevant.len())
                .map_err(|_| "block-parity context count overflow".to_owned())?,
            max_patterns: u64::try_from(max_patterns)
                .map_err(|_| "block-parity pattern budget overflow".to_owned())?,
            source_masks: upload_masks(&sources, dimension, device_index)?,
            target_masks: upload_masks(&targets, dimension, device_index)?,
            incidence: upload_incidence(&incidence, dimension, device_index)?,
        })
    }
}

pub(crate) struct DeviceMatrixBoundaryContext {
    pub(crate) dimension: u32,
    pub(crate) device_index: usize,
    pub(crate) input_count: usize,
    pub(crate) input_columns: Tensor,
    pub(crate) output_rows: Tensor,
}

impl DeviceMatrixBoundaryContext {
    pub(crate) fn new(
        spec: &MatrixBoundarySpec,
        device_index: usize,
    ) -> Result<Self, String> {
        let dimension = spec.dimension();
        if spec.inputs().is_empty() || spec.outputs().is_empty() {
            return Err("matrix-boundary device context is empty".into());
        }
        Ok(Self {
            dimension,
            device_index,
            input_count: spec.inputs().len(),
            input_columns: upload_masks(&[spec.inputs().to_vec()], dimension, device_index)?,
            output_rows: upload_masks(&[spec.outputs().to_vec()], dimension, device_index)?,
        })
    }
}
