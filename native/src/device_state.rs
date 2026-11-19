use crate::alphabet::CnotAlphabet;
use crate::device_context::{
    DeviceBlockParityContext, DeviceLabeledCutContext, DeviceMatrixBoundaryContext,
    DeviceRankIncrementContext, DeviceRestrictedIntersectionContext, DeviceSelectedResidualContext,
};
use crate::schema::FeatureId;
use tch::{Device, Kind, Tensor};

/// A batch of packed CNOT matrices that remains resident on one CUDA device.
///
/// The batch axis is flattened over search states. This type contains no host-side
/// representation of the matrix data.
pub struct DeviceStateBatch {
    packed: Tensor,
    batch: i64,
    dimension: u32,
    words_per_row: u32,
    device_index: usize,
    alphabet_id: Option<String>,
}

pub(crate) fn unpack_matrix(packed: &Tensor, n: i64) -> Tensor {
    let columns = Tensor::arange(n, (Kind::Int64, packed.device()));
    packed
        .index_select(2, &columns.floor_divide_scalar(64))
        .bitwise_right_shift(&columns.remainder(64))
        .bitwise_and(1i64)
}

/// The 20 structural features in family A, retained as device tensors.
pub struct DeviceFamilyA {
    entries: Vec<Tensor>,
}

impl DeviceFamilyA {
    pub(crate) fn shallow_clone(&self) -> Self {
        Self {
            entries: self.entries.iter().map(Tensor::shallow_clone).collect(),
        }
    }

    /// Returns a one-based family ordinal from A01 through A20.
    pub fn feature(&self, ordinal: usize) -> Option<&Tensor> {
        ordinal
            .checked_sub(1)
            .and_then(|index| self.entries.get(index))
    }
}

/// The 12 graph features in family B, retained as device tensors.
pub struct DeviceFamilyB {
    entries: Vec<Tensor>,
    _family_a: DeviceFamilyA,
}

/// GPU-resident C06-C07 outputs for explicit restricted row/column probes.
pub struct DeviceRestrictedIntersectionFeatures {
    c06: Tensor,
    c07: Tensor,
    token_valid: Tensor,
    incidence: Tensor,
}

impl DeviceRestrictedIntersectionFeatures {
    pub fn feature(&self, id: FeatureId) -> Option<&Tensor> {
        match (id.family(), id.ordinal()) {
            ('C', 6) => Some(&self.c06),
            ('C', 7) => Some(&self.c07),
            _ => None,
        }
    }

    pub fn token_valid(&self) -> &Tensor {
        &self.token_valid
    }

    pub fn incidence(&self) -> &Tensor {
        &self.incidence
    }
}

/// GPU-resident C10 outputs for explicit physical CNOT actions.
pub struct DeviceSelectedResidualFeatures {
    value: Tensor,
    token_valid: Tensor,
    incidence: Tensor,
}

impl DeviceSelectedResidualFeatures {
    pub fn feature(&self, id: FeatureId) -> Option<&Tensor> {
        ((id.family(), id.ordinal()) == ('C', 10)).then_some(&self.value)
    }

    pub fn token_valid(&self) -> &Tensor {
        &self.token_valid
    }

    pub fn incidence(&self) -> &Tensor {
        &self.incidence
    }
}

/// GPU-resident C09 outputs for explicit restricted rank-increment probes.
pub struct DeviceRankIncrementFeatures {
    value: Tensor,
    token_valid: Tensor,
    incidence: Tensor,
}

impl DeviceRankIncrementFeatures {
    pub fn feature(&self, id: FeatureId) -> Option<&Tensor> {
        ((id.family(), id.ordinal()) == ('C', 9)).then_some(&self.value)
    }

    pub fn token_valid(&self) -> &Tensor {
        &self.token_valid
    }

    pub fn incidence(&self) -> &Tensor {
        &self.incidence
    }
}

/// GPU-resident D05 node frequencies for target-row and source-column patterns.
pub struct DeviceBlockParityNodeFeatures {
    d05: Tensor,
    status: Tensor,
    _source_masks_lifetime: Tensor,
    _target_masks_lifetime: Tensor,
}

impl DeviceBlockParityNodeFeatures {
    pub fn feature(&self, id: FeatureId) -> Option<&Tensor> {
        ((id.family(), id.ordinal()) == ('D', 5)).then_some(&self.d05)
    }

    /// 1 means computed; 2 means the exact pattern or f32 integer budget was exceeded.
    pub fn status(&self) -> &Tensor {
        &self.status
    }
}

/// GPU-resident D08 physical-block parity weights, range status and wire incidence.
pub struct DeviceBlockParityFeatures {
    d08: Tensor,
    status: Tensor,
    token_valid: Tensor,
    incidence: Tensor,
    _source_masks_lifetime: Tensor,
    _target_masks_lifetime: Tensor,
}

impl DeviceBlockParityFeatures {
    pub fn feature(&self, id: FeatureId) -> Option<&Tensor> {
        ((id.family(), id.ordinal()) == ('D', 8)).then_some(&self.d08)
    }

    /// 1 means computed; 2 means the exact pattern or f32 integer budget was exceeded.
    pub fn status(&self) -> &Tensor {
        &self.status
    }

    pub fn token_valid(&self) -> &Tensor {
        &self.token_valid
    }

    pub fn incidence(&self) -> &Tensor {
        &self.incidence
    }
}

/// GPU-resident C05 exact partial-shear values and applicability status.
/// Status is 0=inapplicable, 1=computed, 2=invalid dependent source rows.
pub struct DevicePartialShearFeatures {
    value: Tensor,
    status: Tensor,
}

impl DevicePartialShearFeatures {
    pub fn feature(&self, id: FeatureId) -> Option<&Tensor> {
        ((id.family(), id.ordinal()) == ('C', 5)).then_some(&self.value)
    }

    pub fn status(&self) -> &Tensor {
        &self.status
    }
}

/// GPU-resident C02-C04 outputs for explicit labeled physical cuts.
pub struct DeviceLabeledCutFeatures {
    entries: Vec<(FeatureId, Tensor)>,
    token_valid: Tensor,
    incidence: Tensor,
}

impl DeviceLabeledCutFeatures {
    pub fn feature(&self, id: FeatureId) -> Option<&Tensor> {
        self.entries
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, value)| value)
    }

    pub fn token_valid(&self) -> &Tensor {
        &self.token_valid
    }

    pub fn incidence(&self) -> &Tensor {
        &self.incidence
    }
}

/// CUDA outputs for context-free features that reuse the family-A structural values.
pub struct DeviceFeatureSlice {
    entries: Vec<(FeatureId, Tensor)>,
    _family_a: DeviceFamilyA,
}

impl DeviceFeatureSlice {
    pub fn feature(&self, id: FeatureId) -> Option<&Tensor> {
        self.entries
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, value)| value)
    }
}

impl DeviceFamilyB {
    /// Returns a one-based family ordinal from B01 through B12.
    pub fn feature(&self, ordinal: usize) -> Option<&Tensor> {
        ordinal
            .checked_sub(1)
            .and_then(|index| self.entries.get(index))
    }
}

fn transitive_closure(adjacency: &Tensor, identity: &Tensor, dimension: i64) -> Tensor {
    let mut closure = adjacency.maximum(identity);
    let target_path_length = dimension - 1;
    let mut covered_path_length = 1_i64;
    while covered_path_length < target_path_length {
        closure = closure.bmm(&closure).gt(0.0).to_kind(Kind::Float);
        covered_path_length = (covered_path_length * 2).min(target_path_length);
    }
    closure
}

pub(crate) fn gf2_inverse(matrix: &Tensor, batch: i64, dimension: i64, device: Device) -> Tensor {
    let identity = Tensor::eye(dimension, (Kind::Int64, device))
        .unsqueeze(0)
        .expand([batch, dimension, dimension], true);
    let left = matrix.to_kind(Kind::Int64);
    let mut augmented = Tensor::cat(&[&left, &identity], 2);
    let row_ids = Tensor::arange(dimension, (Kind::Int64, device)).view([1, dimension, 1]);

    for pivot_column in 0..dimension {
        let column = augmented.select(2, pivot_column);
        let pivot_offset = column
            .narrow(1, pivot_column, dimension - pivot_column)
            .argmax(1, false);
        let pivot_rows = pivot_offset + pivot_column;
        let pivot_selector = pivot_rows
            .view([batch, 1, 1])
            .expand([batch, 1, 2 * dimension], true);
        let pivot_row = augmented.gather(1, &pivot_selector, false);
        let pivot_slot = augmented.narrow(1, pivot_column, 1);
        let at_pivot = row_ids.eq_tensor(&pivot_rows.view([batch, 1, 1]));
        let at_slot = row_ids.eq(pivot_column);
        let after_first_swap = pivot_row
            .expand([batch, dimension, 2 * dimension], true)
            .where_self(&at_slot, &augmented);
        augmented = pivot_slot
            .expand([batch, dimension, 2 * dimension], true)
            .where_self(&at_pivot, &after_first_swap);

        let pivot_row = augmented.narrow(1, pivot_column, 1);
        let elimination_factor = augmented.select(2, pivot_column).unsqueeze(2);
        let eliminated = (&augmented + elimination_factor * pivot_row).remainder(2);
        let non_pivot_rows = row_ids.ne(pivot_column);
        augmented = eliminated.where_self(&non_pivot_rows, &augmented);
    }

    augmented
        .narrow(2, dimension, dimension)
        .to_kind(Kind::Float)
}

fn pair_view_descriptor(matrix: &Tensor) -> Tensor {
    let transpose = matrix.transpose(1, 2);
    let row_weight = matrix.sum_dim_intlist(&[2_i64][..], false, Kind::Float);
    let column_weight = matrix.sum_dim_intlist(&[1_i64][..], false, Kind::Float);
    let row_intersection = matrix.bmm(&transpose);
    let column_intersection = transpose.bmm(matrix);
    let a15 = row_weight.unsqueeze(2) + row_weight.unsqueeze(1) - 2.0 * &row_intersection;
    let a16 = column_weight.unsqueeze(2) + column_weight.unsqueeze(1) - 2.0 * &column_intersection;
    let d03_10 = row_weight.unsqueeze(2) - &row_intersection;
    let d03_01 = row_weight.unsqueeze(1) - &row_intersection;
    Tensor::stack(&[&a15, &a16, &d03_10, &d03_01, &row_intersection], 3)
}

fn global_weight_stats(weights: &Tensor) -> Tensor {
    let weights64 = weights.to_kind(Kind::Double);
    let mean64 = weights64.mean_dim(&[1_i64][..], false, Kind::Double);
    let delta = &weights64 - &mean64.unsqueeze(1);
    let variance64 = (&delta * &delta).mean_dim(&[1_i64][..], false, Kind::Double);
    let mut columns = vec![
        mean64.to_kind(Kind::Float),
        variance64.to_kind(Kind::Float),
        weights.max_dim(1, false).0,
    ];
    for bucket in 0..=4_i64 {
        columns.push(weights.eq(bucket).to_kind(Kind::Float).sum_dim_intlist(
            &[1_i64][..],
            false,
            Kind::Float,
        ));
    }
    let columns = columns.iter().collect::<Vec<_>>();
    Tensor::stack(&columns, 1)
}

impl DeviceStateBatch {
    /// Computes the directly derivable C01, D03, F01, G05 and H01-H06
    /// features from CUDA-resident A-family tensors.
    pub fn feature_slice(&self) -> Result<DeviceFeatureSlice, String> {
        let family_a = self.family_a()?;
        self.feature_slice_from_a(family_a, true)
    }

    pub(crate) fn feature_slice_from_a(
        &self,
        family_a: DeviceFamilyA,
        include_h06: bool,
    ) -> Result<DeviceFeatureSlice, String> {
        let dimension = i64::from(self.dimension);
        let c01 = self.residual_rank()?;

        let row_weights = family_a
            .feature(5)
            .ok_or_else(|| "A05 source tensor is missing".to_owned())?
            .select(2, 0);
        let row_intersection = family_a
            .feature(13)
            .ok_or_else(|| "A13 source tensor is missing".to_owned())?
            .squeeze_dim(-1);
        let row_left = row_weights.unsqueeze(2);
        let row_right = row_weights.unsqueeze(1);
        let d03 = Tensor::stack(
            &[
                &(&row_left - &row_intersection),
                &(&row_right - &row_intersection),
                &row_intersection,
            ],
            3,
        );

        let row_residual = family_a
            .feature(7)
            .ok_or_else(|| "A07 source tensor is missing".to_owned())?
            .squeeze_dim(-1);
        let column_residual = family_a
            .feature(8)
            .ok_or_else(|| "A08 source tensor is missing".to_owned())?
            .squeeze_dim(-1);
        let residual_rows = row_residual.gt(0.0).to_kind(Kind::Float).sum_dim_intlist(
            &[1_i64][..],
            false,
            Kind::Float,
        );
        let residual_columns = column_residual
            .gt(0.0)
            .to_kind(Kind::Float)
            .sum_dim_intlist(&[1_i64][..], false, Kind::Float);
        let f01 = Tensor::stack(&[&residual_rows, &residual_columns], 1);

        let column_weights = family_a
            .feature(6)
            .ok_or_else(|| "A06 source tensor is missing".to_owned())?
            .squeeze_dim(-1);
        let column_intersection = family_a
            .feature(14)
            .ok_or_else(|| "A14 source tensor is missing".to_owned())?
            .squeeze_dim(-1);
        let column_left = column_weights.unsqueeze(2);
        let column_right = column_weights.unsqueeze(1);
        let constant = Tensor::full(
            [self.batch, dimension, dimension],
            dimension as f64,
            (Kind::Float, Device::Cuda(self.device_index)),
        );
        let g05 = Tensor::stack(
            &[
                &(&constant - &column_left - &column_right + &column_intersection),
                &(&column_left - &column_intersection),
                &(&column_right - &column_intersection),
                &column_intersection,
            ],
            3,
        );

        let h01 = family_a
            .feature(1)
            .ok_or_else(|| "A01 source tensor is missing".to_owned())?
            .shallow_clone();
        let h02 = family_a
            .feature(2)
            .ok_or_else(|| "A02 source tensor is missing".to_owned())?
            .shallow_clone();
        let matrix = h01.squeeze_dim(-1);
        let inverse = gf2_inverse(
            &matrix,
            self.batch,
            dimension,
            Device::Cuda(self.device_index),
        );
        let inverse_transpose = inverse.transpose(1, 2);
        let h03 = inverse.unsqueeze(-1);
        let h04 = inverse_transpose.unsqueeze(-1);
        let h05 = Tensor::stack(
            &[
                &matrix,
                &matrix.transpose(1, 2),
                &inverse,
                &inverse_transpose,
            ],
            3,
        );
        let mut entries = vec![
            (FeatureId::new(b'C', 1).unwrap(), c01),
            (FeatureId::new(b'D', 3).unwrap(), d03),
            (FeatureId::new(b'F', 1).unwrap(), f01),
            (FeatureId::new(b'G', 5).unwrap(), g05),
            (FeatureId::new(b'H', 1).unwrap(), h01),
            (FeatureId::new(b'H', 2).unwrap(), h02),
            (FeatureId::new(b'H', 3).unwrap(), h03),
            (FeatureId::new(b'H', 4).unwrap(), h04),
            (FeatureId::new(b'H', 5).unwrap(), h05),
        ];
        if include_h06 {
            let base = pair_view_descriptor(&matrix);
            let h06 = Tensor::cat(
                &[
                    &(pair_view_descriptor(&matrix.transpose(1, 2)) - &base),
                    &(pair_view_descriptor(&inverse) - &base),
                    &(pair_view_descriptor(&inverse_transpose) - &base),
                ],
                3,
            );
            entries.push((FeatureId::new(b'H', 6).unwrap(), h06));
        }
        Ok(DeviceFeatureSlice {
            entries,
            _family_a: family_a,
        })
    }
}

impl DeviceStateBatch {
    /// Computes C01, rank_GF(2)(M-I), on CUDA without host readback.
    pub fn residual_rank(&self) -> Result<Tensor, String> {
        let output = Tensor::empty(
            [self.batch, 1],
            (Kind::Float, Device::Cuda(self.device_index)),
        );
        let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
        let written = unsafe {
            crate::cuda::launch_residual_ranks(
                self.packed.data_ptr().cast::<u64>(),
                u32::try_from(self.batch)
                    .map_err(|_| "CUDA state batch exceeds kernel range".to_owned())?,
                self.dimension,
                self.words_per_row,
                output.data_ptr().cast::<f32>(),
                u64::try_from(self.batch)
                    .map_err(|_| "CUDA rank output exceeds kernel range".to_owned())?,
                stream,
            )
        }
        .map_err(str::to_owned)?;
        if written != self.batch as u64 {
            return Err("CUDA residual-rank kernel wrote an unexpected element count".into());
        }
        Ok(output)
    }
}

impl DeviceStateBatch {
    /// Computes B01-B12 on the same CUDA device and without host readback.
    pub fn family_b(&self) -> Result<DeviceFamilyB, String> {
        self.family_b_from_a(self.family_a()?)
    }

    pub(crate) fn family_b_from_a(&self, family_a: DeviceFamilyA) -> Result<DeviceFamilyB, String> {
        let dimension = i64::from(self.dimension);
        let device = Device::Cuda(self.device_index);
        let matrix = family_a
            .feature(1)
            .ok_or_else(|| "A01 source tensor is missing".to_owned())?
            .squeeze_dim(-1);
        let row_weights = family_a
            .feature(5)
            .ok_or_else(|| "A05 source tensor is missing".to_owned())?
            .select(2, 0);
        let column_weights = family_a
            .feature(6)
            .ok_or_else(|| "A06 source tensor is missing".to_owned())?
            .squeeze_dim(-1);

        let identity = Tensor::eye(dimension, (Kind::Float, device))
            .unsqueeze(0)
            .expand([self.batch, dimension, dimension], true);
        let off_diagonal =
            Tensor::ones([self.batch, dimension, dimension], (Kind::Float, device)) - &identity;
        let adjacency = &matrix * &off_diagonal;
        let adjacency_transpose = adjacency.transpose(1, 2);
        let undirected = adjacency.maximum(&adjacency_transpose);
        let directed_reach = transitive_closure(&adjacency, &identity, dimension);
        let undirected_reach = transitive_closure(&undirected, &identity, dimension);

        let ids = Tensor::arange(dimension, (Kind::Int64, device));
        let component_ids = ids
            .view([1, dimension, 1])
            .expand([self.batch, dimension, dimension], true);
        let member_ids = ids
            .view([1, 1, dimension])
            .expand([self.batch, dimension, dimension], true);
        let node_ids = ids
            .view([1, dimension])
            .expand([self.batch, dimension], true);

        let weak_labels = member_ids
            .masked_fill(&undirected_reach.eq(0.0), dimension)
            .min_dim(2, false)
            .0;
        let strong_relation = directed_reach
            .gt(0.0)
            .logical_and(&directed_reach.transpose(1, 2).gt(0.0));
        let strong_labels = member_ids
            .masked_fill(&strong_relation.eq(0), dimension)
            .min_dim(2, false)
            .0;

        let weak_is_member = weak_labels.unsqueeze(1).eq_tensor(&component_ids);
        let weak_is_representative = weak_labels.eq_tensor(&node_ids).unsqueeze(2);
        let weak_membership = weak_is_member
            .logical_and(&weak_is_representative)
            .to_kind(Kind::Float);
        let strong_is_member = strong_labels.unsqueeze(1).eq_tensor(&component_ids);
        let strong_is_representative = strong_labels.eq_tensor(&node_ids).unsqueeze(2);
        let strong_membership = strong_is_member
            .logical_and(&strong_is_representative)
            .to_kind(Kind::Float);

        let b1 = weak_labels
            .eq_tensor(&node_ids)
            .to_kind(Kind::Float)
            .sum_dim_intlist(&[1_i64][..], true, Kind::Float);
        let weak_sizes = weak_membership.sum_dim_intlist(&[2_i64][..], false, Kind::Float);
        let b2 = weak_sizes.gather(1, &weak_labels, false).unsqueeze(-1);
        let strong_sizes = strong_membership.sum_dim_intlist(&[2_i64][..], false, Kind::Float);
        let b3 = strong_sizes.gather(1, &strong_labels, false).unsqueeze(-1);

        let inter_component_edges = strong_membership
            .bmm(&adjacency)
            .bmm(&strong_membership.transpose(1, 2));
        let dag = inter_component_edges.gt(0.0).to_kind(Kind::Float) * &off_diagonal;
        let mut depth = Tensor::zeros([self.batch, dimension], (Kind::Float, device));
        for _ in 0..self.dimension {
            let candidate_depths = (&depth.unsqueeze(2) + 1.0) * &dag;
            let predecessor_max = candidate_depths.max_dim(1, false).0;
            depth = depth.maximum(&predecessor_max);
        }
        let indegree = dag.sum_dim_intlist(&[1_i64][..], false, Kind::Float);
        let outdegree = dag.sum_dim_intlist(&[2_i64][..], false, Kind::Float);
        let component_depth = depth.gather(1, &strong_labels, false);
        let component_indegree = indegree.gather(1, &strong_labels, false);
        let component_outdegree = outdegree.gather(1, &strong_labels, false);
        let b4 = Tensor::stack(
            &[&component_depth, &component_indegree, &component_outdegree],
            2,
        );

        let reciprocal = (&adjacency * &adjacency_transpose)
            .sum_dim_intlist(&[2_i64][..], false, Kind::Float)
            .unsqueeze(-1);
        let squared = adjacency.bmm(&adjacency);
        let b6 = squared.unsqueeze(-1);
        let b7 = squared.remainder(2.0).unsqueeze(-1);
        let transpose_product = adjacency_transpose.bmm(&adjacency);
        let row_product = adjacency.bmm(&adjacency_transpose);
        let b8 = Tensor::stack(&[&transpose_product, &row_product], 3);
        let b9 = (&squared * &adjacency_transpose)
            .sum_dim_intlist(&[2_i64][..], false, Kind::Float)
            .unsqueeze(-1);

        let common_neighbors = undirected.bmm(&undirected);
        let twice_triangles =
            (&common_neighbors * &undirected).sum_dim_intlist(&[2_i64][..], false, Kind::Float);
        let b10_triangles = (twice_triangles / 2.0).unsqueeze(-1);
        let four_cycle_pairs = &common_neighbors * (&common_neighbors - 1.0) / 2.0;
        let b10_four_cycles = (four_cycle_pairs * &off_diagonal)
            .sum_dim_intlist(&[2_i64][..], false, Kind::Float)
            .unsqueeze(-1);
        let b10 = Tensor::cat(&[&b10_triangles, &b10_four_cycles], 2);

        let diagonal_residual =
            Tensor::ones([self.batch, dimension], (Kind::Float, device)) - matrix.diagonal(0, 1, 2);
        let per_node_stats = Tensor::stack(&[&diagonal_residual, &row_weights, &column_weights], 2);
        let weak_component_stats = weak_membership.bmm(&per_node_stats);
        let b11_indices = weak_labels
            .unsqueeze(-1)
            .expand([self.batch, dimension, 3], true);
        let b11 = weak_component_stats.gather(1, &b11_indices, false);

        let row_stats = global_weight_stats(&row_weights);
        let column_stats = global_weight_stats(&column_weights);
        let b12 = Tensor::cat(&[&row_stats, &column_stats], 1);

        let entries = vec![b1, b2, b3, b4, reciprocal, b6, b7, b8, b9, b10, b11, b12];
        Ok(DeviceFamilyB {
            entries,
            _family_a: family_a,
        })
    }
}

impl DeviceStateBatch {
    /// Wraps an already device-resident packed matrix tensor without copying it.
    ///
    /// Shape is [batch, dimension, ceil(dimension / 64)], with row-major packed
    /// words. Every state must already have valid padding and be invertible.
    /// Structural validation belongs at input admission, outside the search loop.
    pub fn from_packed_tensor(
        packed: Tensor,
        dimension: u32,
        device_index: usize,
    ) -> Result<Self, String> {
        if dimension < 2 {
            return Err("CUDA state dimension must be at least two".into());
        }
        if device_index > i32::MAX as usize {
            return Err("CUDA device index exceeds the supported range".into());
        }
        let words_per_row = dimension.div_ceil(64);
        let shape = packed.size();
        if shape.len() != 3
            || shape[0] <= 0
            || shape[1] != i64::from(dimension)
            || shape[2] != i64::from(words_per_row)
        {
            return Err("packed CUDA state tensor has the wrong shape".into());
        }
        if packed.kind() != Kind::Int64 || !packed.is_contiguous() {
            return Err("packed CUDA state tensor must be contiguous int64".into());
        }
        if packed.device() != Device::Cuda(device_index) {
            return Err("packed state tensor is not on the requested CUDA device".into());
        }

        Ok(Self {
            batch: shape[0],
            packed,
            dimension,
            words_per_row,
            device_index,
            alphabet_id: None,
        })
    }

    /// Wraps packed states with the exact alphabet identity checked by action features.
    pub fn from_packed_tensor_for_alphabet(
        packed: Tensor,
        alphabet: &CnotAlphabet,
        device_index: usize,
    ) -> Result<Self, String> {
        let mut states = Self::from_packed_tensor(packed, alphabet.dimension(), device_index)?;
        states.alphabet_id = Some(alphabet.id().to_owned());
        Ok(states)
    }

    pub fn batch_size(&self) -> i64 {
        self.batch
    }

    pub(crate) fn row(&self, row: i64) -> Result<Self, String> {
        if !(0..self.batch).contains(&row) {
            return Err("resident state row is outside the packed batch".into());
        }
        self.rows(row, 1)
    }

    pub(crate) fn rows(&self, start: i64, count: i64) -> Result<Self, String> {
        if start < 0 || count <= 0 || start > self.batch - count {
            return Err("resident state range is outside the packed batch".into());
        }
        Ok(Self {
            packed: self.packed.narrow(0, start, count),
            batch: count,
            dimension: self.dimension,
            words_per_row: self.words_per_row,
            device_index: self.device_index,
            alphabet_id: self.alphabet_id.clone(),
        })
    }

    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    pub fn tensor(&self) -> &Tensor {
        &self.packed
    }

    pub(crate) fn alphabet_id(&self) -> Option<&str> {
        self.alphabet_id.as_deref()
    }

    /// Computes D05 row/column parity-pattern frequencies on CUDA.
    pub fn block_parity_d05(
        &self,
        context: &DeviceBlockParityContext,
    ) -> Result<DeviceBlockParityNodeFeatures, String> {
        if self.dimension != context.dimension || self.device_index != context.device_index {
            return Err("block-parity context does not match state batch".into());
        }
        if context.context_count != 1 {
            return Err("D05 requires exactly one physical block context".into());
        }
        let device = Device::Cuda(self.device_index);
        if context.source_masks.device() != device || context.target_masks.device() != device {
            return Err("block-parity context is on a different CUDA device".into());
        }
        let context_count_u32 = u32::try_from(context.context_count)
            .map_err(|_| "block-parity context count exceeds kernel range".to_owned())?;
        let output_count = self
            .batch
            .checked_mul(context.context_count)
            .and_then(|count| count.checked_mul(i64::from(self.dimension)))
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| "D05 output size overflow".to_owned())?;
        let output = Tensor::empty(
            [
                self.batch,
                context.context_count,
                i64::from(self.dimension),
                2,
            ],
            (Kind::Float, device),
        );
        let status = Tensor::empty([self.batch, context.context_count], (Kind::Uint8, device));
        let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
        let written = unsafe {
            crate::cuda::launch_block_node_frequencies(
                self.packed.data_ptr().cast::<u64>(),
                context.source_masks.data_ptr().cast::<u64>(),
                context.target_masks.data_ptr().cast::<u64>(),
                u32::try_from(self.batch)
                    .map_err(|_| "CUDA state batch exceeds kernel range".to_owned())?,
                context_count_u32,
                context.max_patterns,
                self.dimension,
                self.words_per_row,
                output.data_ptr().cast::<f32>(),
                status.data_ptr().cast::<u8>(),
                u64::try_from(output_count)
                    .map_err(|_| "D05 output exceeds kernel range".to_owned())?,
                stream,
            )
        }
        .map_err(str::to_owned)?;
        if written != output_count as u64 {
            return Err("CUDA D05 kernel wrote an unexpected element count".into());
        }
        Ok(DeviceBlockParityNodeFeatures {
            d05: output,
            status,
            _source_masks_lifetime: context.source_masks.shallow_clone(),
            _target_masks_lifetime: context.target_masks.shallow_clone(),
        })
    }

    /// Computes D08 = popcount(M[targets, sources]) for immutable physical blocks.
    /// Matrix data and outputs stay on CUDA; status 2 marks counts that cannot be
    /// represented exactly by the f32 feature contract.
    pub fn block_parity_d08(
        &self,
        context: &DeviceBlockParityContext,
    ) -> Result<DeviceBlockParityFeatures, String> {
        if self.dimension != context.dimension || self.device_index != context.device_index {
            return Err("block-parity context does not match state batch".into());
        }
        let device = Device::Cuda(self.device_index);
        if context.source_masks.device() != device
            || context.target_masks.device() != device
            || context.incidence.device() != device
        {
            return Err("block-parity context is on a different CUDA device".into());
        }
        let context_count_u32 = u32::try_from(context.context_count)
            .map_err(|_| "block-parity context count exceeds kernel range".to_owned())?;
        let output_count = self
            .batch
            .checked_mul(context.context_count)
            .ok_or_else(|| "D08 output count overflow".to_owned())?;
        let output = Tensor::empty(
            [self.batch, context.context_count, 1],
            (Kind::Float, device),
        );
        let status = Tensor::empty([self.batch, context.context_count], (Kind::Uint8, device));
        let token_valid = Tensor::empty([self.batch, context.context_count], (Kind::Uint8, device));
        let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
        let written = unsafe {
            crate::cuda::launch_block_parity_weight(
                self.packed.data_ptr().cast::<u64>(),
                context.source_masks.data_ptr().cast::<u64>(),
                context.target_masks.data_ptr().cast::<u64>(),
                u32::try_from(self.batch)
                    .map_err(|_| "CUDA state batch exceeds kernel range".to_owned())?,
                context_count_u32,
                context.max_patterns,
                self.dimension,
                self.words_per_row,
                output.data_ptr().cast::<f32>(),
                status.data_ptr().cast::<u8>(),
                token_valid.data_ptr().cast::<u8>(),
                u64::try_from(output_count)
                    .map_err(|_| "D08 output exceeds kernel range".to_owned())?,
                stream,
            )
        }
        .map_err(str::to_owned)?;
        if written != output_count as u64 {
            return Err("CUDA D08 kernel wrote an unexpected element count".into());
        }
        let incidence = context.incidence.expand(
            [
                self.batch,
                context.context_count,
                i64::from(self.dimension),
                3,
            ],
            true,
        );
        Ok(DeviceBlockParityFeatures {
            d08: output,
            status,
            token_valid,
            incidence,
            _source_masks_lifetime: context.source_masks.shallow_clone(),
            _target_masks_lifetime: context.target_masks.shallow_clone(),
        })
    }

    /// Computes C05 directly from packed CUDA matrices; feature values and
    /// applicability status remain on-device with no scalar host readback.
    pub fn partial_shear_features(&self) -> Result<DevicePartialShearFeatures, String> {
        let output = Tensor::empty(
            [self.batch, 1],
            (Kind::Float, Device::Cuda(self.device_index)),
        );
        let status = Tensor::empty([self.batch], (Kind::Uint8, Device::Cuda(self.device_index)));
        let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
        let written = unsafe {
            crate::cuda::launch_partial_shear_ranks(
                self.packed.data_ptr().cast::<u64>(),
                u32::try_from(self.batch)
                    .map_err(|_| "CUDA state batch exceeds kernel range".to_owned())?,
                self.dimension,
                self.words_per_row,
                output.data_ptr().cast::<f32>(),
                status.data_ptr().cast::<u8>(),
                u64::try_from(self.batch)
                    .map_err(|_| "CUDA partial-shear output exceeds kernel range".to_owned())?,
                stream,
            )
        }
        .map_err(str::to_owned)?;
        if written != self.batch as u64 {
            return Err("CUDA partial-shear kernel wrote an unexpected element count".into());
        }
        Ok(DevicePartialShearFeatures {
            value: output,
            status,
        })
    }

    /// Computes C08 = |inputs| - rank(M[outputs, inputs]) on CUDA.
    /// The context's immutable boundary masks are already device-resident.
    pub fn matrix_boundary_c08(
        &self,
        context: &DeviceMatrixBoundaryContext,
    ) -> Result<Tensor, String> {
        if self.dimension != context.dimension || self.device_index != context.device_index {
            return Err("matrix-boundary context does not match state batch".into());
        }
        let device = Device::Cuda(self.device_index);
        if context.input_columns.device() != device || context.output_rows.device() != device {
            return Err("matrix-boundary context is on a different CUDA device".into());
        }
        let rank = Tensor::empty([self.batch, 1], (Kind::Float, device));
        let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
        let written = unsafe {
            crate::cuda::launch_projected_ranks(
                self.packed.data_ptr().cast::<u64>(),
                context.output_rows.data_ptr().cast::<u64>(),
                context.input_columns.data_ptr().cast::<u64>(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                u32::try_from(self.batch)
                    .map_err(|_| "CUDA state batch exceeds kernel range".to_owned())?,
                1,
                self.dimension,
                self.words_per_row,
                false,
                false,
                false,
                rank.data_ptr().cast::<f32>(),
                u64::try_from(self.batch)
                    .map_err(|_| "CUDA C08 output exceeds kernel range".to_owned())?,
                stream,
            )
        }
        .map_err(str::to_owned)?;
        if written != self.batch as u64 {
            return Err("CUDA C08 kernel wrote an unexpected element count".into());
        }
        let input_count = Tensor::full(
            [self.batch, 1],
            context.input_count as f64,
            (Kind::Float, device),
        );
        Ok(input_count - rank)
    }

    /// Generates every ordered one-CNOT successor on the same CUDA stream.
    ///
    /// The returned [batch * dimension * (dimension - 1), dimension,
    /// words_per_row] tensor stays on the same GPU. This function does not copy
    /// candidate matrices to the host or synchronize the device.
    pub fn one_gate_successors(&self) -> Result<Self, String> {
        let actions = self
            .dimension
            .checked_mul(self.dimension - 1)
            .ok_or_else(|| "CNOT action count overflow".to_owned())?;
        let successor_batch = self
            .batch
            .checked_mul(i64::from(actions))
            .ok_or_else(|| "CNOT successor batch size overflow".to_owned())?;
        let output = Tensor::empty(
            [
                self.batch,
                i64::from(actions),
                i64::from(self.dimension),
                i64::from(self.words_per_row),
            ],
            (Kind::Int64, Device::Cuda(self.device_index)),
        );
        let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
        let output_capacity_words = self
            .batch
            .checked_mul(i64::from(actions))
            .and_then(|value| value.checked_mul(i64::from(self.dimension)))
            .and_then(|value| value.checked_mul(i64::from(self.words_per_row)))
            .ok_or_else(|| "CNOT output allocation size overflow".to_owned())?;
        let written = unsafe {
            crate::cuda::launch_all_to_all_children(
                self.packed.data_ptr().cast::<u64>(),
                output.data_ptr().cast::<u64>(),
                u32::try_from(self.batch)
                    .map_err(|_| "CUDA state batch exceeds kernel range".to_owned())?,
                self.dimension,
                self.words_per_row,
                u64::try_from(output_capacity_words)
                    .map_err(|_| "CUDA output allocation exceeds kernel range".to_owned())?,
                stream,
            )
        }
        .map_err(str::to_owned)?;
        if written != output_capacity_words as u64 {
            return Err("CUDA child kernel wrote an unexpected word count".into());
        }

        let packed = output.view([
            successor_batch,
            i64::from(self.dimension),
            i64::from(self.words_per_row),
        ]);
        Ok(Self {
            packed,
            batch: successor_batch,
            dimension: self.dimension,
            words_per_row: self.words_per_row,
            device_index: self.device_index,
            alphabet_id: self.alphabet_id.clone(),
        })
    }

    /// Computes C06-C07 from a pre-uploaded immutable context.
    pub fn restricted_intersection_features(
        &self,
        context: &DeviceRestrictedIntersectionContext,
    ) -> Result<DeviceRestrictedIntersectionFeatures, String> {
        if context.dimension != self.dimension || context.device_index != self.device_index {
            return Err("restricted-intersection context does not match state batch".into());
        }
        let device = Device::Cuda(self.device_index);
        let row_ranks = self.projected_ranks(&context.row_oriented, &context.row_columns, false)?;
        let transposed_ranks =
            self.projected_ranks(&context.transposed_rows, &context.transposed_columns, false)?;
        let count = context.count;
        let row_intersection = row_ranks.narrow(1, 0, count) + row_ranks.narrow(1, count, count)
            - row_ranks.narrow(1, 2 * count, count);
        let column_intersection = transposed_ranks.narrow(1, 0, count)
            + transposed_ranks.narrow(1, count, count)
            - transposed_ranks.narrow(1, 2 * count, count);
        let incidence = context
            .incidence
            .expand([self.batch, count, i64::from(self.dimension), 3], true);
        let token_valid = Tensor::ones([self.batch, count], (Kind::Bool, device));
        Ok(DeviceRestrictedIntersectionFeatures {
            c06: row_intersection,
            c07: column_intersection,
            token_valid,
            incidence,
        })
    }

    /// Computes C10 using a pre-uploaded action/alphabet context.
    pub fn selected_residual_features(
        &self,
        context: &DeviceSelectedResidualContext,
    ) -> Result<DeviceSelectedResidualFeatures, String> {
        if context.dimension != self.dimension
            || context.device_index != self.device_index
            || self.alphabet_id.as_deref() != Some(context.alphabet_id.as_str())
        {
            return Err(
                "selected-residual alphabet does not match the admitted state batch".into(),
            );
        }
        let device = Device::Cuda(self.device_index);
        let value = self.projected_ranks_with_action(
            &context.row_masks,
            &context.column_masks,
            true,
            &context.gate_controls,
            &context.gate_targets,
        )?;
        let token_valid = Tensor::ones([self.batch, context.probe_count], (Kind::Bool, device));
        let incidence = context.incidence.expand(
            [
                self.batch,
                context.probe_count,
                i64::from(self.dimension),
                2,
            ],
            true,
        );
        Ok(DeviceSelectedResidualFeatures {
            value,
            token_valid,
            incidence,
        })
    }

    /// Computes C09 using a pre-uploaded immutable probe context.
    pub fn rank_increment_features(
        &self,
        context: &DeviceRankIncrementContext,
    ) -> Result<DeviceRankIncrementFeatures, String> {
        if context.dimension != self.dimension || context.device_index != self.device_index {
            return Err("rank-increment context does not match state batch".into());
        }
        let device = Device::Cuda(self.device_index);
        let baseline = self.projected_ranks(&context.row_masks, &context.column_masks, false)?;
        let augmented = self.projected_ranks_with_candidate(
            &context.row_masks,
            &context.column_masks,
            false,
            Some(&context.candidate_masks),
        )?;
        let value = augmented - baseline;
        let token_valid = Tensor::ones([self.batch, context.probe_count], (Kind::Bool, device));
        let incidence = context.incidence.expand(
            [
                self.batch,
                context.probe_count,
                i64::from(self.dimension),
                3,
            ],
            true,
        );
        Ok(DeviceRankIncrementFeatures {
            value,
            token_valid,
            incidence,
        })
    }

    /// Computes C02-C04 from a pre-uploaded immutable cut context.
    pub fn labeled_cut_features(
        &self,
        context: &DeviceLabeledCutContext,
    ) -> Result<DeviceLabeledCutFeatures, String> {
        if context.dimension != self.dimension || context.device_index != self.device_index {
            return Err("labeled-cut context does not match state batch".into());
        }
        let device = Device::Cuda(self.device_index);
        let c02 = self.projected_ranks(&context.side_a, &context.side_b, false)?;
        let c03 = self.projected_ranks(&context.side_b, &context.side_a, false)?;
        let c04 = self.projected_ranks(&context.side_a, &context.side_a, true)?;
        let token_valid = Tensor::ones([self.batch, context.cut_count], (Kind::Bool, device));
        let incidence = context.incidence.expand(
            [self.batch, context.cut_count, i64::from(self.dimension), 2],
            true,
        );
        Ok(DeviceLabeledCutFeatures {
            entries: vec![
                (FeatureId::new(b'C', 2).unwrap(), c02),
                (FeatureId::new(b'C', 3).unwrap(), c03),
                (FeatureId::new(b'C', 4).unwrap(), c04),
            ],
            token_valid,
            incidence,
        })
    }

    pub(crate) fn projected_ranks(
        &self,
        row_masks: &Tensor,
        column_masks: &Tensor,
        subtract_identity: bool,
    ) -> Result<Tensor, String> {
        self.projected_ranks_with_candidate(row_masks, column_masks, subtract_identity, None)
    }

    fn projected_ranks_with_candidate(
        &self,
        row_masks: &Tensor,
        column_masks: &Tensor,
        subtract_identity: bool,
        candidate_masks: Option<&Tensor>,
    ) -> Result<Tensor, String> {
        self.projected_ranks_with_context(
            row_masks,
            column_masks,
            subtract_identity,
            candidate_masks,
            None,
            None,
        )
    }

    fn projected_ranks_with_action(
        &self,
        row_masks: &Tensor,
        column_masks: &Tensor,
        subtract_identity: bool,
        gate_controls: &Tensor,
        gate_targets: &Tensor,
    ) -> Result<Tensor, String> {
        self.projected_ranks_with_context(
            row_masks,
            column_masks,
            subtract_identity,
            None,
            Some(gate_controls),
            Some(gate_targets),
        )
    }

    fn projected_ranks_with_context(
        &self,
        row_masks: &Tensor,
        column_masks: &Tensor,
        subtract_identity: bool,
        candidate_masks: Option<&Tensor>,
        gate_controls: Option<&Tensor>,
        gate_targets: Option<&Tensor>,
    ) -> Result<Tensor, String> {
        let shape = row_masks.size();
        if shape.len() != 2
            || shape[0] <= 0
            || shape[1] != i64::from(self.words_per_row)
            || column_masks.size() != shape
            || row_masks.kind() != Kind::Int64
            || column_masks.kind() != Kind::Int64
            || !row_masks.is_contiguous()
            || !column_masks.is_contiguous()
        {
            return Err("projected-rank masks have an invalid shape or layout".into());
        }
        let device = Device::Cuda(self.device_index);
        if row_masks.device() != device || column_masks.device() != device {
            return Err("projected-rank masks must be on the state CUDA device".into());
        }
        if let Some(candidate) = candidate_masks {
            if candidate.size() != shape
                || candidate.kind() != Kind::Int64
                || !candidate.is_contiguous()
                || candidate.device() != device
            {
                return Err("candidate masks have an invalid shape, layout, or device".into());
            }
        }
        let apply_gate = gate_controls.is_some() || gate_targets.is_some();
        if apply_gate {
            let valid_gate_tensor = |tensor: &Tensor| {
                tensor.size() == [shape[0]]
                    && tensor.kind() == Kind::Int
                    && tensor.is_contiguous()
                    && tensor.device() == device
            };
            if !gate_controls.is_some_and(valid_gate_tensor)
                || !gate_targets.is_some_and(valid_gate_tensor)
            {
                return Err(
                    "selected-action descriptors have invalid shape, layout, or device".into(),
                );
            }
        }
        let gate_controls_ptr =
            gate_controls.map_or(std::ptr::null(), |tensor| tensor.data_ptr().cast::<u32>());
        let gate_targets_ptr =
            gate_targets.map_or(std::ptr::null(), |tensor| tensor.data_ptr().cast::<u32>());
        let cuts = u32::try_from(shape[0])
            .map_err(|_| "projected-rank probe count exceeds CUDA kernel range".to_owned())?;
        let output_count = self
            .batch
            .checked_mul(shape[0])
            .ok_or_else(|| "projected-rank output size overflow".to_owned())?;
        let output = Tensor::empty([output_count], (Kind::Float, device));
        let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
        let candidate_ptr = candidate_masks.map_or(std::ptr::null(), |candidate| {
            candidate.data_ptr().cast::<u64>()
        });
        let written = unsafe {
            crate::cuda::launch_projected_ranks(
                self.packed.data_ptr().cast::<u64>(),
                row_masks.data_ptr().cast::<u64>(),
                column_masks.data_ptr().cast::<u64>(),
                candidate_ptr,
                gate_controls_ptr,
                gate_targets_ptr,
                u32::try_from(self.batch)
                    .map_err(|_| "CUDA state batch exceeds kernel range".to_owned())?,
                cuts,
                self.dimension,
                self.words_per_row,
                subtract_identity,
                candidate_masks.is_some(),
                apply_gate,
                output.data_ptr().cast::<f32>(),
                u64::try_from(output_count)
                    .map_err(|_| "projected-rank output exceeds CUDA range".to_owned())?,
                stream,
            )
        }
        .map_err(str::to_owned)?;
        if written != output_count as u64 {
            return Err("CUDA projected-rank kernel wrote an unexpected element count".into());
        }
        Ok(output.view([self.batch, shape[0], 1]))
    }

    /// Computes A01-A20 directly on the GPU without materializing a host packet.
    pub fn family_a(&self) -> Result<DeviceFamilyA, String> {
        let cells = i64::from(self.dimension)
            .checked_mul(i64::from(self.dimension))
            .ok_or_else(|| "A-family matrix extent overflow".to_owned())?;
        let pair_width = cells
            .checked_mul(17)
            .ok_or_else(|| "A-family pair extent overflow".to_owned())?;
        let node_width = i64::from(self.dimension)
            .checked_mul(10)
            .ok_or_else(|| "A-family node extent overflow".to_owned())?;
        let feature_width = pair_width
            .checked_add(node_width)
            .ok_or_else(|| "A-family feature extent overflow".to_owned())?;
        let capacity = self
            .batch
            .checked_mul(feature_width)
            .ok_or_else(|| "A-family batch extent overflow".to_owned())?;
        let output = Tensor::empty(
            [self.batch, feature_width],
            (Kind::Float, Device::Cuda(self.device_index)),
        );
        let stream = crate::cuda::current_stream(self.device_index).map_err(str::to_owned)?;
        let written = unsafe {
            crate::cuda::launch_family_a(
                self.packed.data_ptr().cast::<u64>(),
                u32::try_from(self.batch)
                    .map_err(|_| "CUDA state batch exceeds kernel range".to_owned())?,
                self.dimension,
                self.words_per_row,
                output.data_ptr().cast::<f32>(),
                u64::try_from(capacity)
                    .map_err(|_| "A-family output exceeds kernel range".to_owned())?,
                stream,
            )
        }
        .map_err(str::to_owned)?;
        if written != capacity as u64 {
            return Err("CUDA A-family kernel wrote an unexpected element count".into());
        }

        let pair = |offset: i64, channels: i64| {
            output.narrow(1, offset * cells, channels * cells).view([
                self.batch,
                i64::from(self.dimension),
                i64::from(self.dimension),
                channels,
            ])
        };
        let node = |offset: i64, channels: i64| {
            output
                .narrow(
                    1,
                    pair_width + offset * i64::from(self.dimension),
                    channels * i64::from(self.dimension),
                )
                .view([self.batch, i64::from(self.dimension), channels])
        };

        let entries = vec![
            pair(0, 1),  // A01
            pair(1, 1),  // A02
            pair(2, 2),  // A03
            pair(4, 1),  // A04
            node(0, 2),  // A05
            node(2, 1),  // A06
            node(3, 1),  // A07
            node(4, 1),  // A08
            node(5, 1),  // A09
            node(6, 1),  // A10
            node(7, 1),  // A11
            node(8, 2),  // A12
            pair(5, 1),  // A13
            pair(6, 1),  // A14
            pair(7, 1),  // A15
            pair(8, 1),  // A16
            pair(9, 2),  // A17
            pair(11, 2), // A18
            pair(13, 2), // A19
            pair(15, 2), // A20
        ];
        Ok(DeviceFamilyA { entries })
    }
}
