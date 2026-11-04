use crate::bilinear::context::{insert_row_basis, words_for, BitMatrix};
use crate::bilinear::BilinearContext;

pub(in crate::bilinear) struct Stage55Reduction {
    pub(in crate::bilinear) residual_rank: usize,
    pub(in crate::bilinear) hidden: usize,
}

pub(in crate::bilinear) fn reduce_stage55(
    context: &BilinearContext,
    c: &BitMatrix,
) -> Result<Stage55Reduction, String> {
    let (m, r, p, t, q) = context.dimensions();
    if c.shape() != (m, r) {
        return Err("i05_selected_block_shape_mismatch".into());
    }

    let uv = context.u.multiply(&context.v)?;
    let uv_columns = uv.transpose()?.rows;
    let v_columns = context.v.transpose()?.rows;
    let mut image_basis = vec![None; p];
    let mut selected = Vec::new();
    let mut e1 = Vec::new();
    let mut coeff1 = Vec::new();
    for (column, image_vector) in uv_columns.iter().enumerate() {
        if insert_row_basis(&mut image_basis, image_vector) {
            selected.push(column);
            e1.push(v_columns[column].clone());
            coeff1.push(basis_vector(q, column)?);
        }
    }
    let h = selected.len();
    let coeff2 = uv.nullspace()?;

    let u_columns = context.u.transpose()?.rows;
    let mut e3 = Vec::new();
    for (coordinate, image_vector) in u_columns.iter().enumerate() {
        if insert_row_basis(&mut image_basis, image_vector) {
            e3.push(basis_vector(t, coordinate)?);
        }
    }
    if image_basis.iter().filter(|pivot| pivot.is_some()).count() != p {
        return Err("i05_source_port_basis_construction_failed".into());
    }

    let mut source_basis = e1;
    source_basis.extend(e3);
    let h_map = context
        .u
        .multiply(&BitMatrix::from_columns(source_basis, t)?)?;
    let h_map_inverse = h_map
        .inverse()
        .ok_or_else(|| "i05_source_port_basis_map_is_not_invertible".to_string())?;
    let a_split = h_map_inverse.multiply(&context.a0)?;
    let a1 = a_split.select_rows(0..h)?;
    let a3 = a_split.select_rows(h..p)?;

    let mut delivery_basis = coeff1;
    delivery_basis.extend(coeff2);
    let d_split = context
        .d0
        .multiply(&BitMatrix::from_columns(delivery_basis, q)?)?;
    let d1 = d_split.select_columns(0..h)?;
    let d2 = d_split.select_columns(h..q)?;
    let hidden = t
        .checked_add(h)
        .and_then(|value| value.checked_sub(p))
        .and_then(|value| value.checked_sub(q))
        .ok_or_else(|| "i05_source_reduction_has_negative_hidden_dimension".to_string())?;

    let left = BitMatrix::from_rows(d2.transpose()?.nullspace()?, m)?;
    let right = BitMatrix::from_columns(a3.nullspace()?, r)?;
    let delivered = d1.multiply(&a1)?;
    let residual_input = c.xor(&delivered)?;
    let residual = left.multiply(&residual_input)?.multiply(&right)?;

    // The I05 packet exposes only residual rank and free hidden width; the
    // alpha/beta ranks used by the source's count channel are not needed here.
    Ok(Stage55Reduction {
        residual_rank: residual.rank(),
        hidden,
    })
}

fn basis_vector(bits: usize, index: usize) -> Result<Vec<u64>, String> {
    if index >= bits {
        return Err("i05_basis_coordinate_out_of_range".into());
    }
    let mut vector = vec![0u64; words_for(bits).ok_or_else(|| "i05_width_overflow".to_string())?];
    vector[index / 64] |= 1u64 << (index % 64);
    Ok(vector)
}
