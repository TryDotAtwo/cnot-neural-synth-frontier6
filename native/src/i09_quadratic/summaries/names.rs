use crate::i09_quadratic::{I09Error, I09ErrorKind};

pub(in crate::i09_quadratic) fn variable_names(
    n: usize,
    horizon: usize,
    variables: usize,
) -> Result<Vec<String>, I09Error> {
    let mut names = Vec::new();
    names.try_reserve_exact(variables).map_err(|_| {
        I09Error::new(
            I09ErrorKind::ResourceLimit,
            "i09_variable_names_allocation_failed",
        )
    })?;
    for slot in 0..horizon {
        names.push(format!("active[{slot}]"));
        for wire in 0..n {
            names.push(format!("control[{slot},{wire}]"));
        }
        for wire in 0..n {
            names.push(format!("target[{slot},{wire}]"));
        }
        for wire in 0..n {
            names.push(format!("readout[{slot},{wire}]"));
        }
        for row in 0..n {
            for column in 0..n {
                names.push(format!("matrix[{slot},{row},{column}]"));
            }
        }
    }
    if names.len() != variables {
        return Err(I09Error::new(
            I09ErrorKind::Internal,
            "i09_variable_name_count_mismatch",
        ));
    }
    Ok(names)
}
