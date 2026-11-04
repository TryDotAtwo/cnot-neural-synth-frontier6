use crate::matrix_boundary::PORT_WIRE_ROLES;

pub(super) fn estimated_output_bytes(ports: usize, n: usize, equations: usize) -> Option<usize> {
    let group_incidence = ports.checked_mul(n)?.checked_mul(PORT_WIRE_ROLES.len())?;
    let equation_incidence = equations
        .checked_mul(n)?
        .checked_mul(PORT_WIRE_ROLES.len())?;
    let float_count = ports
        .checked_mul(2)?
        .checked_add(group_incidence)?
        .checked_add(equations.checked_mul(3)?)?
        .checked_add(equation_incidence)?
        .checked_add(1 + 2 + 3 + 4)?;
    float_count
        .checked_mul(std::mem::size_of::<f32>())?
        .checked_add(ports)?
        .checked_add(equations)
}
