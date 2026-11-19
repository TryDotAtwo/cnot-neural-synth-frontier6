use std::ffi::c_void;

unsafe extern "C" {
    fn cnot103_cuda_launch_boundary_relations(
        states: *const u64,
        port_wires: *const u32,
        protected_ports: *const u8,
        equation_outputs: *const u32,
        batch: u32,
        dimension: u32,
        words_per_row: u32,
        inputs: u32,
        outputs: u32,
        equations: u32,
        values: *mut f32,
        valid: *mut u8,
        incidence: *mut f32,
        ranks: *mut f32,
        group_capacity: u64,
        incidence_capacity: u64,
        stream: *mut c_void,
    ) -> i32;
}

/// Construct source-order canonical boundary equations and their protected rank.
///
/// # Safety
/// All arrays must be live allocations on the stream's CUDA device until the
/// queued work completes. States are packed [batch,n,ceil(n/64)]. Static ports
/// are input_count ordered unique input wires followed by output_count ordered
/// unique output wires. Protected flags mark only input ports. Equation outputs
/// contain outputs ascending, followed by unique selected zero-output indices.
/// Arrays contain only in-range physical wires/indices. Output buffers hold
/// values [batch,equations,3], valid [batch,equations], incidence
/// [batch,equations,n,3], ranks [batch,2], and do not overlap any other buffer.
pub unsafe fn launch_boundary_relations(
    states: *const u64,
    port_wires: *const u32,
    protected_ports: *const u8,
    equation_outputs: *const u32,
    batch: u32,
    dimension: u32,
    words_per_row: u32,
    inputs: u32,
    outputs: u32,
    equations: u32,
    values: *mut f32,
    valid: *mut u8,
    incidence: *mut f32,
    ranks: *mut f32,
    group_capacity: u64,
    incidence_capacity: u64,
    stream: *mut c_void,
) -> Result<(), &'static str> {
    let status = unsafe {
        cnot103_cuda_launch_boundary_relations(
            states,
            port_wires,
            protected_ports,
            equation_outputs,
            batch,
            dimension,
            words_per_row,
            inputs,
            outputs,
            equations,
            values,
            valid,
            incidence,
            ranks,
            group_capacity,
            incidence_capacity,
            stream,
        )
    };
    match status {
        0 => Ok(()),
        1 => Err("invalid boundary-relation kernel arguments/capacity"),
        2 => Err("boundary-relation kernel capacity exceeded"),
        _ => Err("CUDA boundary-relation kernel launch failed"),
    }
}
