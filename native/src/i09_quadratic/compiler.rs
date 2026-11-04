mod equations;
mod layout;

pub(super) use equations::compile_equations;
pub(super) use layout::{equation_bound, slot_width, variable_count, working_bytes_upper_bound};
