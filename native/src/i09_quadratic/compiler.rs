mod equations;
mod layout;

pub(super) use equations::compile_equations;
pub(super) use layout::slot_width;
pub(crate) use layout::{equation_bound, variable_count, working_bytes_upper_bound};
