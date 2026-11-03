pub(super) fn exact_f32_integer(value: usize) -> Option<f32> {
    exact_f32_integer_u128(value as u128)
}

pub(super) fn exact_f32_integer_u128(value: u128) -> Option<f32> {
    (value <= (1 << 24)).then_some(value as f32)
}
