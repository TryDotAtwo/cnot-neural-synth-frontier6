#[test]
fn export_cpu_hamming_fixture_for_ci() {
    let Some(path) = std::env::var_os("CNOT103_TEST_FIXTURE_PATH") else {
        return;
    };
    cnot103_native::frozen_data::write_cpu_hamming_fixture(path)
        .expect("synthetic CPU Hamming fixture should be valid");
}
