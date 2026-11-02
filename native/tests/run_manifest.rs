use cnot103_native::run_config::parse_run_manifest;

#[test]
fn run_manifest_rejects_wrong_schema_before_loading_artifacts() {
    let error = parse_run_manifest(br#"{"schema":"not-a-training-run"}"#)
        .expect_err("wrong schema must fail before artifact I/O");
    assert!(error.to_string().contains("schema"));
}

#[test]
fn run_manifest_rejects_unknown_top_level_fields() {
    let error =
        parse_run_manifest(br#"{"schema":"cnot103_training_run_v1","invented_default":true}"#)
            .expect_err("typos and unknown controls must not be silently ignored");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn run_manifest_rejects_invalid_json() {
    let error = parse_run_manifest(b"{");
    assert!(error.is_err());
}
