use std::path::Path;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let Some(path) = args.next() else {
        eprintln!("Usage: cnot103-validate-manifest <run-manifest.json>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("Usage: cnot103-validate-manifest <run-manifest.json>");
        std::process::exit(2);
    }
    let bytes = match std::fs::read(Path::new(&path)) {
        Ok(bytes) => bytes,
        Err(_) => {
            eprintln!("manifest validation failed: input unavailable");
            std::process::exit(1);
        }
    };
    let manifest = match cnot103_native::run_config::parse_run_manifest(&bytes) {
        Ok(manifest) => manifest,
        Err(_) => {
            eprintln!("manifest validation failed: invalid run manifest");
            std::process::exit(1);
        }
    };
    match manifest.validate_feature_profile() {
        Ok(count) => {
            println!("manifest_profile=valid");
            println!("model_input_slots={count}");
        }
        Err(_) => {
            eprintln!("manifest validation failed: feature profile incomplete or invalid");
            std::process::exit(1);
        }
    }
}
