use std::path::Path;

fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 1 || args[0] == "--help" {
        eprintln!("Usage: cnot103-benchmark <fresh-run-manifest.json>");
        eprintln!("Executes real CUDA updates from scratch for timing; discards the model and writes no checkpoint.");
        std::process::exit(if args.len() == 1 { 0 } else { 2 });
    }
    match cnot103_native::run_config::execute_benchmark_manifest(Path::new(&args[0])) {
        Ok(report) => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        Err(error) => { eprintln!("benchmark failed closed: {error}"); std::process::exit(1); }
    }
}
