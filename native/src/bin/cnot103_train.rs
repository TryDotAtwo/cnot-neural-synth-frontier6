use std::path::Path;

fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && args[0] == "--help" {
        println!("Usage: cnot103-train <run-manifest.json>");
        println!("The manifest pins the dataset and physical 102-input profile. A pinned exact catalog is optional for interval labels and equality-certified labels.");
        println!(
            "CNOT103_STAGE_PROFILE=1 emits synchronized, exclusive wall times as JSON on stderr."
        );
        println!("Each invocation runs bounded updates and writes a verified checkpoint; use resume=true to continue from last.json.");
        println!("With stage profiling, CNOT103_PROFILE_CHUNKS=N stops after N detached candidate chunks. The CLI exits with an error and does not commit the partial update.");
        return;
    }
    if args.len() != 1 {
        eprintln!("Usage: cnot103-train <run-manifest.json>");
        std::process::exit(2);
    }
    match cnot103_native::run_config::execute_run_manifest(Path::new(&args[0])) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("could not serialize training report: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("training run failed closed: {error}");
            std::process::exit(1);
        }
    }
}
