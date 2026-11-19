use std::path::Path;

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let usage = "Usage: cnot103-coverage <run-manifest.json> <train|dev>";
    if args.len() == 1 && args[0] == "--help" {
        println!("{usage}");
        return;
    }
    if args.len() != 2 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    match cnot103_native::run_config::audit_feature_coverage(
        Path::new(&args[0]),
        &args[1].to_string_lossy(),
    ) {
        Ok(report) => println!("{report:#}"),
        Err(e) => {
            eprintln!("feature coverage failed: {e}");
            std::process::exit(1);
        }
    }
}
