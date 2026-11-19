use std::path::Path;
fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("Usage: cnot103-h07 <dev-h07-manifest.json>");
        println!("Pin training_manifest and training_result by path and SHA-256, and supply explicit dev_indices under schema cnot103_dev_h07_v1.");
        return;
    }
    if args.len() != 1 {
        eprintln!("Usage: cnot103-h07 <dev-h07-manifest.json>");
        std::process::exit(2);
    }
    match cnot103_native::run_config::execute_dev_h07_manifest(Path::new(&args[0]))
        .and_then(|v| serde_json::to_string_pretty(&v).map_err(|e| e.to_string()))
    {
        Ok(report) => println!("{report}"),
        Err(e) => {
            eprintln!("H07 diagnostic failed: {e}");
            std::process::exit(1);
        }
    }
}
