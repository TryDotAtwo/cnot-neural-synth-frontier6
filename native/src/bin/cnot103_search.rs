use std::path::Path;

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("Usage: cnot103-search <dev-search-manifest.json>");
        println!("Pin the training manifest and its result by path and SHA-256. Supply dev_indices, search budgets and head (qhat or proposal) under schema cnot103_dev_search_v1.");
        return;
    }
    if args.len() != 1 {
        eprintln!("Usage: cnot103-search <dev-search-manifest.json>");
        std::process::exit(2);
    }
    match cnot103_native::run_config::execute_dev_search_manifest(Path::new(&args[0]))
        .and_then(|v| serde_json::to_string_pretty(&v).map_err(|e| e.to_string()))
    {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("dev search failed: {error}");
            std::process::exit(1);
        }
    }
}
