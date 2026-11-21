use std::path::Path;

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("Usage: cnot103-hamming-search <sealed-hamming-manifest.json>");
        return;
    }
    if args.len() != 1 {
        eprintln!("Usage: cnot103-hamming-search <sealed-hamming-manifest.json>");
        std::process::exit(2);
    }
    match cnot103_native::run_config::execute_hamming_search_manifest(Path::new(&args[0]))
        .and_then(|report| serde_json::to_string_pretty(&report).map_err(|_| "report encoding failed".to_owned()))
    {
        Ok(report) => println!("{report}"),
        Err(_) => {
            eprintln!("CPU Hamming search rejected the pinned manifest or dataset");
            std::process::exit(1);
        }
    }
}
