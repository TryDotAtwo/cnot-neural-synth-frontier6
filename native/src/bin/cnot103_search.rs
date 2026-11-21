use std::path::Path;

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("Usage: cnot103-search <panel-search-manifest.json>");
        println!("Pin the training manifest and result by path and SHA-256. Supply a dev or test panel, its frozen IDs/digest, search budget and head under schema cnot103_panel_search_v2.");
        return;
    }
    if args.len() != 1 {
        eprintln!("Usage: cnot103-search <panel-search-manifest.json>");
        std::process::exit(2);
    }
    match cnot103_native::run_config::execute_dev_search_manifest(Path::new(&args[0]))
        .and_then(|v| serde_json::to_string_pretty(&v).map_err(|e| e.to_string()))
    {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("panel search failed: {error}");
            std::process::exit(1);
        }
    }
}
