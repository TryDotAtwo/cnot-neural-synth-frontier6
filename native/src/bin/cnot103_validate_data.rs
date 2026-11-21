//! Validate a sealed frozen dataset with the canonical Rust loader.
//!
//! This command is intentionally quiet: dataset details and validation errors
//! stay on the remote disk, while Actions receives only an allowlisted status.

use cnot103_native::FrozenDataset;
use std::collections::BTreeMap;
use std::path::PathBuf;

const WIDTHS: [u32; 14] = [8, 10, 13, 25, 26, 34, 45, 50, 58, 72, 90, 108, 122, 144];

fn validate() -> Result<(), ()> {
    let mut args = std::env::args_os().skip(1);
    let first = args.next().ok_or(())?;
    let test_fixture = first == std::ffi::OsStr::new("--test-fixture");
    if test_fixture && !cfg!(debug_assertions) {
        return Err(());
    }
    let path = if test_fixture {
        args.next().map(PathBuf::from).ok_or(())?
    } else {
        PathBuf::from(first)
    };
    let sha256 = args.next().ok_or(())?.into_string().map_err(|_| ())?;
    if args.next().is_some() {
        return Err(());
    }

    let dataset = FrozenDataset::load_pinned(path, &sha256).map_err(|_| ())?;
    if test_fixture {
        return Ok(());
    }
    if dataset.train.len() != 10_000 || dataset.dev.len() != 1_792 || dataset.test.len() != 1_792 {
        return Err(());
    }
    if dataset.train.records().iter().any(|record| {
        record.exact_distance.is_some() || !record.lineage_id.starts_with("rebuild_20261119:")
    }) {
        return Err(());
    }

    let mut counts = BTreeMap::new();
    for record in dataset.train.records() {
        *counts.entry(record.state.dimension()).or_insert(0usize) += 1;
    }
    let expected = WIDTHS
        .iter()
        .enumerate()
        .map(|(i, width)| (*width, if i < 4 { 715 } else { 714 }))
        .collect::<BTreeMap<_, _>>();
    if counts != expected {
        return Err(());
    }
    Ok(())
}

fn main() {
    match validate() {
        Ok(()) => println!("frozen_data_validation=accepted"),
        Err(()) => {
            println!("frozen_data_validation=failed");
            std::process::exit(1);
        }
    }
}
