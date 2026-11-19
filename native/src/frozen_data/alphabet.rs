//! Canonical CNOT alphabet construction shared by wire decoding and symmetry checks.
use super::types::FrozenDataError;
use crate::alphabet::CnotAlphabet;
use std::collections::{hash_map::Entry, HashMap};

pub(super) fn alphabet(
    cache: &mut HashMap<u32, CnotAlphabet>,
    width: u32,
) -> Result<&CnotAlphabet, FrozenDataError> {
    Ok(match cache.entry(width) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(
            CnotAlphabet::all_to_all_unit(width).map_err(|e| FrozenDataError(e.to_string()))?,
        ),
    })
}
