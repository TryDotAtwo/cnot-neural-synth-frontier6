//! Verified training records and exact-distance catalogs.
//!
//! Exact labels are checked against a supplied closed catalog, or certified by
//! a replayed word attaining an independently derived lower bound.

mod bounds;
mod catalog;
mod catalog_bundle;
mod matrix_proof;
mod record;

pub use bounds::elementary_lower_bound;
pub use catalog::{ExactCatalog, ExactCatalogEntry};
pub use catalog_bundle::load_exact_catalogs_pinned;
pub use record::{TrainingRecord, TrainingSplit};

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingDataError(pub String);

impl Display for TrainingDataError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for TrainingDataError {}
