//! Verified training records and exact-distance catalogs.
//!
//! Exact labels are admitted only from a finite, alphabet-closed catalog with
//! replayable reductions and a globally 1-Lipschitz distance potential. This
//! is stronger than treating one short word as a shortest-path proof.

mod bounds;
mod catalog;
mod catalog_bundle;
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
