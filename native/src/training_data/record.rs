use crate::alphabet::{CnotAlphabet, WordCertificate};
use crate::state::MatrixState;

use super::{elementary_lower_bound, ExactCatalog, TrainingDataError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingSplit {
    Train,
    Dev,
    Test,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingRecord {
    pub state: MatrixState,
    pub split: TrainingSplit,
    pub verified_words: Vec<WordCertificate>,
    pub exact_distance: Option<u64>,
    pub lineage_id: String,
}

impl TrainingRecord {
    /// Validate training-only provenance and return its best verified upper bound.
    pub fn validate_for_training(
        &self,
        alphabet: &CnotAlphabet,
        exact_catalog: Option<&ExactCatalog>,
    ) -> Result<u64, TrainingDataError> {
        if self.split != TrainingSplit::Train {
            return Err(TrainingDataError(
                "training record must belong to the train split".into(),
            ));
        }
        if self.lineage_id.trim().is_empty() {
            return Err(TrainingDataError(
                "training record lineage identity is required".into(),
            ));
        }
        if self.state.dimension() != alphabet.dimension()
            || self.state.alphabet_id() != alphabet.id()
        {
            return Err(TrainingDataError(
                "training record state does not match its alphabet".into(),
            ));
        }
        if self.verified_words.is_empty() {
            return Err(TrainingDataError(
                "training record has no verified upper-bound word".into(),
            ));
        }

        let identity = MatrixState::identity(alphabet.dimension(), alphabet.id())
            .map_err(|error| TrainingDataError(error.to_string()))?;
        let mut upper_bound = u64::MAX;
        for certificate in &self.verified_words {
            if certificate.start != self.state || certificate.end != identity {
                return Err(TrainingDataError(
                    "upper-bound word is not a reduction of the training state".into(),
                ));
            }
            certificate
                .verify(alphabet)
                .map_err(|error| TrainingDataError(error.to_string()))?;
            upper_bound = upper_bound.min(certificate.physical_cost);
        }
        let lower_bound = elementary_lower_bound(&self.state)?;
        if lower_bound > upper_bound {
            return Err(TrainingDataError(
                "verified word upper bound is below the elementary lower bound".into(),
            ));
        }

        if let Some(exact_distance) = self.exact_distance {
            if exact_distance < lower_bound || exact_distance > upper_bound {
                return Err(TrainingDataError(
                    "exact label lies outside its verified lower/upper bounds".into(),
                ));
            }
            let proved_by_bounds = lower_bound == exact_distance && upper_bound == exact_distance;
            if !proved_by_bounds {
                let catalog = exact_catalog.ok_or_else(|| {
                    TrainingDataError(
                        "exact label needs a bound-equality proof or verified exact catalog".into(),
                    )
                })?;
                if catalog.alphabet_id() != alphabet.id()
                    || catalog.exact_distance(&self.state) != Some(exact_distance)
                {
                    return Err(TrainingDataError(
                        "exact label differs from the verified exact catalog".into(),
                    ));
                }
            } else if exact_catalog.is_some_and(|catalog| {
                catalog.alphabet_id() != alphabet.id()
                    || catalog.exact_distance(&self.state) != Some(exact_distance)
            }) {
                return Err(TrainingDataError(
                    "exact label differs from the verified exact catalog".into(),
                ));
            }
        }
        Ok(upper_bound)
    }
}
