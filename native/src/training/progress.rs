//! Attempt, acceptance, and rejection counters for guarded optimizer updates.
use super::TrainingError;
use serde::{Deserialize, Serialize};

/// Attempt, acceptance, and rejection counts for guarded optimizer steps.
///
/// The invariant `attempted == accepted + rejected` is maintained by
/// [`record_attempt`](Self::record_attempt), using checked arithmetic.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub attempted: u64,
    pub accepted: u64,
    pub rejected: u64,
}

impl Progress {
    pub fn record_attempt(&mut self, accepted: bool) -> Result<(), TrainingError> {
        if !self.validate() {
            return Err(TrainingError::InvalidProgress);
        }
        let attempted = self
            .attempted
            .checked_add(1)
            .ok_or(TrainingError::CounterOverflow)?;
        let (new_accepted, new_rejected) = if accepted {
            (
                self.accepted
                    .checked_add(1)
                    .ok_or(TrainingError::CounterOverflow)?,
                self.rejected,
            )
        } else {
            (
                self.accepted,
                self.rejected
                    .checked_add(1)
                    .ok_or(TrainingError::CounterOverflow)?,
            )
        };
        self.attempted = attempted;
        self.accepted = new_accepted;
        self.rejected = new_rejected;
        Ok(())
    }

    pub fn validate(&self) -> bool {
        self.accepted.checked_add(self.rejected) == Some(self.attempted)
    }
}
