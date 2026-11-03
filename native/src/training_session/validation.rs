use crate::frozen_data::FrozenDataset;
use crate::nn_training::{NeuralTrainer, TchTrainingError};

pub(super) fn validate_dataset_sampler(
    dataset: &FrozenDataset,
    trainer: &NeuralTrainer,
) -> Result<(), TchTrainingError> {
    if dataset.train.is_empty() {
        return Err(TchTrainingError::Invalid(
            "frozen train panel must be nonempty",
        ));
    }
    if trainer.sampler_snapshot().size != dataset.train.len() {
        return Err(TchTrainingError::Invalid(
            "sampler size differs from the pinned frozen train panel",
        ));
    }
    Ok(())
}
