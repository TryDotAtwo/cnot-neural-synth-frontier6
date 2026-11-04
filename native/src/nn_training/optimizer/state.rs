use super::super::TchTrainingError;
use super::{codec, AdamWConfig};
use tch::nn::VarStore;
use tch::Tensor;

#[derive(Debug)]
pub(super) struct ParameterState {
    pub(super) name: String,
    pub(super) parameter: Tensor,
    pub(super) first_moment: Tensor,
    pub(super) second_moment: Tensor,
    pub(super) steps: u64,
}

#[derive(Debug)]
pub(in crate::nn_training) struct AdamW {
    pub(super) config: AdamWConfig,
    pub(super) steps: u64,
    pub(super) parameters: Vec<ParameterState>,
}

pub(in crate::nn_training) struct DecodedAdamW {
    pub(super) steps: u64,
    pub(super) parameter_steps: Vec<u64>,
    pub(super) moments: Vec<(Tensor, Tensor)>,
}

impl AdamW {
    pub(in crate::nn_training) fn new(
        variables: &VarStore,
        config: AdamWConfig,
    ) -> Result<Self, TchTrainingError> {
        config.validate()?;
        let mut named: Vec<_> = variables
            .variables()
            .into_iter()
            .filter(|(_, tensor)| tensor.requires_grad())
            .collect();
        named.sort_by(|left, right| left.0.cmp(&right.0));
        if named.is_empty() {
            return Err(TchTrainingError::Invalid(
                "model has no trainable parameters",
            ));
        }
        let parameters = named
            .into_iter()
            .map(|(name, parameter)| ParameterState {
                name,
                first_moment: Tensor::zeros_like(&parameter),
                second_moment: Tensor::zeros_like(&parameter),
                steps: 0,
                parameter,
            })
            .collect();
        Ok(Self {
            config,
            steps: 0,
            parameters,
        })
    }

    pub(in crate::nn_training) fn steps(&self) -> u64 {
        self.steps
    }

    pub(in crate::nn_training) fn parameter_refs(&self) -> Vec<&Tensor> {
        self.parameters
            .iter()
            .map(|state| &state.parameter)
            .collect()
    }

    pub(in crate::nn_training) fn encode(&self) -> Result<Vec<u8>, TchTrainingError> {
        codec::encode(self)
    }

    pub(in crate::nn_training) fn decode(
        &self,
        bytes: &[u8],
    ) -> Result<DecodedAdamW, TchTrainingError> {
        codec::decode(self, bytes)
    }

    pub(in crate::nn_training) fn apply_decoded(&mut self, decoded: DecodedAdamW) {
        self.steps = decoded.steps;
        for ((state, (first, second)), steps) in self
            .parameters
            .iter_mut()
            .zip(decoded.moments)
            .zip(decoded.parameter_steps)
        {
            state.first_moment = first;
            state.second_moment = second;
            state.steps = steps;
        }
    }
}

impl DecodedAdamW {
    pub(in crate::nn_training) fn steps(&self) -> u64 {
        self.steps
    }
}
