use super::super::{BasisSide, BasisTemplate};
use crate::model::{EncodedState, ModelError};
use tch::nn::{self};
use tch::{Kind, Tensor};

#[derive(Debug)]
pub(super) struct BasisProposal {
    first: nn::Linear,
    second: nn::Linear,
}

impl BasisProposal {
    pub(super) fn new(path: &nn::Path<'_>, wire_dim: i64, hidden_dim: i64) -> Self {
        let network = path / "network";
        Self {
            // Match the source Sequential(Linear, GELU, Linear) parameter paths.
            first: nn::linear(
                &network / "0",
                4 * wire_dim + 4,
                hidden_dim,
                Default::default(),
            ),
            second: nn::linear(&network / "2", hidden_dim, 1, Default::default()),
        }
    }

    pub(super) fn forward(
        &self,
        encoded: &EncodedState,
        template: &BasisTemplate,
        batch_index: usize,
    ) -> Result<Tensor, ModelError> {
        template.validate()?;
        let shape = encoded.wire.size();
        if shape.len() != 3 || batch_index >= shape[0] as usize {
            return Err(ModelError(
                "basis proposal batch index is outside the encoded batch".into(),
            ));
        }
        if template.n > shape[1] as usize {
            return Err(ModelError(
                "basis template references absent physical wires".into(),
            ));
        }

        let batch = batch_index as i64;
        let wire = encoded.wire.select(0, batch);
        let valid = encoded
            .valid
            .select(0, batch)
            .narrow(0, 0, template.n as i64);
        let valid_count = valid.to_kind(Kind::Int64).sum(Kind::Int64).int64_value(&[]);
        if valid_count != template.n as i64 {
            return Err(ModelError(
                "basis template references absent physical wires".into(),
            ));
        }

        if template.choices.is_empty() {
            return Ok(Tensor::zeros([0], (wire.kind(), wire.device())));
        }

        let indices = |wires: &[usize]| -> Result<Tensor, ModelError> {
            let values = wires
                .iter()
                .map(|&index| {
                    i64::try_from(index).map_err(|_| {
                        ModelError("physical wire index exceeds tensor index range".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Tensor::from_slice(&values).to_device(wire.device()))
        };
        let source = wire
            .index_select(0, &indices(&template.source_wires)?)
            .mean_dim([0i64].as_slice(), false, wire.kind());
        let target = wire
            .index_select(0, &indices(&template.target_wires)?)
            .mean_dim([0i64].as_slice(), false, wire.kind());
        let group_wires = template.group_wires();
        let choice_count = template.choices.len();
        let mut inputs = Vec::with_capacity(choice_count);

        for (position, choice) in template.choices.iter().enumerate() {
            let control = i64::try_from(group_wires[choice.control])
                .map_err(|_| ModelError("control wire exceeds tensor index range".into()))?;
            let target_wire = i64::try_from(group_wires[choice.target])
                .map_err(|_| ModelError("target wire exceeds tensor index range".into()))?;
            let metadata = [
                choice.kind.paid_cost(),
                if template.side == BasisSide::Target {
                    1.0
                } else {
                    0.0
                },
                position as f32 / choice_count.max(1) as f32,
                choice_count as f32,
            ];
            let metadata = Tensor::from_slice(&metadata)
                .to_kind(wire.kind())
                .to_device(wire.device());
            let control_embedding = wire.select(0, control);
            let target_embedding = wire.select(0, target_wire);
            inputs.push(Tensor::cat(
                &[
                    &control_embedding,
                    &target_embedding,
                    &source,
                    &target,
                    &metadata,
                ],
                0,
            ));
        }

        Ok(Tensor::stack(&inputs, 0)
            .apply(&self.first)
            .gelu("none")
            .apply(&self.second)
            .squeeze_dim(-1)
            .sigmoid())
    }
}
