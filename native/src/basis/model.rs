//! Neural basis proposal head and its binding to a physical matrix state.

use super::{
    objective, partial_shear, BasisBinding, BasisLossAudit, BasisLossBackend, BasisSide,
    BasisTemplate,
};
use crate::model::{EncodedState, FieldOutput, FusionDistanceField, ModelConfig, ModelError};
use crate::packet::FeaturePacket;
use crate::schema::FeatureSpec;
use crate::state::MatrixState;
use tch::nn::{self};
use tch::{Device, Kind, Tensor};

#[derive(Debug)]
struct BasisProposal {
    first: nn::Linear,
    second: nn::Linear,
}

impl BasisProposal {
    fn new(path: &nn::Path<'_>, wire_dim: i64, hidden_dim: i64) -> Self {
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

    fn forward(
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

/// Full-width field plus the paid-basis proposal head used by production runs.
#[derive(Debug)]
pub struct BasisField {
    field: FusionDistanceField,
    proposer: BasisProposal,
}

#[derive(Debug)]
pub struct BasisFieldOutput {
    pub field: FieldOutput,
    pub proposal_probabilities: Tensor,
}

impl BasisField {
    pub fn new(
        path: &nn::Path<'_>,
        specs: &[FeatureSpec],
        schema_id: impl Into<String>,
        policy_id: impl Into<String>,
        device: Device,
        config: ModelConfig,
    ) -> Result<Self, ModelError> {
        let field = FusionDistanceField::new(
            &(path / "field"),
            specs,
            schema_id,
            policy_id,
            device,
            config.clone(),
        )?;
        let proposer = BasisProposal::new(
            &(path / "proposer"),
            config.wire_dim,
            config.basis_proposal_hidden_dim,
        );
        Ok(Self { field, proposer })
    }

    pub fn forward(&self, packet: &FeaturePacket) -> Result<FieldOutput, ModelError> {
        self.field.forward(packet)
    }

    pub fn basis_proposal(
        &self,
        packet: &FeaturePacket,
        template: &BasisTemplate,
        batch_index: usize,
    ) -> Result<Tensor, ModelError> {
        let encoded = self.field.encode(packet)?;
        self.proposer.forward(&encoded, template, batch_index)
    }

    pub fn forward_with_basis(
        &self,
        packet: &FeaturePacket,
        template: &BasisTemplate,
        batch_index: usize,
    ) -> Result<BasisFieldOutput, ModelError> {
        let encoded = self.field.encode(packet)?;
        let field = self.field.forward_encoded(&encoded);
        let proposal_probabilities = self.proposer.forward(&encoded, template, batch_index)?;
        Ok(BasisFieldOutput {
            field,
            proposal_probabilities,
        })
    }

    /// Train the basis proposal against the paid cost of a bound shear.
    ///
    /// The packet must contain all 103 resolved features for the same state;
    /// the selected source binding is re-derived before the objective runs.
    #[allow(clippy::too_many_arguments)]
    pub fn basis_loss(
        &self,
        state: &MatrixState,
        packet: &FeaturePacket,
        template: &BasisTemplate,
        block: &[Vec<u8>],
        batch_index: usize,
        binding: BasisBinding,
        backend: BasisLossBackend,
        samples: usize,
        seed: u64,
        max_evaluations: usize,
        max_bytes: usize,
    ) -> Result<(Tensor, BasisLossAudit), ModelError> {
        template.validate()?;
        if state.dimension() as usize != template.n {
            return Err(ModelError("state/template width mismatch".into()));
        }
        let packet_state = packet
            .state_keys
            .get(batch_index)
            .ok_or_else(|| ModelError("basis state batch index is outside packet".into()))?;
        if packet_state != &crate::packet::StateIdentity::from(state) {
            return Err(ModelError(
                "feature packet is not bound to the basis objective state".into(),
            ));
        }
        if block.len() != template.target_wires.len()
            || block.iter().any(|row| {
                row.len() != template.source_wires.len() || row.iter().any(|bit| *bit > 1)
            })
        {
            return Err(ModelError(
                "basis block must be a binary target-by-source matrix".into(),
            ));
        }
        match binding {
            BasisBinding::FullShear => bind_full_shear(state, template, block)?,
            BasisBinding::PartialElimination => {
                let parsed = partial_shear(state)?.ok_or_else(|| {
                    ModelError("matrix has no nonzero partial-elimination shear".into())
                })?;
                if parsed.source_wires != template.source_wires
                    || parsed.target_wires != template.target_wires
                    || parsed.block != block
                {
                    return Err(ModelError(
                        "basis block is not bound to the partial elimination".into(),
                    ));
                }
            }
        }

        let encoded = self.field.encode(packet)?;
        let probabilities = self.proposer.forward(&encoded, template, batch_index)?;
        let result = objective::paid_cost_loss(
            template,
            block,
            &probabilities,
            binding,
            backend,
            samples,
            seed,
            max_evaluations,
            max_bytes,
        )?;
        Ok((result.loss, result.audit))
    }
}

fn bind_full_shear(
    state: &MatrixState,
    template: &BasisTemplate,
    block: &[Vec<u8>],
) -> Result<(), ModelError> {
    let identity = MatrixState::identity(state.dimension(), state.alphabet_id())
        .map_err(|error| ModelError(error.to_string()))?;
    let words_per_row = identity.words_per_row();
    let mut expected_rows = identity.packed_rows().to_vec();
    for (target_index, &target_wire) in template.target_wires.iter().enumerate() {
        for (source_index, &source_wire) in template.source_wires.iter().enumerate() {
            if block[target_index][source_index] == 1 {
                let offset = target_wire * words_per_row + source_wire / 64;
                expected_rows[offset] ^= 1u64 << (source_wire % 64);
            }
        }
    }
    if expected_rows != state.packed_rows() {
        return Err(ModelError(
            "basis block is not bound to the full matrix shear".into(),
        ));
    }
    Ok(())
}
