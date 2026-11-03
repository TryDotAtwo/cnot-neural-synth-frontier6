use super::{EncodedState, Mlp, ModelConfig};
use std::collections::HashMap;
use tch::nn;
use tch::{Kind, Tensor};

#[derive(Debug)]
pub(super) struct FamilyHeads {
    heads: HashMap<char, Mlp>,
    order: Vec<char>,
}

impl FamilyHeads {
    pub(super) fn new(path: &nn::Path<'_>, families: &[char], config: &ModelConfig) -> Self {
        let mut heads = HashMap::new();
        for family in families {
            heads.insert(
                *family,
                Mlp::new(
                    &(path / "heads" / family.to_string().as_str()),
                    config.wire_dim + 2 * config.pair_dim,
                    config.wire_dim,
                    config.wire_dim,
                ),
            );
        }
        Self {
            heads,
            order: families.to_vec(),
        }
    }

    pub(super) fn forward(&self, encoded: &EncodedState) -> (Tensor, Tensor) {
        let size = encoded
            .valid
            .to_kind(Kind::Float)
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
            .clamp_min(1.0)
            .view([-1, 1, 1]);
        let input = Tensor::cat(
            &[
                &encoded.wire,
                &(encoded
                    .pair
                    .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
                    / &size),
                &(encoded
                    .pair
                    .sum_dim_intlist([2i64].as_slice(), false, Kind::Float)
                    / &size),
            ],
            -1,
        );
        let mut tokens = Vec::with_capacity(self.order.len());
        let mut masks = Vec::with_capacity(self.order.len());
        for family in &self.order {
            let family_valid = encoded
                .family_available
                .get(family)
                .expect("all 103 feature families are present")
                .unsqueeze(1)
                .expand_as(&encoded.valid)
                .logical_and(&encoded.valid);
            let token = self
                .heads
                .get(family)
                .expect("family head is constructed")
                .forward(&input)
                * family_valid.to_kind(input.kind()).unsqueeze(-1);
            tokens.push(token);
            masks.push(family_valid);
        }
        (
            Tensor::stack(&tokens.iter().collect::<Vec<_>>(), 2),
            Tensor::stack(&masks.iter().collect::<Vec<_>>(), 2),
        )
    }
}
