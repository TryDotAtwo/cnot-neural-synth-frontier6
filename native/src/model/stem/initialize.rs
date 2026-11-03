use super::StructuralStem;
use crate::model::{pytorch_linear_config, ModelConfig, StructuralBlock};
use crate::schema::{FeatureRole, FeatureSpec};
use std::collections::HashMap;
use tch::nn;

impl StructuralStem {
    pub(in crate::model) fn new(
        path: &nn::Path<'_>,
        specs: &[FeatureSpec],
        config: &ModelConfig,
    ) -> Self {
        let mut projections = HashMap::with_capacity(specs.len());
        let mut group_role_projections = HashMap::new();
        for spec in specs {
            let name = spec.id.to_string();
            let out_dim = if spec.role == FeatureRole::Pair {
                config.pair_dim
            } else {
                config.wire_dim
            };
            projections.insert(
                spec.id,
                nn::linear(
                    path / "projections" / name.as_str(),
                    spec.channels as i64,
                    out_dim,
                    pytorch_linear_config(spec.channels as i64),
                ),
            );
            if spec.role == FeatureRole::Group {
                let mut role_layers = Vec::with_capacity(spec.wire_roles.len());
                for role_index in 0..spec.wire_roles.len() {
                    role_layers.push(nn::linear(
                        path / "group_roles" / name.as_str() / role_index.to_string().as_str(),
                        config.wire_dim,
                        config.wire_dim,
                        nn::LinearConfig {
                            bias: false,
                            ..pytorch_linear_config(config.wire_dim)
                        },
                    ));
                }
                group_role_projections.insert(spec.id, role_layers);
            }
        }

        Self {
            specs: specs.to_vec(),
            projections,
            group_role_projections,
            raw_matrix: nn::linear(
                path / "raw_matrix",
                3,
                config.pair_dim,
                pytorch_linear_config(3),
            ),
            pair_to_wire: nn::linear(
                path / "pair_to_node",
                2 * config.pair_dim,
                config.wire_dim,
                pytorch_linear_config(2 * config.pair_dim),
            ),
            size_projection: nn::linear(
                path / "size_projection",
                2,
                config.wire_dim,
                pytorch_linear_config(2),
            ),
            context_query: nn::linear(
                path / "context_q",
                config.wire_dim,
                config.wire_dim,
                pytorch_linear_config(config.wire_dim),
            ),
            context_key: nn::linear(
                path / "context_k",
                config.wire_dim,
                config.wire_dim,
                pytorch_linear_config(config.wire_dim),
            ),
            context_value: nn::linear(
                path / "context_v",
                config.wire_dim,
                config.wire_dim,
                pytorch_linear_config(config.wire_dim),
            ),
            blocks: (0..config.structural_blocks)
                .map(|index| {
                    let name = index.to_string();
                    StructuralBlock::new(&(path / "blocks" / name.as_str()), config)
                })
                .collect(),
            wire_dim: config.wire_dim,
        }
    }
}
