use super::context::MatrixBoundaryContext;
use super::{MatrixBoundaryError, MatrixBoundarySpec, PORT_WIRE_ROLES};
use crate::features::{FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry};
use crate::packet::FeatureStatus;
use crate::schema::{FeatureId, FeatureRole, FeatureSpec, feature_spec};
use crate::state::MatrixState;
use sha2::{Digest, Sha256};
use std::sync::Arc;

fn producer_version(
    specs: &[MatrixBoundarySpec],
    max_ports: usize,
    max_equations: usize,
) -> String {
    let mut digest = Sha256::new();
    hash_field(&mut digest, b"selected_matrix_boundary_v1");
    hash_field(&mut digest, &(max_ports as u64).to_le_bytes());
    hash_field(&mut digest, &(max_equations as u64).to_le_bytes());
    for spec in specs {
        hash_field(&mut digest, &spec.n.to_le_bytes());
        for wires in [
            &spec.inputs,
            &spec.outputs,
            &spec.protected_inputs,
            &spec.zero_outputs,
        ] {
            hash_field(&mut digest, &(wires.len() as u64).to_le_bytes());
            for wire in wires {
                hash_field(&mut digest, &wire.to_le_bytes());
            }
        }
        if let Some(weights) = &spec.weights {
            hash_field(&mut digest, &[1]);
            hash_field(&mut digest, &(weights.len() as u64).to_le_bytes());
            for (left, right) in weights {
                for weight in [left, right] {
                    hash_field(&mut digest, weight.numerator.as_bytes());
                    hash_field(&mut digest, weight.denominator.as_bytes());
                }
            }
        } else {
            hash_field(&mut digest, &[0]);
        }
    }
    let hash = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("selected_matrix_boundary_v1:{hash}")
}

fn hash_field(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}

struct MatrixBoundaryProducer {
    spec: FeatureSpec,
    id: FeatureId,
    context: Arc<MatrixBoundaryContext>,
}

impl FeatureProducer for MatrixBoundaryProducer {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn symmetry_contract(&self) -> &str {
        "augmentation_only"
    }

    fn compute(&self, state: &MatrixState, policy: &FeaturePolicy) -> FeatureComputation {
        if policy.symmetry_contract != "augmentation_only" {
            return FeatureComputation::Unavailable {
                status: FeatureStatus::Unsupported,
                reason: "matrix_boundary_requires_augmentation_only_symmetry".into(),
            };
        }
        self.context
            .values_for(state, policy.max_bytes)
            .remove(&self.id)
            .unwrap_or_else(|| FeatureComputation::Unavailable {
                status: FeatureStatus::Unsupported,
                reason: "matrix_boundary_provider_did_not_emit_feature".into(),
            })
    }
}

/// Register C08, I01-I04, and I11 together for explicit per-width boundaries.
/// Dynamic I01/I02 group shapes are declared by this provider, never guessed
/// from the static placeholder layout.
pub fn register_matrix_boundaries(
    registry: &mut FeatureRegistry,
    specs: Vec<MatrixBoundarySpec>,
    max_ports: usize,
    max_equations: usize,
) -> Result<(), MatrixBoundaryError> {
    if specs.is_empty() || max_ports == 0 || max_equations == 0 {
        return Err(MatrixBoundaryError(
            "explicit boundaries and finite positive port/equation budgets required".into(),
        ));
    }
    let mut widths = std::collections::HashSet::with_capacity(specs.len());
    for spec in &specs {
        spec.validate()?;
        if !widths.insert(spec.n) {
            return Err(MatrixBoundaryError(
                "at most one explicit boundary per matrix width is allowed".into(),
            ));
        }
    }

    let producer_version = producer_version(&specs, max_ports, max_equations);
    let context = Arc::new(MatrixBoundaryContext::new(specs, max_ports, max_equations));
    let definitions = [
        ((b'C', 8), FeatureRole::Global, 1),
        ((b'I', 1), FeatureRole::Group, 2),
        ((b'I', 2), FeatureRole::Group, 3),
        ((b'I', 3), FeatureRole::Global, 2),
        ((b'I', 4), FeatureRole::Global, 3),
        ((b'I', 11), FeatureRole::Global, 4),
    ];
    for ((family, ordinal), role, channels) in definitions {
        let id = FeatureId::new(family, ordinal).expect("matrix-boundary ID is canonical");
        let mut spec = feature_spec(id);
        spec.role = role;
        spec.channels = channels;
        let wire_roles = if role == FeatureRole::Group {
            PORT_WIRE_ROLES.to_vec()
        } else {
            Vec::new()
        };
        let spec = spec.with_provider_metadata(
            "explicit_y=M[outputs,inputs]x;other_inputs_zero;named_ports;I03_projected_relation_flag_not_generic_zero;I04_positive_product_weights_zero_phase_only;state_counts_as_log2",
            producer_version.clone(),
            ["M", "physical_boundary_spec", "exact_affine_relation"],
            "exact",
            "T2",
            wire_roles,
            false,
        );
        registry.register(Box::new(MatrixBoundaryProducer {
            spec,
            id,
            context: Arc::clone(&context),
        }))?;
    }
    Ok(())
}
