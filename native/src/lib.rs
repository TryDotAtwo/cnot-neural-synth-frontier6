#![deny(unsafe_op_in_unsafe_fn)]

pub mod alphabet;
pub mod basis_gradient;
pub mod block_parities;
pub mod features;
pub mod packet;
pub mod rank_features;
pub mod schema;
pub mod shear;
pub mod state;
pub mod structural;

#[cfg(feature = "torch")]
pub mod basis;

#[cfg(feature = "torch")]
pub mod model;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use alphabet::{AlphabetError, CertificateError, CnotAlphabet, WordCertificate};
pub use basis_gradient::register_basis_gradient;
pub use block_parities::{register_block_parities, BlockContext};
pub use features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureRegistry,
    FeatureValue,
};
pub use packet::{FeatureEntry, FeaturePacket, FeaturePacketError, FeatureStatus, StateIdentity};
pub use rank_features::{
    register_labeled_cuts, register_partial_shear_rank, register_rank_increment_probes,
    register_restricted_intersections, register_selected_residual_ranks, LabeledCut,
    LabeledCutError, RankIncrementProbe, RestrictedIntersection,
};
pub use schema::{all_feature_specs, FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT};
pub use state::{CnotGate, MatrixState};

#[cfg(feature = "torch")]
pub use basis::{
    compile_basis, partial_shear, recognize_shear, BasisBinding, BasisCandidate, BasisChoice,
    BasisChoiceKind, BasisField, BasisFieldOutput, BasisLossAudit, BasisLossBackend, BasisSide,
    BasisTemplate, ShearDecomposition,
};
