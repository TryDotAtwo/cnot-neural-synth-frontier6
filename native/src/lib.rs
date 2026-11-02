#![deny(unsafe_op_in_unsafe_fn)]

pub mod alphabet;
pub mod basis_gradient;
pub mod bilinear;
pub mod block_parities;
pub(crate) mod bound_features;
pub mod checkpoint;
pub mod diffusion;
pub mod features;
pub mod frequency_context;
pub mod frozen_data;
pub mod general_compiler;
pub mod guarded_features;
pub mod i09_quadratic;
pub mod matrix_boundary;
pub(crate) mod numpy_rng;
pub mod packet;
pub mod paid_basis;
pub mod paid_basis_features;
pub mod positive_packet;
pub mod rank_features;
pub mod response;
pub mod schema;
pub mod shared_relation;
pub mod shear;
pub mod stage26;
pub mod stage35;
pub mod stage37;
pub mod stage38;
pub mod state;
pub mod structural;
pub mod symmetry_alignment;
pub mod template_response;
pub mod training;
pub mod training_data;
pub mod view_diagnostic;

#[cfg(feature = "torch")]
pub mod objective;

#[cfg(feature = "torch")]
pub mod basis;

#[cfg(feature = "torch")]
pub mod bellman;

#[cfg(feature = "torch")]
pub mod model;

#[cfg(feature = "torch")]
pub mod nn_training;

#[cfg(feature = "torch")]
pub mod torch_device;

#[cfg(feature = "torch")]
pub mod training_session;

#[cfg(feature = "torch")]
pub use training_session::{
    FullGraphTrainingSession, TrainingSessionConfig, TrainingSessionReport,
    TrainingSessionResources,
};

#[cfg(feature = "cuda")]
pub mod cuda;

pub use alphabet::{AlphabetError, CertificateError, CnotAlphabet, WordCertificate};
pub use basis_gradient::register_basis_gradient;
pub use bilinear::{
    register_bilinear_contexts, register_default_bilinear_contexts, BilinearContext,
    BilinearContextError, STAGE55_SOURCE_SHA256,
};
pub use block_parities::{register_block_parities, BlockContext};
pub use bound_features::register_general_compiler_bounds;
pub use checkpoint::{Checkpoint, CheckpointError, CheckpointManifest, CheckpointStore};
pub use diffusion::{register_diffusion, DiffusionConfig};
pub use features::{
    FeatureBuildError, FeatureComputation, FeaturePolicy, FeatureProducer, FeatureProviderContexts,
    FeatureRegistry, FeatureValue,
};
pub use frequency_context::{
    register_default_frequency_contexts, register_frequency_contexts, FrequencyContext,
    FrequencyContextError,
};
pub use frozen_data::{FrozenDataError, FrozenDataset, FrozenPanel, FrozenRecordMetadata};
pub use general_compiler::{
    compile_general_cnot_word, GeneralCompilerError, GeneralCompilerLimits,
};
pub use guarded_features::{
    register_guarded_one_excess, register_guarded_pair, register_guarded_pair_f02_view,
    register_guarded_zero_excess, register_guarded_zero_excess_with_budget,
};
pub use i09_quadratic::{
    compute_i09_response, i09_source_sha256, register_i09_quadratic_response, I09Context,
    I09ContextError, I09Error, I09ErrorKind, I09Limits, I09Monomial, I09Response,
    I09_STAGE60_SOURCE_SHA256,
};
pub use matrix_boundary::{
    register_matrix_boundaries, MatrixBoundaryError, MatrixBoundarySpec, RationalWeight,
};
pub use packet::{FeatureEntry, FeaturePacket, FeaturePacketError, FeatureStatus, StateIdentity};
pub use paid_basis::{
    compile_paid_shear, optimize_exact_basis, AcceptedBasisMove, BasisSide, ExactBasisOptimization,
    PaidBasisError, PaidShearCandidate, PaidShearConfig,
};
pub use paid_basis_features::{register_paid_basis_features, PaidBasisBlockContext};
pub use positive_packet::{
    register_default_positive_packet_contexts, register_positive_packet_contexts,
    LocalBooleanFactor, OptionalGatePacket, PositivePacketContextError, STAGE59_SOURCE_SHA256,
};
pub use rank_features::{
    register_labeled_cuts, register_partial_shear_rank, register_rank_increment_probes,
    register_restricted_intersections, register_selected_residual_ranks, LabeledCut,
    LabeledCutError, RankIncrementProbe, RestrictedIntersection,
};
pub use response::{evaluate_i10, register_tiny_i10, I10Config, I10OracleResult, I10OracleStatus};
pub use schema::{
    all_feature_specs, all_model_input_specs, FeatureId, FeatureRole, FeatureSpec, FEATURE_COUNT,
    H07_POST_FORWARD_DIAGNOSTIC_ID, MODEL_INPUT_COUNT,
};
pub use shared_relation::{
    register_default_shared_relation_contexts, register_shared_relation_contexts,
    SharedRelationContext, SharedRelationContextError, STAGE56_SOURCE_SHA256,
};
pub use stage26::{
    register_stage26_features, register_stage26_features_with_budget,
    STAGE26_CERTIFICATE_SOURCE_SHA256, STAGE26_CHECKER_SOURCE_SHA256, STAGE26_PROOF_NOTES_SHA256,
};
pub use stage35::{
    recognize_stage35, verify_stage35_result, Stage35Branch, Stage35Error, Stage35Exact,
    Stage35LowerBoundKind, Stage35Result, Stage35Unsupported, STAGE35_PRODUCER,
    STAGE35_THEOREM_SHA256,
};
pub use stage37::{
    recognize_stage37, verify_stage37_result, Stage37Error, Stage37Exact, Stage37NoZeroExcess,
    Stage37Parent, Stage37ParentMode, Stage37Result, Stage37Unsupported, STAGE37_PRODUCER,
    STAGE37_SOURCE_NOTES_SHA256,
};
pub use stage38::{
    recognize_stage38, verify_stage38_result, Stage38Error, Stage38Exact, Stage38NoOneExcess,
    Stage38ResourceLimit, Stage38Result, Stage38Unsupported, STAGE38_PRODUCER,
    STAGE38_SOURCE_IMPLEMENTATION_SHA256, STAGE38_SOURCE_NOTES_SHA256,
};
pub use state::{CnotGate, MatrixState};
pub use symmetry_alignment::register_symmetry_alignment_h06;
pub use template_response::{
    register_template_response, register_template_response_e02, RationalProbability,
    TemplateChoice, TemplateChoiceKind, TemplateResponseBudgets, TemplateResponseConfig,
};
pub use training::{Progress, PythonRandomSnapshot, Sampler, SamplerSnapshot, TrainingError};
pub use training_data::{
    load_exact_catalogs_pinned, ExactCatalog, ExactCatalogEntry, TrainingDataError, TrainingRecord,
    TrainingSplit,
};
pub use view_diagnostic::{
    compute_h07_prediction_spread, H07Diagnostic, H07Error, H07ProbeKind, H07ProbePrediction,
    H07_PRODUCER_VERSION,
};

#[cfg(feature = "torch")]
pub use view_diagnostic::{evaluate_h07_prediction_spread, H07ModelProbe};

#[cfg(feature = "torch")]
pub use objective::{
    ExposureAudit, FullGraphObjective, FullGraphObjectiveConfig, FullGraphObjectiveRunner,
    StateAdmission,
};

#[cfg(feature = "torch")]
pub use basis::{
    compile_basis, partial_shear, recognize_shear, BasisBinding, BasisCandidate, BasisChoice,
    BasisChoiceKind, BasisField, BasisFieldOutput, BasisLossAudit, BasisLossBackend, BasisTemplate,
    ShearDecomposition,
};

#[cfg(feature = "torch")]
pub use bellman::{
    recompute_backup, select_backup, select_backup_with_macros, BellmanBranchAudit,
    BellmanDiagnostics, BellmanError, BellmanLoss, BellmanSelection, MacroProvider,
};
