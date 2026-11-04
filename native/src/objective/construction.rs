use super::{
    objective_config_digest, ExposureLedger, FullGraphObjective, FullGraphObjectiveConfig,
    ObjectiveDiagnostics, ObjectiveFingerprintInput,
};
use crate::alphabet::CnotAlphabet;
use crate::features::{FeaturePolicy, FeatureRegistry};
use crate::packet::StateIdentity;
use crate::schema::all_model_input_specs;
use crate::state::MatrixState;
use crate::training_data::ExactCatalog;
use std::collections::{HashMap, HashSet};

impl FullGraphObjective {
    pub fn new(
        alphabets: Vec<CnotAlphabet>,
        registry: &FeatureRegistry,
        policy: FeaturePolicy,
        exact_catalogs: Vec<ExactCatalog>,
        forbidden_states: Vec<MatrixState>,
        config: FullGraphObjectiveConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let policy_id = policy.id().map_err(|error| error.to_string())?;
        if config.require_all_model_inputs {
            let selected: HashSet<_> = policy.enabled_ids.iter().copied().collect();
            let expected: HashSet<_> = all_model_input_specs()
                .into_iter()
                .map(|spec| spec.id)
                .collect();
            if selected != expected {
                return Err(
                    "full graph training requires every executable model input from the canonical 103-ID manifest".into(),
                );
            }
        }

        let mut alphabet_map = HashMap::with_capacity(alphabets.len());
        for alphabet in alphabets {
            if alphabet_map
                .insert(alphabet.id().to_owned(), alphabet)
                .is_some()
            {
                return Err("duplicate alphabet identity".into());
            }
        }
        if alphabet_map.is_empty() {
            return Err("full graph objective requires at least one alphabet".into());
        }

        let mut exact_map = HashMap::with_capacity(exact_catalogs.len());
        for catalog in exact_catalogs {
            if !alphabet_map.contains_key(catalog.alphabet_id()) {
                return Err(
                    "exact catalog uses an alphabet not registered by the objective".into(),
                );
            }
            let key = catalog.alphabet_id().to_owned();
            if exact_map.insert(key, catalog).is_some() {
                return Err("duplicate exact catalog for one alphabet".into());
            }
        }

        let mut forbidden = HashSet::with_capacity(forbidden_states.len());
        for state in forbidden_states {
            let alphabet = alphabet_map
                .get(state.alphabet_id())
                .ok_or_else(|| "forbidden state uses an unregistered alphabet".to_owned())?;
            if state.dimension() != alphabet.dimension() {
                return Err("forbidden state dimension differs from its alphabet".into());
            }
            forbidden.insert(StateIdentity::from(&state));
        }

        let feature_contract_sha256 = registry.contract_sha256();
        let config_sha256 = objective_config_digest(ObjectiveFingerprintInput {
            alphabets: &alphabet_map,
            catalogs: &exact_map,
            policy_id: &policy_id,
            feature_contract_sha256: &feature_contract_sha256,
            forbidden: &forbidden,
            config: &config,
            training_records_sha256: None,
            frozen_dataset_sha256: None,
        });
        Ok(Self {
            alphabets: alphabet_map,
            exact_catalogs: exact_map,
            feature_contract_sha256,
            training_records_sha256: None,
            frozen_dataset_sha256: None,
            policy,
            forbidden,
            config,
            config_sha256,
            exposures: ExposureLedger::default(),
            last_diagnostics: ObjectiveDiagnostics::default(),
        })
    }
}
