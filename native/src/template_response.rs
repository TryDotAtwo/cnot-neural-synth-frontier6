//! Exact finite-template response features from the research-v3 response contract.
//!
//! E02 is the expected physical cost at the all-half Bernoulli point for an
//! explicit, ordered source-template choice set. It is not a distance label or
//! an optimum over the full CNOT alphabet.

mod compute;
mod config;
mod provider;
mod rational;
mod registration;

const TEMPLATE_RESPONSE_CACHE_CAPACITY: usize = 32;
const TEMPLATE_RESPONSE_IDS: [u8; 6] = [2, 3, 6, 7, 8, 9];
const TEMPLATE_RESPONSE_WIRE_ROLES: [&str; 4] = [
    "first_control",
    "first_target",
    "second_control",
    "second_target",
];

pub use config::{
    TemplateChoice, TemplateChoiceKind, TemplateResponseBudgets, TemplateResponseConfig,
};
pub use rational::RationalProbability;
pub use registration::{register_template_response, register_template_response_e02};
