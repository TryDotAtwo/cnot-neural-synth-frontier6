use crate::model::ModelError;
use crate::state::MatrixState;

pub use crate::shear::{recognize_shear, BasisBinding, ShearDecomposition};

pub fn partial_shear(state: &MatrixState) -> Result<Option<ShearDecomposition>, ModelError> {
    crate::shear::partial_shear(state).map_err(ModelError)
}
