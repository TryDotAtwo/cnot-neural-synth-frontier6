use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrequencyContextError(pub String);

impl fmt::Display for FrequencyContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FrequencyContextError {}

/// Physical wire partition and one labelled central Fourier frequency.
///
/// `central_rows` is an `m x r` matrix over GF(2): row `j` corresponds to the
/// `j`th target wire, and bit `i` of that row corresponds to the `i`th source
/// wire. Rows use little-endian `u64` limbs, matching `MatrixState` bit order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrequencyContext {
    pub(super) n: u32,
    pub(super) sources: Vec<u32>,
    pub(super) internal: Vec<u32>,
    pub(super) targets: Vec<u32>,
    pub(super) central_rows: Vec<Vec<u64>>,
}

impl FrequencyContext {
    pub fn new(
        n: u32,
        sources: Vec<u32>,
        internal: Vec<u32>,
        targets: Vec<u32>,
        central_rows: Vec<Vec<u64>>,
    ) -> Result<Self, FrequencyContextError> {
        let width = usize::try_from(n)
            .map_err(|_| FrequencyContextError("interface width overflow".into()))?;
        if width == 0 || sources.is_empty() || internal.is_empty() || targets.is_empty() {
            return Err(FrequencyContextError(
                "positive width and nonempty source/internal/target blocks required".into(),
            ));
        }

        let mut seen = Vec::new();
        seen.try_reserve_exact(width)
            .map_err(|_| FrequencyContextError("wire-partition allocation failed".into()))?;
        seen.resize(width, false);
        for wire in sources.iter().chain(&internal).chain(&targets) {
            let index = usize::try_from(*wire)
                .map_err(|_| FrequencyContextError("wire index overflow".into()))?;
            let Some(slot) = seen.get_mut(index) else {
                return Err(FrequencyContextError(
                    "interface wire is outside the matrix".into(),
                ));
            };
            if std::mem::replace(slot, true) {
                return Err(FrequencyContextError(
                    "interface blocks must partition physical wires without duplicates".into(),
                ));
            }
        }
        if seen.iter().any(|present| !present) {
            return Err(FrequencyContextError(
                "interface blocks must cover every physical wire".into(),
            ));
        }

        let source_words = sources
            .len()
            .checked_add(63)
            .ok_or_else(|| FrequencyContextError("central-frequency width overflow".into()))?
            / 64;
        if central_rows.len() != targets.len()
            || central_rows.iter().any(|row| row.len() != source_words)
        {
            return Err(FrequencyContextError(
                "central frequency must have one packed source-width row per target".into(),
            ));
        }
        let used_last_word_bits = sources.len() % 64;
        if used_last_word_bits != 0 {
            let allowed = (1u64 << used_last_word_bits) - 1;
            if central_rows
                .iter()
                .any(|row| row[source_words - 1] & !allowed != 0)
            {
                return Err(FrequencyContextError(
                    "central frequency has nonzero padding bits".into(),
                ));
            }
        }

        Ok(Self {
            n,
            sources,
            internal,
            targets,
            central_rows,
        })
    }
}
