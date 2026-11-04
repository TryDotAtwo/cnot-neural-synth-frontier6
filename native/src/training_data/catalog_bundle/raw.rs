//! Strict serde representations for the pinned exact-catalog bundle.

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExactCatalogBundleBody {
    pub(super) schema: String,
    pub(super) provenance: serde_json::Value,
    pub(super) catalogs: Vec<ExactCatalogWidth>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExactCatalogWidth {
    pub(super) dimension: u32,
    pub(super) entries: Vec<ExactCatalogRawEntry>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExactCatalogRawEntry {
    pub(super) packed_rows: Vec<serde_json::Number>,
    pub(super) distance: u64,
    pub(super) gates: Vec<ExactCatalogRawGate>,
    pub(super) producer: String,
    pub(super) lineage_id: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExactCatalogRawGate {
    pub(super) control: u32,
    pub(super) target: u32,
}
