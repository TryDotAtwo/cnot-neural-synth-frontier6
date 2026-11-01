"""Stable ordered IDs only; feature semantics and tensor shapes are separate."""

FAMILY_COUNTS = {
    "A": 20,
    "B": 12,
    "C": 10,
    "D": 14,
    "E": 9,
    "F": 10,
    "G": 10,
    "H": 7,
    "I": 11,
}

FEATURE_IDS = tuple(
    f"{family}{index:02d}"
    for family, count in FAMILY_COUNTS.items()
    for index in range(1, count + 1)
)

FEATURE_ID_MANIFEST_ID = "cnot_feature_ids_a01_i11_v1"

if len(FEATURE_IDS) != 103 or len(set(FEATURE_IDS)) != 103:
    raise RuntimeError("the CNOT feature identity contract must contain 103 unique IDs")
