"""Typed feature packets that make partial feature computation explicit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .layout import FEATURE_LAYOUT
from .schema import FEATURE_IDS, FEATURE_ID_MANIFEST_ID


class FeatureStatus(str, Enum):
    COMPUTED = "computed"
    INAPPLICABLE = "inapplicable"
    UNSUPPORTED = "unsupported"
    RESOURCE_LIMIT = "resource_limit"
    NOT_REQUESTED = "not_requested"


@dataclass(frozen=True)
class FeatureEntry:
    """One feature for a batch; unavailable rows must carry a concrete reason."""

    values: np.ndarray
    available: np.ndarray
    role: str
    exactness: str
    scope: str
    status: tuple[FeatureStatus, ...]
    reason: tuple[str, ...]
    token_valid: np.ndarray | None = None
    incidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        raw_available = np.asarray(self.available)
        if raw_available.dtype != np.dtype(np.bool_):
            raise TypeError("availability must be a boolean array")
        available = np.array(raw_available, copy=True)
        if any(not isinstance(item, str) for item in self.reason):
            raise TypeError("unavailability reasons must be strings")
        status = tuple(FeatureStatus(item) for item in self.status)
        reason = tuple(self.reason)

        if values.ndim < 2:
            raise ValueError("feature values need batch and channel dimensions")
        if values.shape[0] == 0 or values.shape[-1] == 0:
            raise ValueError("feature batch and channel dimensions must be nonempty")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
            raise TypeError("feature values must be real numeric arrays")
        if self.role not in {"node", "pair", "global", "group"}:
            raise ValueError(f"unsupported feature role: {self.role}")
        if not isinstance(self.exactness, str) or not isinstance(self.scope, str):
            raise TypeError("feature exactness and scope must be strings")
        expected_ndim = {"node": 3, "pair": 4, "global": 2, "group": 3}[self.role]
        if values.ndim != expected_ndim:
            raise ValueError(f"{self.role} values need {expected_ndim} dimensions including batch")
        if self.role == "pair" and values.shape[1] != values.shape[2]:
            raise ValueError("pair features must have matching source and target dimensions")
        if available.shape != (values.shape[0],):
            raise ValueError("availability must contain one boolean per batch row")
        if len(status) != values.shape[0] or len(reason) != values.shape[0]:
            raise ValueError("status and reason must contain one entry per batch row")
        if not np.isfinite(values).all():
            raise ValueError("feature values must be finite, including unavailable rows")

        for index, is_available in enumerate(available):
            if is_available:
                if status[index] is not FeatureStatus.COMPUTED:
                    raise ValueError("available feature rows must have computed status")
            elif status[index] is FeatureStatus.COMPUTED or not reason[index].strip():
                raise ValueError("unavailable feature rows need a non-computed status and reason")

        token_valid = None
        incidence = None
        if self.role == "group":
            if self.token_valid is None or self.incidence is None:
                raise ValueError("group features require token validity and wire incidence")
            raw_token_valid = np.asarray(self.token_valid)
            if raw_token_valid.dtype != np.dtype(np.bool_):
                raise TypeError("group token validity must be a boolean array")
            token_valid = np.array(raw_token_valid, copy=True)
            incidence = np.array(self.incidence, copy=True)
            if token_valid.shape != values.shape[:2]:
                raise ValueError("group token validity shape mismatch")
            if incidence.ndim != 4 or incidence.shape[:2] != values.shape[:2]:
                raise ValueError("group incidence must have shape [batch, groups, nodes, roles]")
            if not np.issubdtype(incidence.dtype, np.number) or not np.isfinite(incidence).all():
                raise ValueError("group incidence must be finite and numeric")
            if (incidence < 0).any():
                raise ValueError("group incidence weights must be nonnegative")
        elif self.token_valid is not None or self.incidence is not None:
            raise ValueError("only group features may carry token validity and incidence")

        values.setflags(write=False)
        available.setflags(write=False)
        if token_valid is not None:
            token_valid.setflags(write=False)
            incidence.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "token_valid", token_valid)
        object.__setattr__(self, "incidence", incidence)


class IncompleteFeaturePacket(ValueError):
    """Raised when a caller claims complete-103 coverage with missing values."""

    def __init__(self, missing: Mapping[str, tuple[str, ...]]) -> None:
        self.missing = MappingProxyType(dict(missing))
        summary = "; ".join(
            f"{feature_id}: {', '.join(reasons)}"
            for feature_id, reasons in self.missing.items()
        )
        super().__init__(f"complete-103 input rejected; unavailable features: {summary}")


@dataclass(frozen=True)
class FeaturePacket:
    """An ID-complete packet that may still report per-row unavailable values."""

    features: Mapping[str, FeatureEntry]
    state_keys: tuple[object, ...]
    alphabet_ids: tuple[str, ...]
    valid: np.ndarray
    schema_id: str
    policy_id: str
    feature_id_manifest_id: str = FEATURE_ID_MANIFEST_ID

    def __post_init__(self) -> None:
        if self.feature_id_manifest_id != FEATURE_ID_MANIFEST_ID:
            raise ValueError(f"unsupported feature ID manifest: {self.feature_id_manifest_id}")

        supplied = set(self.features)
        expected = set(FEATURE_IDS)
        missing = expected - supplied
        extra = supplied - expected
        if missing or extra:
            raise ValueError(
                "packet must declare the exact 103 feature IDs; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        ordered = {feature_id: self.features[feature_id] for feature_id in FEATURE_IDS}
        if any(not isinstance(entry, FeatureEntry) for entry in ordered.values()):
            raise TypeError("each feature must be a FeatureEntry")
        batch_sizes = {entry.values.shape[0] for entry in ordered.values()}
        if len(batch_sizes) != 1:
            raise ValueError("all 103 features must describe the same batch")
        batch_size = next(iter(batch_sizes))
        state_keys = tuple(self.state_keys)
        alphabet_ids = tuple(self.alphabet_ids)
        if len(state_keys) != batch_size or len(alphabet_ids) != batch_size:
            raise ValueError("state and alphabet identities must match the feature batch")
        if any(not isinstance(item, str) or not item for item in alphabet_ids):
            raise ValueError("alphabet identities must be nonempty strings")
        try:
            for item in state_keys:
                hash(item)
        except TypeError as exc:
            raise ValueError("state identities must be hashable") from exc
        if not isinstance(self.schema_id, str) or not self.schema_id:
            raise ValueError("legacy feature schema identity is required")
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("feature policy identity is required")
        valid = np.asarray(self.valid)
        if valid.dtype != np.dtype(np.bool_) or valid.ndim != 2 or valid.shape[0] != batch_size:
            raise ValueError("node validity must be a boolean [batch, nodes] array")
        valid = np.array(valid, copy=True)
        node_count = valid.shape[1]
        if node_count == 0:
            raise ValueError("feature packet needs at least one node position")
        for feature_id, entry in ordered.items():
            expected_role, expected_channels = FEATURE_LAYOUT[feature_id]
            if entry.role != expected_role or entry.values.shape[-1] != expected_channels:
                raise ValueError(
                    f"feature layout mismatch for {feature_id}: "
                    f"got ({entry.role}, {entry.values.shape[-1]}), "
                    f"expected ({expected_role}, {expected_channels})"
                )
            if entry.role == "node" and entry.values.shape[1] != node_count:
                raise ValueError(f"node feature width mismatch for {feature_id}")
            if entry.role == "pair" and entry.values.shape[1] != node_count:
                raise ValueError(f"pair feature width mismatch for {feature_id}")
            if entry.role == "group" and entry.incidence.shape[2] != node_count:
                raise ValueError(f"group incidence node width mismatch for {feature_id}")
        valid.setflags(write=False)
        object.__setattr__(self, "state_keys", state_keys)
        object.__setattr__(self, "alphabet_ids", alphabet_ids)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "features", MappingProxyType(ordered))

    @property
    def batch_size(self) -> int:
        return len(self.state_keys)

    @property
    def available_ids(self) -> tuple[str, ...]:
        return tuple(
            feature_id
            for feature_id, entry in self.features.items()
            if bool(entry.available.all())
        )

    @property
    def complete_103(self) -> bool:
        return len(self.available_ids) == len(FEATURE_IDS)

    def require_complete_103(self) -> "FeaturePacket":
        missing = {
            feature_id: tuple(
                f"row {row}: {entry.status[row].value} ({entry.reason[row]})"
                for row, available in enumerate(entry.available)
                if not available
            )
            for feature_id, entry in self.features.items()
            if not entry.available.all()
        }
        if missing:
            raise IncompleteFeaturePacket(missing)
        return self

    def coverage(self) -> dict[str, object]:
        per_feature = {
            feature_id: {
                "available_rows": int(entry.available.sum()),
                "batch_rows": self.batch_size,
                "status_counts": {
                    status.value: sum(item is status for item in entry.status)
                    for status in FeatureStatus
                    if any(item is status for item in entry.status)
                },
                "reasons": tuple(sorted({
                    entry.reason[row]
                    for row, available in enumerate(entry.available)
                    if not available
                })),
            }
            for feature_id, entry in self.features.items()
        }
        return {
            "feature_id_manifest_id": self.feature_id_manifest_id,
            "selected_feature_count": len(self.features),
            "fully_available_feature_count": len(self.available_ids),
            "batch_size": self.batch_size,
            "complete_103": self.complete_103,
            "per_feature": per_feature,
        }
