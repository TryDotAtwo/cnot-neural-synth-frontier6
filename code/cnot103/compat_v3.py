"""Narrow adapter from the legacy research_v3 packet into the strict contract."""

from __future__ import annotations

from collections.abc import Mapping

from .contracts import FeatureEntry, FeaturePacket, FeatureStatus, IncompleteFeaturePacket
from .schema import FEATURE_IDS


_STATUS_MAP = {
    "available": FeatureStatus.COMPUTED,
    "inapplicable": FeatureStatus.INAPPLICABLE,
    "unsupported": FeatureStatus.UNSUPPORTED,
    "resource_limit": FeatureStatus.RESOURCE_LIMIT,
    "not_requested": FeatureStatus.NOT_REQUESTED,
}


def require_research_v3_complete(packet: object) -> object:
    """Check legacy coverage without copying large feature tensors."""

    features = getattr(packet, "features", None)
    audit = getattr(packet, "audit", None)
    if not isinstance(features, Mapping) or not isinstance(audit, Mapping):
        raise TypeError("expected a research_v3 FeaturePacket with feature audit")
    if set(features) != set(FEATURE_IDS):
        raise ValueError("research_v3 packet does not contain exactly the 103 feature IDs")
    per_feature = audit.get("per_feature")
    if not isinstance(per_feature, Mapping):
        raise ValueError("research_v3 packet has no per-feature availability audit")

    missing: dict[str, tuple[str, ...]] = {}
    for feature_id in FEATURE_IDS:
        source = features[feature_id]
        source_audit = per_feature.get(feature_id)
        if source_audit is None:
            raise ValueError(f"missing availability audit for {feature_id}")
        available = getattr(source, "available", None)
        statuses = source_audit.get("availability_by_state")
        reasons = source_audit.get("availability_reason_by_state")
        if available is None or statuses is None or reasons is None:
            raise ValueError(f"incomplete per-state feature evidence for {feature_id}")
        if len(available.shape) != 1 or available.dtype.name != "bool":
            raise ValueError(f"invalid availability mask for {feature_id}")
        if len(statuses) != len(available) or len(reasons) != len(available):
            raise ValueError(f"availability audit length mismatch for {feature_id}")

        unavailable_rows = []
        for row, (is_available, raw_status, raw_reason) in enumerate(zip(available, statuses, reasons)):
            status = _STATUS_MAP.get(str(raw_status))
            if status is None:
                raise ValueError(f"unknown availability status for {feature_id} row {row}: {raw_status}")
            if bool(is_available) != (status is FeatureStatus.COMPUTED):
                raise ValueError(f"status/mask disagreement for {feature_id} row {row}")
            if not is_available:
                unavailable_rows.append(f"row {row}: {status.value} ({raw_reason})")
        if unavailable_rows:
            missing[feature_id] = tuple(unavailable_rows)

    if missing:
        raise IncompleteFeaturePacket(missing)
    return packet


def from_research_v3(packet: object) -> FeaturePacket:
    """Convert a v3 packet while retaining per-row missing-value reasons.

    The legacy packet keeps detailed availability in ``audit['per_feature']``.
    This adapter refuses packets that do not provide that evidence; it never
    treats a feature ID in the config as proof that a value was computed.
    """

    features = getattr(packet, "features", None)
    audit = getattr(packet, "audit", None)
    if not isinstance(features, Mapping) or not isinstance(audit, Mapping):
        raise TypeError("expected a research_v3 FeaturePacket with feature audit")
    if set(features) != set(FEATURE_IDS):
        raise ValueError("research_v3 packet does not contain exactly the 103 feature IDs")

    per_feature = audit.get("per_feature")
    if not isinstance(per_feature, Mapping):
        raise ValueError("research_v3 packet has no per-feature availability audit")

    converted: dict[str, FeatureEntry] = {}
    for feature_id in FEATURE_IDS:
        source = features[feature_id]
        source_audit = per_feature.get(feature_id)
        if source_audit is None:
            raise ValueError(f"missing availability audit for {feature_id}")

        available = getattr(source, "available", None)
        values = getattr(source, "values", None)
        statuses = source_audit.get("availability_by_state")
        reasons = source_audit.get("availability_reason_by_state")
        if available is None or values is None or statuses is None or reasons is None:
            raise ValueError(f"incomplete per-state feature evidence for {feature_id}")
        if len(statuses) != len(available) or len(reasons) != len(available):
            raise ValueError(f"availability audit length mismatch for {feature_id}")

        normalized_status = []
        for row, raw_status in enumerate(statuses):
            status = _STATUS_MAP.get(str(raw_status))
            if status is None:
                raise ValueError(f"unknown availability status for {feature_id} row {row}: {raw_status}")
            if bool(available[row]) != (status is FeatureStatus.COMPUTED):
                raise ValueError(f"status/mask disagreement for {feature_id} row {row}")
            normalized_status.append(status)

        converted[feature_id] = FeatureEntry(
            values=values,
            available=available,
            role=source.role,
            exactness=source.exactness,
            scope=source.scope,
            status=tuple(normalized_status),
            reason=tuple(str(reason) for reason in reasons),
            token_valid=source.token_valid,
            incidence=getattr(packet, "group_incidence", {}).get(feature_id),
        )

    return FeaturePacket(
        features=converted,
        state_keys=tuple(packet.state_keys),
        alphabet_ids=tuple(packet.alphabet_ids),
        valid=packet.valid,
        schema_id=packet.schema_id,
        policy_id=packet.policy_id,
    )
