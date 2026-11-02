"""Stable role/channel layout recorded by the all-103 runtime packet."""

from types import MappingProxyType

from .schema import FEATURE_IDS


def _series(family: str, start: int, stop: int) -> tuple[str, ...]:
    return tuple(f"{family}{index:02d}" for index in range(start, stop + 1))


_layout: dict[str, tuple[str, int]] = {}


def _put(role: str, channels: int, *feature_ids: str) -> None:
    for feature_id in feature_ids:
        if feature_id in _layout:
            raise RuntimeError(f"duplicate layout entry: {feature_id}")
        _layout[feature_id] = (role, channels)


_put("pair", 1, "A01", "A02", "A04", *_series("A", 13, 16))
_put("pair", 2, "A03", *_series("A", 17, 20))
_put("node", 2, "A05", "A12")
_put("node", 1, *_series("A", 6, 11))

_put("global", 1, "B01")
_put("node", 1, "B02", "B03", "B05", "B09")
_put("node", 3, "B04", "B11")
_put("pair", 1, "B06", "B07")
_put("pair", 2, "B08")
_put("node", 2, "B10")
_put("global", 16, "B12")

_put("global", 1, "C01", "C05", "C08")
_put("group", 1, "C02", "C03", "C04", "C06", "C07", "C09", "C10")

_put("group", 1, "D01", "D02", "D03", "D04", "D06", "D08", "D09", "D10")
_put("node", 2, "D05")
_put("group", 5, "D07")
_put("group", 2, "D11", "D12")
_put("group", 12, "D13")
_put("pair", 4, "D14")

_put("global", 1, "E01", "E02", "E03", "E07")
_put("pair", 2, "E04")
_put("global", 4, "E05")
_put("group", 1, "E06")
_put("global", 2, "E08")
_put("global", 5, "E09")

_put("global", 2, "F01", "F05")
_put("global", 1, "F02")
_put("global", 5, "F03", "F10")
_put("pair", 3, "F04")
_put("group", 6, "F06")
_put("global", 3, "F07", "F08")
_put("global", 4, "F09")

_put("node", 28, "G01", "G02")
_put("node", 16, "G03")
_put("node", 3, "G04")
_put("pair", 4, "G05")
_put("pair", 28, "G06")
_put("node", 1, "G07")
_put("pair", 5, "G08")
_put("pair", 2, "G09")
_put("pair", 7, "G10")

_put("pair", 1, *_series("H", 1, 4))
_put("pair", 4, "H05")
_put("pair", 15, "H06")
# The pinned feature schema leaves H07 unimplemented. Its group/1 entry is
# only the schema placeholder; it is not an executable producer contract.
_put("group", 1, "H07")

_put("group", 2, "I01", "I03")
_put("group", 3, "I02")
_put("global", 4, "I04", "I11")
_put("global", 2, "I05", "I07")
_put("group", 5, "I06")
_put("group", 4, "I08")
_put("group", 132, "I09")
_put("global", 8, "I10")

if set(_layout) != set(FEATURE_IDS):
    raise RuntimeError(
        "feature layout mismatch; "
        f"missing={sorted(set(FEATURE_IDS) - set(_layout))}, "
        f"extra={sorted(set(_layout) - set(FEATURE_IDS))}"
    )

FEATURE_LAYOUT_ID = "runtime_packet_layout_v23"
FEATURE_LAYOUT = MappingProxyType(_layout)
