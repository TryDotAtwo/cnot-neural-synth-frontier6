#!/usr/bin/env python3
"""Validate a public HF feature-profile candidate without exposing its contents."""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from build_training_manifest import validate_profile

REPOSITORY = "TryDotAtwo/quantum-generator"
SOURCE_REPOSITORY = "TryDotAtwo/cnot-neural-synth-frontier6"
REVISION_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/main"
TREE_PREFIX = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/"
FILE_PREFIX = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
RAW_GITHUB_PREFIX = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/"
USER_AGENT = "TheQuantumProject-ProfileMetadata/1.0"
HUB_HOSTS = {"huggingface.co", "hf.co"}
PROFILE_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".toml"}
MAX_PAGE_BYTES = 4 * 1024 * 1024
MAX_PROFILE_BYTES = 8 * 1024 * 1024
MAX_PROFILE_CANDIDATES = 32
ROW_CONTEXT_KEYS = {
    "labeled_cuts",
    "restricted_intersections",
    "rank_increment_probes",
    "block_parity_contexts",
    "matrix_boundaries",
    "paid_basis_contexts",
    "bilinear_contexts",
    "shared_relation_contexts",
    "frequency_contexts",
    "positive_packet_contexts",
    "i09_contexts",
    "selected_residual_additional_contexts",
}
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_TREE_PAGES = 20
MAX_TREE_ENTRIES = 20_000
NEXT_LINK = re.compile(r"<([^<>]+)>\s*;\s*rel\s*=\s*\"?next\"?", re.IGNORECASE)


class ProfileCheckError(RuntimeError):
    pass


def _candidate_priority(path: str) -> int:
    from pathlib import PurePosixPath

    parsed = PurePosixPath(path)
    basename = parsed.name.casefold().replace("-", "_")
    parents = {part.casefold().replace("-", "_") for part in parsed.parts[:-1]}
    if "profile" in basename:
        return 0
    if "feature" in basename and "context" in basename:
        return 1
    if any("profile" in part for part in parents):
        return 2
    if any("feature" in part and ("context" in part or "profile" in part) for part in parents):
        return 3
    return 4


def is_candidate_path(path: str) -> bool:
    if not isinstance(path, str) or "\\x00" in path:
        return False
    from pathlib import PurePosixPath

    parsed = PurePosixPath(path)
    suffix = parsed.suffix.casefold()
    basename = parsed.name.casefold().replace("-", "_")
    parents = {part.casefold().replace("-", "_") for part in parsed.parts[:-1]}
    named_profile = "profile" in basename or (
        "feature" in basename and "context" in basename
    )
    profile_directory = any("profile" in part for part in parents)
    feature_context_directory = any(
        "feature" in part and ("context" in part or "profile" in part)
        for part in parents
    )
    return suffix in PROFILE_SUFFIXES and (
        named_profile or profile_directory or feature_context_directory
    )


def _next_page_url(link_header: str | None, current_url: str) -> str | None:
    if not link_header:
        return None
    matches = NEXT_LINK.findall(link_header)
    if len(matches) > 1:
        raise ProfileCheckError("ambiguous Hub pagination")
    if not matches:
        return None
    target = urljoin(current_url, matches[0])
    parsed = urlparse(target)
    if parsed.scheme != "https" or parsed.hostname not in HUB_HOSTS:
        raise ProfileCheckError("Hub pagination left the allowed host")
    return target


def _fetch_tree_page(url: str) -> tuple[list[str], str | None]:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            if urlparse(response.geturl()).hostname not in HUB_HOSTS:
                raise ProfileCheckError("tree response host is invalid")
            payload = response.read(MAX_PAGE_BYTES + 1)
            link_header = response.headers.get("Link")
    except ProfileCheckError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError):
        raise ProfileCheckError("Hub tree request failed") from None
    if len(payload) > MAX_PAGE_BYTES:
        raise ProfileCheckError("Hub tree response is too large")
    try:
        entries = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ProfileCheckError("Hub tree response is invalid") from None
    if not isinstance(entries, list):
        raise ProfileCheckError("Hub tree response shape is invalid")
    paths = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ProfileCheckError("Hub tree entry is invalid")
        if entry.get("type") == "file":
            paths.append(entry["path"])
    return paths, _next_page_url(link_header, url)


def list_candidate_paths(revision: str, fetch_page=_fetch_tree_page) -> list[str]:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProfileCheckError("Hub revision is invalid")
    url = (
        f"{TREE_PREFIX}{quote(revision, safe='')}?"
        f"{urlencode({'recursive': 'true', 'expand': 'false'})}"
    )
    seen = set()
    total = 0
    candidates = []
    for _ in range(MAX_TREE_PAGES):
        if url in seen:
            raise ProfileCheckError("Hub pagination repeated")
        seen.add(url)
        paths, next_url = fetch_page(url)
        total += len(paths)
        if total > MAX_TREE_ENTRIES:
            raise ProfileCheckError("Hub tree entry limit exceeded")
        candidates.extend(path for path in paths if is_candidate_path(path))
        if next_url is None:
            return candidates
        url = next_url
    raise ProfileCheckError("Hub tree page limit exceeded")


def _fetch_limited(url: str, limit: int, allowed_host) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            host = urlparse(response.geturl()).hostname or ""
            if not allowed_host(host):
                raise ProfileCheckError("download redirect host is invalid")
            declared = response.headers.get("Content-Length")
            if declared and (not declared.isdigit() or int(declared) > limit):
                raise ProfileCheckError("download size limit exceeded")
            body = response.read(limit + 1)
    except ProfileCheckError:
        raise
    except HTTPError as error:
        if error.code == 404:
            raise ProfileCheckError("source file was not found") from None
        raise ProfileCheckError("download request failed") from None
    except (URLError, TimeoutError, OSError):
        raise ProfileCheckError("download request failed") from None
    if len(body) > limit:
        raise ProfileCheckError("download size limit exceeded")
    return body


def _revision() -> str:
    body = _fetch_limited(REVISION_URL, MAX_PAGE_BYTES, lambda host: host in HUB_HOSTS)
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ProfileCheckError("Hub revision response is invalid") from None
    revision = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProfileCheckError("Hub revision is invalid")
    return revision


def _candidate_url(revision: str, path: str) -> str:
    return f"{FILE_PREFIX}{quote(revision, safe='')}/{quote(path, safe='/')}"


def _source_url(revision: str, path: str) -> str:
    return f"{RAW_GITHUB_PREFIX}{quote(revision, safe='')}/{quote(path, safe='/')}"


def validate_profile_bytes(payload: bytes) -> dict:
    try:
        profile = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ProfileCheckError("candidate JSON is invalid") from None
    try:
        contexts = validate_profile(profile)
    except (ValueError, TypeError, KeyError):
        raise ProfileCheckError("candidate does not match the training profile schema") from None
    return {
        "profile": profile,
        "contexts": contexts,
        "source_revision": profile["source"]["revision"],
        "source_path": profile["source"]["path"],
    }


def _safe_download_failure(stage: str, error: ProfileCheckError) -> str:
    reason = {
        "source file was not found": "not_found",
        "download size limit exceeded": "size_limit",
        "download redirect host is invalid": "redirect_rejected",
    }.get(str(error), "request_failed")
    return f"unavailable_{stage}_{reason}"


def _allowed_hub_download_host(host: str) -> bool:
    return (
        host in HUB_HOSTS
        or host.endswith(".hf.co")
        or host.endswith(".huggingface.co")
    )


def _source_exists(
    revision: str,
    path: str,
    fetch_file=_fetch_limited,
) -> bool:
    try:
        body = fetch_file(
            _source_url(revision, path),
            MAX_SOURCE_BYTES,
            lambda host: host == "raw.githubusercontent.com",
        )
    except ProfileCheckError as error:
        if str(error) == "source file was not found":
            return False
        raise
    return bool(body)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _same_source(left: dict, right: dict) -> bool:
    return (
        left["source_revision"] == right["source_revision"]
        and left["source_path"] == right["source_path"]
    )


def _context_extends(left: dict, right: dict) -> bool:
    for key in set(left) | set(right):
        if key in ROW_CONTEXT_KEYS:
            left_rows = left.get(key, [])
            right_rows = right.get(key, [])
            if not isinstance(left_rows, list) or not isinstance(right_rows, list):
                return False
            left_counts = {}
            for row in left_rows:
                signature = _canonical_json(row)
                left_counts[signature] = left_counts.get(signature, 0) + 1
            right_counts = {}
            for row in right_rows:
                signature = _canonical_json(row)
                right_counts[signature] = right_counts.get(signature, 0) + 1
            if any(left_counts.get(signature, 0) < count
                   for signature, count in right_counts.items()):
                return False
        else:
            if key not in left or key not in right:
                if key not in left and key not in right:
                    continue
                return False
            if _canonical_json(left[key]) != _canonical_json(right[key]):
                return False
    return True


def _profile_semantic_signature(profile: dict) -> str:
    return _canonical_json(
        {
            "contexts": profile["contexts"],
            "source_revision": profile["source_revision"],
            "source_path": profile["source_path"],
        }
    )


def _select_pinned_profile(candidates: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    unique = {}
    for path, profile in candidates:
        signature = _profile_semantic_signature(profile)
        unique.setdefault(signature, (path, profile))
    profiles = list(unique.values())
    if not profiles:
        return None
    if len(profiles) == 1:
        return profiles[0]

    dominators = []
    for candidate in profiles:
        _path, profile = candidate
        if all(
            _same_source(profile, other_profile)
            and _context_extends(profile["contexts"], other_profile["contexts"])
            for _other_path, other_profile in profiles
        ):
            dominators.append(candidate)
    return dominators[0] if len(dominators) == 1 else None


def _profile_selection_status(candidates: list[tuple[str, dict]]) -> str:
    if _select_pinned_profile(candidates) is not None:
        return "schema_source_pinned"
    sources = {
        (profile["source_revision"], profile["source_path"])
        for _path, profile in candidates
    }
    if len(sources) > 1:
        return "candidate_source_ambiguous"
    return "candidate_contexts_incomparable"


def inspect_profile(
    fetch_page=_fetch_tree_page,
    fetch_file=_fetch_limited,
    source_exists=_source_exists,
    revision_getter=_revision,
) -> str:
    try:
        revision = revision_getter()
    except ProfileCheckError:
        return "unavailable_revision"

    try:
        candidates = list_candidate_paths(revision, fetch_page=fetch_page)
    except ProfileCheckError:
        return "unavailable_tree"

    json_candidates = sorted(
        {path for path in candidates if path.casefold().endswith(".json")}
    )
    if not candidates:
        return "not_found"
    if not json_candidates:
        return "candidate_unverified"

    saw_schema_valid = False
    for priority in sorted({_candidate_priority(path) for path in json_candidates}):
        tier = [
            path for path in json_candidates
            if _candidate_priority(path) == priority
        ]
        if len(tier) > MAX_PROFILE_CANDIDATES:
            return "candidate_limit_exceeded"
        pinned_profiles = []
        for path in tier:
            try:
                raw = fetch_file(
                    _candidate_url(revision, path),
                    MAX_PROFILE_BYTES,
                    _allowed_hub_download_host,
                )
            except ProfileCheckError as error:
                return _safe_download_failure("profile", error)
            try:
                profile = validate_profile_bytes(raw)
            except ProfileCheckError:
                continue
            saw_schema_valid = True
            try:
                is_pinned = source_exists(
                    profile["source_revision"], profile["source_path"]
                )
            except ProfileCheckError as error:
                return _safe_download_failure("source", error)
            if is_pinned:
                pinned_profiles.append((path, profile))
        if pinned_profiles:
            return _profile_selection_status(pinned_profiles)

    return "candidate_source_unverified" if saw_schema_valid else "candidate_schema_unverified"


if __name__ == "__main__":
    status = inspect_profile()
    if status not in {
        "candidate_ambiguous", "candidate_contexts_incomparable",
        "candidate_limit_exceeded", "candidate_schema_unverified",
        "candidate_source_ambiguous", "candidate_source_unverified",
        "candidate_unverified", "not_found",
        "schema_source_pinned", "unavailable",
        "unavailable_profile_not_found",
        "unavailable_profile_redirect_rejected",
        "unavailable_profile_request_failed",
        "unavailable_profile_size_limit",
        "unavailable_revision",
        "unavailable_tree",
        "unavailable_source_not_found",
        "unavailable_source_redirect_rejected",
        "unavailable_source_request_failed",
        "unavailable_source_size_limit",
    }:
        status = "unavailable"
    print(f"profile_validation={status}")
