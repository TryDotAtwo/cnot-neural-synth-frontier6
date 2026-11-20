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
MAX_PROFILE_BYTES = 1 * 1024 * 1024
MAX_PROFILE_CANDIDATES = 32
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_TREE_PAGES = 20
MAX_TREE_ENTRIES = 20_000
NEXT_LINK = re.compile(r"<([^<>]+)>\s*;\s*rel\s*=\s*\"?next\"?", re.IGNORECASE)


class ProfileCheckError(RuntimeError):
    pass


def is_candidate_path(path: str) -> bool:
    if not isinstance(path, str) or "\x00" in path:
        return False
    from pathlib import PurePosixPath

    suffix = PurePosixPath(path).suffix.casefold()
    return suffix in PROFILE_SUFFIXES and (
        "context" in path.casefold() or "profile" in path.casefold()
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


def inspect_profile(
    fetch_page=_fetch_tree_page,
    fetch_file=_fetch_limited,
    source_exists=_source_exists,
    revision_getter=_revision,
) -> str:
    try:
        revision = revision_getter()
        candidates = list_candidate_paths(revision, fetch_page=fetch_page)
        json_candidates = sorted(
            {path for path in candidates if path.casefold().endswith(".json")}
        )
        if not candidates:
            return "not_found"
        if not json_candidates:
            return "candidate_unverified"
        if len(json_candidates) > MAX_PROFILE_CANDIDATES:
            return "candidate_ambiguous"

        pinned_signatures = set()
        for path in json_candidates:
            raw = fetch_file(
                _candidate_url(revision, path),
                MAX_PROFILE_BYTES,
                _allowed_hub_download_host,
            )
            try:
                profile = validate_profile_bytes(raw)
            except ProfileCheckError:
                continue
            if source_exists(profile["source_revision"], profile["source_path"]):
                signature = json.dumps(
                    {
                        "contexts": profile["contexts"],
                        "source_revision": profile["source_revision"],
                        "source_path": profile["source_path"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                pinned_signatures.add(signature)
                if len(pinned_signatures) > 1:
                    return "candidate_ambiguous"

        return "schema_source_pinned" if pinned_signatures else "candidate_unverified"
    except ProfileCheckError:
        return "unavailable"


if __name__ == "__main__":
    status = inspect_profile()
    if status not in {
        "candidate_ambiguous", "candidate_unverified", "not_found",
        "schema_source_pinned", "unavailable",
    }:
        status = "unavailable"
    print(f"profile_validation={status}")
