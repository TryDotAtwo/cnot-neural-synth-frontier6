#!/usr/bin/env python3
"""Metadata-only search for a feature-context profile in the public holdout repo.

This queries Hugging Face's repository tree API. It never downloads file contents
and reports only a small status enum; file names and API payloads stay private.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

REPOSITORY = "TryDotAtwo/quantum-generator"
REVISION_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/main"
TREE_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/"
USER_AGENT = "TheQuantumProject-ProfileMetadata/1.0"
ALLOWED_HOSTS = {"huggingface.co", "hf.co"}
PROFILE_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".toml"}
MAX_PAGE_BYTES = 4 * 1024 * 1024
MAX_TREE_PAGES = 20
MAX_TREE_ENTRIES = 20_000
NEXT_LINK = re.compile(r"<([^<>]+)>\s*;\s*rel\s*=\s*\"?next\"?", re.IGNORECASE)


class MetadataQueryError(RuntimeError):
    pass


def is_profile_candidate(path: str) -> bool:
    if not isinstance(path, str) or "\x00" in path:
        return False
    candidate = PurePosixPath(path)
    return (
        candidate.suffix.casefold() in PROFILE_SUFFIXES
        and ("context" in path.casefold() or "profile" in path.casefold())
    )


def next_page_url(link_header: str | None, current_url: str) -> str | None:
    if not link_header:
        return None
    matches = NEXT_LINK.findall(link_header)
    if len(matches) > 1:
        raise MetadataQueryError("ambiguous pagination link")
    if not matches:
        return None
    target = urljoin(current_url, matches[0])
    parsed = urlparse(target)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise MetadataQueryError("pagination link left the public Hub")
    return target


def fetch_tree_page(url: str) -> tuple[list[str], str | None]:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            host = urlparse(response.geturl()).hostname
            if host not in ALLOWED_HOSTS:
                raise MetadataQueryError("tree response host is not allowed")
            payload = response.read(MAX_PAGE_BYTES + 1)
            link_header = response.headers.get("Link")
    except MetadataQueryError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError):
        raise MetadataQueryError("tree metadata query failed") from None
    if len(payload) > MAX_PAGE_BYTES:
        raise MetadataQueryError("tree metadata response is too large")
    try:
        entries = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise MetadataQueryError("tree metadata response is invalid") from None
    if not isinstance(entries, list):
        raise MetadataQueryError("tree metadata shape is invalid")
    paths = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise MetadataQueryError("tree metadata entry is invalid")
        if entry.get("type") == "file":
            paths.append(entry["path"])
    return paths, next_page_url(link_header, url)


def find_profile_candidate(revision: str, fetch_page=fetch_tree_page) -> str:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise MetadataQueryError("repository revision is invalid")
    query = urlencode({"recursive": "true", "expand": "false"})
    url = f"{TREE_URL}{quote(revision, safe='')}?{query}"
    seen = set()
    total = 0
    for _ in range(MAX_TREE_PAGES):
        if url in seen:
            raise MetadataQueryError("tree pagination repeated")
        seen.add(url)
        paths, next_url = fetch_page(url)
        total += len(paths)
        if total > MAX_TREE_ENTRIES:
            raise MetadataQueryError("tree metadata entry limit exceeded")
        if any(is_profile_candidate(path) for path in paths):
            return "candidate_found"
        if next_url is None:
            return "not_found"
        url = next_url
    raise MetadataQueryError("tree pagination limit exceeded")


def query_status() -> str:
    request = Request(
        REVISION_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            host = urlparse(response.geturl()).hostname
            if host not in ALLOWED_HOSTS:
                return "unavailable"
            payload = json.loads(response.read(MAX_PAGE_BYTES + 1))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return "unavailable"
    if not isinstance(payload, dict):
        return "unavailable"
    revision = payload.get("sha")
    try:
        return find_profile_candidate(revision)
    except MetadataQueryError:
        return "unavailable"


def main() -> int:
    checked = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    status = query_status()
    outcome = "completed" if status != "unavailable" else "unavailable"
    print(f"checked_at_utc={checked}")
    print(f"query_outcome={outcome}")
    print(f"feature_profile_candidate={status}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as summary:
                summary.write(
                    "## Public feature-profile metadata\\n"
                    f"- checked_at_utc: {checked}\\n"
                    f"- query_outcome: {outcome}\\n"
                    f"- feature_profile_candidate: {status}\\n"
                )
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
