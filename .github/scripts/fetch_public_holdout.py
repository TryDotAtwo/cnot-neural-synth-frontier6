#!/usr/bin/env python3
"""Fetch the immutable public holdout release onto a remote run disk."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REPOSITORY = "TryDotAtwo/quantum-generator"
FILENAME = "full_shared_native_holdout_20261001.sqlite"
API_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/main"
MAX_BYTES = 1 << 30
USER_AGENT = "TheQuantumProject-Remote-Data-Preparation/1.0"


class FetchError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_revision() -> str:
    request = Request(API_URL, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read(2 * 1024 * 1024))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
        raise FetchError("could not query the public holdout revision") from None
    revision = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise FetchError("public holdout revision response is invalid")
    return revision


def download(path: pathlib.Path, revision: str) -> None:
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{FILENAME}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"})
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FetchError("preserving an existing holdout or partial file")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(request, timeout=60) as response:
            host = urlparse(response.geturl()).hostname or ""
            if host not in {"hf.co", "huggingface.co"} and not host.endswith((".hf.co", ".huggingface.co")):
                raise FetchError("holdout download redirected outside Hugging Face")
            length = response.headers.get("Content-Length")
            if length and (not length.isdigit() or int(length) > MAX_BYTES):
                raise FetchError("holdout download exceeds the configured size limit")
            total = 0
            prefix = bytearray()
            with temporary.open("xb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    if len(prefix) < 16:
                        prefix.extend(block[: 16 - len(prefix)])
                    total += len(block)
                    if total > MAX_BYTES:
                        raise FetchError("holdout download exceeds the configured size limit")
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
    except FetchError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError):
        raise FetchError("holdout download failed") from None
    if total == 0 or bytes(prefix) != b"SQLite format 3\x00":
        raise FetchError("holdout download is empty or not a SQLite database")
    os.rename(temporary, path)


def write_receipt(path: pathlib.Path, receipt: dict) -> None:
    if path.exists():
        raise FetchError("preserving an existing holdout receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FetchError("preserving an existing partial receipt")
    raw = json.dumps(receipt, separators=(",", ":")).encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)


def run(root: pathlib.Path) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise FetchError("remote experiment directory does not exist")
    input_dir = root / "input"
    path = input_dir / FILENAME
    receipt_path = input_dir / "holdout-source.receipt.json"
    if any(item.exists() for item in (
        path,
        path.with_name(path.name + ".partial"),
        receipt_path,
        receipt_path.with_name(receipt_path.name + ".partial"),
    )):
        raise FetchError("preserving an existing input file or receipt")
    revision = get_revision()
    download(path, revision)
    digest = sha256_file(path)
    write_receipt(receipt_path, {
        "schema": "cnot103_public_holdout_source_v1",
        "repository": REPOSITORY,
        "filename": FILENAME,
        "revision": revision,
        "sha256": digest,
        "bytes": path.stat().st_size,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    try:
        run(pathlib.Path(args.experiment_root))
        print("holdout_download=available")
        return 0
    except Exception as error:
        try:
            root = pathlib.Path(args.experiment_root).resolve()
            root.mkdir(parents=True, exist_ok=True)
            with (root / "holdout-fetch.private.log").open("a", encoding="utf-8") as stream:
                stream.write(f"{type(error).__name__}: {error}\n")
        except OSError:
            pass
        print("holdout_download=failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
