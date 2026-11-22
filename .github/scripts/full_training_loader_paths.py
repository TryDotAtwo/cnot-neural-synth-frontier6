"""Run the Rust frozen-data validator only for changes in its actual inputs."""
from __future__ import annotations

import argparse
import re
import subprocess
from typing import Iterable

_SHA = re.compile(r"^[0-9a-f]{40}$")
_DOCUMENT_SUFFIXES = {".md", ".rst", ".txt"}


def loader_validation_required_for_paths(paths: Iterable[str]) -> bool:
    """Return whether a changed path can affect the compiled native loader."""
    for raw_path in paths:
        if not isinstance(raw_path, str):
            continue
        path = raw_path.replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        if path in {"Cargo.toml", "Cargo.lock", "native/Cargo.toml", "native/Cargo.lock"}:
            return True
        if path.startswith(".cargo/"):
            return True
        if path.startswith("native/"):
            leaf = path.rsplit("/", 1)[-1]
            suffix = "." + leaf.rsplit(".", 1)[-1] if "." in leaf else ""
            if suffix.lower() not in _DOCUMENT_SUFFIXES:
                return True
    return False


def loader_validation_required_for_range(before: str, after: str) -> bool:
    """Fail closed to a full loader check when a push range cannot be inspected."""
    if not isinstance(before, str) or not isinstance(after, str):
        return True
    if _SHA.fullmatch(before) is None or _SHA.fullmatch(after) is None:
        return True
    if before == "0" * 40:
        return True
    try:
        completed = subprocess.run(
            ["git", "diff", "--name-only", before, after],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return loader_validation_required_for_paths(completed.stdout.splitlines())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    args = parser.parse_args()
    print("true" if loader_validation_required_for_range(args.before, args.after) else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
