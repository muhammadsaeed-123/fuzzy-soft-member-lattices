#!/usr/bin/env python3
"""Verify every file listed in manifest.sha256."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            digest, name = raw.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed manifest line {line_number}: {raw!r}") from exc
        if len(digest) != 64:
            raise ValueError(f"Invalid SHA-256 on manifest line {line_number}")
        entries.append((digest.lower(), name))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if not manifest.exists():
        print(f"Manifest not found: {manifest}", file=sys.stderr)
        return 1
    failures: list[str] = []
    entries = parse_manifest(manifest)
    for expected, name in entries:
        path = ROOT / name
        if not path.exists():
            failures.append(f"missing: {name}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"checksum mismatch: {name}")
    if failures:
        print("Manifest verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"All {len(entries)} manifest entries verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
