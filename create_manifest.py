#!/usr/bin/env python3
"""Create SHA-256 manifest for source, raw data and outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.sha256"
EXCLUDED_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", "reproduction_output",
    "smoke_output", "smoke_analysis", ".mypy_cache", ".ruff_cache",
}
EXCLUDED_FILES = {
    "manifest.sha256",
    "data/DATA_VALIDATION.json",
    # Citation metadata is edited after the Zenodo DOI is issued.
    "CITATION.cff",
    ".zenodo.json",
}
GENERATED_PREFIXES = set()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    name = relative.as_posix()
    if any(name.startswith(prefix) for prefix in GENERATED_PREFIXES):
        return False
    return name not in EXCLUDED_FILES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
