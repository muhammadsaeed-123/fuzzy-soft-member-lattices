#!/usr/bin/env python3
"""Optional recovery downloader for the canonical DataCo transaction input.

The input is already included in the package. Network retrieval is not needed
for the declared experiment. Downloads must pass the canonical checksum and
structural validation before use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

DATASET_ID = "8gx2fvg2k6"
DATASET_VERSION = 5
FILENAME = "DataCoSupplyChainDataset.csv"
EXPECTED_DATA_ROWS = 180_519
EXPECTED_SIZE = 95_910_149
# Public mirrors report these values for the Version-5 transaction file.
EXPECTED_SHA256 = "fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6"
EXPECTED_MD5 = "354eb3cd362619ef120ca2201337ae92"
REQUIRED_COLUMNS = {
    "Order Id",
    "order date (DateOrders)",
    "Market",
    "Department Name",
    "Shipping Mode",
    "Delivery Status",
    "Order Status",
    "Type",
    "Customer Id",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Sales",
    "Order Item Discount",
    "Order Item Total",
    "Order Profit Per Order",
    "Order Item Quantity",
    "Order Item Id",
}

MIRRORS = [
    (
        "Zenodo preservation mirror",
        "https://zenodo.org/records/18835167/files/DataCoSupplyChainDataset.csv?download=1",
    ),
    (
        "Hugging Face mirror",
        "https://huggingface.co/idoyaaran/dataco-supply-chain-model/resolve/main/DataCoSupplyChainDataset.csv?download=true",
    ),
    (
        "GitHub mirror",
        "https://raw.githubusercontent.com/ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS/main/DataCoSupplyChainDataset.csv",
    ),
]


def _request(url: str, *, accept: str | None = None) -> urllib.request.Request:
    headers = {
        "User-Agent": "FODM-fuzzy-soft-lattice-reproducibility/1.0",
        "Accept-Encoding": "identity",
    }
    if accept:
        headers["Accept"] = accept
    return urllib.request.Request(url, headers=headers)


def _json_get(url: str) -> object:
    with urllib.request.urlopen(_request(url, accept="application/json"), timeout=90) as response:
        return json.load(response)


def mendeley_download_url() -> str | None:
    """Resolve the public file URL using the Mendeley Data API."""
    candidates = [
        f"https://api.data.mendeley.com/datasets/publics/{DATASET_ID}/files?version={DATASET_VERSION}",
        f"https://api.data.mendeley.com/datasets/{DATASET_ID}/files?version={DATASET_VERSION}",
    ]
    payload: object | None = None
    for endpoint in candidates:
        try:
            payload = _json_get(endpoint)
            break
        except Exception as exc:  # network/API differences are handled by mirrors
            print(f"Mendeley metadata endpoint unavailable: {endpoint} ({exc})", file=sys.stderr)
    if payload is None:
        return None

    if isinstance(payload, dict):
        records = payload.get("records") or payload.get("results") or payload.get("files") or []
    else:
        records = payload
    if not isinstance(records, list):
        return None

    selected: dict | None = None
    for record in records:
        if not isinstance(record, dict):
            continue
        name = record.get("filename") or record.get("name") or record.get("file_name")
        if name == FILENAME:
            selected = record
            break
    if selected is None:
        return None

    # Some API versions expose a ready-made download link.
    for key in ("download_url", "downloadUrl", "url"):
        value = selected.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value

    file_id = selected.get("id") or selected.get("file_id") or selected.get("uuid")
    if not file_id:
        return None
    return (
        f"https://api.data.mendeley.com/datasets/{DATASET_ID}/files/"
        f"{file_id}/file_downloaded?version={DATASET_VERSION}"
    )


def _human_size(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(_request(url), timeout=180) as response, tmp_path.open("wb") as out:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            read = 0
            last_print = 0.0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                read += len(chunk)
                now = time.monotonic()
                if now - last_print > 1.0:
                    if total:
                        print(
                            f"  downloaded {_human_size(read)} / {_human_size(total)} "
                            f"({100.0 * read / total:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(f"  downloaded {_human_size(read)}", flush=True)
                    last_print = now
        os.replace(tmp_path, destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def file_hash(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def structural_validation(path: Path) -> tuple[int, set[str]]:
    with path.open("r", encoding="latin1", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Downloaded file is empty") from exc
        row_count = sum(1 for _ in reader)
    columns = set(header)
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"Downloaded file is missing required columns: {sorted(missing)}")
    if row_count != EXPECTED_DATA_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_DATA_ROWS:,} data rows but found {row_count:,}. "
            "This is not the transaction file used in the article."
        )
    return row_count, columns


def validate(path: Path, *, strict_hash: bool = False) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    sha256 = file_hash(path, "sha256")
    md5 = file_hash(path, "md5")
    rows, columns = structural_validation(path)
    hash_match = sha256 == EXPECTED_SHA256 or md5 == EXPECTED_MD5
    size_match = size == EXPECTED_SIZE
    if strict_hash and not hash_match:
        raise ValueError(
            "The file has the correct structure but does not match either published mirror checksum. "
            f"SHA-256={sha256}; MD5={md5}."
        )
    return {
        "file": str(path),
        "bytes": size,
        "expected_bytes": EXPECTED_SIZE,
        "size_match": size_match,
        "sha256": sha256,
        "expected_sha256": EXPECTED_SHA256,
        "md5": md5,
        "expected_md5": EXPECTED_MD5,
        "checksum_match": hash_match,
        "data_rows": rows,
        "columns": len(columns),
        "dataset_doi": "10.17632/8gx2fvg2k6.5",
    }


def write_validation(metadata: dict[str, object], destination: Path) -> None:
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def candidate_urls() -> Iterable[tuple[str, str]]:
    official = mendeley_download_url()
    if official:
        yield "Mendeley Data Version 5", official
    yield from MIRRORS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_path = Path(__file__).resolve().parents[1] / "data" / FILENAME
    parser.add_argument("--output", type=Path, default=default_path, help="Destination CSV path")
    parser.add_argument("--force", action="store_true", help="Replace an existing file")
    parser.add_argument(
        "--strict-hash",
        action="store_true",
        help="Require the downloaded bytes to match a published mirror checksum",
    )
    parser.add_argument(
        "--url",
        help="Use one explicit download URL instead of the official API and fallback mirrors",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    validation_path = output.with_name("DATA_VALIDATION.json")
    if output.exists() and not args.force:
        print(f"Validating existing file: {output}")
        metadata = validate(output, strict_hash=args.strict_hash)
        write_validation(metadata, validation_path)
        print(json.dumps(metadata, indent=2))
        return 0

    sources = [("user-specified URL", args.url)] if args.url else list(candidate_urls())
    errors: list[str] = []
    for label, url in sources:
        if not url:
            continue
        print(f"Trying {label}: {url}")
        try:
            download(url, output)
            metadata = validate(output, strict_hash=args.strict_hash)
            metadata["download_source"] = label
            metadata["download_url"] = url
            write_validation(metadata, validation_path)
            print("Data file downloaded and validated.")
            print(json.dumps(metadata, indent=2))
            return 0
        except Exception as exc:
            output.unlink(missing_ok=True)
            message = f"{label}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print(f"  failed: {message}", file=sys.stderr)

    print("\nNo automated source succeeded.", file=sys.stderr)
    print(
        "Download DataCoSupplyChainDataset.csv from DOI 10.17632/8gx2fvg2k6.5, "
        f"place it at {output}, then run this command again without --force.",
        file=sys.stderr,
    )
    if errors:
        print("Attempt log:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
