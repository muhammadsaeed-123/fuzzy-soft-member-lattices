from __future__ import annotations

import csv
from pathlib import Path

from download_data import EXPECTED_DATA_ROWS, REQUIRED_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def test_description_file_has_expected_core_fields() -> None:
    path = ROOT / "data" / "DescriptionDataCoSupplyChain.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # The unchanged source descriptor pads several field names with spaces.
    names = {row["FIELDS"].strip() for row in rows}
    assert REQUIRED_COLUMNS <= names
    assert EXPECTED_DATA_ROWS == 180_519


def test_required_fields_match_the_actual_transaction_header() -> None:
    path = ROOT / "data" / "DataCoSupplyChainDataset.csv"
    with path.open(encoding="latin-1", newline="") as handle:
        header = next(csv.reader(handle))
    assert REQUIRED_COLUMNS <= set(header)
    assert len(header) == 53
