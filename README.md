# Online Resource 2 — Reproducibility repository

**Complete Fuzzy-Soft Member Lattices: Isomorphic and Pullback Collaboration with Exact Path Optimization**

This repository contains the transaction input, analysis code, exact mathematical examples, native TikZ figures, numerical tables and validation material supporting the article and Online Resource 1.

**Authors:** Amna Wasim, Muhammad Saeed (corresponding author, muhammad.saeed@umt.edu.pk) and Muhammad Haris Saeed — University of Management and Technology, Lahore, Pakistan.

**How to cite:** use the *Cite this repository* button on the GitHub page (generated from `CITATION.cff`) or the citation shown on the Zenodo record, which carries the permanent DOI of each archived release.

**Large file:** `data/DataCoSupplyChainDataset.csv` is about 96 MB. If it is missing from your copy, run `python code/download_data.py`; the script retrieves the Version 5 file and checks its SHA-256 before use.

## Environment and prerequisites

The validation environment used Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, Matplotlib 3.10.8 and pytest 9.0.2. Install the pinned Python dependencies with `python -m pip install -r requirements.txt`. A TeX distribution containing pdfLaTeX, TikZ, standalone and Latin Modern is required to compile the mathematical figures. Poppler commands `pdftoppm` and `pdftops` must be on PATH for PNG/EPS export. Numerical calculations do not require network access because the input CSV is included.

## Complete reproduction

From this repository directory:

```sh
python verify_manifest.py
python -m pytest -q
python code/reproduce.py --out reproduction_output
```

The complete run uses 2,000 customer-cluster resamples for held-out associations and exploratory contrasts, 1,000 resamples for support transport, 500 full-pipeline DataCo refits and 200 synthetic refits. DataCo seed: 20260819; synthetic seed: 20260825. It regenerates every reported numerical CSV, PDF/EPS/PNG figure and numerical LaTeX table. The output directory must be empty; no existing analysis is silently deleted.

For a faster mathematical and figure check using the included full-run tables:

```sh
python code/reproduce.py --smoke --out smoke_output
```

This mode does not rerun raw-data resampling. The shell, PowerShell and batch wrappers invoke the same Python entry point.

## Contents

- `data/`: canonical transaction CSV, variable descriptions, attribution and checksum.
- `code/`, `tests/`, `config/`: executable methods, regression tests and figure-number mapping.
- `figure_source/`: six editable standalone mathematical TikZ sources.
- `outputs/reference/`: complete DataCo run, fitted transform specification and all reference tables.
- `outputs/synthetic/`: seeded simulation, refits, comparisons, recurrence and timing outputs.
- `outputs/theory/`: exact rational example audits and rendered diagrams.
- `outputs/publication_figures/`: article Figures 1–11 and supplementary Figures S1–S18 in PDF, EPS and PNG.
- `outputs/manuscript_tables/`: native LaTeX numerical table fragments for the article and Online Resource 1.
- `docs/`: reproduction protocol, software provenance and known limitations.
- `validation/`: numerical reproduction and package verification records.

## Interpretation

The supported DataCo order has 181 nodes, 485 covers and 24,870 admissible source-to-maximal paths. Its unique ten-node optimizer has mean 0.5179137537919203. A separate internal-bound audit, rather than ambient nonclosure alone, establishes that both support-filtered orders are not lattices. Only seven of 500 DataCo refits recover the entire path. The common-cohort associations on 976 holdout orders have intervals crossing zero. Exact fixed-size subsets use distance followed by ascending order identifier, an outcome-blind rule; equal subset sizes do not mean identical observations. These are descriptive profiles, not causal policies or evidence of method superiority.

The synchronized arithmetic-mean example declares all nine utilities and has six tied optimizers of mean 11/20. The chain–diamond example has a unique optimal mean of 0.7078125. These small examples are distinct from the DataCo empirical utility.

The immutable reference CSVs retain full numerical precision. Timing results depend on hardware. `code/verify_outputs.py` compares all reference CSV cells, not merely table row counts. A byte-identical PDF is not required because PDF metadata can differ; numerical tables, graph relationships and mathematical inputs are the scientific checks.

## Data and license

Code is released under the MIT License (`LICENSE`). The DataCo dataset attribution and its license terms are given in `DATA_LICENSE.md` and `data/README.md`.
