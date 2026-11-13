# Validation report

## Scope

The supplied transaction data, the mathematical examples and the reproducibility repository were examined together. This report records executed numerical and software checks. It is not an assertion of external peer review or universal mathematical completeness.

## Input and complete DataCo execution

The unchanged transaction CSV contains 180,519 item rows and 53 columns, aggregating to 65,752 orders. Its SHA-256 is `fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6`. The chronological, market-stratified split contains 52,599 training orders and 13,153 holdout orders. The descriptor is also unchanged; whitespace is stripped only when testing its padded field names.

The full experiment was executed with the declared 2,000 customer-cluster replicates, 1,000 support-transport replicates, 500 full-pipeline refits, and seed 20260819. A separate full execution from the included input matched all 44 DataCo CSV tables, cell by cell (relative tolerance 1e-9 and absolute tolerance 1e-10 for floating-point values), and all headline results. `data_reproduction.json` records 89 passing checks: 36 result fields, 44 tables and nine figure-presence checks. Numerical agreement is not inferred from filenames or table shapes.

The baseline supported order has 181 nodes and 485 cover relations. Exhaustive enumeration of 24,870 source-to-maximal paths agrees with the length-indexed exact optimizer. The unique optimum contains ten nodes and has mean utility 0.5179137537919203. The second-best mean is 0.5173623177959996; the relative gap is 0.10647255298460678 percent. Only seven of 500 full-pipeline refits recover the exact path (1.4 percent); median node Jaccard overlap is 6/17, approximately 0.3529411765.

## Internal-order checks

The exact full pullback is a nonempty lattice under the proved homomorphism construction. Its finite support restrictions require their own internal-bound audits; ambient nonclosure alone is insufficient.

| Structure | Nodes | Unordered distinct pairs | Pairs lacking an internal join | Pairs lacking an internal meet | Intrinsic lattice |
|---|---:|---:|---:|---:|---|
| Full pullback | 3,200 | 5,118,400 | 0, by theorem | 0, by theorem | Yes |
| Marginal support | 512 | 130,816 | 1,536 | 576 | No |
| Joint support | 181 | 16,290 | 2,096 | 1,046 | No |

The filtered-set conclusions use executable pairwise internal-bound searches and complete witness CSVs. The full pullback conclusion is theorem-based; no claim is made that its 5,118,400 pairs were all brute-force searched. Join- and meet-failure counts can overlap and must not be added to obtain a distinct-pair count.

The Boolean-lattice diagram contains precisely its 12 subset-cover edges. The selected four-element set is intrinsically a lattice while failing closure under the ambient join. The isomorphic collaboration diagram uses index-preserving maps and explicitly states phi23 = phi13 composed with inverse(phi12). The pullback theorem and semantic restrictions explicitly require nonempty retained domains when a lattice is asserted. Derived payload joins are recomputed from the order indices; they need not equal scalarwise maxima of every redundant attribute.

## Explicit small-graph calculations

The synchronized illustration declares the chains alpha = (0.2, 0.5, 0.8), beta = (0.3, 0.6, 0.9), arithmetic-mean node utility and zero edge cost. All nine utilities and all six saturated paths are included. Every path has five nodes and exact mean 11/20 = 0.55; the highlighted path is one of six tied optimizers. The unavailable utility specification behind the earlier value 0.77025 is not inferred from the unrelated transaction dataset and is not used to support a result.

The chain–diamond illustration enumerates all three source-to-target paths. The unique optimum is (gamma1, gamma3, gamma5, gamma6), with exact mean 453/640 = 0.7078125. The other two means are 0.6884375 and 0.688125.

## Holdout and cohort checks

All four distance-to-path comparisons use the same 976 exactly coordinated holdout orders. Their customer-cluster association intervals include zero. Fixed-coverage groups contain exactly 48, 97 or 195 orders for the 5, 10 or 20 percent protocols. Distance ties are resolved by ascending order identifier, without reading outcomes. Equal group sizes do not make the selected observations identical; the rule is a declared convention, not a causal identification strategy. Full numerical tables, bootstrap replicates and coverage profiles are included.

The frozen holdout evidence degree uses the training support denominator and is capped at one, consistently in the executable code and mathematical description. The baseline radius-one subgroup has 116 orders (0.8819280772 percent of the full holdout); its signed-status mean is descriptive, not a validated probability or evidence of universal method superiority.

## Synthetic execution

The seeded controlled benchmark was executed with 2,000 observations, a 1,400/600 training/holdout split, seed 20260825 and 200 full refits. It produces 137 fitted nodes, 359 covers and 159,986 source-to-target paths. The selected eleven-node path has mean 0.5824276912297703. Exact path recovery is 0/200 and median node Jaccard is 1/3. All 14 synthetic numerical tables are included. Runtime measurements are machine-dependent; the timing table and plot use measurements from this execution.

## Figures and tables

The article has 11 figures; Online Resource 1 has 18. All 29 figure slots have PDF, EPS and PNG files (87 files across the two documents). Six distinct mathematical diagrams are compiled from editable TikZ sources. Their PNGs are rendered at 1200 dpi; numerical PNGs are rendered at 600 dpi. Mathematical drawings are not raster enlargements. Font embedding, figure-number mappings and agreement between each document and repository assembly were checked. `figure_integrity.json` records all image dimensions, DPI and format checks.

The portable entry point was executed in table-based mode. All 29 PNG figure slots and all 23 generated native numerical LaTeX table fragments matched the packaged counterparts byte for byte. This check is separate from the complete raw-data run; it is not presented as raw-data resampling. The PDFs/EPS may differ in metadata when compiled again, so byte identity is not required for those formats.

## Software tests

All 47 repository tests pass. These include exact illustrative-path values, internal versus ambient lattice audits, graph algorithms, canonical input checksum, complete refit validity, exact-size cohort tie handling, mathematical diagram relationships, source-to-figure assembly and manifest verification. Detailed records are supplied in this directory. Runtime environment and commands are in the repository README and `docs/REPRODUCTION_PROTOCOL.md`.

## Data and interpretation notes

Public availability of the identifier-bearing dataset is not a guarantee of anonymization. DataCo is a historical observational benchmark, not a causal experiment or an observed physical multi-echelon transition system. The results support conditional mathematical and descriptive statements, not automatic operational superiority.

Data source: DataCo version 5, https://data.mendeley.com/datasets/8gx2fvg2k6/5 . Dataset attribution and license information are retained in `DATA_LICENSE.md`.
