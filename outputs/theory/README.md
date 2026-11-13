# Exact small-graph calculations

`c3xc3_node_utilities.csv` contains all nine utilities for the explicitly declared arithmetic-mean illustration. The two equally spaced chains are (0.2, 0.5, 0.8) and (0.3, 0.6, 0.9). With utility u(i,j) = (alpha(i) + beta(j))/2 and zero edge cost, every saturated bottom-to-top path has five nodes and mean 11/20 = 0.55. `c3xc3_paths.csv` lists all six paths; all six are optimal. The highlighted chain is one member of this optimal family, not a unique optimum.

`chain_diamond_paths.csv` lists all three source-to-target paths, their exact fractions and decimal means. The unique optimal path is (1,3,5,6), with mean 453/640 = 0.7078125.

`illustrative_audit.json` records both checks. The numerical illustration is defined by these inputs, not by fitting invented utilities to an unavailable earlier score. All six mathematical figures have editable TikZ sources and vector PDF/EPS and 1200-dpi PNG outputs in `figures/`.

Rebuild the numerical checks with `python code/audit_illustrative_paths.py` and the figures with `python code/generate_theory_figures.py` (see each command's `--help`).
