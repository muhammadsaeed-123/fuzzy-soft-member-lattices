# Data attribution and license

The experiment uses **DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS**, version 5, by Fabian Constante, Fernando Silva and António Pereira (2019), Mendeley Data. Dataset DOI: https://doi.org/10.17632/8gx2fvg2k6.5 . Official landing page: https://data.mendeley.com/datasets/8gx2fvg2k6/5 . The landing page identifies the source license as **Creative Commons Attribution 4.0 International**: https://creativecommons.org/licenses/by/4.0/ .

The transaction CSV and variable-description CSV were supplied for this analysis and are included unchanged with this attribution. `tokenized_access_logs.csv` is excluded because no reported computation uses it. The original transaction file includes identifier-bearing fields; public availability is not a guarantee of anonymization. Direct personal details are not used as model attributes. Order and customer keys are used to aggregate orders, resolve deterministic distance ties, and define customer-level resampling units. Do not infer new permissions beyond the source license and applicable data-handling obligations.

Transaction file: `DataCoSupplyChainDataset.csv`

Rows: 180,519; columns: 53; bytes: 95,910,149.

SHA-256: `fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6`.

The included reference results are derived from this file by the code and declared specifications. The software's MIT license does not replace the dataset license. Dataset license and journal requirements were consulted on 7 September 2026.

The source descriptor has trailing spaces in some `FIELDS` entries (including `Days for shipping (real)` and `Order Item Total`). Descriptor checks strip surrounding whitespace when comparing names; neither the descriptor nor transaction file is modified. The transaction header is checked separately against the required analysis columns.
