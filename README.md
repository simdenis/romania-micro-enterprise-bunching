# Bunching at a Moving Threshold: Firm Responses to Romania's Micro-Enterprise Tax, 2014–2025

[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22863290.svg)](https://doi.org/10.5281/zenodo.22863290)

Replication package for the working paper of the same title (Denis Siminiuc, September 2026). It contains the
data pipeline, the harmonised firm-year panel, the trade-register linkage with anonymised keys, every table and
figure in the paper, and the legal timeline with Monitorul Oficial references.

## What is in the package

| Folder | Contents |
|---|---|
| `code/` | All analysis scripts (Python). `build_panel.py` builds the panel from the raw statement files; `bunching_check.py`, `diff_in_bunching.py`, `checks_round*.py`, `revision*_checks.py` produce every number in the paper; `make_tables.py` writes the LaTeX tables from `outputs/`. |
| `data/` | The harmonised panel and linkage tables (Parquet), the BNR year-end EUR/RON rates, and a manifest of the source files with their data.gov.ro URLs and upload dates. Parquet files are hosted on Zenodo (https://doi.org/10.5281/zenodo.22863290) because of their size; the manifest and rates are in the repository. |
| `outputs/` | Every CSV and PNG behind the tables and figures. |
| `paper/` | `main.tex`, `main.pdf`, the generated tables, figures, and `legal_timeline.md`. |

## Data sources (all public, data.gov.ro)

- Ministry of Finance, *Situații financiare* 2014–2025: firm-level annual financial statements (micro-entity and full
  forms). `data/source_manifest.csv` lists the exact resource URL and upload date used for each year.
- ANAF, *Date de identificare plătitori* (June 2023 and June 2024 editions): taxpayer registry with tax-vector flags.
- ONRC, *Firme înregistrate la Registrul Comerțului* snapshot of 2 September 2026: firms, status, legal representatives.
- BNR reference exchange rates (curs.bnr.ro), last quotation of each year.

The portal is reachable only from European IP addresses and serves large files unreliably; `code/download_statements.py`
and `code/download_onrc.py` handle the downloads with retries.

## Panel files

`statements_2014_2025.parquet` — one row per firm-year (9,578,769 rows, 1,556,982 firms): `year`, `form` (UU = micro-entity
form, BL = full form), `cui` (fiscal code, a public firm identifier), `caen`, the 20 statement indicators in lei
(`turnover`, `revenue`, `costs`, `profit_gross`, `loss_gross`, `profit_net`, `loss_net`, `employees`, balance-sheet totals),
and derived `pretax`, `net`, `tax`.

`regime_inferred.parquet` — statement-based regime classification per firm-year (`micro`, `micro_loss`, `profit`, `no_tax`,
`ambiguous`, `other`) with the implied tax rates.

`onrc_firms.parquet` — one row per registered firm: `cui`, registration code, registration date, legal form, county,
status code, and `addr_id`.

`onrc_admins.parquet`, `onrc_person_firm.parquet` — administrators and legal representatives linked to firms through
`person_id`.

### Anonymisation

`person_id` and `addr_id` are keyed hashes (SHA-256 of a random secret salt concatenated with the source key). The salt is
not distributed, so the keys cannot be re-linked to the public register even though the source data are public. No names,
birth dates, birth places or street addresses appear anywhere in the package. The ANAF registry files are not redistributed;
the scripts document how they were used and the cross-check table is in `outputs/regime_crosscheck_2023.csv`.

## Reproducing the paper

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# download the Parquet files from https://doi.org/10.5281/zenodo.22863290 into data/
.venv/bin/python code/bunching_check.py
.venv/bin/python code/diff_in_bunching.py && .venv/bin/python code/diff_in_bunching_counts.py
.venv/bin/python code/checks_round2.py && .venv/bin/python code/checks_round3.py && .venv/bin/python code/checks_round4.py
.venv/bin/python code/revision_checks.py && .venv/bin/python code/revision2_checks.py && .venv/bin/python code/revision3_checks.py
.venv/bin/python code/revision4_checks.py && .venv/bin/python code/revision5_checks.py && .venv/bin/python code/revision6_checks.py
.venv/bin/python code/splitting_test.py
.venv/bin/python code/make_tables.py && cd paper && tectonic -X compile main.tex
```

Scripts that expect the raw files (`build_panel.py`, `build_onrc_links.py`, `build_onrc_links_anaf.py`, `build_onrc_links_v2.py`)
are included for transparency; the panel they produce is the one distributed.

## Legal timeline

`paper/legal_timeline.md` gives, for every change to the micro-enterprise regime from 2013 to 2026, the effective date,
threshold, rates, conditions, the amending act and its Monitorul Oficial reference, the exchange-rate provisions with the
Romanian text, and the VAT threshold history. Items that could not be verified against a primary text are marked.

## Disclosure

The manuscript, code and legal timeline were drafted with substantial assistance from a large language model (Anthropic
Claude). The author verified legal references against primary sources and numerical claims against the data and is
responsible for the content.

## Citation

Siminiuc, D. (2026). Bunching at a Moving Threshold: Firm Responses to Romania's Micro-Enterprise Tax, 2014–2025.
Working paper. Code: https://github.com/simdenis/romania-micro-enterprise-bunching. Data: Zenodo, https://doi.org/10.5281/zenodo.22863290.

## License

Code: MIT. Derived data: subject to the terms of the source datasets on data.gov.ro (open licence, attribution required).
