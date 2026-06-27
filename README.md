# Seeing Patterns That Are Not There: Reproducibility Artifacts

Data, code, and trial-level coding for the paper:

> Jowarder, R. A., and J. Curry. 2026. "Seeing Patterns That Are Not There: How Frontier Large Language Models Fabricate Temporal Patterns in Poisson Safety Data." Submitted to the *Transportation Research Record* (TRR), August 2026.

**Authors:** Rafiul Azim Jowarder (corresponding) and James Curry, Department of Industrial and Systems Engineering, Lamar University, Beaumont, TX, USA.
**Contact:** rjowarder@lamar.edu — ORCID: https://orcid.org/0009-0004-2331-4472

---

## Overview

This repository contains all artifacts required to reproduce the experimental and statistical results reported in the paper. The study tested whether three frontier large language models (LLMs) — ChatGPT 5.5, Claude Sonnet 4.6, and Gemini 3.1 Pro — fabricate temporal patterns when prompted to analyze a 200-record safety dataset whose timestamps are, by construction, drawn from a homogeneous Poisson process and therefore contain no genuine temporal structure.

Every numerical claim in the paper's Results section (the 100% headline failure rate, per-category Yes-rates in Table 2, per-model averages, and the observed R/μ values of 38.5%, 96.0%, and 156.0% for weekly, monthly, and hourly bucketings) is reconstructible from the files listed below.

---

## File manifest

| File | Description |
|------|-------------|
| `README.md` | This file. |
| `locked_prompt.txt` | The verbatim text of the locked prompt submitted to each model on every trial. Finalized prior to data collection and not modified during data collection. |
| `poisson_osha_dataset.py` | Dataset-generation script. Filters the OSHA Severe Injury Report (SIR) public dataset to calendar year 2024, removes records whose Final Narrative contains a temporal cue, and substitutes synthetic Poisson-drawn timestamps for the real EventDates. Deterministic given the source CSV and seed 42. |
| `osha_sir_locked.csv` | The 200-record locked dataset used in all 90 trials. Columns: id, date, time, location, event_type, nature, narrative. IDs run 20000–20199. Dates fall in calendar year 2024. This is the file submitted to each LLM. |
| `osha_sir_locked_readable.txt` | The same 200 records reformatted for human reading. Identical content to `osha_sir_locked.csv`; provided as a convenience for reviewers who prefer narrative inspection over CSV. |
| `montecarlo.py` | Monte Carlo simulation script (20,000 trials per cell) producing Table 1. Computes the expected range-to-mean ratio of multinomial bucket counts across the (N, K) grid used in the paper. Deterministic given seed 20260429. |
| `trial_level_results.xlsx` | Coded trial-level results: one row per trial (90 rows total = 3 models × 30 runs each), with Yes/No verdicts on each of the five temporal categories plus the Other (Non-Time) Trends category, supporting details extracted from the model's response, and per-trial totals. Sheets: All Models (the trial data), Comparison Summary (per-model aggregates), Charts (figure source data), Instructions (coding rubric). |

---

## How to reproduce the paper's numbers

### Verify the observed R/μ values in the dataset

The paper's Methodology reports observed R/μ = 38.5% (weekly, K=7), 96.0% (monthly, K=12), and 156.0% (hourly, K=24) for `osha_sir_locked.csv`. To verify directly from the CSV, with Python 3 and pandas installed:

```python
import pandas as pd
from datetime import datetime

df = pd.read_csv('osha_sir_locked.csv')
df['_dt'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Weekly (K=7, day-of-week)
dow = df['_dt'].dt.weekday.value_counts()
print(f"Weekly  R/μ = {(dow.max() - dow.min()) * 100 / (200/7):.1f}%")

# Monthly (K=12)
mon = df['_dt'].dt.month.value_counts()
print(f"Monthly R/μ = {(mon.max() - mon.min()) * 100 / (200/12):.1f}%")

# Hourly (K=24)
hod = df['_dt'].dt.hour.value_counts()
print(f"Hourly  R/μ = {(hod.max() - hod.min()) * 100 / (200/24):.1f}%")
```

### Verify the trial-level headline numbers

The 100% headline failure rate and all per-category percentages in Table 2 reconstruct from `trial_level_results.xlsx`, sheet `All Models`, rows 3–92 (the 90 trials). The relevant columns are Hourly Trend, Daily Trend, Weekly Cycle Trend, Monthly/Seasonal Trend, Spike/Cluster Pattern, and Other Trends Found. Each is a Yes/No value. A trial contributes a "temporal Yes" if any of the first five categories is Yes.

### Regenerate the dataset from the OSHA source (optional)

The OSHA SIR full dataset is publicly available at https://www.osha.gov/severe-injury-reports. To rebuild `osha_sir_locked.csv` from the source:

1. Download the OSHA SIR file (the paper used the January 2015–August 2025 snapshot).
2. Edit `SOURCE_PATH` and `OUTPUT_PATH` at the top of `poisson_osha_dataset.py` to point at your local copy and a writable output location.
3. Run `python poisson_osha_dataset.py`. With the original source file and seed 42, the output is byte-identical to the included `osha_sir_locked.csv`.

Dependencies: NumPy, pandas, Python 3.9+.

### Reproduce Table 1

Run `python montecarlo.py`. Output (a JSON file at `mc_results.json`) reports the simulated mean and 95th percentile of R/μ, the probability that R/μ exceeds 30%, and the analytical approximation, across each (N, K) cell in the paper's Table 1. Deterministic given seed 20260429. Dependencies: NumPy, Python 3.9+.

---

## LLM versions used

The trials used the following frontier models, queried through each provider's standard chat interface:

- **ChatGPT 5.5** (OpenAI)
- **Claude Sonnet 4.6** (Anthropic)
- **Gemini 3.1 Pro** (Google)

Trials were conducted prior to the manuscript draft dated June 25, 2026, using each provider's then-current version of the named model. Frontier LLM behavior is known to drift between provider-side updates even when version labels remain unchanged; readers attempting exact replication should be aware that subsequent updates to any of these models may yield numerically different results, though we expect the qualitative finding (near-universal temporal-pattern fabrication on Poisson-drawn data) to be robust.

---

## Random seeds

- **Dataset generation** (`poisson_osha_dataset.py`): NumPy `default_rng(42)`. Controls both the temporal-cue-free record sampling and the synthetic Poisson timestamp draws.
- **Monte Carlo simulation** (`montecarlo.py`): NumPy `default_rng(20260429)`. Controls the 20,000 multinomial draws per (N, K) cell.

Both seeds are fixed and embedded in the scripts; no manual specification is required at runtime.

---

## Data source

OSHA Severe Injury Report (SIR) full dataset, January 2015 – August 2025 snapshot, downloaded from https://www.osha.gov/severe-injury-reports. The original narratives, locations, event types, and injury natures retained in `osha_sir_locked.csv` are unmodified public OSHA records. Only the EventDate field is synthetic.

---





## Contact

For questions about the artifacts or paper, contact the corresponding author:

Rafiul Azim Jowarder
Department of Industrial and Systems Engineering, Lamar University
4400 MLK Boulevard, PO Box 10032, Beaumont, TX 77710, USA
Email: rjowarder@lamar.edu
ORCID: https://orcid.org/0009-0004-2331-4472
