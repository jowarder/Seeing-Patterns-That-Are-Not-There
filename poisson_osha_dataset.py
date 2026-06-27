"""
Generate the experimental OSHA-derived dataset for Section 7.

This script constructs a hybrid dataset:
  - Real OSHA Severe Injury Report (SIR) records from calendar year 2024
    (employer, location, event type, narrative, etc.) — UNMODIFIED.
  - SYNTHETIC timestamps drawn from a homogeneous Poisson process,
    REPLACING the original EventDate.

Rationale:
The paper extends the synthetic-data demonstration to data with realistic
surface features. By holding the narrative metadata fixed (real OSHA
records) and randomizing only the temporal structure (homogeneous Poisson),
we test whether the LLM pattern-fabrication failure mode persists when
records "look like" real practitioner data.

Filtering step (important):
Records whose Final Narrative contains any temporal cue (dates, times,
days-of-week, months, seasons, AM/PM, etc.) are EXCLUDED from the sampling
pool before sampling. This prevents an LLM from spotting a mismatch between
the synthetic Poisson timestamp and a date embedded in the narrative,
which would otherwise contaminate the "no real temporal pattern by
construction" property of the dataset.

Narratives are NOT modified. We restrict sampling to records whose original
narratives are temporal-cue-free, then sample N from that filtered pool.
All retained fields (employer activity, equipment, body part, injury type,
sequence of events) remain 100% real OSHA content.

Source data: OSHA SIR full dataset (January 2015 - August 2025), available
from https://www.osha.gov/severe-injury-reports

Reproducibility: seed = 42 (numpy default_rng). The filter step is
deterministic, so the same seed always yields the same N records.

Usage:
    1. Edit SOURCE_PATH and OUTPUT_PATH below to point at your files.
    2. Click the Run Python File button in VS Code (or run `python poisson_osha_dataset.py`).

Output columns:
    id, date, time, location, event_type, nature, narrative
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# EDIT THESE TWO PATHS BEFORE RUNNING
# ============================================================
# Path to the OSHA source CSV (the real dataset downloaded from OSHA)
SOURCE_PATH = Path(r"C:\Users\rafin\Desktop\Osha Data Generation\January2015toAugust2025_osha_orginal.csv")

# Path where the generated experiment dataset will be written
OUTPUT_PATH = Path(r"C:\Users\rafin\Desktop\Osha Data Generation\osha_sir_locked.csv")
# ============================================================


# ---------- locked parameters (do not change for reproducibility) ----------
SEED = 42
N_RECORDS = 200
YEAR = 2024
DAYS_IN_YEAR = 366  # 2024 is a leap year
LAMBDA_PER_DAY = N_RECORDS / DAYS_IN_YEAR  # ≈ 0.547 incidents/day
NARRATIVE_MAX_CHARS = 300  # truncate to keep API prompts tractable
# ----------------------------------------------------------------------------


# Temporal-cue patterns. A narrative containing ANY of these is EXCLUDED
# from the sampling pool. Deliberately generous — better to exclude a few
# false-positive narratives than retain one that anchors to a specific time.
TEMPORAL_CUE_PATTERNS = [
    # Month name + day, with or without year (e.g. "January 5, 2024")
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}",
    # Abbreviated month + day (e.g. "Jan 5")
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}\b",
    # Numeric date MM/DD/YY or MM/DD/YYYY (requires year — avoids catching "3/8 inch")
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    # MM-DD-YYYY form
    r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
    # Year alone (2020-2025)
    r"\b(?:2020|2021|2022|2023|2024|2025)\b",
    # Day of week
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    # Time of day in HH:MM form (with or without AM/PM)
    r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?\b",
    # Standalone month name (covers "in February", "during October", etc.)
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    # Season words
    r"\b(?:spring|summer|fall|autumn|winter)\b",
    # Relative time
    r"\b(?:yesterday|today|tomorrow|this\s+(?:morning|afternoon|evening|week|month|year)|last\s+(?:night|week|month|year))\b",
    # AM/PM alone
    r"\b(?:a\.m\.|p\.m\.|AM|PM)\b",
    # Time-of-day descriptors
    r"\b(?:overnight|midnight|noon|dawn|dusk)\b",
    # Shift references
    r"\b(?:morning|afternoon|evening|night|day)\s+shift\b",
]
TEMPORAL_CUE_REGEX = re.compile("|".join(TEMPORAL_CUE_PATTERNS), re.IGNORECASE)


def has_temporal_cue(text: str) -> bool:
    """Return True if the narrative contains any temporal cue."""
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(TEMPORAL_CUE_REGEX.search(text))


def build_locked_sample(source_path: Path, out_path: Path) -> None:
    rng = np.random.default_rng(SEED)

    # 1. Load full OSHA SIR file
    print(f"Reading source: {source_path}")
    df = pd.read_csv(source_path, low_memory=False)
    df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
    df = df.dropna(subset=["EventDate"]).copy()

    # 2. Filter to calendar year 2024
    df_2024 = df[df["EventDate"].dt.year == YEAR].copy().reset_index(drop=True)
    print(f"OSHA SIR rows in {YEAR}: {len(df_2024):,}")

    # 3. Filter to narratives with no temporal cues.
    narrative_col = df_2024["Final Narrative"].fillna("").astype(str)
    has_cue_mask = narrative_col.apply(has_temporal_cue)
    df_clean = df_2024[~has_cue_mask].copy().reset_index(drop=True)
    n_excluded = len(df_2024) - len(df_clean)
    pct_excluded = 100.0 * n_excluded / len(df_2024)
    print(
        f"Excluded {n_excluded:,} records ({pct_excluded:.1f}%) containing "
        f"temporal cues in narrative."
    )
    print(f"Clean pool size: {len(df_clean):,}")

    if len(df_clean) < N_RECORDS:
        raise RuntimeError(
            f"Only {len(df_clean)} date-free OSHA records in {YEAR}; "
            f"need at least {N_RECORDS}."
        )

    # 4. Random sample of N_RECORDS without replacement from the clean pool
    sample_idx = rng.choice(len(df_clean), size=N_RECORDS, replace=False)
    sample = df_clean.iloc[sample_idx].copy().reset_index(drop=True)
    print(f"Sampled {N_RECORDS} records from clean pool (seed={SEED}).")

    # 5. Draw synthetic Poisson event times uniformly over 2024.
    #    Homogeneous Poisson process: event count is Poisson(λ * T); conditional
    #    on N events, event times are uniform over the observation window.
    #    We fix N = N_RECORDS (so the dataset is exactly the size we sampled).
    start = datetime(YEAR, 1, 1)
    total_hours = DAYS_IN_YEAR * 24
    hours = np.sort(rng.uniform(0, total_hours, size=N_RECORDS))
    event_times = [start + timedelta(hours=float(h)) for h in hours]

    # 6. Assemble output rows. Real metadata, synthetic timestamps, untouched narratives.
    rows = []
    for i, t in enumerate(event_times):
        src = sample.iloc[i]

        # Build a compact, prompt-friendly location field
        city = str(src.get("City", "")).strip()
        state = str(src.get("State", "")).strip()
        location = ", ".join(
            p for p in [city, state] if p and p.lower() != "nan"
        )

        # Truncate narrative to keep prompts manageable. The narrative content
        # itself is NOT modified — we only collapse whitespace and trim length.
        narrative = str(src.get("Final Narrative", "")).strip()
        if narrative and narrative.lower() != "nan":
            narrative = " ".join(narrative.split())
            if len(narrative) > NARRATIVE_MAX_CHARS:
                narrative = narrative[: NARRATIVE_MAX_CHARS - 1].rstrip() + "…"
        else:
            narrative = ""

        rows.append({
            "id": 20000 + i,
            "date": t.strftime("%Y-%m-%d"),
            "time": t.strftime("%H:%M"),
            "location": location,
            "event_type": str(src.get("EventTitle", "")).strip(),
            "nature": str(src.get("NatureTitle", "")).strip(),
            "narrative": narrative,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows.")

    # 7. Sanity report — confirm the dataset is, as constructed, pattern-free
    #    on the temporal dimension AND temporal-cue-free in narratives.
    out_df["_dt"] = pd.to_datetime(out_df["date"] + " " + out_df["time"])
    dow_counts = out_df["_dt"].dt.dayofweek.value_counts().sort_index()
    month_counts = out_df["_dt"].dt.month.value_counts().sort_index()
    print()
    print("Day-of-week counts (Mon=0):")
    print(dow_counts.to_dict())
    print(f"Monthly counts: {month_counts.to_dict()}")
    print(
        "(Variation here is pure Poisson noise — no underlying temporal "
        "pattern by construction.)"
    )

    # 8. Verify the output narratives are temporal-cue-free
    leaked = out_df["narrative"].apply(has_temporal_cue).sum()
    print()
    if leaked == 0:
        print(f"VERIFIED: 0/{N_RECORDS} output narratives contain temporal cues.")
    else:
        print(
            f"WARNING: {leaked}/{N_RECORDS} narratives still contain temporal "
            f"cues after filtering. Review the regex patterns."
        )


if __name__ == "__main__":
    build_locked_sample(SOURCE_PATH, OUTPUT_PATH)
