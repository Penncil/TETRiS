"""
Generate synthetic SCCS-style daily observation data for TETRiS demonstration.

Outputs:
  stroke_cases.csv       -- one row per patient
  daily_observations.csv -- one row per (patient, day) up to end of follow-up
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================
SEED = 42
N_CASES = 1000
START_DATE = datetime(2022, 8, 31)
END_DATE = datetime(2023, 2, 4)
FOLLOWUP_DAYS = 90

RISK_WINDOW_1 = (1, 21)    # days post-vaccination
CONTROL_WINDOW = (22, 90)  # everything after RW1

AGE_GROUP_PROBS = {"65-74": 0.50, "75-84": 0.35, "≥85": 0.15}
AGE_GROUP_RANGES = {"65-74": (65, 75), "75-84": (75, 85), "≥85": (85, 95)}
AGE_GROUP_RISK_MULT = {"65-74": 1.0, "75-84": 1.5, "≥85": 2.0}

SEX_PROBS = {"F": 0.56, "M": 0.44}

VACCINE_PROBS = {"Pfizer-BioNTech": 0.6, "Moderna": 0.4}

STROKE_TYPE_PROBS = {"nonhemorrhagic": 0.65, "TIA": 0.25, "hemorrhagic": 0.10}

# Incidence rate ratios by (vaccine brand, window)
IRR = {
    ("Pfizer-BioNTech", "RW1"): 1.05,
    ("Moderna",         "RW1"): 0.90,
    "control":                  1.00,
}

# Multiplicative seasonality factor by quarter
SEASON_FACTOR = {1: 1.25, 2: 0.90, 3: 0.85, 4: 1.20}

BASE_DAILY_RISK = 1e-4

OUTPUT_CASES = "data/stroke_cases.csv"
OUTPUT_DAILY = "data/daily_observations.csv"


# =============================================================================
# Helpers
# =============================================================================
def sample_from(prob_dict, rng):
    """Sample a key from a dict mapping key -> probability."""
    keys = list(prob_dict.keys())
    probs = list(prob_dict.values())
    return keys[rng.choice(len(keys), p=probs)]


def quarter_of(dt):
    return (dt.month - 1) // 3 + 1


def window_of(day):
    if RISK_WINDOW_1[0] <= day <= RISK_WINDOW_1[1]:
        return "1-21 days", "RW1"
    return "22-90 days", "control"


def irr_for(vaccine, window_key):
    if window_key == "control":
        return IRR["control"]
    return IRR[(vaccine, window_key)]


# =============================================================================
# Patient generation
# =============================================================================
def sample_patient(patient_id, rng):
    """Sample one patient's demographics, vaccination date, and event day."""
    age_group = sample_from(AGE_GROUP_PROBS, rng)
    lo, hi = AGE_GROUP_RANGES[age_group]
    age = rng.integers(lo, hi)

    sex = sample_from(SEX_PROBS, rng)
    vaccine = sample_from(VACCINE_PROBS, rng)
    stroke_type = sample_from(STROKE_TYPE_PROBS, rng)

    # Vaccination date: uniform within the window that leaves room for full follow-up
    max_vax_date = END_DATE - timedelta(days=FOLLOWUP_DAYS)
    max_offset = (max_vax_date - START_DATE).days
    vaccination_date = START_DATE + timedelta(days=int(rng.integers(0, max_offset + 1)))

    # Day-by-day hazard, modulated by window IRR and calendar-quarter seasonality
    base = BASE_DAILY_RISK * AGE_GROUP_RISK_MULT[age_group]
    daily_hazards = np.empty(FOLLOWUP_DAYS, dtype=float)
    for day in range(1, FOLLOWUP_DAYS + 1):
        obs_date = vaccination_date + timedelta(days=day)
        _, window_key = window_of(day)
        daily_hazards[day - 1] = (
            base * irr_for(vaccine, window_key) * SEASON_FACTOR[quarter_of(obs_date)]
        )

    # Force exactly one event per patient (SCCS case-only framing)
    probs = daily_hazards / daily_hazards.sum()
    days_to_stroke = int(rng.choice(np.arange(1, FOLLOWUP_DAYS + 1), p=probs))
    stroke_date = vaccination_date + timedelta(days=days_to_stroke)

    return {
        "beneficiary_id": patient_id,
        "age": int(age),
        "age_group": age_group,
        "gender": sex,
        "vaccine_brand": vaccine,
        "vaccination_date": vaccination_date,
        "stroke_type": stroke_type,
        "days_to_stroke": days_to_stroke,
        "stroke_date": stroke_date,
    }


def generate_cases(n, rng):
    return pd.DataFrame([sample_patient(i + 1, rng) for i in range(n)])


# =============================================================================
# Daily expansion
# =============================================================================
def expand_to_daily(cases_df):
    """Expand one row per patient into one row per (patient, day)."""
    rows = []
    for _, case in cases_df.iterrows():
        for day in range(1, FOLLOWUP_DAYS + 1):
            obs_date = case["vaccination_date"] + timedelta(days=day)
            if obs_date > END_DATE:
                continue
            window_label, _ = window_of(day)
            rows.append({
                "beneficiary_id":   case["beneficiary_id"],
                "vaccine_brand":    case["vaccine_brand"],
                "stroke_type":      case["stroke_type"],
                "observation_day":  day,
                "observation_date": obs_date,
                "risk_window":      window_label,
                "quarter":          f"Q{quarter_of(obs_date)}",
                "event":            int(day == case["days_to_stroke"]),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main():
    rng = np.random.default_rng(SEED)
    cases_df = generate_cases(N_CASES, rng)
    daily_df = expand_to_daily(cases_df)

    cases_df.to_csv(OUTPUT_CASES, index=False)
    daily_df.to_csv(OUTPUT_DAILY, index=False)

    print(f"Patients generated:      {len(cases_df):,}")
    print(f"Daily observation rows:  {len(daily_df):,}")
    print(f"Events:                  {daily_df['event'].sum():,}")
    print(f"By vaccine brand:")
    print(cases_df["vaccine_brand"].value_counts().to_string())
    print(f"\nWrote {OUTPUT_CASES}, {OUTPUT_DAILY}")


if __name__ == "__main__":
    main()