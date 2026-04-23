"""
TETRiS configuration.

All data-specific and analysis-specific choices live here. The core math
(tetris_core.py), sponsor pipeline (sponsor_side.py), and regulator pipeline
(regulator_side.py) read from this file and are otherwise data-agnostic.

To run on a different dataset (e.g., the real SCRI use case), edit only this
file; the rest of the code does not need to change.
"""

# =============================================================================
# Data
# =============================================================================
DATA_PATH = "data/daily_observations.csv"
PATIENT_ID_COL = "beneficiary_id"
EVENT_COL = "event"

# =============================================================================
# Preprocessing: categorical columns -> binary indicators
#
# For each source column, each entry (new_col_name -> value) creates a binary
# indicator that is 1 when the source column equals that value, 0 otherwise.
# The reference (omitted) level is whichever value has no entry here.
# =============================================================================
CATEGORICAL_TO_BINARY = {
    "risk_window": {
        "exposure_RW1": "1-21 days",
        # reference: "22-90 days" (control window)
    },
    "quarter": {
        "season2": "Q2",
        "season3": "Q3",
        "season4": "Q4",
        # reference: Q1
    },
}

# Covariates used in the model, in order. The parameter vector beta has one
# entry per covariate, in this order.
COVARIATE_COLS = [
    "exposure_RW1",
    "season2",
    "season3",
    "season4",
]

# =============================================================================
# Subgroups: sponsor builds one TT submission per subgroup value
# =============================================================================
SUBGROUP_COL = "vaccine_brand"
SUBGROUP_VALUES = ["Pfizer-BioNTech", "Moderna"]

# =============================================================================
# Submodels: regulator-side analyses, each a subset of covariates to keep
# (omitted covariates are fixed at 0 during optimization).
# =============================================================================
SUBMODELS = {
    "full":      ["exposure_RW1", "season2", "season3", "season4"],
    "no_season": ["exposure_RW1"],
}

# =============================================================================
# TT-cross parameters
# =============================================================================
GRID_SIZE = 51              # Chebyshev nodes per dimension (M + 1)
BETA_LOWER = -20.0          # parameter box lower bound
BETA_UPPER = 20.0           # parameter box upper bound
PRIMARY_EPS = 1e-2          # DMRG-cross accuracy tolerance
DMRG_MAX_TRIES = 10

# =============================================================================
# I/O
# =============================================================================
SUBMISSION_PATH = "data/submission_package.npz"