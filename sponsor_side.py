"""
TETRiS sponsor-side pipeline (Algorithm 1A).

Reads patient-level records, partitions them by prespecified subgroup, runs
TT-cross on the conditional negative log-likelihood of each subgroup, and
writes the compact submission package to disk.

This script should be run ONCE by the external data entity. The output file
is the entire submission to the regulator.
"""

from time import perf_counter
import numpy as np
import pandas as pd

import config as cfg
import tetris_core as tc


# =============================================================================
# Data loading
# =============================================================================
def preprocess(df):
    """Apply categorical -> binary expansion as specified in config."""
    df = df.copy()
    for src_col, mapping in cfg.CATEGORICAL_TO_BINARY.items():
        for new_col, value in mapping.items():
            df[new_col] = (df[src_col] == value).astype(int)
    return df


def extract_patient_arrays(df, covariate_cols):
    """Return (data_list, event_indices, n_patients).

    data_list[i]: (n_intervals_i, d) design matrix for patient i.
    event_indices[i]: row within data_list[i] where event occurred.
    """
    ids = df[cfg.PATIENT_ID_COL].unique()
    n = len(ids)
    id_to_idx = {pid: k for k, pid in enumerate(ids)}

    data_list = [None] * n
    event_indices = [None] * n

    for pid, sub in df.groupby(cfg.PATIENT_ID_COL, sort=False):
        i = id_to_idx[pid]
        X = sub[covariate_cols].to_numpy(dtype=float)
        ev_rows = np.where(sub[cfg.EVENT_COL].to_numpy() == 1)[0]
        if len(ev_rows) == 0:
            continue
        data_list[i] = X
        event_indices[i] = int(ev_rows[0])

    # Drop any patients with no event (shouldn't happen in a well-formed SCCS/SCRI
    # dataset, but guard against it)
    keep = [i for i in range(n) if data_list[i] is not None]
    data_list = [data_list[i] for i in keep]
    event_indices = [event_indices[i] for i in keep]
    return data_list, event_indices, len(data_list)


# =============================================================================
# TT construction for a single subgroup
# =============================================================================
def build_subgroup_cores(data_list, event_indices, d, grid_n, cheb_pts, lo, hi,
                         eps, max_tries):
    """Run DMRG-cross and return (cores, ranks, size_bytes, elapsed_seconds)."""
    query_fn = tc.make_tt_query(data_list, event_indices, cheb_pts, lo, hi)
    t0 = perf_counter()
    tt_result = tc.run_dmrg_cross(query_fn, [grid_n] * d, eps, max_tries)
    elapsed = perf_counter() - t0
    if tt_result is None:
        raise RuntimeError("DMRG-cross failed after max retries")
    cores = tc.tt_cores_numpy(tt_result)
    ranks = tc.tt_ranks_from_cores(cores)
    size_bytes = tc.tt_size_bytes(cores)
    return cores, ranks, size_bytes, elapsed


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("TETRiS SPONSOR-SIDE PIPELINE")
    print("=" * 60)

    # Load and preprocess
    df = pd.read_csv(cfg.DATA_PATH)
    df = preprocess(df)
    print(f"Loaded {len(df):,} rows from {cfg.DATA_PATH}")

    d = len(cfg.COVARIATE_COLS)
    grid_n = cfg.GRID_SIZE
    cheb_pts = tc.chebyshev_points(grid_n)
    lo, hi = cfg.BETA_LOWER, cfg.BETA_UPPER

    # Build TT cores for each subgroup
    package = {
        "meta__covariate_cols": np.array(cfg.COVARIATE_COLS),
        "meta__subgroup_col": np.array(cfg.SUBGROUP_COL),
        "meta__subgroup_values": np.array(cfg.SUBGROUP_VALUES),
        "meta__grid_n": np.array(grid_n),
        "meta__beta_lo": np.array(lo),
        "meta__beta_hi": np.array(hi),
        "meta__eps": np.array(cfg.PRIMARY_EPS),
    }

    for sg_value in cfg.SUBGROUP_VALUES:
        print("\n" + "-" * 60)
        print(f"Subgroup: {cfg.SUBGROUP_COL} = {sg_value}")
        print("-" * 60)

        sub_df = df[df[cfg.SUBGROUP_COL] == sg_value]
        data_list, event_indices, n_patients = extract_patient_arrays(
            sub_df, cfg.COVARIATE_COLS
        )
        print(f"Patients: {n_patients}")

        cores, ranks, size_bytes, elapsed = build_subgroup_cores(
            data_list, event_indices, d, grid_n, cheb_pts, lo, hi,
            cfg.PRIMARY_EPS, cfg.DMRG_MAX_TRIES,
        )
        print(f"TT ranks:       {ranks}")
        print(f"File size:      {size_bytes / 1024:.2f} KB")
        print(f"Elapsed:        {elapsed:.1f} s")

        # Store under keys like "sg0__n_patients", "sg0__core0", ...
        key = f"sg{cfg.SUBGROUP_VALUES.index(sg_value)}"
        package[f"{key}__value"] = np.array(sg_value)
        package[f"{key}__n_patients"] = np.array(n_patients)
        for j, core in enumerate(cores):
            package[f"{key}__core{j}"] = core

    # Save
    np.savez(cfg.SUBMISSION_PATH, **package)

    # Summary
    dense_bytes = (grid_n ** d) * 8
    total_tt_bytes = sum(
        v.size * 8 for k, v in package.items() if "__core" in k
    )
    print("\n" + "=" * 60)
    print("SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"Output file:            {cfg.SUBMISSION_PATH}")
    print(f"Subgroups:              {len(cfg.SUBGROUP_VALUES)}")
    print(f"Dimension d:            {d}")
    print(f"Dense-grid size (hyp.): {dense_bytes / 1024:.1f} KB per subgroup")
    print(f"Total TT file size:     {total_tt_bytes / 1024:.2f} KB "
          f"(all subgroups combined)")
    print(f"Compression ratio:      "
          f"{(dense_bytes * len(cfg.SUBGROUP_VALUES)) / total_tt_bytes:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
