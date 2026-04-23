"""
TETRiS regulator-side pipeline (Algorithm 1B).

Reads the compact submission produced by sponsor_side.py and carries out
- the primary analysis (combining subgroup functionals, patient-weighted),
- per-subgroup analyses (each from its own cores),
- submodel analyses (drop a covariate subset by fixing those coordinates at 0).

All inference is done locally from the submission file; no further data
access or computation from the external entity is required.
"""

import numpy as np

import config as cfg
import tetris_core as tc


# =============================================================================
# Load submission package
# =============================================================================
def load_submission(path):
    """Return (meta, subgroups) where subgroups is a list of dicts:
        { "value": ..., "n_patients": int, "cores": [np.ndarray, ...] }
    """
    npz = np.load(path, allow_pickle=True)
    meta = {
        "covariate_cols": [str(x) for x in npz["meta__covariate_cols"]],
        "subgroup_col":   str(npz["meta__subgroup_col"]),
        "subgroup_values": [str(x) for x in npz["meta__subgroup_values"]],
        "grid_n":         int(npz["meta__grid_n"]),
        "beta_lo":        float(npz["meta__beta_lo"]),
        "beta_hi":        float(npz["meta__beta_hi"]),
        "eps":            float(npz["meta__eps"]),
    }
    d = len(meta["covariate_cols"])
    n_sg = len(meta["subgroup_values"])

    subgroups = []
    for g in range(n_sg):
        cores = [npz[f"sg{g}__core{j}"] for j in range(d)]
        subgroups.append({
            "value": str(npz[f"sg{g}__value"]),
            "n_patients": int(npz[f"sg{g}__n_patients"]),
            "cores": cores,
        })
    return meta, subgroups


# =============================================================================
# Printing
# =============================================================================
def print_result(label, covariate_names, active_idx, rr, ci_low, ci_high):
    print(f"\n{label}")
    print("-" * len(label))
    for j, k in enumerate(active_idx):
        name = covariate_names[k]
        print(f"  {name:<15}  RR = {rr[j]:.4f}   95% CI = [{ci_low[j]:.4f}, {ci_high[j]:.4f}]")


def print_submodels(header, functional, d, n_eff, lo, hi, covariate_names, submodels):
    """Run each configured submodel against a given functional and print results."""
    for sm_name, keep_cols in submodels.items():
        drop_idx = [k for k, name in enumerate(covariate_names) if name not in keep_cols]
        fit = tc.fit_and_wald_ci(functional, d, n_eff, lo, hi,
                                 fixed_zero=drop_idx if drop_idx else None)
        print_result(
            f"{header}  |  submodel: {sm_name}",
            covariate_names, fit["active_idx"],
            fit["rr"], fit["ci_low"], fit["ci_high"],
        )


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("TETRiS REGULATOR-SIDE PIPELINE")
    print("=" * 60)

    meta, subgroups = load_submission(cfg.SUBMISSION_PATH)
    d = len(meta["covariate_cols"])
    grid_n = meta["grid_n"]
    lo, hi = meta["beta_lo"], meta["beta_hi"]
    cov_names = meta["covariate_cols"]

    print(f"Loaded {cfg.SUBMISSION_PATH}")
    print(f"Dimension d:       {d}")
    print(f"Grid per dim:      {grid_n}")
    print(f"Parameter box:     [{lo}, {hi}]^{d}")
    print(f"DMRG-cross eps:    {meta['eps']}")
    print(f"Subgroups:         {[sg['value'] for sg in subgroups]}")
    print(f"Patients/subgroup: {[sg['n_patients'] for sg in subgroups]}")

    # One-time setup: transform each subgroup's cores to Chebyshev coefficients
    cheb_pts = tc.chebyshev_points(grid_n)
    Q = tc.chebyshev_transform_matrix(cheb_pts, grid_n)

    per_subgroup_fn = []
    for sg in subgroups:
        coeff_cores = tc.cores_to_cheb_coeffs(sg["cores"], Q)
        per_subgroup_fn.append(tc.make_tt_functional(coeff_cores, grid_n))

    # Primary: patient-weighted combination of subgroup functionals
    weights = [sg["n_patients"] for sg in subgroups]
    n_total = sum(weights)
    primary_fn = tc.combine_functionals(per_subgroup_fn, weights)

    # ---- Primary analysis (full model) ----
    print("\n" + "=" * 60)
    print("PRIMARY ANALYSIS (pooled across subgroups)")
    print("=" * 60)
    fit = tc.fit_and_wald_ci(primary_fn, d, n_total, lo, hi)
    print_result("Primary, full model", cov_names, fit["active_idx"],
                 fit["rr"], fit["ci_low"], fit["ci_high"])

    # ---- Submodel analyses on primary ----
    print("\n" + "=" * 60)
    print("SUBMODEL ANALYSES (pooled)")
    print("=" * 60)
    print_submodels("Primary", primary_fn, d, n_total, lo, hi, cov_names, cfg.SUBMODELS)

    # ---- Per-subgroup analyses (full model and submodels) ----
    print("\n" + "=" * 60)
    print("SUBGROUP ANALYSES")
    print("=" * 60)
    for sg, fn in zip(subgroups, per_subgroup_fn):
        header = f"Subgroup: {meta['subgroup_col']} = {sg['value']}  (n={sg['n_patients']})"
        print("\n" + header)
        print("-" * len(header))
        for sm_name, keep_cols in cfg.SUBMODELS.items():
            drop_idx = [k for k, name in enumerate(cov_names) if name not in keep_cols]
            fit = tc.fit_and_wald_ci(fn, d, sg["n_patients"], lo, hi,
                                     fixed_zero=drop_idx if drop_idx else None)
            print_result(
                f"  submodel: {sm_name}",
                cov_names, fit["active_idx"],
                fit["rr"], fit["ci_low"], fit["ci_high"],
            )

    print("\n" + "=" * 60)
    print("All analyses complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
