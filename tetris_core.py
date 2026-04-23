"""
TETRiS core math.

Shared between sponsor-side (TT construction) and regulator-side
(reconstruction and inference). No I/O, no config access.
"""

from math import log
import numpy as np
import torch
import torchtt
import numdifftools as nd
from scipy.optimize import minimize
from scipy.stats import norm


# =============================================================================
# Rescaling between parameter box [a, b] and unit box [-1, 1]
# =============================================================================
def rescale_to_beta(x, lo, hi):
    """Map x in [-1, 1] to beta in [lo, hi]."""
    return np.asarray(x) * (hi - lo) / 2 + (lo + hi) / 2


def hessian_rescale_factor(lo, hi):
    """Multiplier to convert a Hessian computed on [-1, 1] back to beta scale."""
    return 4.0 / ((hi - lo) ** 2)


# =============================================================================
# SCCS / SCRI conditional log-likelihood
# =============================================================================
def conditional_loglike(beta, data_list, event_indices):
    """Average conditional log-likelihood over patients.

    L(beta) = (1/n) sum_i log[ exp(x_{i,t_i}^T beta) / sum_j exp(x_{i,j}^T beta) ].
    """
    total = 0.0
    for X, ev in zip(data_list, event_indices):
        eta = np.exp(X @ beta)
        total += log(eta[ev] / eta.sum())
    return total / len(data_list)


def make_direct_objective(data_list, event_indices, lo, hi):
    """Negative log-likelihood on [-1, 1]^d, for direct patient-level optimization."""
    def obj(x):
        beta = rescale_to_beta(x, lo, hi)
        return -conditional_loglike(beta, data_list, event_indices)
    return obj


# =============================================================================
# Chebyshev grid
# =============================================================================
def chebyshev_points(n):
    """First-kind Chebyshev nodes on [-1, 1], in increasing order."""
    pts = np.cos((np.pi * (np.arange(1, n + 1) - 0.5)) / n)
    return pts[::-1]


def chebyshev_transform_matrix(cheb_pts, n):
    """Discrete Chebyshev transform: maps values on cheb_pts to Chebyshev coefficients."""
    Q = np.ones((n, n))
    for i in range(2, n + 1):
        Q[i - 1, :] = np.cos((i - 1) * np.arccos(cheb_pts))
    Q[0, :] *= 0.5
    Q *= 2.0 / n
    return Q


# =============================================================================
# TT-cross
# =============================================================================
def make_tt_query(data_list, event_indices, cheb_pts, lo, hi):
    """Callable for torchtt.interpolate.dmrg_cross: grid indices -> -loglike."""
    def query(I):
        if len(I) == 1:
            beta = rescale_to_beta(cheb_pts[np.array(I[0])], lo, hi)
            val = -conditional_loglike(beta, data_list, event_indices)
            return torch.tensor(val, dtype=torch.float64)
        values = []
        for idx in I:
            beta = rescale_to_beta(cheb_pts[np.array(idx)], lo, hi)
            values.append(-conditional_loglike(beta, data_list, event_indices))
        return torch.tensor(values, dtype=torch.float64)
    return query


def run_dmrg_cross(query_fn, shape, eps, max_tries):
    """DMRG-cross with retry on transient RuntimeErrors."""
    for _ in range(max_tries):
        try:
            return torchtt.interpolate.dmrg_cross(query_fn, shape, eps=eps)
        except RuntimeError:
            continue
    return None


def tt_cores_numpy(tt_result, round_tol=1e-10):
    """Return TT cores as a list of numpy arrays after rounding."""
    return [np.array(c) for c in tt_result.round(round_tol).cores]


def tt_ranks_from_cores(cores):
    """TT rank sequence [r0, r1, ..., rd] from a list of cores."""
    ranks = [cores[0].shape[0]]
    for c in cores:
        ranks.append(c.shape[2])
    return ranks


def tt_size_bytes(cores):
    """Total storage in bytes for a list of float64 TT cores."""
    return sum(c.size for c in cores) * 8


# =============================================================================
# Core reconstruction: function-value cores -> Chebyshev-coefficient cores
# =============================================================================
def cores_to_cheb_coeffs(cores, Q):
    """Apply Chebyshev transform Q along the middle mode of each TT core."""
    coeff_cores = []
    for core in cores:
        new_core = np.empty_like(core)
        r_front, _, r_rear = core.shape
        for a in range(r_front):
            for b in range(r_rear):
                new_core[a, :, b] = Q @ core[a, :, b]
        coeff_cores.append(new_core)
    return coeff_cores


def make_tt_functional(coeff_cores, grid_n):
    """Return f(x) = reconstructed neg-log-likelihood at x in [-1, 1]^d."""
    def cheb_basis(z):
        return np.cos(np.arange(grid_n) * np.arccos(z))

    def f(x):
        out = np.tensordot(coeff_cores[0], cheb_basis(x[0]), axes=([1], [0]))
        for k in range(1, len(coeff_cores)):
            out = out @ np.tensordot(coeff_cores[k], cheb_basis(x[k]), axes=([1], [0]))
        return out[0, 0]
    return f


def combine_functionals(functionals, weights):
    """Return weighted sum of functionals. Weights are normalized to sum to 1.

    Used to combine per-subgroup negative log-likelihoods into a pooled one.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    def combined(x):
        return sum(wk * fk(x) for wk, fk in zip(w, functionals))
    return combined


# =============================================================================
# Inference
# =============================================================================
def fit_and_wald_ci(objective, d, n_eff, lo, hi, fixed_zero=None):
    """Minimize objective on [-1, 1]^d, return point estimates and Wald CIs.

    Args:
        objective: function of x in [-1, 1]^d returning negative log-likelihood
            (already averaged over patients).
        d: dimension of the full parameter space.
        n_eff: effective sample size used to scale the Hessian (patient count).
        lo, hi: parameter box bounds.
        fixed_zero: optional list of coordinate indices to fix at 0 (beta = 0,
            i.e. x = -(lo+hi)/(hi-lo)). Used for submodel analysis.

    Returns:
        dict with keys: beta, rr, ci_low, ci_high, active_idx.
    """
    active_idx = [k for k in range(d) if fixed_zero is None or k not in fixed_zero]
    n_active = len(active_idx)

    # x = 0 in rescaled space corresponds to beta = (lo + hi) / 2.
    # For beta = 0 we need x = -(lo + hi) / (hi - lo).
    x_for_zero = -(lo + hi) / (hi - lo)

    def embed(x_active):
        x_full = np.full(d, x_for_zero, dtype=float)
        for j, k in enumerate(active_idx):
            x_full[k] = x_active[j]
        return x_full

    def restricted_obj(x_active):
        return objective(embed(x_active))

    x0 = np.zeros(n_active)
    bounds = [(-1.0, 1.0)] * n_active
    opt = minimize(fun=restricted_obj, x0=x0, method="L-BFGS-B", bounds=bounds)

    # Hessian on the active subspace, in rescaled coordinates
    H_rescaled = nd.Hessian(restricted_obj)(opt.x)
    H = H_rescaled * hessian_rescale_factor(lo, hi)

    # Wald CIs on the log scale, per manuscript convention
    cov_log = np.linalg.inv(H) / n_eff
    sd_log = np.sqrt(np.diag(cov_log))
    z = norm.ppf(0.975)

    beta_full = rescale_to_beta(embed(opt.x), lo, hi)
    beta_active = beta_full[active_idx]

    return {
        "beta_full": beta_full,
        "beta_active": beta_active,
        "active_idx": active_idx,
        "rr": np.exp(beta_active),
        "ci_low": np.exp(beta_active - z * sd_log),
        "ci_high": np.exp(beta_active + z * sd_log),
    }
