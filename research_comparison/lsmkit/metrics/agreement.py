"""
agreement.py -- target vs surrogate LSM on identical circuits and inputs.

State level (all instances): CKA between liquid-state matrices, per-neuron spike-count
correlation across instances (needs many instances), spike-count correlation across neurons
within an instance, population rate ratio.
Spike level (L1-recorded instances only): Kistler coincidence factor Gamma per neuron.
"""
import numpy as np


def cka(a, b):
    """linear CKA between (n, d1) and (n, d2) state matrices"""
    a = a - a.mean(0)
    b = b - b.mean(0)
    hsic = np.linalg.norm(a.T @ b) ** 2
    den = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    return float(hsic / den) if den > 0 else float("nan")


def count_corr(ct, cs):
    """per-neuron Pearson r of spike counts across instances; nan where either is constant"""
    ct = ct.astype(float); cs = cs.astype(float)
    at = ct - ct.mean(0); as_ = cs - cs.mean(0)
    den = np.sqrt((at ** 2).sum(0) * (as_ ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (at * as_).sum(0) / den, np.nan)


def gamma_coinc(t_ref, t_mod, delta, T):
    """Kistler et al. 1997 coincidence factor (Jolivet 2008 form); times and delta in the
    same units, T = duration. 1 = identical, ~0 = chance. nan if either train is empty."""
    n1, n2 = len(t_ref), len(t_mod)
    if n1 == 0 or n2 == 0:
        return float("nan")
    t_mod = np.sort(t_mod)
    j = np.searchsorted(t_mod, t_ref)
    lo = np.abs(t_ref - t_mod[np.clip(j - 1, 0, n2 - 1)])
    hi = np.abs(t_mod[np.clip(j, 0, n2 - 1)] - t_ref)
    nc = int((np.minimum(lo, hi) <= delta).sum())
    nu = n2 / T
    norm = 1 - 2 * nu * delta
    if norm <= 0:
        return float("nan")
    return (nc - 2 * nu * delta * n1) / (0.5 * (n1 + n2) * norm)


def gamma_table(sp_t_ref, sp_n_ref, sp_t_mod, sp_n_mod, dt, delta_ms, T_ms, t0_ms=0.0):
    """Gamma for every (recorded instance, neuron) -> (K, N), nan for silent pairs.
    Spikes are restricted to the common window [t0, t0 + T) (T may be per instance);
    the caller must make sure the capped L1 buffers cover that window."""
    K, N = sp_n_ref.shape
    g = np.full((K, N), np.nan)
    for k in range(K):
        T = T_ms[k] if np.ndim(T_ms) else T_ms
        for i in range(N):
            a = sp_t_ref[k, i, :sp_n_ref[k, i]] * dt
            b = sp_t_mod[k, i, :sp_n_mod[k, i]] * dt
            a = a[(a >= t0_ms) & (a < t0_ms + T)]
            b = b[(b >= t0_ms) & (b < t0_ms + T)]
            g[k, i] = gamma_coinc(a, b, delta_ms, T)
    return g


def rate_corr_neurons(ct, cs):
    """per instance: Pearson r over neurons of spike counts (does the surrogate keep WHICH
    neurons are active?) -> (K,)"""
    ct = ct.astype(float); cs = cs.astype(float)
    out = np.full(len(ct), np.nan)
    for k in range(len(ct)):
        if ct[k].std() > 0 and cs[k].std() > 0:
            out[k] = np.corrcoef(ct[k], cs[k])[0, 1]
    return out
