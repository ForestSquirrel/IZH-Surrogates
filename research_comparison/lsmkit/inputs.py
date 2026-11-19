"""
inputs.py -- input spike-train generators (times in ms, channel ids 0..nin-1).
"""
import numpy as np


def poisson(nin, rate_hz, T_ms, rng):
    """independent homogeneous Poisson trains -> (times, channels)"""
    n = rng.poisson(rate_hz * T_ms / 1000.0, nin)
    t = np.concatenate([np.sort(rng.uniform(0, T_ms, k)) for k in n]) if n.sum() else np.zeros(0)
    ch = np.repeat(np.arange(nin), n)
    return t, ch


def warp_jitter(template, scale, jitter_ms, rng, t_max=None):
    """Maass 2003 §18.5.1: multiply the time axis by `scale`, then move every spike by
    N(0, jitter_ms); spikes before 0 (or after t_max) are dropped."""
    t, ch = template
    t = t * scale + rng.normal(0.0, jitter_ms, len(t))
    keep = t >= 0
    if t_max is not None:
        keep &= t < t_max
    return t[keep], ch[keep]


def sin_warp(template, rng, f_hz=2.0, K_range=(0.5, 2.0)):
    """Maass 2003 footnote ††: t' = B + K (t + sin(2 pi f t + phi) / (2 pi f)), g(0) = 0."""
    t, ch = template
    K = rng.uniform(*K_range)
    phi = rng.uniform(0, 2 * np.pi)
    w = 2 * np.pi * f_hz / 1000.0
    g = lambda x: K * (x + np.sin(w * x + phi) / w)
    return g(t) - g(0.0), ch
