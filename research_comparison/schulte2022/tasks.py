"""
tasks.py -- spike-pattern tasks of Haeusler & Maass 2007 as replicated by Schulte 2022 (§2.2.1).

One continuous input per run (no reset between trials): DISCARD_SEG + 15 * (1500 + 300)
segments of 30 ms. For each stream and each of the 15 segment positions two templates
(bit 0 / 1) of 40 trains are fixed; a random bit per segment selects one. Every input spike
is jittered by N(0, 1 ms) once (all target neurons see the same jittered train). A template
holds round(40 * 30 ms * 20 Hz) spikes (fractional part as a probability, CODE) at uniform
times on uniformly chosen channels. Each stream has its own templates (paper; the authors'
code reuses stream 1's for stream 2).

Targets per trial (state read at the trial's end): tcl_s = bit of segment 15 of stream s,
delayed tcl_s = bit of segment 14, XOR = tcl_1 xor tcl_2.
"""
import numpy as np

import spec as S


def template(rng):
    lam = S.N_CH * S.T_SEG / 1000.0 * S.RATE
    n = int(lam) + (rng.random() < lam - int(lam))
    return np.sort(rng.uniform(0.0, S.T_SEG, n)), rng.integers(0, S.N_CH, n)


def stream(rng, nseg, ch0):
    tmpl = [[template(rng) for _ in range(2)] for _ in range(S.N_SEG)]
    bits = rng.integers(0, 2, nseg)
    t, ch = [], []
    for k in range(nseg):
        tt, cc = tmpl[k % S.N_SEG][bits[k]]
        t.append(k * S.T_SEG + tt + rng.normal(0.0, S.JITTER, len(tt)))
        ch.append(cc + ch0)
    t, ch = np.concatenate(t), np.concatenate(ch)
    keep = t >= 0
    return t[keep], ch[keep], bits


def make(rng, n_trials=S.N_TRAIN + S.N_TEST, bits=False):
    """-> (times, channels), targets per trial, duration ms [, (bits1, bits2) per segment]"""
    nseg = S.DISCARD_SEG + S.N_SEG * n_trials
    t1, c1, b1 = stream(rng, nseg, 0)
    t2, c2, b2 = stream(rng, nseg, S.N_CH)
    end = S.DISCARD_SEG + S.N_SEG * np.arange(1, n_trials + 1) - 1     # last segment index
    y = {"tcl1": b1[end], "tcl2": b2[end], "tcl1(t-1)": b1[end - 1], "tcl2(t-1)": b2[end - 1]}
    y["XOR"] = y["tcl1"] ^ y["tcl2"]
    out = (np.concatenate([t1, t2]), np.concatenate([c1, c2])), y, nseg * S.T_SEG
    return out + ((b1, b2),) if bits else out
