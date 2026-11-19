"""
v3_regime.py -- regime stability across the protocol amplitude ladder (§53.5).

WHAT THIS IS FOR

The pipeline is built around exactly ONE bifurcation: the onset of firing at
i_rh. Every duration-normalized feature assumes that above rheobase the cell
occupies a single regime whose properties vary smoothly with drive. The schema
then probes gamma in [1.02, 4.0] -- an f-I sweep across eight amplitudes, a pulse
pair at gamma=3, a reference step at gamma=2 that defines T_cycle and normalizes
everything else, and a hysteresis triangle that SWEEPS gamma = 0..3.

A second regime boundary anywhere inside that range makes the feature vector a
collage of several dynamical systems rather than a description of one (§51 for
Hindmarsh-Rose, §52 for the in-family `D phasic bursting` preset).

THIS DOES NOT GATE ANYTHING.

An unstable verdict is a statement about the FEATURE VECTOR, not a prediction of
failure, and the distinction is load-bearing:

  * `AdEx tonic` scores 17 on this diagnostic -- unstable -- and is matched almost
    exactly (w1_shape 0.0017, d2 0.20). It shares Izhikevich's state dimension,
    threshold and reset convention, so an Izhikevich cell can straddle the same
    boundary the same way and reproduce the collage. §53.3.
  * A posterior drawn from an unstable target may be perfectly usable for a
    particular purpose -- it may reproduce the reference step well while failing
    the f-I curve, which is exactly the right answer if the reference step is
    what the application needs.

So the pipeline draws the posterior regardless and reports that its conditioning
vector is internally inconsistent, leaving the usability judgement to whoever
knows the application. Refusing to answer would be a stronger claim than the
evidence supports.

WHAT IT MEASURES

Sweep gamma over the ladder; at each amplitude record the uncapped spike count
and the number of DISTINCT settled ISIs. The verdict keys on the largest
step-to-step change in the distinct-ISI count.

Count jump alone is NOT predictive and is reported only as context: Hindmarsh-Rose
at r=0.008 has a 26.5x count jump with an ISI jump of 4, and matches well
(w1_shape 0.0115). What predicts spike-train fidelity is the ISI jump
(spearman +0.723 across the 14-target suite, +0.771 within HR alone), and it
predicts d2 not at all (+0.086) -- regime instability corrupts the spike train,
which only W1 sees (§8.1, §54.2).

Cost: len(GAMMA_LADDER) step simulations. No flow, no atlas.
"""
import numpy as np

import v3_config as C

# The amplitudes the schema actually probes, in rheobase units. FI_GAMMAS plus
# the reference step and the pulse/hysteresis amplitude; 2.5 fills the gap the
# f-I sweep leaves between 2 and 3.
GAMMA_LADDER = np.array([1.02, 1.05, 1.10, 1.20, 1.50, 2.00, 2.50, 3.00, 4.00])

# Thresholds from §53.2. Across the 14-target suite the split at 10 separated
# median w1_shape 0.0009 (stable) from 0.2346 (unstable), a 268x gap. The
# 5..10 band is a caution zone rather than a measured one -- no target in the
# suite occupies it, so it is a hedge, not a result.
JUMP_MARGINAL = 5
JUMP_UNSTABLE = 10

# Which protocols read which amplitudes, for the per-protocol breakdown.
PROTOCOL_AT = {
    1.02: "f-I onset", 1.05: "f-I", 1.10: "f-I", 1.20: "f-I, P8' test step",
    1.50: "f-I", 2.00: "P1 ref step (defines T_cycle), P3, P5 ramp",
    2.50: "(between f-I samples)", 3.00: "f-I, P4 pulse pair, P7 hysteresis peak",
    4.00: "f-I",
}


def _verdict(isi_jump):
    if isi_jump >= JUMP_UNSTABLE:
        return "unstable"
    if isi_jump >= JUMP_MARGINAL:
        return "marginal"
    return "stable"


ADVISORY_BLOCK = (
    "DEPOLARIZATION BLOCK above gamma = {g:g}: the cell fires below this "
    "amplitude and is silent at and above it, so every protocol sampling there "
    "measures a silent cell. Those columns are sentinel or zero by the mask "
    "rules, which is correct, but the surviving vector describes only the "
    "sub-block part of the amplitude range.")

ADVISORY = {
    "stable": (
        "Regime stable across the probed amplitude range. The feature vector "
        "describes one dynamical system."),
    "marginal": (
        "Regime changes somewhere in the probed amplitude range. The feature "
        "vector may mix regimes; the posterior is drawn as usual, and agreement "
        "with the target is worth checking per protocol rather than in aggregate."),
    "unstable": (
        "REGIME UNSTABLE across the probed amplitude range: the conditioning "
        "vector is assembled from several dynamical regimes of the same cell, so "
        "it does not describe any single one of them. The posterior is drawn and "
        "reported as usual and is NOT invalidated by this -- a target can be "
        "matched well despite an unstable verdict when the surrogate family can "
        "reproduce the same regime change (AdEx tonic scores 17 here and matches "
        "at w1_shape 0.0017). Expect agreement to vary by protocol; whether the "
        "result is usable depends on which protocol the application cares about, "
        "and that judgement needs the application, not this diagnostic. Spike-"
        "train fidelity (W1) is the metric this predicts; d2 is uninformative "
        "about it."),
}


def summarize(counts, n_isi, gammas=None):
    """Build the report from a completed ladder scan.

    Two DIFFERENT things can break the ladder and they are reported separately,
    because they mean different things:

      * a regime change between two FIRING states -- the collage case (§51/§52)
      * DEPOLARIZATION BLOCK, where the cell stops firing above some amplitude
        and every protocol sampling above it measures silence

    Wilson exposed the need for the split: it fires 199 spikes at gamma=2 and
    ZERO at 2.5, 3.0 and 4.0, so the naive "largest ISI change" was measuring a
    transition into silence and calling it a regime change. Its f-I points at 3.0
    and 4.0, and both P4 and P7 at gamma=3, are taken on a silent cell.

    The ISI jump is therefore computed over the FIRING range only, and block is
    its own flag. Either one makes the vector inconsistent, so the verdict is the
    worse of the two.
    """
    g = GAMMA_LADDER if gammas is None else np.asarray(gammas)
    counts = np.asarray(counts, float)
    n_isi = np.asarray(n_isi, float)
    prev = np.maximum(counts[:-1], 1.0)
    count_jump = float(np.max(counts[1:] / prev)) if len(counts) > 1 else 1.0

    # depolarization block: fires somewhere, then goes silent and stays silent
    fires = counts > 0
    block_at = None
    if fires.any():
        last = int(np.max(np.flatnonzero(fires)))
        if last + 1 < len(counts) and not fires[last + 1:].any():
            block_at = float(g[last + 1])
    silent_protocols = [PROTOCOL_AT.get(float(x), "") for x, f in zip(g, fires) if not f]

    # ISI jump over the firing range only
    fi = np.flatnonzero(fires)
    if len(fi) > 1:
        sub_n = n_isi[fi[0]:fi[-1] + 1]
        sub_g = g[fi[0]:fi[-1] + 1]
        isi_jump = int(np.max(np.abs(np.diff(sub_n))))
        k = int(np.argmax(np.abs(np.diff(sub_n))))
        jb = (float(sub_g[k]), float(sub_g[k + 1]))
    else:
        isi_jump, jb = 0, None

    v = _verdict(isi_jump)
    if block_at is not None:
        v = "unstable"          # half the ladder measuring silence is not "stable"
    return dict(
        gammas=g.tolist(), spike_counts=counts.astype(int).tolist(),
        distinct_isis=n_isi.astype(int).tolist(),
        count_jump=count_jump, isi_jump=isi_jump,
        block_above=block_at, silent_protocols=silent_protocols,
        jump_between=jb,
        jump_protocols=(PROTOCOL_AT.get(jb[0], ""), PROTOCOL_AT.get(jb[1], ""))
                       if jb else None,
        n_distinct_regimes=int(len(set(int(x) for x in n_isi if x > 0))),
        verdict=v,
        advisory=(ADVISORY_BLOCK.format(g=block_at) + " " + ADVISORY[v])
                 if block_at is not None else ADVISORY[v],
        gates_posterior=False)


def _scan(step_fn, i_rh, t_cycle, gammas):
    """step_fn(amp, t_trans, t_end, t_disc) -> (n_total, distinct_settled_isis)."""
    trans = C.K_TRANS * t_cycle
    win = C.K_WIN * t_cycle
    disc = C.K_DISCARD * t_cycle
    counts, nisi = [], []
    for g in gammas:
        n, d = step_fn(g * i_rh, trans, trans + win, trans + disc)
        counts.append(n)
        nisi.append(d)
    return counts, nisi


def regime_report_izh(rhs, params, i_rh, t_cycle, bufs, gammas=GAMMA_LADDER,
                      rheo_hi=None):
    """Ladder scan for an Izhikevich-family (reset) model."""
    from v3_core import integrate_v3
    from v3_config import STIM_PARAMS_LEN, STATE_DIM, V_REST, DT
    th = np.ascontiguousarray(params, dtype=np.float64)
    st = bufs["spike_times"]

    def step(amp, t_trans, t_end, t_ss):
        sp = np.zeros(STIM_PARAMS_LEN)
        sp[0] = t_trans; sp[1] = amp
        y = np.empty(STATE_DIM); y[0] = V_REST; y[1] = th[1] * V_REST
        ns, nt, inv = integrate_v3(rhs, 0, sp, 1.0, y, 0.0, t_end, DT, st,
                                   th[2], th[3], th)
        if inv:
            return 0, 0
        t = st[:ns]
        tail = t[t >= t_ss]
        isis = np.diff(tail) if len(tail) > 1 else np.empty(0)
        d = len(np.unique(np.round(isis, 3))) if len(isis) else 0
        return int(nt), int(d)

    c, n = _scan(step, i_rh, t_cycle, gammas)
    return summarize(c, n, gammas)


def regime_report_an(spec, i_rh, t_cycle, gammas=GAMMA_LADDER, buf_n=1 << 16):
    """Ladder scan for a reset-free foreign model (the B4 path)."""
    from v3_an_core import integrate_an, make_sp
    st = np.empty(buf_n)

    def step(amp, t_trans, t_end, t_ss):
        n, nt, inv = integrate_an(spec.rhs, C.STIM_STEP, make_sp(t_trans, amp),
                                  1.0, spec.y0, 0.0, t_end, spec.h, spec.v_th,
                                  st, spec.params)
        if inv:
            return 0, 0
        t = st[:n]
        tail = t[t >= t_ss]
        isis = np.diff(tail) if len(tail) > 1 else np.empty(0)
        d = len(np.unique(np.round(isis, 3))) if len(isis) else 0
        return int(nt), int(d)

    c, n = _scan(step, i_rh, t_cycle, gammas)
    return summarize(c, n, gammas)


def format_report(rep, name=""):
    """One-screen human summary."""
    L = []
    tag = {"stable": "STABLE", "marginal": "MARGINAL", "unstable": "UNSTABLE"}[rep["verdict"]]
    L.append(f"regime: {tag}   ISI jump {rep['isi_jump']}   "
             f"count jump {rep['count_jump']:.1f}x   "
             f"{rep['n_distinct_regimes']} distinct ISI-counts across the ladder")
    L.append("  gamma : " + " ".join(f"{g:>5g}" for g in rep["gammas"]))
    L.append("  spikes: " + " ".join(f"{c:>5d}" for c in rep["spike_counts"]))
    L.append("  ISIs  : " + " ".join(f"{d:>5d}" for d in rep["distinct_isis"]))
    if rep.get("block_above") is not None:
        L.append(f"  DEPOLARIZATION BLOCK above gamma={rep['block_above']:g}"
                 f"  ({sum(1 for c in rep['spike_counts'] if c == 0)} of "
                 f"{len(rep['spike_counts'])} ladder points silent)")
    if rep["jump_between"] and rep["isi_jump"] >= JUMP_MARGINAL:
        a, b = rep["jump_between"]
        pa, pb = rep["jump_protocols"]
        L.append(f"  largest change between gamma={a:g} [{pa}] and gamma={b:g} [{pb}]")
    L.append("  " + rep["advisory"])
    L.append("  posterior: DRAWN (this diagnostic never gates inference)")
    return "\n".join(L)
