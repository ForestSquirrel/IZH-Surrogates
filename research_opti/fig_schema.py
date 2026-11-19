"""
fig_schema.py -- what the eighteen features measure, drawn on the dynamics.

The companion figure to the optimizer results. It answers "what does this feature set
capture about a neuron?" without appealing to the atlas, which belongs to the amortized
inference work and has no place in a paper about the schema and its protocols.

Every panel is one protocol run on an Izhikevich cell chosen because it *displays* the
feature in question -- Izhikevich's own canonical regimes, from `v3_presets`. That choice
is deliberate: a single out-of-family target would leave several panels showing a feature
that happens to be zero for it (Hodgkin-Huxley does not adapt, does not rebound, and
accommodates so hard that `gamma_ramp` saturates), which demonstrates nothing. This is a
visual account of what each quantity is, not a claim about any one model.

The drives come from the pipeline's own `stim_v3`, the counts from its own kernels, and
the annotated values are the features the pipeline actually measured on that cell.

    $PY fig_schema.py            -> figs/fig5_features.{pdf,png}
"""
import os
import argparse
import numpy as np

import common as K
import v3_config as C
import traces as TR

import sys
import style                                                   # noqa: E402,F401 (rcParams)
import matplotlib.pyplot as plt                                # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#c9c7c1"
ACC, ACC2, ACC3 = "#1c5cab", "#d94801", "#2a9d4a"

# the cell each panel uses, picked so the feature is visible rather than incidental
# Chosen from the measured feature values over all twenty presets, not from their names.
# Two things that choice has to respect. Izhikevich's "R accomodation" and "P bistability"
# have b >= 1 and fire at zero current, so their rheobase is ~1e-14 and every gamma-scaled
# drive collapses -- they cannot be used at all. And "F spike freq adapt" shows A ~ 1e-11
# under a gamma = 2 step: it is periodic from its first interval, so the canonical name
# does not imply the feature is visible. "N rebound burst" is the only in-box preset with
# strong adaptation (A = 1.18), so it carries that panel whatever its name says.
CELLS = {
    "latency":   "H Class 2",            # ell  = 0.553, the longest in-box delay
    "adapt":     "N rebound burst",      # A    = 1.18
    "burst":     "C tonic bursting",     # p    = 4, rho = 43.7
    "fi_lo":     "G Class 1",            # beta = 0.593, lam = 10.7
    "fi_hi":     "H Class 2",            # beta = 0.048, lam = 1.41
    "pulse":     "A tonic spiking",
    "ramp":      "M rebound spike",      # gamma_ramp = 1.28
    "rebound":   "D phasic bursting",    # N_reb = 9, g_reb = 3
    "hyst":      "K resonator",          # eta  = -0.871
}


def theta_of(name):
    from v3_presets import PRESETS
    p = PRESETS[name]
    return np.array([p["a"], p["b"], p["c"], p["d"]], float)


def landmarks(thetas):
    """(i_rh, T_cycle) per cell, from the pipeline's own PASS 0."""
    import v3_gpu as G
    lm = G.landmarks_gpu(np.ascontiguousarray(np.asarray(thetas, float)))
    return lm[:, 0].astype(float), lm[:, 1].astype(float)


def features(theta):
    from v3_core import alloc_buffers
    from v3_features import compute_features_v3
    from v3_models import izh_rhs_v3
    out = np.full(C.N_COLS, C.NAN_SENTINEL)
    f, it = compute_features_v3(izh_rhs_v3, np.ascontiguousarray(theta),
                                alloc_buffers(), out=out)
    return dict(zip(C.FEATURE_NAMES, np.asarray(f[:C.N_ACTIVE], float))), it


def run(theta, stim_id, sp, t_end, h=None):
    from v3_models import izh_rhs_v3
    th = np.asarray(theta, float)
    y0 = np.array([C.V_REST, th[1] * C.V_REST])
    return TR.run_stim(izh_rhs_v3, th, y0, stim_id, np.asarray(sp, float),
                       h or C.DT, t_end, C.V_TH, reset=(th[2], th[3]), v_th=C.V_TH)


def count(theta, stim_id, sp, t_end, t_lo, t_hi):
    from v3_core import count_in_window_uncapped
    from v3_models import izh_rhs_v3
    th = np.ascontiguousarray(np.asarray(theta, float))
    return count_in_window_uncapped(izh_rhs_v3, stim_id, np.asarray(sp, float), 1.0,
                                    th, 0.0, t_end, C.DT, t_lo, t_hi)


def _tag(ax, name, extra="", loc="upper left"):
    """The cell and the measured value, inside the axes.

    Above the axes they collided with the titles and with each other once the values
    were added; inside, each panel owns its own space.
    """
    th = theta_of(name)
    txt = f"{name[2:]} ({th[0]:g}, {th[1]:g}, {th[2]:g}, {th[3]:g})"
    if extra:
        txt += "\n" + extra
    # above the axes, under the title: inside, every corner of every panel has data in
    # it at some point, and the tag lands on a trace
    _ = loc
    ax.text(0.0, 1.015, txt, transform=ax.transAxes,
            fontsize=5.6, color=INK2, ha="left", va="bottom", linespacing=1.5)


def main():
    names = sorted({v for v in CELLS.values()})
    th_all = np.array([theta_of(n) for n in names])
    irh, tcy = landmarks(th_all)
    L = {n: (irh[i], tcy[i]) for i, n in enumerate(names)}
    F = {n: features(theta_of(n))[0] for n in names}

    fig, axs = plt.subplots(2, 4, figsize=(7.9, 5.0),
                            gridspec_kw=dict(hspace=1.00, wspace=0.52))

    # ------------------------------------------------- (a) P1 latency ---------
    ax = axs[0, 0]
    nm = CELLS["latency"]
    i_rh, tc = L[nm]
    t_on = 1.2 * tc
    t, v, I, spk = run(theta_of(nm), C.STIM_STEP, [t_on, C.GAMMA_STEP * i_rh],
                       t_on + 3.5 * tc)
    ax.plot(t / tc, v, color=INK, lw=0.7)
    ax.axvline(t_on / tc, color=GRID, lw=0.8)
    if spk.size:
        ax.axvspan(t_on / tc, spk[0] / tc, color=ACC2, alpha=0.16, lw=0)
        ax.text(0.5 * (t_on + spk[0]) / tc, 0.90, r"$\ell$",
                transform=ax.get_xaxis_transform(), fontsize=8, color=ACC2,
                ha="center", va="center")
    _tag(ax, nm, rf"$\ell$ = {F[nm]['ell']:.2f}", loc="upper right")
    ax.set_title("P1  first-spike latency", fontsize=7.5, pad=25)
    ax.set_ylabel("$v$ (mV)", fontsize=8)
    ax.set_xlabel(r"$t/T_{\rm cyc}$", fontsize=8)

    # ------------------------------------------------- (b) P1 adaptation ------
    ax = axs[0, 1]
    nm = CELLS["adapt"]
    i_rh, tc = L[nm]
    t, v, I, spk = run(theta_of(nm), C.STIM_STEP, [C.K_TRANS * tc, C.GAMMA_STEP * i_rh],
                       C.K_TRANS * tc + 40 * tc)
    isi = np.diff(spk) / tc
    ax.plot(np.arange(len(isi)), isi, "o-", ms=2.4, lw=0.7, color=INK, mew=0)
    if len(isi) > 6:
        tail = float(np.mean(isi[-5:]))
        ax.axhline(isi[0], color=ACC2, lw=0.7, ls="--")
        ax.axhline(tail, color=ACC, lw=0.7, ls="--")
        x = 0.55 * len(isi)
        ax.annotate("", xy=(x, tail), xytext=(x, isi[0]),
                    arrowprops=dict(arrowstyle="<->", color=ACC2, lw=0.8))
        ax.text(x + 0.03 * len(isi), 0.5 * (isi[0] + tail), "$A$", fontsize=7.5,
                color=ACC2, va="center")
    _tag(ax, nm, f"$A$ = {F[nm]['A']:+.2f}", loc="lower right")
    ax.set_title("P1  adaptation over the train", fontsize=7.5, pad=25)
    ax.set_xlabel("interval index", fontsize=8)
    ax.set_ylabel(r"ISI / $T_{\rm cyc}$", fontsize=8)

    # ------------------------------------------------- (c) P1 burst shape -----
    # The intervals themselves, not the trace: within one burst the short and long
    # intervals differ by a factor of 43, which on a voltage trace is a sub-pixel gap
    # against a 90-cycle window. A log axis shows both, and the repeating pattern of
    # p = 4 levels is what `p` counts.
    ax = axs[0, 2]
    nm = CELLS["burst"]
    i_rh, tc = L[nm]
    t, v, I, spk = run(theta_of(nm), C.STIM_STEP,
                       [C.K_TRANS * tc, C.GAMMA_STEP * i_rh],
                       C.K_TRANS * tc + 6.0 * tc)
    isi = np.diff(spk) / tc
    n_show = min(len(isi), 14)
    ax.semilogy(np.arange(n_show), isi[:n_show], "o-", ms=3.0, lw=0.8, color=INK, mew=0)
    pp = int(F[nm]["p"]) if F[nm]["p"] > C.SENT_CUT else 0
    if pp > 1 and n_show > pp:
        for q in range(0, n_show - 1, pp):
            ax.axvspan(q - 0.4, q + pp - 0.6, color=ACC, alpha=0.10, lw=0)
        one = isi[:pp]
        ax.axhline(float(one.max()), color=ACC2, lw=0.7, ls="--")
        ax.axhline(float(one.min()), color=ACC, lw=0.7, ls="--")
        ax.annotate("", xy=(n_show - 1.2, float(one.min())),
                    xytext=(n_show - 1.2, float(one.max())),
                    arrowprops=dict(arrowstyle="<->", color=ACC2, lw=0.8))
        ax.text(n_show - 1.6, float(np.sqrt(one.min() * one.max())), r"$\rho$",
                fontsize=8, color=ACC2, ha="right", va="center")
    _tag(ax, nm, f"$p$ = {F[nm]['p']:.0f},  " rf"$\rho$ = {F[nm]['rho']:.1f}",
         loc="upper right")
    ax.set_title(r"P1  interval pattern: $p$, $\rho$", fontsize=7.5, pad=25)
    ax.set_xlabel("interval index", fontsize=8)
    ax.set_ylabel(r"ISI / $T_{\rm cyc}$", fontsize=8)

    # ------------------------------------------------- (d) P2 f--I ------------
    ax = axs[0, 3]
    for nm, col in ((CELLS["fi_lo"], ACC2), (CELLS["fi_hi"], ACC)):
        i_rh, tc = L[nm]
        g_tr, g_win = C.K_TRANS * tc, C.K_WIN * tc
        f = np.array([count(theta_of(nm), C.STIM_STEP, [g_tr, g * i_rh],
                            g_tr + g_win, g_tr, g_tr + g_win) / g_win
                      for g in C.FI_GAMMAS])
        gam = np.array(C.FI_GAMMAS)
        m = f > 0
        ax.plot(gam[m] - 1.0, f[m] * tc, "o-", ms=2.4, lw=0.8, color=col, mew=0,
                label=f"{nm[2:]}  " rf"$\beta$={F[nm]['beta']:.2f}, "
                      rf"$\lambda$={F[nm]['lam']:.1f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=5.2, loc="lower right", handlelength=1.2,
              labelspacing=0.25)
    ax.set_title(r"P2  f--I shape: $\beta$, $\chi$, $\lambda$", fontsize=7.5, pad=25)
    ax.set_xlabel(r"$\gamma - 1$", fontsize=8)
    ax.set_ylabel(r"rate $\times\,T_{\rm cyc}$", fontsize=8)

    # ------------------------------------------------- (e) P4 pulse pair ------
    ax = axs[1, 0]
    nm = CELLS["pulse"]
    i_rh, tc = L[nm]
    g_tr = C.K_TRANS * tc
    w = C.K_PULSE_WIDTH * tc
    d_max = C.K_DELTA_MAX * tc
    step = d_max / (C.DELTA_GRID_N - 1)
    deltas = np.arange(C.DELTA_GRID_N) * step
    resp = np.array([count(theta_of(nm), C.STIM_PULSE_PAIR,
                           [g_tr, g_tr + d, w, C.GAMMA_PULSE * i_rh],
                           g_tr + d + w + C.K_WIN_PULSE * tc,
                           g_tr, g_tr + d + w + C.K_WIN_PULSE * tc)
                     for d in deltas])
    ax.plot(deltas / tc, resp, "o-", ms=2.4, lw=0.8, color=INK, mew=0)
    ds = F[nm]["delta_star"]
    if ds > C.SENT_CUT:
        ax.axvline(ds, color=ACC2, lw=0.9, ls="--")
        ax.annotate(r"$\delta^*$", xy=(ds, float(resp.min() + 0.5 * np.ptp(resp))),
                    xytext=(ds + 0.30 * C.K_DELTA_MAX,
                            float(resp.min() + 0.25 * np.ptp(resp))),
                    fontsize=7.5, color=ACC2,
                    arrowprops=dict(arrowstyle="-|>", color=ACC2, lw=0.7))
    _tag(ax, nm, rf"$\delta^*$ = {ds:.1f}", loc="lower left")
    ax.set_title(r"P4  pulse pair: $\delta^*$", fontsize=7.5, pad=25)
    ax.set_xlabel(r"separation / $T_{\rm cyc}$", fontsize=8)
    ax.set_ylabel("spikes", fontsize=8)

    # ------------------------------------------------- (f) P5 ramp ------------
    ax = axs[1, 1]
    nm = CELLS["ramp"]
    i_rh, tc = L[nm]
    t_ramp = C.K_RAMP * tc
    t, v, I, spk = run(theta_of(nm), C.STIM_RAMP_HOLD,
                       [0.4 * tc, 0.4 * tc + t_ramp, C.RAMP_GAMMA_F * i_rh],
                       0.4 * tc + 1.25 * t_ramp)
    ax.plot(t / tc, v, color=INK, lw=0.7)
    ax2 = ax.twinx()
    ax2.plot(t / tc, I / i_rh, color=ACC, lw=0.9)
    ax2.axhline(1.0, color=GRID, lw=0.7, ls=":")
    ax2.text(0.97, 0.52, r"$I/i_{\rm rh}$", transform=ax2.transAxes, fontsize=6.6,
             color=ACC, ha="right", va="bottom")
    ax2.tick_params(labelsize=6.0, colors=ACC)
    if spk.size:
        g_at = float(np.interp(spk[0], t, I)) / i_rh
        ax2.plot(spk[0] / tc, g_at, "o", ms=4.5, mfc="none", mec=ACC2, mew=1.1)
        ax2.annotate(r"$\gamma_{\rm ramp}$", xy=(spk[0] / tc, g_at),
                     xytext=(spk[0] / tc - 0.42 * t_ramp / tc, g_at + 0.35),
                     fontsize=7, color=ACC2,
                     arrowprops=dict(arrowstyle="-|>", color=ACC2, lw=0.7))
    _tag(ax, nm, rf"$\gamma_{{\rm ramp}}$ = {F[nm]['gamma_ramp']:.2f}",
         loc="upper right")
    ax.set_title("P5  ramp: the dynamic threshold", fontsize=7.5, pad=25)
    ax.set_xlabel(r"$t/T_{\rm cyc}$", fontsize=8)
    ax.set_ylabel("$v$ (mV)", fontsize=8)

    # ------------------------------------------------- (g) P6 rebound ---------
    ax = axs[1, 2]
    nm = CELLS["rebound"]
    i_rh, tc = L[nm]
    for g, col, ls in ((1.5, GRID, "-"), (3.5, ACC, "-"), (9.0, ACC2, "-")):
        t, v, I, spk = run(theta_of(nm), C.STIM_HYPER_REL,
                           [0.4 * tc, 0.4 * tc + C.K_HYPER * tc, -g * i_rh],
                           0.4 * tc + C.K_HYPER * tc + 2.2 * tc)
        ax.plot(t / tc, v, lw=0.7, color=col, label=rf"$\gamma = -{g:g}$")
    ax.axvline((0.4 * tc + C.K_HYPER * tc) / tc, color=INK2, lw=0.7, ls="--")
    ax.legend(frameon=False, fontsize=5.4, loc="center left", handlelength=1.2,
              labelspacing=0.2)
    _tag(ax, nm, rf"$N_{{\rm reb}}$ = {F[nm]['N_reb']:.0f}, "
                 rf"$g_{{\rm reb}}$ = {F[nm]['g_reb']:.1f}", loc="lower right")
    ax.set_title(r"P6  release: $N_{\rm reb}$, $g_{\rm reb}$", fontsize=7.5, pad=25)
    ax.set_xlabel(r"$t/T_{\rm cyc}$", fontsize=8)
    ax.set_ylabel("$v$ (mV)", fontsize=8)

    # ------------------------------------------------- (h) P7 hysteresis ------
    ax = axs[1, 3]
    nm = CELLS["hyst"]
    i_rh, tc = L[nm]
    t_half = 40.0 * tc
    t, v, I, spk = run(theta_of(nm), C.STIM_HYST,
                       [0.0, t_half, 2 * t_half, C.HYST_GAMMA_MAX * i_rh], 2 * t_half)
    ax.plot(I / i_rh, v, color=GRID, lw=0.3)
    if spk.size:
        i_on = float(np.interp(spk[0], t, I)) / i_rh
        i_off = float(np.interp(spk[-1], t, I)) / i_rh
        for xv, col, nmm in ((i_on, ACC2, r"$I_{\rm on}$"),
                             (i_off, ACC, r"$I_{\rm off}$")):
            ax.axvline(xv, color=col, lw=0.9, ls="--")
            ax.text(xv, 0.015, nmm, transform=ax.get_xaxis_transform(), fontsize=6.4,
                    color=col, ha="center", va="bottom")
        y = float(np.min(v)) + 0.18 * (float(np.max(v)) - float(np.min(v)))
        ax.annotate("", xy=(i_off, y), xytext=(i_on, y),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
        ax.text(0.5 * (i_on + i_off), y + 0.07 * (float(np.max(v)) - float(np.min(v))),
                r"$\eta$", fontsize=7.5, color=INK2, ha="center")
    _tag(ax, nm, rf"$\eta$ = {F[nm]['eta']:+.2f}", loc="upper right")
    ax.set_title(r"P7  hysteresis: $\eta$", fontsize=7.5, pad=25)
    ax.set_xlabel(r"$I/i_{\rm rh}$ (up, then down)", fontsize=8)
    ax.set_ylabel("$v$ (mV)", fontsize=8)

    for n, ax in enumerate(axs.ravel()):
        ax.spines["top"].set_visible(False)
        if n not in (5,):                       # (f) keeps its right-hand axis
            ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6.4)
        ax.text(-0.13, 1.33, f"({chr(97 + n)})", transform=ax.transAxes, fontsize=8.5,
                ha="right", va="bottom")
    fig.text(0.5, 1.02, "each panel: one protocol on the Izhikevich regime that displays "
             "the feature; the quoted value is the one the pipeline measured",
             fontsize=6.8, color=INK2, ha="center")

    os.makedirs(K.FIGS, exist_ok=True)
    out = os.path.join(K.FIGS, "fig5_features")
    for e in ("pdf", "png"):
        fig.savefig(f"{e}" and f"{out}.{e}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig5_features.pdf/.png")
    for nm in names:
        print(f"  {nm:22s} i_rh {L[nm][0]:8.4g}  T_cyc {L[nm][1]:8.4g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    main()
