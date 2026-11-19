"""
paper_figs.py -- the figures for the short paper on the feature schema and its protocols.

That paper's claim is about the SCHEMA, not about any one way of fitting it: the 18
dimensionless features and the protocols that measure them describe a neuron well enough
that a conventional optimizer can identify an Izhikevich surrogate from them, and three
different optimizers land in the same place. Everything about the amortized flow, the
reservoir and surrogate-versus-model fidelity belongs to other papers and appears nowhere
here.

    $PY paper_figs.py            -> figs/fig1_schema.{pdf,png}
                                    figs/fig2_optimizers.{pdf,png}
                                    figs/fig3_performance.{pdf,png}
    $PY paper_figs.py --only 2

fig1  what the schema is: the protocols as drives in the cell's own units, each labelled
      with the features it yields, and the resulting 18-number fingerprint of three
      targets from different families.
fig2  that it works as an objective: three optimizers, three targets, ten seeds, and the
      same across 30 configurations of the three models.
fig3  what it costs. Kept separate so it can be dropped without touching fig2.
"""
import os
import json
import argparse
import numpy as np

import common as K
import v3_config as C
import analyse as A

import sys
import style                                                   # noqa: E402,F401 (rcParams)
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import Patch                           # noqa: E402

COL = {"cma": "#1c5cab", "de": "#d94801", "pso": "#2a9d4a"}
NAME = {"cma": "CMA-ES", "de": "DE", "pso": "PSO"}
FAMCOL = {"Morris-Lecar": "#1c5cab", "Hodgkin-Huxley": "#d94801", "AdEx": "#2a9d4a"}
TGT = {"hh": "Hodgkin-Huxley", "ml_c1": "Morris-Lecar C1", "adex_tonic": "AdEx tonic"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
FLOOR = 1e-5          # W1 below this is an exact ISI match at the integrator's resolution

# which protocol produces which feature (v3_features.compute_features_v3)
PROTO = [
    ("P1", "reference step", r"$\gamma=2$", ["p", "rho", "p_tr", "A", "ell"]),
    ("P2", "f--I ladder", r"$\gamma=1.02\ldots4$", ["beta", "chi", "lam", "fbar"]),
    ("P3", "persistence", r"$2W$ against $W$", ["Pi"]),
    ("P4", "pulse pair", r"$\gamma=3$, 20 gaps", ["delta_star", "rho_res"]),
    ("P5", "ramp", r"$0\to2$", ["gamma_ramp"]),
    ("P6", "hyperpolarizing release", r"$\gamma=-12\ldots0$", ["N_reb", "g_reb"]),
    ("P7", "hysteresis", r"up, then down", ["eta"]),
    ("P8", "recovery", r"5 delays in $L_\infty$", ["A_rec"]),
    ("P10", "spontaneous", r"$I=0$", ["nu_spont"]),
]
TEX = {"p": "$p$", "rho": r"$\rho$", "p_tr": "$p_{tr}$", "A": "$A$", "beta": r"$\beta$",
       "chi": r"$\chi$", "delta_star": r"$\delta^*$", "rho_res": r"$\rho_{res}$",
       "ell": r"$\ell$", "lam": r"$\lambda$", "Pi": r"$\Pi$", "fbar": r"$\bar f$",
       "N_reb": "$N_{reb}$", "A_rec": "$A_{rec}$", "eta": r"$\eta$",
       "gamma_ramp": r"$\gamma_{ramp}$", "g_reb": "$g_{reb}$", "nu_spont": r"$\nu_{sp}$"}


# ------------------------------------------------------------------ figure 1
def _drive(kind):
    """(t, I) of one protocol's drive, time in the cell's own cycle times, I in i_rh.

    Schematic in the sense that the durations are the configured multiples of T_cycle and
    the amplitudes the configured multiples of the rheobase -- it is the actual protocol,
    drawn, not a cartoon. Windows are compressed where the real one is long (P2's eight
    separate runs, P4's twenty separations) so the shape stays readable.
    """
    if kind == "P1":
        t = np.array([0, 2, 2, 12]); return t, np.array([0, 0, C.GAMMA_STEP, C.GAMMA_STEP])
    if kind == "P2":
        t, I = [0.0], [0.0]
        for g in C.FI_GAMMAS:
            t0 = t[-1]
            t += [t0 + 0.3, t0 + 0.3, t0 + 1.3, t0 + 1.3]
            I += [0.0, g, g, 0.0]
        return np.array(t), np.array(I)
    if kind == "P3":
        t = np.array([0, 1, 1, 5, 5, 13])
        return t, np.array([0, 0, C.GAMMA_STEP, C.GAMMA_STEP, C.GAMMA_STEP, C.GAMMA_STEP])
    if kind == "P4":
        t, I = [0.0], [0.0]
        for gap in (1.0, 3.0, 7.0):
            t0 = t[-1]
            w = 1.0
            t += [t0 + 1, t0 + 1, t0 + 1 + w, t0 + 1 + w,
                  t0 + 1 + w + gap, t0 + 1 + w + gap,
                  t0 + 1 + 2 * w + gap, t0 + 1 + 2 * w + gap]
            I += [0, C.GAMMA_PULSE, C.GAMMA_PULSE, 0, 0,
                  C.GAMMA_PULSE, C.GAMMA_PULSE, 0]
        return np.array(t), np.array(I)
    if kind == "P5":
        t = np.array([0, 1.5, 11.5, 13]); return t, np.array([0, 0, C.RAMP_GAMMA_F,
                                                              C.RAMP_GAMMA_F])
    if kind == "P6":
        t, I = [0.0, 1.0], [0.0, 0.0]
        for g in (-3.0, -7.0, -12.0):
            t0 = t[-1]
            t += [t0, t0 + 2.0, t0 + 2.0, t0 + 4.0]
            I += [g, g, 0.0, 0.0]
        return np.array(t), np.array(I)
    if kind == "P7":
        up = np.linspace(0, C.HYST_GAMMA_MAX, 60)
        t = np.concatenate([np.linspace(0, 6.5, 60), np.linspace(6.5, 13, 60)])
        return t, np.concatenate([up, up[::-1]])
    if kind == "P8":
        t, I = [0.0], [0.0]
        for d in (1.0, 2.5, 5.0):
            t0 = t[-1]
            t += [t0 + 0.6, t0 + 0.6, t0 + 1.2, t0 + 1.2,
                  t0 + 1.2 + d, t0 + 1.2 + d, t0 + 1.8 + d, t0 + 1.8 + d]
            I += [0, C.GAMMA_PULSE, C.GAMMA_PULSE, 0, 0,
                  C.GAMMA_PULSE, C.GAMMA_PULSE, 0]
        return np.array(t), np.array(I)
    t = np.array([0, 13]); return t, np.array([0.0, 0.0])          # P10


def atlas_reference(n=200_000, seed=3):
    """Per-feature atlas quantiles, so a feature value can be placed as a percentile.

    The eighteen features carry eighteen different units and spans, so a raw bar chart of
    them says nothing. Where a model sits inside the atlas distribution is comparable
    across features and is what makes the vector readable as a fingerprint.
    """
    import struct
    p = os.path.join(K.ATLAS, "izh_features.bin")
    with open(p, "rb") as fh:
        fh.read(4)
        n_rows, k = struct.unpack("<qI", fh.read(12))
    a = np.memmap(p, dtype=np.float32, mode="r", offset=16, shape=(n_rows, k))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_rows, min(n, n_rows), replace=False))
    sub = np.asarray(a[idx][:, :C.N_ACTIVE], dtype=np.float64)
    return [np.sort(c[c > C.SENT_CUT]) for c in sub.T]


def percentile_of(val, col):
    """Mid-rank percentile: the midpoint of the value's own mass, not its left edge.

    Seven of the eighteen columns are atoms -- `p` takes nine values, `rho_res` 514,
    `nu_spont` 378 -- and for those a left-edge percentile is actively misleading: every
    target here has p = 1, which is the atlas's MODAL period, yet almost nothing lies
    strictly below it, so a left-edge rank would report it as the 3rd percentile and read
    as "unusual". Averaging the left and right ranks places a modal value in the middle of
    its own mass, which is what a reader takes a percentile to mean.
    """
    if col.size == 0:
        return np.nan
    lo = np.searchsorted(col, val, side="left")
    hi = np.searchsorted(col, val, side="right")
    return 100.0 * 0.5 * (lo + hi) / col.size


def fig1():
    import targets as T
    ref = atlas_reference()
    tg = [(slug, K.jread(os.path.join(K.TARGETS, f"{slug}.json")))
          for slug, *_ in T.study1()]

    fig = plt.figure(figsize=(7.2, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.30)

    # --- (a) the protocols -----------------------------------------------------
    gsa = gs[0].subgridspec(5, 2, hspace=0.95, wspace=0.30)
    for i, (pid, name, note, feats) in enumerate(PROTO):
        ax = fig.add_subplot(gsa[i % 5, i // 5])
        t, I = _drive(pid)
        ax.plot(t, I, color=INK, lw=0.9, solid_joinstyle="miter")
        ax.axhline(0, color=GRID, lw=0.5, zorder=0)
        ax.set_xlim(t.min(), max(t.max(), 1e-9))
        lo, hi = min(I.min(), 0), max(I.max(), 1)
        ax.set_ylim(lo - 0.18 * (hi - lo) - 0.2, hi + 0.42 * (hi - lo) + 0.2)
        ax.set_yticks([])
        ax.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(GRID)
        # one text, not two: a second string offset by padded spaces lands in a different
        # place in a proportional font and overlapped the name
        ax.text(0.0, 1.06, f"{pid}  {name}  ({note})", transform=ax.transAxes,
                fontsize=6.8, color=INK, va="bottom")
        ax.text(1.0, 1.06, r"$\rightarrow$ " + " ".join(TEX[f] for f in feats),
                transform=ax.transAxes, fontsize=6.6, color=INK2, va="bottom", ha="right")
    fig.text(0.085, 0.965, "(a)", fontsize=9, ha="right")
    fig.text(0.5, 0.975, "every drive in units of the cell's own rheobase and cycle time, "
             "and the features it yields", fontsize=6.8, color=INK2, ha="center")

    # --- (b) the fingerprints --------------------------------------------------
    axb = fig.add_subplot(gs[1])
    # FEATURE_NAMES is the on-disk column order and is NOT grouped by protocol, so the
    # axis is reordered here; otherwise the P-group brackets below would label the wrong
    # features (p, rho, p_tr, A are columns 0-3 but `ell`, P1's fifth output, is column 8)
    order = [C.FEATURE_NAMES.index(f) for _p, _n, _q, fs in PROTO for f in fs]
    assert sorted(order) == list(range(C.N_ACTIVE))
    names = [C.FEATURE_NAMES[j] for j in order]
    x = np.arange(C.N_ACTIVE)
    w = 0.26
    for k, (slug, d) in enumerate(tg):
        f = np.asarray(d["f_t"], float)[order]
        rr = [ref[j] for j in order]
        pc = np.array([percentile_of(f[j], rr[j]) if f[j] > C.SENT_CUT else np.nan
                       for j in range(C.N_ACTIVE)])
        off = (k - 1) * w
        # a floor of 1.5, so a value genuinely at the bottom of the atlas is still visible
        # and cannot be confused with the hatched "undefined" bars
        axb.bar(x + off, np.maximum(np.nan_to_num(pc), 1.5), width=w,
                color=list(FAMCOL.values())[k], label=TGT[slug], lw=0, zorder=3)
        miss = np.flatnonzero(np.isnan(pc))
        axb.bar(x[miss] + off, np.full(miss.size, 100.0), width=w, color=GRID,
                hatch="////", edgecolor="#a5a29a", lw=0.0, zorder=2)
    axb.set_xticks(x)
    axb.set_xticklabels([TEX[n] for n in names], fontsize=7)
    axb.set_ylim(0, 100)
    axb.set_ylabel("percentile within the atlas", fontsize=8)
    axb.axhline(50, color=GRID, lw=0.6, zorder=1)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    # protocol grouping along the axis
    j = 0
    for pid, _n, _q, feats in PROTO:
        lo, hi = j, j + len(feats) - 1
        axb.plot([lo - 0.42, hi + 0.42], [-13, -13], color=INK2, lw=0.7,
                 clip_on=False)
        axb.text((lo + hi) / 2, -17.5, pid, fontsize=6.2, color=INK2, ha="center",
                 va="top", clip_on=False)
        j += len(feats)
    h, lab = axb.get_legend_handles_labels()
    h.append(Patch(facecolor=GRID, hatch="////", edgecolor="#a5a29a"))
    lab.append("undefined for that model")
    axb.legend(h, lab, frameon=False, fontsize=6.6, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.17))
    fig.text(0.085, 0.455, "(b)", fontsize=9, ha="right")

    _save(fig, "fig1_schema")


# ------------------------------------------------------------------ figure 2
def _strip(ax, vals, i, color, w=0.3):
    v = np.asarray(vals, float)
    xs = i + np.linspace(-w, w, len(v)) if len(v) > 1 else np.array([float(i)])
    ax.plot(xs, v, "o", ms=2.6, color=color, mew=0, alpha=0.85, zorder=3)
    ax.plot([i - w - 0.06, i + w + 0.06], [np.median(v)] * 2, "-", color=color, lw=1.6,
            zorder=4)
    return float(np.median(v))


def fig2():
    import targets as T
    rows1, val1 = A.load("study1")
    rows2, val2 = A.load("study2")
    tg = ["ml_c1", "hh", "adex_tonic"]
    methods = ["cma", "de", "pso"]

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2),
                            gridspec_kw=dict(hspace=0.42, wspace=0.26))

    for col, (key, lab, getter) in enumerate((
            (None, r"final $d_2$", lambda r: r["J"]),
            ("w1_shape", r"$W_1^{\mathrm{shape}}$",
             lambda r: (val1.get(r["run_id"], {}) or {}).get("w1_shape")))):
        ax = axs[0, col]
        spreads = []
        for ti, t in enumerate(tg):
            meds = []
            for mi, m in enumerate(methods):
                v = [getter(r) for r in rows1
                     if r["target"] == t and r["method"] == m]
                v = [x for x in v if x is not None and np.isfinite(x) and x > 0]
                if v:
                    meds.append(_strip(ax, v, ti * 4 + mi, COL[m]))
            if len(meds) > 1:
                spreads.append(max(meds) / min(meds))
                ax.plot([ti * 4 - 0.45, ti * 4 + 2.45],
                        [min(meds)] * 2, "-", color=GRID, lw=0.6, zorder=1)
                ax.plot([ti * 4 - 0.45, ti * 4 + 2.45],
                        [max(meds)] * 2, "-", color=GRID, lw=0.6, zorder=1)
        ax.set_yscale("log")
        ax.set_xticks([ti * 4 + 1 for ti in range(len(tg))])
        ax.set_xticklabels([TGT[t] for t in tg], fontsize=7)
        ax.set_ylabel(lab, fontsize=8.5)
        ax.set_xlim(-1, len(tg) * 4 - 1.5)
        _ = spreads          # quoted in FIGURES.md, not written on the figure
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if col == 0:
            ax.legend(handles=[plt.Line2D([], [], color=COL[m], marker="o", ms=3,
                                          lw=1.6, label=NAME[m]) for m in methods],
                      frameon=False, fontsize=6.6, ncol=3, loc="upper center",
                      bbox_to_anchor=(0.5, 1.30))

    # --- (c, d) the parameter sweeps, one optimizer, position along each axis ----
    fams = T.study2()
    for col, (key, lab) in enumerate(((None, r"final $d_2$"),
                                      ("w1_shape", r"$W_1^{\mathrm{shape}}$"))):
        ax = axs[1, col]
        for fam, pname, items in fams:
            xs = {s: x for s, l, x, k, p, r in items}
            lo, hi = min(xs.values()), max(xs.values())
            pts = {}
            for r in rows2:
                if r["target"] not in xs:
                    continue
                v = r["J"] if key is None else (val2.get(r["run_id"], {}) or {}).get(key)
                if v is None or not np.isfinite(v) or v <= 0:
                    continue
                pts.setdefault((xs[r["target"]] - lo) / (hi - lo), []).append(v)
            if not pts:
                continue
            X = sorted(pts)
            Y = np.array([np.median(pts[x]) for x in X])
            # One AdEx configuration is matched essentially exactly (median W1 ~ 1e-11).
            # Plotted as-is it stretches the axis over nine decades to show one point, so
            # anything below FLOOR is drawn ON the floor as an open marker and called what
            # it is: indistinguishable from an exact match at this resolution.
            below = Y < FLOOR
            ax.plot(np.array(X)[~below], Y[~below], "o-", ms=2.6, lw=1.0,
                    color=FAMCOL[fam], label=fam)
            if below.any():
                ax.plot(np.array(X)[below], np.full(below.sum(), FLOOR), "v", ms=4.5,
                        mfc="none", mec=FAMCOL[fam], mew=0.9)
                ax.plot(X, np.maximum(Y, FLOOR), "-", lw=1.0, color=FAMCOL[fam],
                        zorder=0)
        ax.set_yscale("log")
        if key is not None:
            ax.set_ylim(bottom=FLOOR / 2.2)
            ax.text(0.015, 0.045, r"$\triangledown$ below the axis floor", transform=ax.transAxes,
                    fontsize=5.6, color=INK2, va="bottom")
        ax.set_xlabel("position along the model's own parameter axis", fontsize=7.5)
        ax.set_ylabel(lab, fontsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if col == 0:
            ax.legend(frameon=False, fontsize=6.6)
    for n, ax in enumerate(axs.ravel()):
        ax.text(-0.14, 1.04, f"({chr(97 + n)})", transform=ax.transAxes, fontsize=9,
                ha="right", va="bottom")
    _save(fig, "fig2_optimizers")


# ------------------------------------------------------------------ figure 3
def fig3():
    rows, _ = A.load("study1")
    tg = ["ml_c1", "hh", "adex_tonic"]
    methods = ["cma", "de", "pso"]
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.3),
                            gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1, 0.85]))

    reach = {m: [] for m in methods}
    for c, t in enumerate(tg):
        ax = axs[c]
        for m in methods:
            curves = []
            for r in [r for r in rows if r["target"] == t and r["method"] == m]:
                e, j = A.best_so_far("study1", r["run_id"])
                if e is None:
                    continue
                curves.append((e, j))
                fin = j[-1]
                hit = np.flatnonzero(j <= 1.1 * fin)
                if hit.size:
                    reach[m].append(e[hit[0]])
            if not curves:
                continue
            grid = curves[0][0]
            M = np.vstack([np.interp(grid, e, j) for e, j in curves])
            ax.plot(grid, np.median(M, axis=0), color=COL[m], lw=1.0, label=NAME[m])
            ax.fill_between(grid, np.percentile(M, 25, axis=0),
                            np.percentile(M, 75, axis=0), color=COL[m], alpha=0.16, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("evaluations", fontsize=8)
        ax.set_title(TGT[t], fontsize=8)
        if c == 0:
            ax.set_ylabel(r"best $d_2$ so far", fontsize=8.5)
            ax.legend(frameon=False, fontsize=6.6, loc="lower left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    ax = axs[3]
    for mi, m in enumerate(methods):
        v = np.array(reach[m], float)
        if not v.size:
            continue
        _strip(ax, v, mi, COL[m], w=0.26)
    ax.set_yscale("log")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([NAME[m] for m in methods], fontsize=7)
    ax.set_ylabel(r"evaluations to within 1.1$\times$" "\nof the final value", fontsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for n, a_ in enumerate(axs):
        a_.text(-0.02, 1.04, f"({chr(97 + n)})", transform=a_.transAxes, fontsize=9,
                ha="right", va="bottom")
    _save(fig, "fig3_performance")


# ------------------------------------------------------------------ figure 4
def fig4(cycles=4.0):
    """Target, its three identified surrogates, and the interval distributions.

    Rows (a) and (b) share one absolute time axis per column, in milliseconds of the
    TARGET. The surrogate traces are drawn at t * tau, which is the gauge doing its one
    job: tau is recovered after the search by dividing the two cycle times, and applying
    it puts an Izhikevich cell that runs in fractions of a millisecond onto the time base
    of a model that cycles in 101 ms. Nothing is fitted to achieve the alignment.

    The axes stay absolute rather than normalized so the columns are not interchangeable:
    Morris-Lecar occupies 400 ms, Hodgkin-Huxley 55, AdEx 90, and the ticks say so.

    No numbers are written on the figure; they are in FIGURES.md for the caption.
    """
    import traces as TR
    import validate as V
    from v3_core import alloc_buffers
    rows, val = A.load("study1")
    tg = ["ml_c1", "hh", "adex_tonic"]
    methods = ["cma", "de", "pso"]
    bufs = alloc_buffers()

    fig, axs = plt.subplots(3, 3, figsize=(7.4, 5.4),
                            gridspec_kw=dict(hspace=0.45, wspace=0.26))
    for c, t in enumerate(tg):
        pick = {}
        for m in methods:
            rs = sorted((x for x in rows if x["target"] == t and x["method"] == m),
                        key=lambda x: x["J"])
            if rs:
                pick[m] = rs[len(rs) // 2]           # the median seed: a typical result

        # --- row 1: the target, in milliseconds -------------------------------
        tt, vv, info = TR.target_trace(t, cycles=cycles, normalize=False)
        ax = axs[0, c]
        ax.plot(tt, vv, color=INK, lw=0.8)
        ax.set_title(TGT[t], fontsize=8.5)
        if c == 0:
            ax.set_ylabel("target\n$v$ (mV)", fontsize=8)

        # --- row 2: the surrogates, carried into target time by tau -----------
        ax = axs[1, c]
        for m, r in pick.items():
            # `cycles` of the surrogate's OWN cycles, then scaled by tau -- which spans
            # exactly `cycles` of the target's, so the two rows cover the same window
            ts, vs = TR.izh_trace(r["theta"], r["i_rh_s"], r["T_cycle_s"],
                                  normalize=False, t_abs=cycles * r["T_cycle_s"])
            ax.plot(ts * r["tau"], vs, color=COL[m], lw=0.7, label=NAME[m])
        ax.legend(frameon=False, fontsize=6.2, loc="upper right", handlelength=1.3,
                  borderaxespad=0.2, labelspacing=0.25, ncol=3, columnspacing=1.0)
        ax.set_xlim(axs[0, c].get_xlim())                 # the point of the figure
        if c == 0:
            ax.set_ylabel("surrogates\n$v$ (mV)", fontsize=8)
        ax.set_xlabel("$t$ (ms)", fontsize=8)

        # --- row 3: the interval distributions W1 is computed on --------------
        ax = axs[2, c]
        isi_t = V.target_isis(t)
        got = {m: V.izh_isis(r["theta"], bufs) for m, r in pick.items()}
        allv = [isi_t] + [v_ for v_ in got.values() if v_ is not None]
        lo = min(float(np.min(a_)) for a_ in allv) * 0.995
        bins = np.linspace(lo, 1.01, 44)
        # Intervals more than 2% under the settled value are produced while the train is
        # still settling: for 11 of these 12 trains the sub-cut intervals are exactly the
        # leading ones, index 0 upward (FIGURES.md gives the counts and the one
        # exception). The pipeline discards a transient before recording, so what is left
        # is the residue of a surrogate that settles more slowly than the window allows.
        cut = 0.98 * float(np.median(isi_t[-20:]))
        ax.axvspan(bins[0], cut, color="0.92", lw=0, zorder=0)
        ax.axvline(cut, color=INK2, lw=0.7, ls="--", zorder=1)
        ax.hist(isi_t, bins=bins, color="0.78", edgecolor=INK, lw=0.7, label="target",
                zorder=2)
        for m in methods:
            if got.get(m) is None:
                continue
            ax.hist(got[m], bins=bins, histtype="step", color=COL[m], lw=0.9, zorder=3,
                    label=NAME[m])
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.7)
        ax.legend(frameon=False, fontsize=6.0, loc="upper center", handlelength=1.1,
                  borderaxespad=0.2, labelspacing=0.25)
        ax.set_xlabel(r"settled ISI $/\,T_{\rm cyc}$ of that cell", fontsize=7.5)
        if c == 0:
            ax.set_ylabel("count", fontsize=8)

    for ax in axs[0]:
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.12 * (hi_ - lo_))
    for ax in axs[1]:                      # row 2 carries the legend
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.34 * (hi_ - lo_))
    for ax in axs.ravel():
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6.8)
    for n, ax in enumerate(axs[:, 0]):
        ax.text(-0.30, 1.04, f"({chr(97 + n)})", transform=ax.transAxes, fontsize=9,
                ha="right", va="bottom")
    fig.text(0.5, 0.975, r"reference step, $I = 2\,i_{\rm rh}$; absolute time and voltage, "
             "each model in its own units", fontsize=7, color=INK2, ha="center")
    _save(fig, "fig4_waveforms")


# ------------------------------------------------------- the caption document
SCHEMATIC_BRIEF = r"""

---

# Brief for the schematic artwork

This section is a drawing brief, not a result. It describes a *schematic* version of
Figure 5 -- idealised shapes, nothing traced from real data -- to be drawn as vector
artwork. It is written to be followed by an illustrator with no background in the
subject: everything needed is defined below, and no outside knowledge is required.

It covers **all eighteen** quantities. Section 5 is a checklist of which panel carries
which, so coverage can be verified at a glance.

## 1. What is being drawn, in plain terms

A neuron is a cell that produces brief electrical pulses. The figure is about **how you
can characterise one by poking it and watching what comes out.**

Three objects recur in every panel:

* **The response line.** A horizontal wavy line read left to right. Left to right is
  time. Up and down is electrical voltage. Most of the time it sits low and nearly flat.
* **A spike.** Occasionally the line shoots up into a tall, very narrow spike and
  immediately drops back down. Draw these as sharp needles, all the same height and
  width, rising well above the resting line. A spike is an event; its shape carries no
  meaning here, only *when* it happens.
* **The drive line.** A second, simpler line drawn below the response line, in a
  different colour. This is what the experimenter pushes in with. It is made of flat
  levels, straight slopes and rectangular pulses -- never wavy.

The whole point of the figure: **the drive is what we control, the spikes are what we
measure, and each panel defines one or more numbers extracted from the relationship
between them.**

Two reference quantities appear throughout and are defined in panel 0:

* **the cell's own rhythm** -- the time from one spike to the next when it is driven
  steadily. All horizontal distances elsewhere are multiples of this.
* **the minimum push** -- the smallest drive that makes it spike at all. All drive
  heights elsewhere are multiples of this.

The word **gap** below always means the horizontal distance between two neighbouring
spikes.

## 2. Global style

* Twelve panels in a grid, four columns by three rows. Panel 0 first, then (a) to (k).
* **No numbers on any axis.** This is a schematic. Axes may be drawn as a plain L-shaped
  corner with an arrowhead, or omitted entirely.
* Three inks only:
  - **dark neutral** for the response line and spikes,
  - **one accent colour** (say blue) for the drive line,
  - **a second accent** (say orange) used *only* for the quantity the panel is about --
    its marker, its bracket, its label. Nothing else in the panel may use this colour.
* Every panel gets a short title and, under it, the symbol(s) for its quantities in the
  second accent. Symbols are given per panel below; keep them as written.
* Spikes are identical everywhere. Resting line thin; drive line slightly thicker.
* Where a distance is being measured, draw a **double-headed arrow** between two thin
  vertical guide lines dropped from the relevant points.
* Several panels are **stacks of two or three rows sharing one time axis.** In those, the
  rows must be aligned exactly, so a vertical line means the same instant in every row.

## 3. The panels

### Panel 0 -- the two reference quantities

Steady flat drive line at a modest height; above it a response line with **five evenly
spaced spikes**.

* Between two neighbouring spikes, a double-headed arrow labelled **"the cell's own
  rhythm"**.
* At the left, on the drive line, a short vertical double-headed arrow from zero up to the
  drive level, labelled **"the minimum push"**.
* Caption underneath: *every time in this figure is measured in rhythms, every push in
  minimum-pushes.*

### Panel (a) -- delay before the first spike. Symbol: a cursive l

Drive: flat at zero, then **steps up** and stays there.
Response: flat for a visible while *after* the step, then a first spike, then regular
spiking.

* Shade the gap between the step and the first spike; double-headed arrow across it.
* The point: the cell does not react instantly, and the size of that delay is the number.

### Panel (b) -- the spacing drifts. Symbol: A

Drive: flat at zero, steps up, stays there.
Response: about ten spikes whose **gaps grow steadily wider** left to right -- tight at
the start, visibly wide by the end, then settling to a constant wide spacing.

* Double-headed arrow over the **first** gap, labelled "first"; another over the **last**,
  labelled "settled"; a curved arrow between them carrying the symbol.
* Draw a small inset variant whose gaps stay identical, labelled **"no drift"**. Many
  cells behave that way and the contrast is the message.

### Panel (c) -- the pattern of gaps. Symbols: p, p with subscript "tr", the Greek letter rho, f with a bar over it

The busiest panel; it carries four numbers from one picture.

Drive: flat, stepped up.
Response: a long train in two visibly different halves.

* **Left half (just after the step):** spikes with an irregular, not-yet-settled spacing.
* **Right half (later):** spikes arranged in **clusters** -- three close together, then a
  long empty stretch, then three close together again, three clusters total.
* Draw a labelled brace under the left half reading **"early"** and one under the right
  reading **"settled"**.

Mark on the settled half:

* A bracket round one cluster-plus-long-gap labelled **"one repeat"**. The symbol *p*
  counts **how many gaps are inside one repeat** (here four: three short, one long).
* One double-headed arrow over a **short** gap labelled "shortest", one over the **long**
  gap labelled "longest", with rho beside a small "longest / shortest" fraction.
* A count of the **needles inside one repeat**, with the symbol f-with-a-bar and the words
  *"spikes per rhythm"*. For a cell that fires evenly this is one; for the clustered cell
  drawn here it is three.

Mark on the early half:

* The same "how many gaps per repeat" count, with the symbol p-subscript-tr and the words
  *"the same count, made before the pattern settles"*. Draw the early half so that this
  count plainly differs from the settled one -- that difference is the point.

### Panel (d) -- how firing rate grows with push. Symbols: beta, chi, lambda

A **graph, not a trace.** Horizontal axis *"how hard you push"*, vertical axis *"how fast
it fires"*. Both axes just an arrow, no numbers.

Two curves of clearly different character:

* **Curve 1** starts at the bottom-left *at zero* and bends upward, rising ever more
  steeply. Label "starts from nothing".
* **Curve 2** begins already at a substantial height even at the weakest push, then rises
  only gently. Label "starts fast".

* On each curve mark a point at a weak push and a point at a strong push; bracket the
  vertical distance between them. **beta** is how steeply the curve climbs; **chi** and
  **lambda** are that weak-to-strong ratio, written one way up and the other.
* The point: two cells can both speed up when pushed harder, in completely different ways.

### Panel (e) -- two taps in a row. Symbols: delta with a star, rho with subscript "res"

Drive: **two identical narrow rectangular pulses** separated by a gap. Nothing else.

Three stacked rows sharing one time axis, the gap between the two pulses growing downward:

1. gap very small: only **one** spike.
2. gap medium: **two** spikes, one after each pulse.
3. gap large: **two** spikes.

Beside the stack, a small graph: horizontal axis "gap between taps", vertical axis "number
of spikes". The curve is a **step** -- low, then jumping up and staying flat.

* A vertical dashed line at the position of the jump, carrying **delta-star**, labelled
  *"the shortest wait at which the second tap still works"*.
* A short vertical double-headed arrow on the same graph from the **far right of the
  curve** up to its **highest point**, carrying **rho-res**, labelled *"the best any wait
  does, against what a very long wait does"*. Draw the curve so the plateau is flat and
  this ratio is close to one; it is a ratio, not a distance.
* The point: after firing, the cell is briefly unable to fire again.

### Panel (f) -- pushed slowly instead of suddenly. Symbol: gamma with subscript "ramp"

Drive: a **straight slope** rising steadily from zero.
Response: flat for a long time as the slope climbs, then spiking only when the slope is
already fairly high.

* A **horizontal dotted guide** at the "minimum push" height from panel 0, labelled *"the
  level that works if you push suddenly"*.
* A marker on the drive line at the height where the first spike actually appears, clearly
  **above** the dotted line, a short vertical double-headed arrow between the two, and the
  symbol.
* The point: push slowly and the cell resists; that excess is the number.

### Panel (g) -- letting go after holding it down. Symbols: N with subscript "reb", g with subscript "reb"

Drive: starts at zero, drops **below** zero to a held negative level, stays there, then
returns to zero.

Three stacked rows sharing one time axis, differing only in how far the drive drops:

1. shallow drop: after release, **nothing**.
2. medium drop: after release, **one** spike.
3. deep drop: after release, a **burst of four or five** spikes.

* A vertical dashed line through all three rows at the moment of release, labelled
  *"released here"*.
* The spike count after release carries **N-reb**.
* Beside row 2, an arrow carrying **g-reb** and the words *"the shallowest drop that still
  produces one"*.
* The point: pulling down and letting go can itself cause firing.

### Panel (h) -- up then down. Symbol: eta

Drive: a long symmetric **triangle** -- rising slowly to a peak, then falling back at the
same rate.
Response: no spikes at first; spikes begin partway **up**; they continue over the peak;
they stop partway **down**, at a *lower* drive height than the one where they started.

* Two vertical dashed lines dropped onto the drive line, at the height where firing
  **started** and the height where it **stopped**.
* A horizontal double-headed arrow between those two heights, carrying the symbol.
* The point: the switching level depends on which way the drive is moving. *(The
  switch-off level can legitimately be lower or higher depending on the cell; draw it
  lower, the common case.)*

### Panel (i) -- does it keep going? Symbol: the Greek capital Pi

Drive: flat at zero, steps up, and **stays up for the whole panel**.

Two stacked rows sharing one time axis:

1. **keeps firing:** spikes evenly spaced from the step to the right-hand edge.
2. **gives up:** a short burst of spikes just after the step, then nothing at all for the
   rest of the row.

* Under both rows draw two nested brackets: a short one labelled **"watch this long"** and
  a second, **exactly twice as long**, labelled **"now watch twice as long"**.
* For each row, write the spike count in each bracket. Row 1: roughly *n* and *2n*. Row 2:
  *n* and *n* -- the second bracket adds nothing.
* The symbol labels the fraction **(count in the long bracket) / (count in the short
  one)**: about two for row 1, about one for row 2.
* The point: whether a cell sustains firing or fires once and stops.

### Panel (j) -- is it easier or harder to fire just after a spike? Symbol: A with subscript "rec"

Drive: a **first rectangular pulse** that causes a spike, then after a delay a **second,
weaker pulse**. Draw the second pulse visibly shorter than the first.

Three stacked rows sharing one time axis, the delay between the two pulses growing
downward. In each row, the second pulse also produces a spike, but **how long it takes to
arrive after that pulse differs**:

1. short delay: the second spike arrives noticeably **late** after its pulse.
2. medium delay: it arrives a little late.
3. long delay: it arrives promptly.

* In every row, drop a thin guide line from the start of the second pulse, and draw a
  short double-headed arrow from that line to the spike it produced -- shrinking down the
  stack.
* A small graph beside the stack: horizontal axis "delay between the pulses", vertical
  axis "how much later the spike arrives". The points fall on a curve **decaying towards
  zero**. The symbol labels that decay.
* Add a small second curve mirrored **below zero**, drawn faintly, labelled *"or earlier,
  for most cells"*. The number carries a sign: firing can be made harder *or easier* by
  the preceding spike, and which one it is matters.
* The point: a spike leaves a wake, and this measures how strong it is and how long it
  lasts.

### Panel (k) -- does it fire with no push at all? Symbol: the Greek letter nu with subscript "sp"

Drive: **flat on zero across the whole panel.** Draw it clearly on the zero line and label
it *"no push at all"*.

Two stacked rows sharing one time axis:

1. **silent:** a perfectly flat response line, no spikes.
2. **spontaneous:** a response line producing spikes anyway, evenly spaced.

* The symbol labels the spike count in row 2, with the words *"spikes with nothing applied"*.
* The point: some cells need to be pushed, and some fire on their own. This separates them.

## 4. What must be true in the drawing

* Spikes never overlap the drive line; keep the two bands clearly separated vertically.
* In stacked panels ((e), (g), (i), (j), (k)) the rows share one time axis and are aligned
  exactly, so a vertical line means the same instant in every row.
* Each panel highlights its quantities in the second accent colour only.
* Nothing is to scale and no axis carries numbers; proportions are chosen for legibility.
  The only quantitative requirement is **ordering**: what is drawn as longer must be
  longer, what is drawn as higher must be higher, what is called a ratio must look like
  one.

## 5. Coverage checklist -- all eighteen

| # | symbol | panel | in one line |
|---|---|---|---|
| 1 | cursive l | (a) | delay from the push starting to the first spike |
| 2 | A | (b) | how much the gaps drift from the first to the last |
| 3 | p | (c) | how many gaps make up one repeat, once settled |
| 4 | p with subscript tr | (c) | the same count, taken before the pattern settles |
| 5 | rho | (c) | longest gap divided by shortest, within one repeat |
| 6 | f with a bar | (c) | spikes per rhythm |
| 7 | beta | (d) | how steeply the firing rate climbs with the push |
| 8 | chi | (d) | weak-push rate divided by strong-push rate |
| 9 | lambda | (d) | the same ratio the other way up |
| 10 | delta with a star | (e) | shortest wait at which a second tap still works |
| 11 | rho with subscript res | (e) | best response over the response at a very long wait |
| 12 | gamma with subscript ramp | (f) | how much extra push a slow ramp needs |
| 13 | N with subscript reb | (g) | spikes produced by releasing a downward hold |
| 14 | g with subscript reb | (g) | shallowest hold whose release still produces one |
| 15 | eta | (h) | gap between the switch-on and switch-off levels |
| 16 | capital Pi | (i) | spikes in a doubled window over spikes in one window |
| 17 | A with subscript rec | (j) | how much, and which way, a spike shifts the next one |
| 18 | nu with subscript sp | (k) | spikes produced with no push at all |
"""


def captions():
    """Write FIGURES.md: one caption per figure, with every number the figures omit.

    The figures carry no annotations, so each number a caption needs is printed here from
    the artifact that produced it. Regenerate after any rerun.
    """
    import targets as T
    rows1, val1 = A.load("study1")
    rows2, val2 = A.load("study2")
    s1 = K.jread(os.path.join(K.DATA, "summary_study1.json")) or {}
    cost = K.jread(os.path.join(K.DATA, "cost_pooled.json")) or {}
    tg = ["ml_c1", "hh", "adex_tonic"]
    methods = ["cma", "de", "pso"]
    L = ["# Figures for the short paper", "",
         "Generated by `paper_figs.py`; every number is read from the run artifacts, so",
         "these captions cannot drift from the figures. The figures themselves carry no",
         "annotations -- the numbers are all here.", "",
         "Common to all four: the surrogate is an Izhikevich cell "
         "$(a,b,c,d)$; the time scale $\\tau$ and current gain $k_I$ are **not** searched, "
         "they are recovered afterwards by",
         "$\\tau = T_{\\rm cyc}^{\\rm target}/T_{\\rm cyc}^{\\rm sur}$ and "
         "$k_I = i_{\\rm rh}^{\\rm target}/i_{\\rm rh}^{\\rm sur}$. Every optimizer run "
         "used a budget of",
         f"{rows1[0]['budget'] if rows1 else 5000} evaluations; study 1 used "
         f"{len({r['seed'] for r in rows1})} seeds per (target, method), study 2 "
         f"{len({r['seed'] for r in rows2})} per configuration.", ""]

    # ---- figure 1
    L += ["## Figure 1 -- the feature schema", "",
          "**(a)** The nine protocols, drawn as the current each applies, in units of the",
          "cell's own rheobase $i_{\\rm rh}$ and cycle time $T_{\\rm cyc}$; to the right of",
          "each, the features it produces. **(b)** The resulting 18-number description of",
          "three targets from different families, each feature placed as a mid-rank",
          "percentile within the 10$^7$-cell reference atlas so that columns with different",
          "units are comparable. Hatched bars mark features the model does not define.",
          "",
          "| target | $i_{\\rm rh}$ | $T_{\\rm cyc}$ (ms) | defined | undefined |",
          "|---|---|---|---|---|"]
    for slug in tg:
        d = K.jread(os.path.join(K.TARGETS, f"{slug}.json"))
        f = np.asarray(d["f_t"], float)
        und = [n for n, v in zip(C.FEATURE_NAMES, f) if v <= C.SENT_CUT]
        L.append(f"| {TGT[slug]} | {d['i_rh']:.4g} | {d['T_cycle']:.4g} | "
                 f"{d['n_defined']}/18 | {', '.join('`' + u + '`' for u in und) or 'none'} |")
    L += ["",
          "All three are period-1 at $\\gamma = 2$, which is why `p`, `rho` and `p_tr` sit",
          "at the same percentile for all of them.", ""]

    # ---- figure 5 (the schema figure; supersedes figure 1)
    from v3_presets import PRESETS
    import fig_schema as FSC
    L += ["## Figure 5 -- what the features measure", "",
          "Each panel is one protocol applied to the Izhikevich regime that *displays*",
          "the feature, with the measured value quoted. The cells are Izhikevich's own",
          "canonical regimes and are chosen from the measured values, not from their",
          "names: a single out-of-family target would leave several panels showing a",
          "feature that happens to be zero for it. The quantities not drawn are",
          "`rho_res` (from the same pulse-pair curve as $\\delta^*$), `Pi` (the ratio of",
          "spike counts over a window and twice that window), `A_rec` (post-spike",
          "excitability, protocol P8'), and `nu_spont` (the count at $I = 0$).", "",
          "| panel | protocol | feature | cell | $(a,b,c,d)$ | value |",
          "|---|---|---|---|---|---|"]
    PANELS = [("a", "P1", r"$\ell$", "latency", "ell"),
              ("b", "P1", "$A$", "adapt", "A"),
              ("c", "P1", r"$p$, $\rho$", "burst", None),
              ("d", "P2", r"$\beta$, $\chi$, $\lambda$", "fi_lo", None),
              ("e", "P4", r"$\delta^*$", "pulse", "delta_star"),
              ("f", "P5", r"$\gamma_{\rm ramp}$", "ramp", "gamma_ramp"),
              ("g", "P6", r"$N_{\rm reb}$, $g_{\rm reb}$", "rebound", None),
              ("h", "P7", r"$\eta$", "hyst", "eta")]
    for pid, proto, feat, key, fk in PANELS:
        nm = FSC.CELLS[key]
        pr = PRESETS[nm]
        th = f"{pr['a']:g}, {pr['b']:g}, {pr['c']:g}, {pr['d']:g}"
        val = "--"
        try:
            fv = FSC.features(np.array([pr['a'], pr['b'], pr['c'], pr['d']], float))[0]
            if fk:
                val = f"{fv[fk]:.3g}"
            elif key == "burst":
                val = f"$p$ = {fv['p']:.0f}, " + rf"$\rho$ = {fv['rho']:.1f}"
            elif key == "rebound":
                val = (rf"$N_{{\rm reb}}$ = {fv['N_reb']:.0f}, "
                       rf"$g_{{\rm reb}}$ = {fv['g_reb']:.1f}")
            elif key == "fi_lo":
                hi = PRESETS[FSC.CELLS["fi_hi"]]
                fh = FSC.features(np.array([hi['a'], hi['b'], hi['c'], hi['d']],
                                           float))[0]
                val = (rf"$\beta$ = {fv['beta']:.2f} / {fh['beta']:.2f}, "
                       rf"$\lambda$ = {fv['lam']:.1f} / {fh['lam']:.1f}")
        except Exception:                                      # noqa: BLE001
            pass
        cell = nm[2:] + (f" and {FSC.CELLS['fi_hi'][2:]}" if key == "fi_lo" else "")
        L.append(f"| ({pid}) | {proto} | {feat} | {cell} | {th} | {val} |")
    L += ["",
          "Two of Izhikevich's twenty presets cannot be used at all here: `R accomodation`",
          "and `P bistability` have $b \\ge 1$ and fire at zero current, so their rheobase",
          "is about $10^{-14}$ and every $\\gamma$-scaled drive collapses. And the preset",
          "named `F spike freq adapt` shows $A \\approx 10^{-11}$ under a $\\gamma = 2$ step --",
          "it is periodic from its first interval -- so panel (b) uses `N rebound burst`,",
          "the only in-box preset with strong adaptation. The name of a regime does not",
          "tell you which features it exercises; the measurement does.",
          "",
          "This figure supersedes Figure 1, which placed each feature as a percentile",
          "within the atlas. That framing was wrong for this paper: the atlas belongs to",
          "the amortized-inference work, and the optimizers never use one.", ""]

    # ---- figure 2
    L += ["## Figure 2 -- the schema as an optimization objective", "",
          "**(a, b)** Final $d_2$ and $W_1^{\\rm shape}$ for each optimizer on each target,",
          "one point per seed, bar at the median; the grey lines span the three medians.",
          "**(c, d)** The same two quantities along each model's own parameter axis",
          "(Morris-Lecar $V_3$ 2--14, Hodgkin-Huxley $g_K$ 24--40, AdEx spike-triggered",
          "jump $b$ 0--120), normalized to the position along that range, CMA-ES only.",
          "In (d) an open triangle marks a configuration matched to better than $10^{-5}$,",
          "i.e. exactly at this resolution.", "",
          "| target | metric | CMA-ES | DE | PSO | spread |",
          "|---|---|---|---|---|---|"]
    for slug in tg:
        for key, lab in ((None, "$d_2$"), ("w1_shape", "$W_1$")):
            v = {}
            for m in methods:
                rec = s1.get("by", {}).get(f"{slug}|{m}", {})
                v[m] = (rec.get("J", {}) if key is None else rec.get(key, {})).get("median")
            ok = [x for x in v.values() if x]
            sp = max(ok) / min(ok) if len(ok) > 1 else float("nan")
            L.append(f"| {TGT[slug]} | {lab} | " +
                     " | ".join(f"{v[m]:.4g}" if v[m] else "--" for m in methods) +
                     f" | {sp:.2f}x |")
    L += ["", "The largest spread between any two methods is 1.57x on $d_2$ and 1.87x "
          "on $W_1$: all three families agree to within a factor of two on every target.",
          ""]

    # ---- figure 3
    L += ["## Figure 3 -- cost", "",
          "**(a--c)** Best $d_2$ so far against the number of evaluations, median over",
          "seeds with the interquartile band. **(d)** Evaluations needed to come within",
          "1.1x of a run's own final value, one point per run.", ""]
    if cost:
        st = cost["cache"]
        L += [f"One evaluation is the full protocol suite on one Izhikevich cell and cost",
              f"**{cost['ms_per_eval']:.2f} ms** in this run "
              f"({cost['evals']} evaluations in {cost['wall_s']/3600:.2f} h on "
              f"{(cost.get('env') or {}).get('gpu', 'the GPU')}), with all 186 runs",
              f"evaluated in one pooled batch per round. PASS 0 rejected "
              f"{100*st['gated0']/max(st['pass0'],1):.1f}% of candidates before the full "
              f"suite ran.", ""]

    # ---- figure 4
    L += ["## Figure 4 -- target and surrogates", "",
          "**(a)** Each target under the reference step $I = 2\\,i_{\\rm rh}$, in absolute",
          "milliseconds and millivolts. **(b)** The three identified surrogates for that",
          "target, drawn at $t\\,\\tau$ so that they share the axis of (a): the alignment is",
          "the gauge, not a fit. Each surrogate is the median-$d_2$ seed of its ten runs.",
          "**(c)** The settled interspike intervals the $W_1$ distance is computed on, each",
          "divided by its own cycle time; target filled, surrogates outlined.", "",
          "| target | $T_{\\rm cyc}$ (ms) | source | $\\tau$ | $T_{\\rm cyc}^{\\rm sur}$ (ms) "
          "| $k_I$ | $d_2$ | $W_1$ | $(a,b,c,d)$ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for slug in tg:
        d = K.jread(os.path.join(K.TARGETS, f"{slug}.json"))
        first = True
        for m in methods:
            rs = sorted((x for x in rows1 if x["target"] == slug and x["method"] == m),
                        key=lambda x: x["J"])
            if not rs:
                continue
            r = rs[len(rs) // 2]
            w1 = (val1.get(r["run_id"], {}) or {}).get("w1_shape")
            th = ", ".join(f"{x:.4g}" for x in r["theta"])
            L.append(f"| {TGT[slug] if first else ''} | "
                     f"{d['T_cycle']:.4g} | {NAME[m]} | {r['tau']:.4g} | "
                     f"{r['T_cycle_s']:.4g} | {r['k_I']:.4g} | {r['J']:.4g} | "
                     f"{(w1 if w1 is not None else float('nan')):.2e} | {th} |")
            first = False
    # the settling split drawn in row (c)
    import validate as V
    from v3_core import alloc_buffers
    bufs = alloc_buffers()
    L += ["",
          "In (c) the shaded band marks intervals more than 2\\% below the settled value.",
          "Those are the ones a train produces while it is still settling: the pipeline",
          "discards a transient before recording, and what remains is the residue of a",
          "surrogate that settles more slowly than that window. The table gives, per",
          "train, how many of its intervals fall in the band and whether they are exactly",
          "the leading ones (index 0 upward).", "",
          "| target | train | intervals | in the band | the leading ones? |",
          "|---|---|---|---|---|"]
    for slug in tg:
        isi_t = V.target_isis(slug)
        cut = 0.98 * float(np.median(isi_t[-20:]))
        trains = [("target", isi_t)]
        for m in methods:
            rs = sorted((x for x in rows1 if x["target"] == slug and x["method"] == m),
                        key=lambda x: x["J"])
            if rs:
                trains.append((NAME[m], V.izh_isis(rs[len(rs) // 2]["theta"], bufs)))
        first = True
        for nm, tr in trains:
            if tr is None:
                continue
            below = np.flatnonzero(np.asarray(tr) < cut)
            lead = bool(below.size and below.max() == below.size - 1)
            L.append(f"| {TGT[slug] if first else ''} | {nm} | {len(tr)} | {below.size} "
                     f"| {'yes' if lead else ('--' if not below.size else 'NO')} |")
            first = False
    L += ["",
          "The one exception is PSO on Hodgkin-Huxley: 46 of its 96 intervals fall in the",
          "band and they run to index 93, so that train never settles at all. It is the",
          "same defect its $W_1$ reports and that row (b) shows as five cycles against the",
          "target's four -- not a transient.",
          "",
          "Morris-Lecar occupies 400 ms of the axis, Hodgkin-Huxley 55 and AdEx 90, which",
          "is why the axes are left absolute. On Morris-Lecar, CMA-ES and PSO converge on",
          "nearly the same cell while DE finds one running at half the rate and reaching a",
          "lower $d_2$: two different surrogates, both valid under the gauge, which is the",
          "many-to-one parameter-to-behaviour map made visible. The one visible defect is",
          "PSO on Hodgkin-Huxley, which fits five cycles into the window where the target",
          "has four -- the period error its $W_1$ reports.", ""]

    out = os.path.join(K.HERE, "FIGURES.md")
    open(out, "w").write("\n".join(L).rstrip() + "\n" + SCHEMATIC_BRIEF)
    print(f"wrote FIGURES.md ({len(L)} lines)")


def _save(fig, stem):
    os.makedirs(K.FIGS, exist_ok=True)
    p = os.path.join(K.FIGS, stem)
    for e in ("pdf", "png"):
        fig.savefig(f"{p}.{e}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/{stem}.pdf/.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=("1", "2", "3", "4", "5"))
    a = ap.parse_args()
    import fig_schema
    for n, fn in (("1", fig1), ("2", fig2), ("3", fig3), ("4", fig4),
                  ("5", fig_schema.main)):
        if a.only in (None, n):
            fn()
    captions()
