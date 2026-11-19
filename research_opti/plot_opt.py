"""
plot_opt.py -- the figures, from the logged artifacts only (no GPU, no re-simulation).

    $PY plot_opt.py --study study1     -> figs/opt_study1.{pdf,png}
    $PY plot_opt.py --study study2     -> figs/opt_study2.{pdf,png}

study 1, one column per target:
  (a) best-so-far J against evaluations: median over seeds, IQR band, one colour per
      method, with the cINN's best-of-N draw as a horizontal line. The line is not a
      competitor on this axis -- the cINN spends no evaluations -- it marks what the
      amortized arm reaches for free, so the x position where a method crosses it is the
      per-target price of not having an atlas.
  (b) final J per method (points, one per seed) and the chosen theta's w1_shape, which
      d2 never saw.
  (c) the spread of theta across seeds in u units: the degeneracy, not an error bar.

study 2: each metric against the target model's own parameter, optimizer and cINN
overlaid, one column per family.
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

UNITS = {"V3": r"$V_3$ (mV)", "gK": r"$g_K$ (mS/cm$^2$)",
         "b": r"spike-triggered jump $b$ (pA)"}
COL = {"cma": "#1c5cab", "de": "#d94801", "pso": "#2a9d4a"}
NAME = {"cma": "CMA-ES", "de": "DE", "pso": "PSO"}
CINN = "#6b6b6b"


def _panel_label(ax, s):
    ax.text(-0.02, 1.03, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def study1(study="study1"):
    rows, val = A.load(study)
    CB = A.cinn()
    tgts = sorted({r["target"] for r in rows})
    methods = [m for m in ("cma", "de", "pso") if any(r["method"] == m for r in rows)]
    n = len(tgts)
    # wspace is wide because row (b) carries a twin axis on every panel: at the default
    # the right-hand W1 ticks of one panel land on the left-hand d2 ticks of the next
    fig, axs = plt.subplots(3, n, figsize=(3.05 * n, 6.4),
                            gridspec_kw=dict(hspace=0.45, wspace=0.62))
    axs = np.atleast_2d(axs)
    if n == 1:
        axs = axs.reshape(3, 1)

    for c, tg in enumerate(tgts):
        lbl = next((r.get("label") or tg for r in rows if r["target"] == tg), tg)

        # (a) convergence
        ax = axs[0, c]
        for m in methods:
            rs = [r for r in rows if r["target"] == tg and r["method"] == m]
            curves = []
            for r in rs:
                e, j = A.best_so_far(study, r["run_id"])
                if e is not None:
                    curves.append((e, j))
            if not curves:
                continue
            grid = curves[0][0]
            M = np.vstack([np.interp(grid, e, j) for e, j in curves])
            ax.plot(grid, np.median(M, axis=0), color=COL[m], lw=1.0, label=NAME[m])
            if len(M) > 1:
                ax.fill_between(grid, np.percentile(M, 25, axis=0),
                                np.percentile(M, 75, axis=0), color=COL[m], alpha=0.16,
                                lw=0)
        cb = CB.get(tg, {}).get("J_best")
        if cb:
            ax.axhline(cb, color=CINN, lw=0.8, ls="--")
            ax.text(0.98, cb, " cINN best of "
                    f"{CB[tg]['n_posterior']}", color=CINN, fontsize=6,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("evaluations")
        if c == 0:
            ax.set_ylabel(r"best $d_2$ so far")
        ax.set_title(lbl, fontsize=8)
        if c == 0:
            ax.legend(frameon=False, loc="lower left", fontsize=6.5)

        # (b) final J and w1_shape
        ax = axs[1, c]
        ax2 = ax.twinx()
        for i, m in enumerate(methods):
            rs = [r for r in rows if r["target"] == tg and r["method"] == m]
            J = np.array([r["J"] for r in rs])
            x = i + 0.16 * np.linspace(-1, 1, max(len(J), 1))
            ax.plot(x - 0.1, J, "o", ms=2.6, color=COL[m], mew=0)
            ax.plot([i - 0.32, i - 0.08], [np.median(J)] * 2, "-", color=COL[m], lw=1.2)
            w = np.array([val[r["run_id"]]["w1_shape"] for r in rs
                          if r["run_id"] in val
                          and val[r["run_id"]]["w1_shape"] is not None])
            if w.size:
                xw = i + 0.1 + 0.16 * np.linspace(-1, 1, len(w))
                ax2.plot(xw, w, "^", ms=2.6, color=COL[m], mew=0, alpha=0.55)
                ax2.plot([i + 0.08, i + 0.32], [np.median(w)] * 2, "-", color=COL[m],
                         lw=1.2, alpha=0.55)
        if cb:
            ax.axhline(cb, color=CINN, lw=0.9, ls="--",
                       label=r"cINN $d_2$" if c == 0 else None)
        cw = CB.get(tg, {}).get("w1_shape")
        if cw:
            ax2.axhline(cw, color=CINN, lw=0.9, ls=":",
                        label=r"cINN $W_1$" if c == 0 else None)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([NAME[m] for m in methods], fontsize=7)
        # linear, with a floor at zero: over the narrow range these spread, a log axis
        # labels every minor tick (6x10^0, 4x10^0, ...) and the labels collide with the
        # neighbouring panel's right-hand axis
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(0, max(hi_, (cb or 0) * 1.08))
        if c == 0:
            ax.set_ylabel(r"final $d_2$  ($\circ$, dashed)")
        ax2.set_yscale("log")
        if c == n - 1:
            ax2.set_ylabel(r"$W_1^{\mathrm{shape}}$  ($\triangle$, dotted)", fontsize=7.5)
        ax2.tick_params(labelsize=6.5, pad=1.5)
        ax.tick_params(labelsize=6.5)
        if c == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, loc="lower left")

        # (c) theta spread in u units
        ax = axs[2, c]
        w = 0.8 / max(len(methods), 1)
        for i, m in enumerate(methods):
            rs = [r for r in rows if r["target"] == tg and r["method"] == m]
            U = np.array([r["u"] for r in rs if r["u"]])
            if U.size == 0:
                continue
            iqr = np.percentile(U, 75, axis=0) - np.percentile(U, 25, axis=0)
            ax.bar(np.arange(4) + (i - (len(methods) - 1) / 2) * w, iqr, width=w,
                   color=COL[m], label=NAME[m], lw=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["$a$", "$b$", "$c$", "$d$"])
        ax.set_ylim(0, 1)
        if c == 0:
            ax.set_ylabel("IQR across seeds\n(fraction of the box)")

    for k, ax in enumerate(axs[:, 0]):
        _panel_label(ax, f"({chr(97 + k)})")
    out = os.path.join(K.FIGS, f"opt_{study}")
    os.makedirs(K.FIGS, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{out}.{e}", bbox_inches="tight")
    print(f"wrote figs/opt_{study}.pdf/.png")


def study2(study="study2"):
    import targets as T
    rows, val = A.load(study)
    CB = A.cinn()
    fams = T.study2()
    fig, axs = plt.subplots(2, len(fams), figsize=(2.6 * len(fams), 4.3),
                            gridspec_kw=dict(hspace=0.38, wspace=0.34))
    axs = np.atleast_2d(axs)
    for c, (fam, pname, items) in enumerate(fams):
        xs = {s: x for s, l, x, k, p, r in items}
        for ri, (key, lab, logy) in enumerate(
                ((None, r"$d_2$", True), ("w1_shape", r"$W_1^{\mathrm{shape}}$", True))):
            ax = axs[ri, c]
            pts, cpts, med = {}, {}, []
            for r in rows:
                if r["target"] not in xs:
                    continue
                v = (r["J"] if key is None else
                     (val.get(r["run_id"], {}) or {}).get(key))
                if v is None:
                    continue
                pts.setdefault(xs[r["target"]], []).append(v)
            for s, x in xs.items():
                cv = (CB.get(s, {}).get("J_best") if key is None
                      else CB.get(s, {}).get(key))
                if cv is not None:
                    cpts[x] = cv
            if pts:
                X = sorted(pts)
                med = [np.median(pts[x]) for x in X]
                lo = [np.min(pts[x]) for x in X]
                hi = [np.max(pts[x]) for x in X]
                ax.fill_between(X, lo, hi, color=COL["cma"], alpha=0.18, lw=0)
                ax.plot(X, med, "o-", ms=3, color=COL["cma"], lw=1.0, label="optimizer")
            if cpts:
                X = sorted(cpts)
                ax.plot(X, [cpts[x] for x in X], "s--", ms=2.6, color=CINN, lw=0.8,
                        label="cINN")
            if logy:
                ax.set_yscale("log")
                # the band is min-max over seeds, and a single near-exact seed (W1 ~ 1e-11
                # on AdEx) would otherwise set the bottom of the axis four decades below
                # everything else; the limits follow the medians and the cINN line, and
                # the band is allowed to run off the bottom
                ref = [v for v in list(med) + list(cpts.values()) if v and v > 0]
                if ref:
                    ax.set_ylim(bottom=min(ref) / 3.0)
            if ri == 1:
                ax.set_xlabel(f"{fam}: {UNITS.get(pname, pname)}")
            if c == 0:
                ax.set_ylabel(lab)
            if ri == 0:
                ax.set_title(fam, fontsize=8)
                if c == 0:
                    ax.legend(frameon=False, fontsize=6.5)
    for k, ax in enumerate(axs[:, 0]):
        _panel_label(ax, f"({chr(97 + k)})")
    out = os.path.join(K.FIGS, f"opt_{study}")
    os.makedirs(K.FIGS, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{out}.{e}", bbox_inches="tight")
    print(f"wrote figs/opt_{study}.pdf/.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="study1")
    a = ap.parse_args()
    (study1 if a.study == "study1" else study2)(a.study)
