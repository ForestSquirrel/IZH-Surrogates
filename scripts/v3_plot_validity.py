"""
v3_plot_validity.py -- the (tau, k_I) validity rectangle (§49, §55).

The figure states the method's domain of validity in one picture: over what range
of the two post-hoc scales does the feature vector stay invariant, and where does
it stop.

FORM. A heatmap, because the job is the SHAPE of a region -- a table of the same
numbers cannot show where the walls are or that they are perpendicular.

COLOR. Deviation magnitude is a sequential encoding: one blue hue, light -> dark,
lightest = "indistinguishable from the reference". "DEAD" (no anchor recovered)
is NOT a step on that ramp -- it is a different state, not an extreme value -- so
it is drawn as a hatched neutral and named separately in the legend. Encoding it
as dark blue would claim it is "very deviant", which is a different and false
statement.

Two panels: the shipped bootstrap, and §50's auto-widening. The kappa wall is a
bracket limit and does not move; the tau wall is a window limit and does.
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))   # pipeline_final/scripts
ROOT = os.path.dirname(HERE)                       # pipeline_final
PKG     = os.path.join(ROOT, "pipeline")
DATA    = os.path.join(ROOT, "data")
MODELS  = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
PLOTS   = os.path.join(ROOT, "plots")
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)   # sibling runners (v3_plot_pair -> v3_plot_family)
import v3_config as C

# dataviz reference palette -- sequential blue ramp, steps 100..700
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
DEAD_FILL = "#d8d7d2"
DEAD_EDGE = "#8d8b84"

# deviation bands, in percent
BANDS  = [0.0, 0.01, 0.1, 1.0, 5.0, 15.0, 40.0, 100.0]
LABELS = ["exact\n(<0.01%)", "<0.1%", "<1%", "<5%", "<15%", "<40%", ">=40%"]
CMAP   = LinearSegmentedColormap.from_list("dev", [BLUE[0], BLUE[3], BLUE[6],
                                                   BLUE[8], BLUE[10], BLUE[12]])


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "g_*.json"))):
        rows += json.load(open(f))
    return rows


def build(rows, auto):
    taus = sorted({r["tau"] for r in rows})
    kaps = sorted({r["kappa"] for r in rows})
    ref = [r for r in rows if r["tau"] == 1.0 and r["kappa"] == 1.0 and r["auto"] == auto]
    ref = ref[0] if ref else [r for r in rows if r["tau"] == 1.0 and r["kappa"] == 1.0][0]
    rf = np.array(ref["feat"]); dm = rf > C.SENT_CUT
    M = np.full((len(taus), len(kaps)), np.nan)
    D = np.zeros_like(M, dtype=bool)
    R = np.zeros_like(M)
    for r in rows:
        if r["auto"] != auto:
            continue
        i, j = taus.index(r["tau"]), kaps.index(r["kappa"])
        R[i, j] = r["retries"]
        f = np.array(r["feat"])
        if r["gated"] or r["T_cycle"] < C.SENT_CUT or r["defined"] <= 1:
            D[i, j] = True
            continue
        m = dm & (f > C.SENT_CUT)
        if m.sum() < 5:
            D[i, j] = True
            continue
        e = np.abs(f[m] - rf[m]) / np.maximum(np.abs(rf[m]), 1e-12)
        M[i, j] = float(np.median(e)) * 100.0
    return np.array(taus), np.array(kaps), M, D, R


def panel(ax, taus, kaps, M, D, title, norm):
    ax.set_facecolor(SURFACE)
    Mm = np.ma.masked_invalid(M)
    im = ax.pcolormesh(np.arange(len(kaps) + 1), np.arange(len(taus) + 1), Mm,
                       cmap=CMAP, norm=norm, edgecolors=SURFACE, linewidth=2)
    for i in range(len(taus)):
        for j in range(len(kaps)):
            if D[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=DEAD_FILL,
                                           edgecolor=DEAD_EDGE, hatch="///",
                                           linewidth=0.8, zorder=3))
    ax.set_xticks(np.arange(len(kaps)) + 0.5)
    ax.set_xticklabels([f"{k:g}" for k in kaps], fontsize=8, color=INK_2)
    ax.set_yticks(np.arange(len(taus)) + 0.5)
    ax.set_yticklabels([f"{t:g}" for t in taus], fontsize=8, color=INK_2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(PLOTS, "v3_tau_kappa_validity.png"))
    args = ap.parse_args()
    rows = load(args.dir)
    norm = BoundaryNorm(BANDS, CMAP.N, extend="max")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.74, bottom=0.16, left=0.075, right=0.88, wspace=0.09)
    for ax, auto, ttl in zip(axes, (0, 1),
                             ("shipped bootstrap  (BOOT_T_WIN = 500)",
                              "with auto-widening  (§50)")):
        taus, kaps, M, D, R = build(rows, auto)
        im = panel(ax, taus, kaps, M, D, ttl, norm)
        n_ok = int((~D).sum())
        ax.set_xlabel("$k_I$  (current gain $\\kappa$)", fontsize=9, color=INK_2)
        ax.text(0.99, -0.135, f"{n_ok}/{D.size} cells recover an anchor",
                transform=ax.transAxes, ha="right", fontsize=8, color=INK_2)
    axes[0].set_ylabel("$\\tau$  (time dilation)", fontsize=9, color=INK_2)

    cax = fig.add_axes([0.895, 0.16, 0.016, 0.58])
    cb = fig.colorbar(im, cax=cax, ticks=BANDS, extend="max")
    cb.ax.set_yticklabels([f"{b:g}" for b in BANDS], fontsize=7.5, color=INK_2)
    cb.set_label("median feature deviation from ($\\tau$=1, $\\kappa$=1)   [%]",
                 fontsize=8.5, color=INK_2)
    cb.outline.set_visible(False)
    fig.legend(handles=[Patch(facecolor=DEAD_FILL, edgecolor=DEAD_EDGE, hatch="///",
                              label="no anchor recovered")],
               loc="lower left", bbox_to_anchor=(0.075, 0.015), frameon=False,
               fontsize=8.5, labelcolor=INK_2)
    fig.text(0.075, 0.945, "Domain of validity of the $(\\tau,\\ k_I)$ factorization",
             fontsize=13, color=INK, ha="left", va="top")
    fig.text(0.075, 0.885,
             "The excluded region is a STAIRCASE, not a rectangle: under the "
             "shipped bootstrap the two walls are coupled, because a starved\n"
             "window needs more current to fire and pushes $i_{rh}$ out of its "
             "bracket. Widening the bootstrap decouples them and the interior "
             "becomes exact.",
             fontsize=8.8, color=INK_2, ha="left", va="top", linespacing=1.5)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")

    for auto, nm in ((0, "shipped"), (1, "auto")):
        taus, kaps, M, D, R = build(rows, auto)
        print(f"\n{nm}:  rows=tau {taus.tolist()}  cols=kappa {kaps.tolist()}")
        for i, t in enumerate(taus):
            cells = []
            for j in range(len(kaps)):
                cells.append("  DEAD" if D[i, j] else
                             ("     -" if np.isnan(M[i, j]) else f"{M[i,j]:6.2f}"))
            print(f"  tau={t:<7g} " + " ".join(cells))


if __name__ == "__main__":
    main()
