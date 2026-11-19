"""
v3_plot_validity_wide.py -- the (tau, k_I) domain of validity, +-5 decades.

One panel, auto-widening on: the INTRINSIC domain of the factorization rather
than the behaviour of any particular bootstrap seed (§49's two-panel figure
covers that). DT and the bootstrap seed both scale with tau, so every cell costs
the same number of steps and the axes are a fair sweep rather than a cost ramp.

Encoding is unchanged from the narrow figure: sequential blue for deviation
magnitude, hatched neutral for "no anchor recovered" -- a different state, not an
extreme value, so it never takes a step on the ramp.
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

BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
DEAD_FILL, DEAD_EDGE = "#d8d7d2", "#8d8b84"
BANDS = [0.0, 0.01, 0.1, 1.0, 5.0, 15.0, 40.0, 100.0]
CMAP = LinearSegmentedColormap.from_list("dev", [BLUE[0], BLUE[3], BLUE[6],
                                                 BLUE[8], BLUE[10], BLUE[12]])


def _fmt(v):
    e = int(round(np.log10(v)))
    return f"$10^{{{e}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(PLOTS,
                                                  "v3_tau_kappa_validity_wide.png"))
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "w_*.json"))):
        rows += json.load(open(f))
    taus = sorted({r["tau"] for r in rows})
    kaps = sorted({r["kappa"] for r in rows})
    ref = [r for r in rows if abs(r["tau"] - 1.0) < 1e-9 and abs(r["kappa"] - 1.0) < 1e-9][0]
    rf = np.array(ref["feat"]); dm = rf > C.SENT_CUT

    M = np.full((len(taus), len(kaps)), np.nan)
    D = np.zeros_like(M, dtype=bool)
    ND = np.zeros_like(M)
    for r in rows:
        i, j = taus.index(r["tau"]), kaps.index(r["kappa"])
        ND[i, j] = r["defined"]
        f = np.array(r["feat"])
        if r["gated"] or r["T_cycle"] < C.SENT_CUT or r["defined"] <= 1:
            D[i, j] = True; continue
        m = dm & (f > C.SENT_CUT)
        if m.sum() < 5:
            D[i, j] = True; continue
        e = np.abs(f[m] - rf[m]) / np.maximum(np.abs(rf[m]), 1e-12)
        M[i, j] = float(np.median(e)) * 100.0

    fig = plt.figure(figsize=(9.4, 7.4))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.115, 0.115, 0.70, 0.66])
    ax.set_facecolor(SURFACE)
    norm = BoundaryNorm(BANDS, CMAP.N, extend="max")
    im = ax.pcolormesh(np.arange(len(kaps) + 1), np.arange(len(taus) + 1),
                       np.ma.masked_invalid(M), cmap=CMAP, norm=norm,
                       edgecolors=SURFACE, linewidth=2)
    for i in range(len(taus)):
        for j in range(len(kaps)):
            if D[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=DEAD_FILL,
                                           edgecolor=DEAD_EDGE, hatch="///",
                                           linewidth=0.8, zorder=3))
    ax.set_xticks(np.arange(len(kaps)) + 0.5)
    ax.set_xticklabels([_fmt(k) for k in kaps], fontsize=9, color=INK_2)
    ax.set_yticks(np.arange(len(taus)) + 0.5)
    ax.set_yticklabels([_fmt(t) for t in taus], fontsize=9, color=INK_2)
    ax.set_xlabel("$k_I$   (current gain $\\kappa$)", fontsize=10, color=INK_2,
                  labelpad=8)
    ax.set_ylabel("$\\tau$   (time dilation)", fontsize=10, color=INK_2, labelpad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    cax = fig.add_axes([0.845, 0.115, 0.022, 0.66])
    cb = fig.colorbar(im, cax=cax, ticks=BANDS, extend="max")
    cb.ax.set_yticklabels([f"{b:g}" for b in BANDS], fontsize=8, color=INK_2)
    cb.set_label("median feature deviation from ($\\tau$=1, $\\kappa$=1)   [%]",
                 fontsize=9, color=INK_2)
    cb.outline.set_visible(False)

    n_ok = int((~D).sum())
    exact = int(np.nansum(M[~D] < 0.01))
    fig.text(0.115, 0.955, "Domain of validity of the $(\\tau,\\ k_I)$ factorization",
             fontsize=14, color=INK, ha="left", va="top")
    lo_dead = int(D[:, 0].sum())
    hi_part = ""
    if len(kaps) > 1:
        nd_hi = ND[~D[:, -1], -1]
        if len(nd_hi):
            hi_part = (f" At the high-$\\kappa$ wall the anchor survives and only the "
                       f"four $I_{{RH\\_MIN}}$-gated columns\n($\\eta$, $N_{{reb}}$, "
                       f"$g_{{reb}}$, $\\gamma_{{ramp}}$) drop, leaving "
                       f"{int(np.median(nd_hi))} of {C.N_ACTIVE}.")
    fig.text(0.115, 0.905,
             f"{exact} of {n_ok} recovered cells are exact to <0.01%, over "
             f"{len(taus)} decades of $\\tau$ and {len(kaps)} of $\\kappa$. "
             f"$\\tau$ carries no intrinsic\nbound once the bootstrap scales with the "
             f"cell; $\\kappa$ has two walls, and they fail differently — below, the "
             f"rheobase search ceiling\ntakes the anchor and 17 of 18 columns with it."
             + hi_part,
             fontsize=9.2, color=INK_2, ha="left", va="top", linespacing=1.6)
    fig.legend(handles=[Patch(facecolor=DEAD_FILL, edgecolor=DEAD_EDGE,
                              hatch="///", label="no anchor recovered")],
               loc="lower left", bbox_to_anchor=(0.115, 0.018), frameon=False,
               fontsize=9, labelcolor=INK_2)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")

    print(f"\nrows = tau {[f'{t:g}' for t in taus]}")
    print(f"cols = kappa {[f'{k:g}' for k in kaps]}")
    for i, t in enumerate(taus):
        cells = ["  DEAD" if D[i, j] else
                 ("     -" if np.isnan(M[i, j]) else f"{M[i,j]:6.2f}")
                 for j in range(len(kaps))]
        print(f"  tau={t:<9g} " + " ".join(cells))
    print("\ndefined-feature count:")
    for i, t in enumerate(taus):
        print(f"  tau={t:<9g} " + " ".join(f"{int(ND[i,j]):3d}" for j in range(len(kaps))))


if __name__ == "__main__":
    main()
