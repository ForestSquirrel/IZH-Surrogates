"""
v3_plot_pair.py -- one WIDE two-panel model/surrogate figure for a single
family-sweep configuration.

The family figures (v3_plot_family.py) show four configurations at once, which
costs each panel its horizontal resolution: at ~3.5 in wide, twelve cycles of a
burster read as a picket fence and the intra-burst structure -- the thing worth
looking at -- is below a pixel. This renders ONE configuration across the full
width so that structure is visible.

Nothing is recomputed. The surrogate theta is read from the family run's JSON,
so this figure and the corresponding panel of the family figure are the same
draw; re-inferring would risk showing a different one (the flow is stochastic and
run_surrogate is nondeterministic on GPU).

Traces, protocol geometry and the cycle-normalized x axis are v3_plots'.

    python scripts/v3_plot_pair.py --family Wilson --label gT=2.00
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from v3_core import alloc_buffers
import v3_plots as P
import v3_plot_family as F

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
C_MODEL, C_SURR, C_I, C_TH = "#2a78d6", "#1baf7a", "#898781", "#d03b3b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--label", required=True,
                    help="substring of the config label, e.g. 'gT=2.00'")
    ap.add_argument("--protocol", default="refstep")
    ap.add_argument("--cycles", type=float, default=None,
                    help="override the protocol's span, in T_cycle units")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fams = {n: (pn, it) for n, pn, it in F.families()}
    match = [n for n in fams if args.family.lower() in n.lower()]
    if not match:
        raise SystemExit(f"no family matching {args.family!r}; have {sorted(fams)}")
    family = match[0]
    pname, items = fams[family]

    jf = os.path.join(RESULTS, f"v3_family_{F.slug(family)}.json")
    if not os.path.exists(jf):
        raise SystemExit(f"{jf} missing -- run v3_plot_family.py --family {family} first")
    D = json.load(open(jf))
    pan = [p for p in D["panels"] if args.label.lower() in p["label"].lower()]
    if not pan:
        raise SystemExit(f"no panel matching {args.label!r}; have "
                         f"{[p['label'] for p in D['panels']]}")
    pan = pan[0]
    if not pan.get("ok"):
        raise SystemExit(f"{pan['label']} was not scorable: {pan.get('why')}")
    rec = [r for r in D["records"] if r["label"] == pan["label"]][0]
    idx = D["labels"].index(pan["label"])
    lbl, x, kind, payload = items[idx]

    bufs = alloc_buffers()
    theta = np.ascontiguousarray(np.asarray(pan["best"]["theta"], float))
    surr = P.izh_side(theta, "best draw", bufs)
    if surr is None:
        raise SystemExit("saved surrogate theta has no anchor")
    model = F.model_side(kind, payload, rec["i_rh"], rec["T_cycle"],
                         surr["l_inf"], lbl)

    proto = [p for p in P.protocols() if p["key"] == args.protocol][0]
    if args.cycles:
        proto = dict(proto, span=args.cycles)

    fig, axes = plt.subplots(2, 1, figsize=(17.0, 7.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    b = pan["best"]; r = pan["regime"]
    sides = [(model, C_MODEL, f"{lbl}   (model)"),
             (surr, C_SURR,
              "surrogate   "
              f"a={theta[0]:.4g}  b={theta[1]:.4g}  c={theta[2]:.4g}  d={theta[3]:.4g}")]
    for ax, (side, col, lab) in zip(axes, sides):
        xx, v, I, sp = P.trace(side, proto)
        ax.set_facecolor(SURFACE)
        ax.plot(xx, v, color=col, lw=1.0, zorder=3)
        ax.axhline(side["v_th"], color=C_TH, lw=0.9, ls="--", alpha=0.75, zorder=2)
        ax.text(0.999, side["v_th"], f" v_th={side['v_th']:g}", color=C_TH,
                fontsize=8, ha="right", va="bottom", zorder=4,
                transform=ax.get_yaxis_transform())
        for s in sp:
            ax.axvline(s, color=col, alpha=0.10, lw=0.6, zorder=1)
        ax2 = ax.twinx()
        ax2.plot(xx, I, color=C_I, lw=0.8, alpha=0.55, zorder=0)
        ax2.set_ylabel("I", color=C_I, fontsize=8.5)
        ax2.tick_params(axis="y", labelsize=7.5, colors=C_I, length=0)
        for s in ax2.spines.values():
            s.set_visible(False)
        ax.set_title(f"{lab}   —   {len(sp)} spikes", fontsize=10, color=INK,
                     loc="left", pad=6)
        ax.set_ylabel("V", fontsize=9.5, color=INK_2)
        ax.grid(alpha=0.14)
        ax.tick_params(labelsize=8.5, colors=INK_2, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("t   (units of each side's own $T_{cycle}$)",
                        fontsize=9.5, color=INK_2)

    fig.text(0.055, 0.972, f"{family}  {pan['label'].split()[-1]}  —  "
             f"{proto['title']}", fontsize=14, color=INK, ha="left", va="top")
    fig.text(0.055, 0.932,
             f"τ={b['tau']:.4g}   $k_I$={b['k_I']:.4g}   d²={b['d2']:.4g} "
             f"({b['n_dims']}d)   ·   regime {r['verdict'].upper()}, ISI jump "
             f"{r['isi_jump']}   ·   x axis is each side's own $T_{{cycle}}$, so "
             "the stimulus events align by construction",
             fontsize=9.2, color=INK_2, ha="left", va="top")
    fig.subplots_adjust(left=0.055, right=0.945, top=0.875, bottom=0.085, hspace=0.30)

    outp = args.out or os.path.join(
        PLOTS, "family",
        f"v3_pair_{F.slug(family)}_{F.slug(pan['label'].split()[-1])}_{args.protocol}.png")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=150, facecolor=SURFACE)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
