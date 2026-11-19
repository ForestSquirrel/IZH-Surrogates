"""
v3_plot_family.py -- one figure per out-of-family FAMILY.

    row 1 (spans 4 cols) : regime-diagnostic heatmap over the family's own
                           parameter sweep x the protocol amplitude ladder
    row 2 (4 panels)     : the MODEL's reference-step trace at four sampled
                           parameter values
    row 3 (4 panels)     : the SURROGATE drawn for each of those four

WHY A (parameter x gamma) HEATMAP AND NOT A CURVE

The finding this has to show is §51/§52's: a single cell hands the schema several
regimes AT ONCE, which is a statement about variation along the AMPLITUDE axis.
Collapsing gamma into one verdict per parameter value would integrate away the
very thing being displayed. A column that changes colour vertically is a feature
vector assembled from more than one dynamical system; a column of one colour is
the case the pipeline was designed for.

THE FOUR SAMPLED COLUMNS SPAN THE DIAGNOSTIC, THEY DO NOT FLATTER IT

Picked as the two lowest and two highest ISI-jump columns, so every figure shows
the method working and failing on the same family. For Hindmarsh-Rose that puts
the good r-band and the chaotic band side by side.

Colour is the distinct settled-ISI count -- an ordinal magnitude, so ONE hue
light->dark (never a rainbow). "Silent at this drive" is a hatched neutral, never
a step of the ramp: a cell that does not fire is in a different state, not at an
extreme of the same one. §55 is the record of what conflating those two does --
it read Wilson's depolarization block as a regime change.

Traces, protocol geometry and the x-axis convention are v3_plots'; this module
imports them rather than restating them, so the panels here are the same objects
as plots/*family/*_refstep.png.

    python scripts/v3_plot_family.py --family Hindmarsh-Rose
    python scripts/v3_plot_family.py --all
"""
import os, sys, json, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

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
from v3_features import compute_features_v3
from v3_landmarks import landmarks as izh_landmarks
from v3_models import izh_rhs_v3
from v3_an_features import V3Spec, compute_features_an, landmarks_an
from v3_regime import GAMMA_LADDER, regime_report_izh, regime_report_an
from v3_scoring import d2
import v3_plots as P

# --- design tokens: the sequential blue ramp, matching the validity figures ---
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
DEAD_FILL, DEAD_EDGE = "#d8d7d2", "#8d8b84"      # silent: the cell does not fire
NOANC_FILL, NOANC_EDGE = "#eceae4", "#a5a29a"    # no anchor: it fires, but the
                                                 # window holds no settled cycle,
                                                 # so nothing can be normalized
C_MARK = "#d03b3b"                                    # status-critical
CMAP = LinearSegmentedColormap.from_list(
    "reg", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
BANDS = [1, 2, 3, 4, 6, 9, 15, 30, 200]               # distinct settled ISIs

# Wilson's spike-detection threshold. Not a model property -- v_th is where we
# CHOOSE to call a spike, and -0.5 (i.e. -50 mV, V being in units of 100 mV) is
# the settled choice from §49. sweeps.py still carries 0.0 for v2's sake.
V_TH_OVERRIDE = {"Wilson": -0.5}


def slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


# ---------------------------------------------------------------- families ----
def families():
    """[(name, pname, [(label, x, kind, payload)])].

    kind "resetfree" -> a V3Spec through the B4 port
    kind "reset"     -> (rhs, params, rheo_hi) through v3's Izhikevich path

    AdEx is the odd one out and stays that way: it has four NAMED regimes, not a
    smooth sweep, so its x axis is categorical. Forcing a continuous parameter on
    it would invent a sweep the model was never characterized over.
    """
    from sweeps import resetfree_sweeps
    out = []
    for sw in resetfree_sweeps():
        cfgs = sw["configs"]
        Pm = np.array([np.asarray(c[1], float) for c in cfgs])
        vi = int(np.argmax(Pm.std(axis=0)))            # the swept coordinate
        pname = cfgs[0][0].split("=")[0].split()[-1]
        vth = V_TH_OVERRIDE.get(sw["family"], sw["v_th"])
        items = []
        for lbl, pr in cfgs:
            sp = V3Spec(rhs=sw["rhs"], y0=np.asarray(sw["y0"], float),
                        params=np.asarray(pr, float), v_th=vth,
                        rheo_lo=0.0, rheo_hi=sw["rheo_hi"], h=sw["h"],
                        boot_t_trans=sw["t_discard"], boot_t_win=sw["t_win"],
                        boot_t_disc=sw["t_discard"], name=lbl)
            items.append((lbl, float(pr[vi]), "resetfree", sp))
        out.append((sw["family"], pname, items))

    from v3_an_zoo import reset_targets
    items = [(lbl, float(i), "reset", (rhs, pr, rh))
             for i, (lbl, rhs, pr, rh) in enumerate(reset_targets())]
    out.append(("AdEx", "regime", items))
    return out


# ------------------------------------------------------------ measurement ----
def anchor_and_regime(kind, payload, bufs):
    """(i_rh, T_cycle, regime report or None)."""
    if kind == "resetfree":
        sp = payload
        lm = landmarks_an(sp, np.empty(1 << 16), np.empty(1 << 16), auto_boot=True)
        if lm["T_cycle"] < C.SENT_CUT or lm["i_rh"] < C.SENT_CUT:
            return lm.get("i_rh", C.NAN_SENTINEL), C.NAN_SENTINEL, None
        rep = regime_report_an(sp, lm["i_rh"], lm["T_cycle"])
        return lm["i_rh"], lm["T_cycle"], rep
    rhs, pr, rh = payload
    th = np.ascontiguousarray(np.asarray(pr, float))
    lm = izh_landmarks(izh_rhs_v3 if rhs is None else rhs, th, bufs, rheo_hi=rh)
    if lm["T_cycle"] < C.SENT_CUT or lm["i_rh"] < C.SENT_CUT:
        return lm.get("i_rh", C.NAN_SENTINEL), C.NAN_SENTINEL, None
    rep = regime_report_izh(rhs, th, lm["i_rh"], lm["T_cycle"], bufs, rheo_hi=rh)
    return lm["i_rh"], lm["T_cycle"], rep


def sigma(atlas, sig_path):
    """Robust per-feature scale, with the std fallback for zero-MAD columns."""
    from data import AtlasView
    view = AtlasView(atlas)
    sig = json.load(open(sig_path))
    rng = np.random.default_rng(1)
    A0 = np.asarray(view._feats[np.sort(rng.choice(view.n, 200_000, replace=False))],
                    float)[:, view.keep]
    S = np.zeros(C.N_ACTIVE)
    for j, nm in enumerate(C.FEATURE_NAMES):
        s = sig.get(nm, 0.0)
        if s > 0:
            S[j] = s
        else:
            c = A0[:, j]; d = c > C.SENT_CUT
            S[j] = float(np.std(c[d])) if d.sum() > 100 else 1.0
    return S


def posterior_best(kind, payload, S, bufs, model_dir, n_post):
    """Target features -> posterior -> best-matching draw. None if unscorable."""
    import run_surrogate as rs
    out = np.full(C.N_COLS, C.NAN_SENTINEL)
    if kind == "resetfree":
        f_t, it = compute_features_an(payload)
    else:
        rhs, pr, rh = payload
        f_t, it = compute_features_v3(rhs, np.ascontiguousarray(np.asarray(pr, float)),
                                      bufs, out=out, rheo_hi=rh)
    f_t = np.asarray(f_t[:C.N_ACTIVE], float).copy()
    if it["gated"] or (f_t > C.SENT_CUT).sum() < 5:
        return None
    theta, _ = rs.cinn_infer(model_dir, f_t, n_post)
    theta = np.asarray(theta, float)
    best = None
    for k in range(len(theta)):
        th = np.ascontiguousarray(theta[k])
        fk, ik = compute_features_v3(izh_rhs_v3, th, bufs, out=out)
        if ik["gated"]:
            continue
        fk = np.asarray(fk[:C.N_ACTIVE], float).copy()
        if (fk > C.SENT_CUT).sum() < 5:
            continue
        v, nd = d2(f_t, fk, S)
        if best is None or v < best[0]:
            best = (v, nd, th.copy())
    if best is None:
        return None
    d2v, nd, th = best
    lm = izh_landmarks(izh_rhs_v3, th, bufs)
    if lm["T_cycle"] < C.SENT_CUT or lm["i_rh"] < C.SENT_CUT:
        return None
    return dict(d2=d2v, n_dims=nd, theta=th,
                tau=float(it["T_cycle"] / lm["T_cycle"]),
                k_I=float(it["i_rh"] / lm["i_rh"]),
                n_defined=int((f_t > C.SENT_CUT).sum()))


def model_side(kind, payload, i_rh, tc, l_inf, label):
    if kind == "resetfree":
        sp = payload
        return dict(rhs=sp.rhs, params=sp.params, y0=sp.y0, v_th=sp.v_th, h=sp.h,
                    tc=tc, i_rh=i_rh, l_inf=l_inf, label=label)
    rhs, pr, rh = payload
    pr = np.asarray(pr, float)
    return dict(rhs=rhs, params=pr, y0=np.array([C.V_REST, pr[1] * C.V_REST]),
                v_th=C.V_TH, h=C.DT, reset_v=pr[2], reset_du=pr[3],
                tc=tc, i_rh=i_rh, l_inf=l_inf, label=label)


# ----------------------------------------------------------------- figure ----
def render(family, pname, items, cols_data, NIS, SIL, NOA, picked, outp):
    refstep = [p for p in P.protocols() if p["key"] == "refstep"][0]
    xs_lbl = [it[1] for it in items]
    ng, npv = len(GAMMA_LADDER), len(items)

    fig = plt.figure(figsize=(15.4, 10.8))
    fig.patch.set_facecolor(SURFACE)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.2, 1.0, 1.0],
                           hspace=0.60, wspace=0.24,
                           left=0.065, right=0.885, top=0.835, bottom=0.075)

    # ---- row 1: the regime landscape ----
    ax0 = fig.add_subplot(gs[0, :]); ax0.set_facecolor(SURFACE)
    norm = BoundaryNorm(BANDS, CMAP.N, extend="max")
    im = ax0.pcolormesh(np.arange(npv + 1), np.arange(ng + 1),
                        np.ma.masked_invalid(NIS), cmap=CMAP, norm=norm,
                        edgecolors=SURFACE, linewidth=1.5)
    # Two different failures, drawn differently on purpose. A SILENT cell does
    # not fire at that amplitude. A NO-ANCHOR cell fires but never settles, so
    # T_cycle is undefined and the whole column is unmeasurable -- HR r<=0.006
    # is the documented case. Painting both as one grey would repeat §55's error
    # of reading "stopped firing" as "changed regime".
    for j in range(npv):
        if NOA[j]:
            ax0.add_patch(plt.Rectangle((j, 0), 1, ng, facecolor=NOANC_FILL,
                                        edgecolor=NOANC_EDGE, hatch="xx",
                                        linewidth=0.8, zorder=3))
            ax0.text(j + 0.5, ng * 0.5, "no settled cycle", rotation=90,
                     ha="center", va="center", fontsize=8.5, color=INK_2,
                     zorder=4)
            continue
        for i in range(ng):
            if SIL[i, j]:
                ax0.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=DEAD_FILL,
                                            edgecolor=DEAD_EDGE, hatch="///",
                                            linewidth=0.6, zorder=3))
    ax0.set_xlim(0, npv); ax0.set_ylim(0, ng)
    ax0.set_xticks(np.arange(npv) + 0.5)
    ax0.set_xticklabels([f"{v:.4g}" if pname != "regime" else items[k][0]
                         for k, v in enumerate(xs_lbl)],
                        fontsize=8.5, color=INK_2,
                        rotation=0 if pname != "regime" else 12)
    ax0.set_yticks(np.arange(ng) + 0.5)
    ax0.set_yticklabels([f"{g:g}" for g in GAMMA_LADDER], fontsize=8.5, color=INK_2)
    ax0.set_xlabel(f"{pname}   (family parameter sweep)", fontsize=9.5, color=INK_2)
    ax0.set_ylabel("$\\gamma$   (drive / $i_{rh}$)", fontsize=9.5, color=INK_2)
    for s in ax0.spines.values():
        s.set_visible(False)
    ax0.tick_params(length=0)

    tags = "ABCD"
    for n, idx in enumerate(picked):
        ax0.plot([idx + 0.5], [ng + 0.22], marker="v", ms=9, color=C_MARK,
                 clip_on=False, zorder=6)
        ax0.text(idx + 0.5, ng + 0.62, tags[n], ha="center", va="bottom",
                 fontsize=10.5, color=C_MARK, fontweight="bold", clip_on=False)
        for e in (idx, idx + 1):
            ax0.axvline(e, color=C_MARK, lw=1.1, alpha=0.6, zorder=5)

    pos = gs[0, :].get_position(fig)
    cax = fig.add_axes([0.895, pos.y0, 0.013, pos.height])
    cb = fig.colorbar(im, cax=cax, ticks=BANDS, extend="max")
    cb.ax.set_yticklabels([f"{v:g}" for v in BANDS], fontsize=7.5, color=INK_2)
    cb.set_label("distinct settled ISIs", fontsize=8.5, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)

    # ---- rows 2-3: model over surrogate ----
    for n, (idx, e) in enumerate(zip(picked, cols_data)):
        axm = fig.add_subplot(gs[1, n]); axs = fig.add_subplot(gs[2, n])
        for ax in (axm, axs):
            ax.set_facecolor(SURFACE); ax.grid(alpha=0.14)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(labelsize=7.5, colors=INK_2, length=0)
        rep = e["regime"]
        vtag = rep["verdict"].upper() if rep else "NO ANCHOR"
        axm.set_title(f"{tags[n]}   {pname}={items[idx][1]:.4g}\n"
                      f"ISI jump {rep['isi_jump'] if rep else '--'} · {vtag}"
                      if pname != "regime" else
                      f"{tags[n]}   {items[idx][0]}\n"
                      f"ISI jump {rep['isi_jump'] if rep else '--'} · {vtag}",
                      fontsize=9, color=INK, loc="left", pad=6)
        if e.get("ok"):
            for ax, side, col in ((axm, e["m"], "#1f77b4"), (axs, e["s"], "#2ca02c")):
                x, v, I, sp = P.trace(side, refstep)
                ax.plot(x, v, color=col, lw=0.85)
                ax.axhline(side["v_th"], color="#d62728", lw=0.7, ls="--", alpha=0.65)
                for t in sp:
                    ax.axvline(t, color=col, alpha=0.13, lw=0.5)
                ax.text(0.015, 0.04, f"{len(sp)} spikes", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=7.5, color=col)
            b = e["best"]
            a, bb, cc, dd = (float(x) for x in b["theta"][:4])
            axs.set_title(f"$\\tau$={b['tau']:.4g}   $k_I$={b['k_I']:.4g}\n"
                          f"a={a:.4g}  b={bb:.4g}  c={cc:.4g}  d={dd:.4g}\n"
                          f"d²={b['d2']:.3g} ({b['n_dims']}d)",
                          fontsize=7.6, color=INK_2, loc="left", pad=5)
        else:
            for ax, t in ((axm, e.get("why", "no anchor")), (axs, "no surrogate")):
                ax.text(0.5, 0.5, t, transform=ax.transAxes, ha="center",
                        va="center", fontsize=9, color=DEAD_EDGE)
        if n == 0:
            axm.set_ylabel("V   model", fontsize=8.5, color=INK_2)
            axs.set_ylabel("V   surrogate", fontsize=8.5, color=INK_2)
        axs.set_xlabel("t / $T_{cycle}$", fontsize=8.5, color=INK_2)

    fig.text(0.065, 0.972, f"{family} — the regime landscape, and what the "
             "surrogate recovers across it", fontsize=14.5, color=INK,
             ha="left", va="top")
    fig.text(0.065, 0.930,
             "Top: distinct settled ISIs at each (family parameter, drive). A column that changes colour "
             "vertically is a feature vector assembled from several regimes of one cell.\n"
             "Bottom: reference-step traces (γ=2) for the four marked columns — the two lowest and the two "
             "highest ISI jump — model above, best posterior draw below.",
             fontsize=9, color=INK_2, ha="left", va="top", linespacing=1.6)
    fig.legend(handles=[Patch(facecolor=DEAD_FILL, edgecolor=DEAD_EDGE,
                              hatch="///", label="silent at this drive"),
                        Patch(facecolor=NOANC_FILL, edgecolor=NOANC_EDGE,
                              hatch="xx",
                              label="no settled cycle — column unmeasurable")],
               ncol=2,
               loc="lower left", bbox_to_anchor=(0.065, 0.006), frameon=False,
               fontsize=8.5, labelcolor=INK_2)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=140, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------- main ----
def do_family(family, pname, items, S, bufs, args):
    ng, npv = len(GAMMA_LADDER), len(items)
    NIS = np.full((ng, npv), np.nan)
    SIL = np.zeros((ng, npv), bool)
    NOA = np.zeros(npv, bool)
    recs = []
    t0 = time.time()
    for k, (lbl, x, kind, payload) in enumerate(items):
        irh, tc, rep = anchor_and_regime(kind, payload, bufs)
        if rep is None:
            NOA[k] = True
        else:
            NIS[:, k] = rep["distinct_isis"]
            SIL[:, k] = np.array(rep["spike_counts"]) == 0
        recs.append(dict(label=lbl, x=x, i_rh=float(irh), T_cycle=float(tc),
                         regime=rep))
        print(f"    [{k+1}/{npv}] {lbl:<22s} "
              f"{'no anchor' if rep is None else rep['verdict'] + ' jump=' + str(rep['isi_jump'])}"
              f"   {time.time()-t0:.0f}s", flush=True)

    ok = [i for i, r in enumerate(recs) if r["regime"] is not None]
    ok.sort(key=lambda i: (recs[i]["regime"]["isi_jump"],
                           recs[i]["regime"]["block_above"] is not None))
    if len(ok) >= 4:
        picked = [ok[0], ok[1], ok[-2], ok[-1]]
    else:
        picked = ok + [i for i in range(npv) if i not in ok]
    picked = sorted(set(picked[:4])) if len(set(picked[:4])) == 4 else picked[:4]

    cols = []
    for idx in picked:
        r = recs[idx]; lbl, x, kind, payload = items[idx]
        e = dict(regime=r["regime"], label=lbl, x=x, ok=False)
        if r["regime"] is None:
            e["why"] = "no anchor"
        else:
            b = posterior_best(kind, payload, S, bufs, args.model_dir,
                               args.n_posterior)
            if b is None:
                e["why"] = "no scorable draw"
            else:
                s = P.izh_side(b["theta"], "best draw", bufs)
                if s is None:
                    e["why"] = "surrogate has no anchor"
                else:
                    m = model_side(kind, payload, r["i_rh"], r["T_cycle"],
                                   s["l_inf"], lbl)
                    e.update(ok=True, m=m, s=s,
                             best=dict(b, theta=np.asarray(b["theta"]).tolist()))
        print(f"    posterior {lbl:<22s} "
              + ("ok  d2=%.3g" % e["best"]["d2"] if e["ok"] else e.get("why", "")),
              flush=True)
        cols.append(e)

    outp = os.path.join(args.out, f"v3_family_{slug(family)}.png")
    render(family, pname, items, cols, NIS, SIL, NOA, picked, outp)
    print(f"  wrote {outp}", flush=True)

    json.dump(dict(family=family, pname=pname,
                   x=[it[1] for it in items], labels=[it[0] for it in items],
                   gammas=GAMMA_LADDER.tolist(), distinct_isis=NIS.tolist(),
                   silent=SIL.tolist(), no_anchor=NOA.tolist(), records=recs,
                   picked=[int(i) for i in picked],
                   panels=[{k: v for k, v in e.items() if k not in ("m", "s")}
                           for e in cols]),
              open(os.path.join(RESULTS, f"v3_family_{slug(family)}.json"), "w"),
              indent=2, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-posterior", type=int, default=128)
    ap.add_argument("--model-dir", default=os.path.join(MODELS, "cinn_v3_p015"))
    ap.add_argument("--atlas", default=os.path.join(DATA, "atlas_v3_1e7"))
    ap.add_argument("--sigma", default=os.path.join(DATA, "atlas_v3_1e7", "sigma_v3.json"))
    ap.add_argument("--out", default=os.path.join(PLOTS, "family"))
    args = ap.parse_args()

    fams = families()
    names = [f[0] for f in fams]
    if not args.all:
        if not args.family:
            raise SystemExit(f"--family <one of {names}>  or  --all")
        fams = [f for f in fams if args.family.lower() in f[0].lower()]
        if not fams:
            raise SystemExit(f"no family matching {args.family!r}; have {names}")

    S = sigma(args.atlas, args.sigma)
    bufs = alloc_buffers()
    for family, pname, items in fams:
        print(f"\n=== {family} ({len(items)} configs) ===", flush=True)
        do_family(family, pname, items, S, bufs, args)


if __name__ == "__main__":
    main()
