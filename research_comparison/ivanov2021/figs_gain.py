"""
figs_gain.py -- figures of the synaptic-gain section (tex/sec_gain.tex), from the diagnostics on
disk (diag_sweep.py, diag_matched.py, campaign matched_w.json via gain.py, diag_waveforms.py).

    iv_gain.pdf
        a  liquid rate vs tau_u at equal charge per spike (open loop, identical input): the target
           depends on the synaptic time scale, the surrogate does not
        b  liquid rate vs w at tau_u = 1 ms (closed loop): w_t, w_s at the operating rate
        c  the prediction (gain.py) for HH: target r(nu) at w*_t and its fixed point nu*,
           surrogate r(w) at nu*
        d  g vs tau_u: network vs prediction
        e  predicted vs network g
        f  per-neuron count r vs tau_u: same w vs predicted w_s

    $PY figs_gain.py [--out tex/figs]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import figs as F
import matplotlib.pyplot as plt

TAUS = (1.0, 2.0, 5.0, 10.0, 20.0)


def load(d, z, t):
    f = os.path.join(HERE, "data", d, f"{z}_tau{t:g}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def logx_tau(a):
    a.set_xscale("log"); a.set_xticks(TAUS, [f"{t:g}" for t in TAUS]); a.minorticks_off(); a.set_xlabel("$\\tau_u$ (ms)")


def main():
    ap = argparse.ArgumentParser()
    import texdirs as TD
    ap.add_argument("--out", default=TD.figs("gain"))
    a_ = ap.parse_args()
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 4.6), gridspec_kw=dict(hspace=0.62, wspace=0.42))
    ax = ax.ravel()
    # a: rate vs tau_u, equal charge, open loop
    a = ax[0]
    for z in F.ZOO:
        rows = [(t, load("diag_sweep", z, t)) for t in TAUS]
        rows = [(t, r["open"][0]) for t, r in rows if r]
        a.plot([t for t, _ in rows], [m["rate_t"] for _, m in rows], "-o", ms=3, color=F.ZCOL[z], label=F.ZLAB[z])
        a.plot([t for t, _ in rows], [m["rate_s"] for _, m in rows], "--o", ms=3, mfc="w", color=F.ZCOL[z])
    a.set_yscale("log"); logx_tau(a); a.set_ylabel("liquid rate (Hz)")
    a.plot([], [], "-", color="k", lw=1.0, label="target"); a.plot([], [], "--", color="k", lw=0.9, label="surrogate")
    a.legend(fontsize=5.5, frameon=False, loc="upper right"); F.label(a, "a")
    # b: rate vs w at tau_u = 1 ms
    a = ax[1]
    for z in F.ZOO:
        r = load("diag_matched", z, 1.0)
        if r is None:
            continue
        a.plot(r["scan_w"], r["scan_t"], "-", color=F.ZCOL[z], lw=1.0)
        a.plot(r["scan_w"], r["scan_s"], "--", color=F.ZCOL[z], lw=0.9)
        if r.get("w_t"):
            a.annotate("", xy=(r["w_s"], r["R"]), xytext=(r["w_t"], r["R"]),
                       arrowprops=dict(arrowstyle="->", color=F.ZCOL[z], lw=0.8))
            a.plot([r["w_t"]], [r["R"]], "o", color=F.ZCOL[z], ms=3)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(0.1, 400)
    a.set_xlabel("$w$ (target current units)"); a.set_ylabel("liquid rate (Hz)")
    a.plot([], [], "-", color="k", lw=1.0, label="target"); a.plot([], [], "--", color="k", lw=0.9, label="surrogate")
    a.legend(fontsize=5.5, frameon=False, loc="lower right"); F.label(a, "b")
    # c: the prediction for HH (campaign matched_w.json)
    a = ax[2]
    mw = os.path.join(HERE, "data", "campaign_mnist", "matched_w.json")
    if os.path.exists(mw):
        p = json.load(open(mw)).get("sur:hh@fastin")
        nu, rt = np.array(p["nu_grid"]), np.array(p["r_t_nu"])
        a.plot(nu, rt, "-", color=F.ZCOL["hh"], lw=1.0, label="target $r(\\nu)$ at $w^*_t$")
        a.plot(nu, nu, color="0.6", lw=0.5, ls=":")
        a.plot([p["nu"]], [p["nu"]], "o", color=F.ZCOL["hh"], ms=3.5)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Poisson input rate $\\nu$ (Hz)")
        a.set_ylabel("single-neuron rate (Hz)")
        b = a.inset_axes([0.62, 0.24, 0.34, 0.3])
        b.plot(p["w_grid"], p["r_s_w"], "--", color=F.ZCOL["hh"], lw=0.9)
        b.axhline(p["nu"], color="0.6", lw=0.5, ls=":"); b.plot([p["w_s"]], [p["nu"]], "o", mfc="w", color=F.ZCOL["hh"], ms=3)
        b.set_xscale("log"); b.set_yscale("log"); b.tick_params(labelsize=4.5)
        a.legend(fontsize=5, frameon=False, loc="upper left")
    F.label(a, "c")
    # d: g vs tau_u, network vs prediction
    a = ax[3]
    for z in F.ZOO:
        rows = [(t, load("diag_matched", z, t)) for t in TAUS]
        rows = [(t, r) for t, r in rows if r and r.get("g")]
        a.plot([t for t, _ in rows], [r["g"] for _, r in rows], "-o", ms=3, color=F.ZCOL[z], label=F.ZLAB[z])
        a.plot([t for t, r in rows if r.get("g_mf")], [r["g_mf"] for _, r in rows if r.get("g_mf")], "--o", ms=3, mfc="w", color=F.ZCOL[z])
    a.set_yscale("log"); logx_tau(a); a.axhline(1, color="0.7", lw=0.4)
    a.set_yticks([1, 2, 3], ["1", "2", "3"]); a.minorticks_off()
    a.plot([], [], "-", color="k", lw=1.0, label="network"); a.plot([], [], "--", color="k", lw=0.9, label="prediction")
    a.set_ylabel("gain $g = w_s / w_t$")
    a.legend(fontsize=5.5, frameon=False); F.label(a, "d")
    # e: predicted vs network g
    a = ax[4]
    P = []
    for z in F.ZOO:
        for t in TAUS:
            r = load("diag_matched", z, t)
            if r and r.get("g") and r.get("g_mf"):
                a.plot(r["g"], r["g_mf"], "o", ms=3.5, color=F.ZCOL[z]); P.append((r["g"], r["g_mf"]))
    if P:
        lo, hi = min(min(p) for p in P) * 0.85, max(max(p) for p in P) * 1.15
        a.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls=":")
        err = np.array([abs(b / a_ - 1) for a_, b in P])
        a.text(0.05, 0.92, f"median |error| {np.median(err) * 100:.1f} %, max {err.max() * 100:.0f} %",
               transform=a.transAxes, fontsize=5.5)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks([1, 2, 3], ["1", "2", "3"]); a.set_yticks([1, 2, 3], ["1", "2", "3"])
    a.minorticks_off(); a.set_xlabel("network $g$"); a.set_ylabel("predicted $g$"); F.label(a, "e")
    # f: count r same w vs predicted
    a = ax[5]
    for z in F.ZOO:
        rows = [(t, load("diag_matched", z, t)) for t in TAUS]
        rows = [(t, r) for t, r in rows if r and "same" in r]
        a.plot([t for t, _ in rows], [r["same"]["count_r"] for _, r in rows], ":s", ms=2.8, color=F.ZCOL[z])
        a.plot([t for t, r in rows if "predicted" in r], [r["predicted"]["count_r"] for _, r in rows if "predicted" in r],
               "-o", ms=3, color=F.ZCOL[z])
    a.plot([], [], ":s", color="k", ms=2.8, label="same $w$"); a.plot([], [], "-o", color="k", ms=3, label="predicted $w_s$")
    logx_tau(a); a.set_ylim(0, 1.02); a.set_ylabel("per-neuron count $r$")
    a.legend(fontsize=5.5, frameon=False, loc="lower right")
    F.label(a, "f")
    os.makedirs(a_.out, exist_ok=True)
    F.save(fig, a_.out, "iv_gain")
    print("wrote", os.path.join(a_.out, "iv_gain.pdf"))


if __name__ == "__main__":
    main()
