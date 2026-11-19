"""
diag_sweep.py -- start-up transient and synaptic time scale (diag_rates.py setup: identical input
spikes, target w* as physical current, closed and open loop), MNIST test digits.

  start-up  spike counts / Gamma from t0 on (t0 = 0, 25, 50, 100 ms discarded per sample)
  tau_u     synaptic current kernel exp(-t/tau_u)/tau_u at the same charge per spike (weights
            are divided by tau_u: the mean drive is unchanged, fluctuations get slower and
            smaller)
Per (model, tau_u): metrics for every t0 + population PSTH (5 ms bins) of target and surrogate.

    $PY diag_sweep.py [--n 200] [--taus 1,2,5,10,20] [--t0 0,25,50,100]
    $PY diag_sweep.py --plot
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import spec as S
import circuit as CI
import run as R
import diag_rates as D
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec
from lsmkit.metrics import agreement as AG

OUT = os.path.join(HERE, "data", "diag_sweep")
DT = D.DT
RMAX = 160
BIN = 5.0


def sim(net, n, y0, ev):
    o = engine_gpu.simulate(net, np.zeros(n, np.int32), np.full(n, int(round(S.T_SAMPLE / DT)), np.int64),
                            y0, ev, DT, Rec(spikes=tuple(range(n)), rmax=RMAX))
    assert (o["spikes"][2] < RMAX).all(), "spike buffer capped"
    return o["spikes"][1], o["spikes"][2]


def counts_after(sp, t0):
    spt, spn = sp
    valid = np.arange(spt.shape[2])[None, None] < spn[..., None]
    return ((spt * DT >= t0) & valid).sum(-1).astype(float)


def psth(sp):
    spt, spn = sp
    valid = np.arange(spt.shape[2])[None, None] < spn[..., None]
    h = np.histogram(spt[valid] * DT, bins=np.arange(0, S.T_SAMPLE + BIN, BIN))[0]
    return h / (spt.shape[0] * spt.shape[1] * BIN / 1000)          # Hz per neuron


def metrics(a, b, t0):
    ct, cs = counts_after(a, t0), counts_after(b, t0)
    T = S.T_SAMPLE - t0
    g = AG.gamma_table(a[0], a[1], b[0], b[1], DT, 2.0, T, t0_ms=t0)
    return dict(t0=t0, rate_t=float(ct.mean() * 1000 / T), rate_s=float(cs.mean() * 1000 / T), cka=AG.cka(ct, cs),
                count_r=float(np.nanmedian(AG.count_corr(ct, cs))), rate_r=float(np.nanmean(AG.rate_corr_neurons(ct, cs))),
                gamma=float(np.nanmedian(g)))


def run(z, tau, n, t0s, data, nin):
    f = os.path.join(OUT, f"{z}_tau{tau:g}.json")
    if os.path.exists(f):
        return json.load(open(f))
    S.TAU_U = tau
    w = json.load(open(os.path.join(HERE, "data", "campaign_mnist", "best_w.json")))[f"{z}@fastin"]
    idx = 60000 + np.arange(n)
    circ = CI.build(nin, w, DT, np.random.default_rng(100), pulse=R.PULSE_U, input_neurons=False)
    oc = D.open_loop(circ)
    bt, _ = R.binding(z, S.N_LIQ, DT)
    bs, _ = R.binding(f"sur:{z}", S.N_LIQ, DT)
    net = lambda c, b: LS.stack([c], [b], CI.receptors())
    yt, ys = R.y0_for(z, n, S.N_LIQ), R.y0_for(f"sur:{z}", n, S.N_LIQ)
    ev = R.input_layer(f"{z}@fastin", z, nin, data, idx, "mnist", DT)
    tc = sim(net(circ, bt), n, yt, ev)
    sc = sim(net(circ, bs), n, ys, ev)
    rev = D.replay_events(ev, (None,) + tc, nin)
    so = sim(net(oc, bs), n, ys, rev)
    r = dict(model=z, tau_u=tau, w=w, n=n, closed=[metrics(tc, sc, t) for t in t0s], open=[metrics(tc, so, t) for t in t0s],
             psth=dict(target=psth(tc).tolist(), sur_closed=psth(sc).tolist(), sur_open=psth(so).tolist()), bin_ms=BIN)
    json.dump(r, open(f, "w"), indent=1)
    S.TAU_U = 1.0
    return r


def plot(models, taus):
    import figs as F
    import matplotlib.pyplot as plt
    L = {(z, t): json.load(open(os.path.join(OUT, f"{z}_tau{t:g}.json"))) for z in models for t in taus
         if os.path.exists(os.path.join(OUT, f"{z}_tau{t:g}.json"))}
    os.makedirs(os.path.join(HERE, "figs_diag"), exist_ok=True)
    # 1. start-up (paper tau_u = 1 ms): PSTH + metrics vs t0
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 4.2), gridspec_kw=dict(hspace=0.5, wspace=0.35))
    for c, z in enumerate(models):
        r = L.get((z, taus[0]))
        if r is None:
            continue
        x = np.arange(len(r["psth"]["target"])) * r["bin_ms"] + r["bin_ms"] / 2
        a = ax[0, c]
        a.plot(x, r["psth"]["target"], color=F.ZCOL[z], lw=1.2, label="target")
        a.plot(x, r["psth"]["sur_open"], color=F.ZCOL[z], lw=1.0, ls="--", label="surrogate, open loop")
        a.plot(x, r["psth"]["sur_closed"], color=F.ZCOL[z], lw=0.8, ls=":", label="surrogate, closed loop")
        a.set_title(F.ZLAB[z], fontsize=8); a.set_xlabel("time in sample (ms)")
        if c == 0:
            a.set_ylabel("liquid rate (Hz)"); a.legend(fontsize=5.5, frameon=False)
        b = ax[1, c]
        t0 = [m["t0"] for m in r["open"]]
        for key, mk, lab in (("count_r", "o", "count $r$"), ("cka", "s", "CKA"), ("gamma", "^", "$\\Gamma$ (2 ms)")):
            b.plot(t0, [m[key] for m in r["open"]], marker=mk, ms=3, lw=0.9, color="k", label=lab + " open")
            b.plot(t0, [m[key] for m in r["closed"]], marker=mk, ms=3, lw=0.9, color="0.6", ls="--", label=lab + " closed")
        rr = [m["rate_s"] / m["rate_t"] if m["rate_t"] > 0 else np.nan for m in r["open"]]
        b.plot(t0, rr, marker="d", ms=3, lw=0.9, color=F.ZCOL[z], label="rate sur/tgt open")
        b.set_xlabel("discarded start $t_0$ (ms)"); b.set_ylim(-0.1, 1.25); b.axhline(1, color="0.8", lw=0.4)
        if c == 0:
            b.set_ylabel("agreement"); b.legend(fontsize=5, frameon=False, ncol=1, loc="lower right")
    for i, a in enumerate(ax.flat):
        a.text(-0.2, 1.1, "abcdef"[i], transform=a.transAxes, fontweight="bold", fontsize=9)
    fig.suptitle("start-up transient, $\\tau_u$ = 1 ms (paper), identical input, target $w^*$", fontsize=8)
    fig.savefig(os.path.join(HERE, "figs_diag", "startup.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "figs_diag", "startup.png"), dpi=160, bbox_inches="tight")
    # 2. synaptic time scale: rates, count r, Gamma vs tau_u (t0 = 0 and 50 ms) + HH PSTHs
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 4.2), gridspec_kw=dict(hspace=0.5, wspace=0.38))
    T0 = (0, 50)
    for z in models:
        ts = [t for t in taus if (z, t) in L]
        g = lambda loop, k, t0: [next(m for m in L[(z, t)][loop] if m["t0"] == t0)[k] for t in ts]
        ax[0, 0].plot(ts, g("open", "rate_t", 0), "-o", ms=3, color=F.ZCOL[z], label=f"{F.ZLAB[z]} target")
        ax[0, 0].plot(ts, g("open", "rate_s", 0), "--o", ms=3, mfc="w", color=F.ZCOL[z], label=f"{F.ZLAB[z]} surr. (open)")
        ax[0, 1].plot(ts, g("closed", "rate_s", 0), "--o", ms=3, mfc="w", color=F.ZCOL[z])
        ax[0, 1].plot(ts, g("closed", "rate_t", 0), "-o", ms=3, color=F.ZCOL[z])
        for t0, ls in zip(T0, ("-", ":")):
            ax[1, 0].plot(ts, g("open", "count_r", t0), ls, marker="o", ms=3, color=F.ZCOL[z])
            ax[1, 1].plot(ts, g("closed", "count_r", t0), ls, marker="o", ms=3, color=F.ZCOL[z])
            ax[1, 2].plot(ts, g("open", "gamma", t0), ls, marker="o", ms=3, color=F.ZCOL[z])
    for a, ttl in ((ax[0, 0], "rate, open loop"), (ax[0, 1], "rate, closed loop")):
        a.set_yscale("log"); a.set_ylabel("liquid rate (Hz)"); a.set_title(ttl, fontsize=8)
    ax[0, 0].legend(fontsize=5, frameon=False, ncol=1)
    for a, ttl in ((ax[1, 0], "count $r$, open loop"), (ax[1, 1], "count $r$, closed loop"), (ax[1, 2], "$\\Gamma$ (2 ms), open loop")):
        a.set_title(ttl, fontsize=8); a.set_ylim(-0.1, 1.02)
    ax[1, 0].plot([], [], "k-", label="$t_0$ = 0"); ax[1, 0].plot([], [], "k:", label="$t_0$ = 50 ms")
    ax[1, 0].legend(fontsize=5.5, frameon=False)
    a = ax[0, 2]
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate([t for t in taus if ("hh", t) in L]):
        r = L[("hh", t)]
        x = np.arange(len(r["psth"]["target"])) * r["bin_ms"] + r["bin_ms"] / 2
        col = cmap(i / max(len(taus) - 1, 1))
        a.plot(x, r["psth"]["target"], color=col, lw=1.0, label=f"{t:g} ms")
        a.plot(x, r["psth"]["sur_open"], color=col, lw=0.8, ls="--")
    a.set_title("HH PSTH (— target, - - surr. open)", fontsize=7.5); a.set_xlabel("time in sample (ms)")
    a.legend(fontsize=5, frameon=False, title="$\\tau_u$", title_fontsize=5.5)
    for a in ax.flat:
        if a is not ax[0, 2]:
            a.set_xscale("log"); a.set_xticks(taus, [f"{t:g}" for t in taus]); a.minorticks_off()
            a.set_xlabel("$\\tau_u$ (ms)")
    for i, a in enumerate(ax.flat):
        a.text(-0.2, 1.1, "abcdef"[i], transform=a.transAxes, fontweight="bold", fontsize=9)
    fig.suptitle("synaptic time scale at equal charge per spike (identical input, target $w^*$)", fontsize=8)
    fig.savefig(os.path.join(HERE, "figs_diag", "tau_u.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "figs_diag", "tau_u.png"), dpi=160, bbox_inches="tight")
    print("wrote figs_diag/startup.pdf, figs_diag/tau_u.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--taus", default="1,2,5,10,20")
    ap.add_argument("--t0", default="0,25,50,100")
    ap.add_argument("--models", default=",".join(D.ZOO))
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    models, taus = a.models.split(","), [float(x) for x in a.taus.split(",")]
    if not a.plot:
        os.makedirs(OUT, exist_ok=True)
        data, _, nin = R.load("mnist")
        for z in models:
            for t in taus:
                r = run(z, t, a.n, [float(x) for x in a.t0.split(",")], data, nin)
                for loop in ("closed", "open"):
                    print(f"{z:14s} tau_u {t:<4g} {loop:6s} " + "  ".join(
                        f"t0 {m['t0']:<3g}: tgt {m['rate_t']:6.1f} sur {m['rate_s']:6.1f} r {m['count_r']:.2f} G {m['gamma']:.2f}"
                        for m in r[loop]), flush=True)
    plot(models, taus)


if __name__ == "__main__":
    main()
