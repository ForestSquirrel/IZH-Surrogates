"""
diag_gain.py -- is the surrogate's synaptic gain back-solvable? (diag_rates.py setup: identical
input spikes, MNIST test digits; weights as physical current, surrogate via 'sur:')

  scan    closed-loop liquid rate vs w for target and surrogate, per tau_u
  match   for target operating rates R* the weights w_t, w_s giving R* (log-interpolated on the
          rising branch); gain g = w_s / w_t; agreement at (w_t, w_s) vs at the same w; rate
          ratio rho = r_s(w_t) / r_t(w_t) at the same w (the observable that would back-solve g)
  psc     single-neuron prediction without a network: smallest single-PSC charge (kernel
          exp(-t/tau_u)/tau_u, circuit weight units = physical charge) that fires the neuron
          from rest, target vs surrogate -> q_s / q_t
  xcorr   target-vs-surrogate spike cross-correlogram per neuron (open loop): is there a lag?

    $PY diag_gain.py [--taus 1,2,5,10,20] [--n 100]
    $PY diag_gain.py --plot
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

OUT = os.path.join(HERE, "data", "diag_gain")
DT = D.DT
RSTAR = (5.0, 10.0, 20.0, 40.0)
RUNAWAY = 300.0
NS = int(round(S.T_SAMPLE / DT))


def counts(z, model, w, ev, n, nin):
    circ = CI.build(nin, w, DT, np.random.default_rng(100), pulse=R.PULSE_U, input_neurons=False)
    b, _ = R.binding(model, S.N_LIQ, DT)
    o = engine_gpu.simulate(LS.stack([circ], [b], CI.receptors()), np.zeros(n, np.int32), np.full(n, NS, np.int64),
                            R.y0_for(model, n, S.N_LIQ), ev, DT, Rec())
    return o["counts"].astype(float)


def hz(c):
    return float(c.mean() * 1000 / S.T_SAMPLE)


def scan(z, model, w0, ev, n, nin):
    ws, rs = [], []
    for w in w0 * np.sqrt(2.0) ** np.arange(-8, 13):
        r = hz(counts(z, model, w, ev, n, nin))
        ws.append(float(w)); rs.append(r)
        if r > RUNAWAY:
            break
    return ws, rs


def w_at(ws, rs, R_):
    """first upward crossing of R_ (log-log interpolation); None if never reached"""
    for i in range(1, len(ws)):
        if rs[i - 1] < R_ <= rs[i]:
            if rs[i - 1] <= 0:
                return ws[i]
            f = (np.log(R_) - np.log(rs[i - 1])) / (np.log(rs[i]) - np.log(rs[i - 1]))
            return float(np.exp(np.log(ws[i - 1]) + f * (np.log(ws[i]) - np.log(ws[i - 1]))))
    return None


def r_at(ws, rs, w):
    lw, lr = np.log(ws), np.log(np.maximum(rs, 1e-3))
    return float(np.exp(np.interp(np.log(w), lw, lr)))


def psc_threshold(model, tau, K=384):
    """smallest single-PSC charge (circuit weight units) that evokes a spike from rest"""
    S.TAU_U = tau
    b, _ = R.binding(model, K, DT)
    lo, hi = 1e-3, 1e5
    for _ in range(3):                       # three geometric grid passes
        q = np.geomspace(lo, hi, K)
        circ = LS.Circuit(N=K, nin=1, types=np.ones(K, np.int8), src=np.full(K, K, np.int32),
                          post=np.arange(K, dtype=np.int32), w=q / tau, dly=np.ones(K, np.int32),
                          rec=np.zeros(K, np.int32), U=np.ones(K), D=np.ones(K), F=np.ones(K),
                          stp=np.zeros(K, np.int32), meta={})
        ev = (np.array([0, 1], np.int64), np.array([int(round(20 / DT))], np.int64), np.zeros(1, np.int32))
        o = engine_gpu.simulate(LS.stack([circ], [b], CI.receptors()), np.zeros(1, np.int32),
                                np.array([int(round((20 + 10 * tau + 100) / DT))], np.int64),
                                R.y0_for(model, 1, K), ev, DT, Rec())
        fired = o["counts"][0] > 0
        if not fired.any():
            return float("nan")
        k = int(np.argmax(fired))
        if k == 0:
            return float(q[0])
        lo, hi = q[k - 1], q[k]
    return float(np.sqrt(lo * hi))


def xcorr(z, tau, n, data, nin, maxlag=50.0, bw=1.0):
    S.TAU_U = tau
    w = json.load(open(os.path.join(HERE, "data", "campaign_mnist", "best_w.json")))[f"{z}@fastin"]
    idx = 60000 + np.arange(n)
    circ = CI.build(nin, w, DT, np.random.default_rng(100), pulse=R.PULSE_U, input_neurons=False)
    oc = D.open_loop(circ)
    bt, _ = R.binding(z, S.N_LIQ, DT)
    bs, _ = R.binding(f"sur:{z}", S.N_LIQ, DT)
    ev = R.input_layer(f"{z}@fastin", z, nin, data, idx, "mnist", DT)
    sim = lambda c, b, m, e: engine_gpu.simulate(LS.stack([c], [b], CI.receptors()), np.zeros(n, np.int32),
                                                 np.full(n, NS, np.int64), R.y0_for(m, n, S.N_LIQ), e, DT,
                                                 Rec(spikes=tuple(range(n)), rmax=160))["spikes"]
    tc = sim(circ, bt, z, ev)
    so = sim(oc, bs, f"sur:{z}", D.replay_events(ev, tc, nin))
    edges = np.arange(-maxlag, maxlag + bw, bw)
    H = np.zeros(len(edges) - 1)
    chance = 0.0
    for k in range(n):
        for j in range(S.N_LIQ):
            a = tc[1][k, j, :tc[2][k, j]] * DT
            b = so[1][k, j, :so[2][k, j]] * DT
            if len(a) and len(b):
                H += np.histogram((b[None] - a[:, None]).ravel(), edges)[0]
                chance += len(a) * len(b) * bw / S.T_SAMPLE
    lag = 0.5 * (edges[1:] + edges[:-1])
    return dict(lag=lag.tolist(), ratio=(H / chance).tolist() if chance else [], peak_lag=float(lag[np.argmax(H)]) if chance else None)


def main_run(models, taus, n):
    os.makedirs(OUT, exist_ok=True)
    data, _, nin = R.load("mnist")
    bw = json.load(open(os.path.join(HERE, "data", "campaign_mnist", "best_w.json")))
    idx = 60000 + np.arange(n)
    for z in models:
        f = os.path.join(OUT, f"{z}.json")
        res = json.load(open(f)) if os.path.exists(f) else {}
        ev = R.input_layer(f"{z}@fastin", z, nin, data, idx, "mnist", DT)
        w0 = bw[f"{z}@fastin"]
        for tau in taus:
            key = f"{tau:g}"
            if key in res:
                continue
            S.TAU_U = tau
            wt, rt = scan(z, z, w0, ev, n, nin)
            wsur, rsur = scan(z, f"sur:{z}", w0, ev, n, nin)
            match = []
            for Rs in RSTAR:
                a, b = w_at(wt, rt, Rs), w_at(wsur, rsur, Rs)
                if a is None or b is None:
                    continue
                S.TAU_U = tau
                ct, cs, cs_same = counts(z, z, a, ev, n, nin), counts(z, f"sur:{z}", b, ev, n, nin), counts(z, f"sur:{z}", a, ev, n, nin)
                match.append(dict(R=Rs, w_t=a, w_s=b, g=b / a, rho_same=hz(cs_same) / max(hz(ct), 1e-9),
                                  rate_t=hz(ct), rate_s=hz(cs), rate_s_same=hz(cs_same),
                                  count_r=float(np.nanmedian(AG.count_corr(ct, cs))), cka=AG.cka(ct, cs),
                                  count_r_same=float(np.nanmedian(AG.count_corr(ct, cs_same))), cka_same=AG.cka(ct, cs_same)))
            qt, qs = psc_threshold(z, tau), psc_threshold(f"sur:{z}", tau)
            res[key] = dict(tau_u=tau, scan_t=[wt, rt], scan_s=[wsur, rsur], match=match, psc_t=qt, psc_s=qs, psc_ratio=qs / qt)
            print(f"{z:14s} tau_u {tau:<4g} psc q_t {qt:9.3f} q_s {qs:9.3f} ratio {qs / qt:7.3f} | " + "  ".join(
                f"R {m['R']:g}: g {m['g']:.3f} rho {m['rho_same']:.3f} r {m['count_r']:.2f} (same w {m['count_r_same']:.2f})"
                for m in match), flush=True)
            json.dump(res, open(f, "w"), indent=1)
        S.TAU_U = 1.0
    xf = os.path.join(OUT, "xcorr.json")
    xc = json.load(open(xf)) if os.path.exists(xf) else {}
    for z in models:
        for tau in (1.0, 10.0):
            key = f"{z}_{tau:g}"
            if key not in xc:
                xc[key] = xcorr(z, tau, min(n, 100), data, nin)
                print(f"xcorr {z:14s} tau_u {tau:g}: peak lag {xc[key]['peak_lag']:+.1f} ms, peak/chance {max(xc[key]['ratio']):.2f}", flush=True)
                json.dump(xc, open(xf, "w"), indent=1)
    S.TAU_U = 1.0


def plot(models, taus):
    import figs as F
    import matplotlib.pyplot as plt
    L = {z: json.load(open(os.path.join(OUT, f"{z}.json"))) for z in models if os.path.exists(os.path.join(OUT, f"{z}.json"))}
    xc = json.load(open(os.path.join(OUT, "xcorr.json"))) if os.path.exists(os.path.join(OUT, "xcorr.json")) else {}
    fig = plt.figure(figsize=(7.1, 6.0))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.4)
    cmap = plt.get_cmap("viridis")
    tcol = {t: cmap(i / max(len(taus) - 1, 1)) for i, t in enumerate(taus)}
    for c, z in enumerate(models):                                   # a-c: rate vs w
        a = fig.add_subplot(gs[0, c])
        for t in taus:
            r = L.get(z, {}).get(f"{t:g}")
            if r is None:
                continue
            a.plot(*r["scan_t"], "-", color=tcol[t], lw=1.0, label=f"{t:g}")
            a.plot(*r["scan_s"], "--", color=tcol[t], lw=0.9)
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(0.1, 400)
        a.set_title(f"{F.ZLAB[z]} (— target, - - surr.)", fontsize=7.5); a.set_xlabel("$w$ (physical)")
        if c == 0:
            a.set_ylabel("liquid rate (Hz)"); a.legend(fontsize=5, frameon=False, title="$\\tau_u$ (ms)", title_fontsize=5.5)
    a = fig.add_subplot(gs[1, 0])                                    # d: g vs tau, network vs psc prediction
    for z in models:
        ts = [t for t in taus if f"{t:g}" in L.get(z, {})]
        for Rs, mk in zip(RSTAR, "v^os"):
            pts = [(t, m["g"]) for t in ts for m in L[z][f"{t:g}"]["match"] if m["R"] == Rs]
            if pts:
                a.plot(*zip(*pts), marker=mk, ms=2.5, lw=0.6, color=F.ZCOL[z], alpha=0.8)
        a.plot(ts, [L[z][f"{t:g}"]["psc_ratio"] for t in ts], ls="--", lw=1.4, color=F.ZCOL[z])
    a.set_xscale("log"); a.set_yscale("log"); a.axhline(1, color="0.7", lw=0.4)
    a.set_xticks(taus, [f"{t:g}" for t in taus]); a.minorticks_off()
    a.set_xlabel("$\\tau_u$ (ms)"); a.set_ylabel("gain $g = w_s / w_t$")
    a.set_title("network $g$ (markers: $R^*$ 5/10/20/40 Hz)\n- - single-PSC $q_s/q_t$", fontsize=6.5)
    a = fig.add_subplot(gs[1, 1])                                    # e: g vs rho at same w
    for z in models:
        for t in taus:
            for m in L.get(z, {}).get(f"{t:g}", {}).get("match", []):
                a.plot(m["rho_same"], m["g"], "o", ms=3, color=F.ZCOL[z], mfc=tcol[t], mew=0.8)
    a.set_xscale("log"); a.set_yscale("log"); a.axhline(1, color="0.7", lw=0.4); a.axvline(1, color="0.7", lw=0.4)
    a.set_xlabel("rate ratio at same $w$, $\\rho = r_s / r_t$"); a.set_ylabel("gain $g$")
    a.set_title("edge: model, fill: $\\tau_u$", fontsize=6.5)
    a = fig.add_subplot(gs[1, 2])                                    # f: g vs psc ratio
    allp = []
    for z in models:
        for t in taus:
            r = L.get(z, {}).get(f"{t:g}")
            if r is None:
                continue
            for m in r["match"]:
                a.plot(r["psc_ratio"], m["g"], "o", ms=3, color=F.ZCOL[z], mfc=tcol[t], mew=0.8)
                allp.append((r["psc_ratio"], m["g"]))
    if allp:
        lo, hi = min(min(p) for p in allp) * 0.7, max(max(p) for p in allp) * 1.4
        a.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls=":")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("single-PSC charge ratio $q_s / q_t$"); a.set_ylabel("gain $g$"); a.set_title("prediction without a network", fontsize=6.5)
    a = fig.add_subplot(gs[2, 0:2])                                  # g: agreement same w vs matched
    xt, lab = 0, []
    for z in models:
        for t in taus:
            ms_ = L.get(z, {}).get(f"{t:g}", {}).get("match", [])
            m = next((m for m in ms_ if m["R"] == 20.0), ms_[-1] if ms_ else None)
            if m is None:
                continue
            a.bar(xt - 0.2, m["count_r_same"], 0.38, color=F.ZCOL[z], alpha=0.35)
            a.bar(xt + 0.2, m["count_r"], 0.38, color=F.ZCOL[z])
            lab.append(f"{F.ZLAB[z]}\n{t:g}"); xt += 1
    a.set_xticks(range(len(lab)), lab, fontsize=5); a.set_ylim(0, 1.02)
    a.set_ylabel("count $r$"); a.set_title("pale: surrogate at the same $w$; solid: at the rate-matched $w_s$ ($R^*$ = 20 Hz, else highest)", fontsize=6.5)
    a = fig.add_subplot(gs[2, 2])                                    # h: xcorr
    for key, v in xc.items():
        z, t = key.rsplit("_", 1)
        if v["ratio"]:
            a.plot(v["lag"], v["ratio"], color=F.ZCOL.get(z, "k"), lw=0.9, ls="-" if t == "1" else "--",
                   label=f"{F.ZLAB.get(z, z)} {t} ms")
    a.axhline(1, color="0.7", lw=0.4); a.axvline(0, color="0.7", lw=0.4)
    a.set_xlabel("lag $t_s - t_t$ (ms)"); a.set_ylabel("coincidences / chance"); a.set_title("cross-correlogram, open loop", fontsize=6.5)
    a.legend(fontsize=4.5, frameon=False)
    for i, a in enumerate(fig.axes):
        a.text(-0.22, 1.12, "abcdefgh"[i], transform=a.transAxes, fontweight="bold", fontsize=9)
    os.makedirs(os.path.join(HERE, "figs_diag"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figs_diag", "gain.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "figs_diag", "gain.png"), dpi=160, bbox_inches="tight")
    print("wrote figs_diag/gain.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--taus", default="1,2,5,10,20")
    ap.add_argument("--models", default=",".join(D.ZOO))
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    models, taus = a.models.split(","), [float(x) for x in a.taus.split(",")]
    if not a.plot:
        main_run(models, taus, a.n)
    plot(models, taus)


if __name__ == "__main__":
    main()
