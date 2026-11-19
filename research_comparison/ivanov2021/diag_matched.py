"""
diag_matched.py -- rate-matched static LSM, target vs surrogate (diag_rates.py setup: identical
input spikes, weights as physical current, surrogate through 'sur:'), MNIST test digits.

Per (model, tau_u):
  scan     closed-loop liquid rate vs w (target, surrogate; one batched GPU call each, n_scan
           digits) -> operating rate R* (20 Hz, else 10, else 5: the highest both reach) and
           the network gain g = w_s / w_t at R*
  mf       network-free prediction of g ("fluctuation gauge"): each liquid neuron alone, its
           recurrent in-edges driven by independent Poisson trains at rate R* (the operating
           rate, self-consistency), its input edges by the real input layer; w solving
           <r_out>(w) = R* for target and surrogate -> g_mf
  matched  n digits, closed loop: target at w_t; surrogate at w_t (same w), at w_t g (rate-
           matched), at w_t g_mf (predicted). CKA, per-neuron count r, rate r over neurons,
           Gamma (2 ms) raw and after removing the cross-correlogram lag; wall per sample
  xcorr    target-vs-surrogate spike cross-correlogram (1 ms bins, FFT) -> lag
Figure figs_diag/ivanov_matched.pdf: test digit 0 rasters at tau_u = 1 ms (paper), rate
matching, rates, agreement, g vs tau_u (network vs prediction), agreement vs tau_u, lag, compute.

    $PY diag_matched.py [--n 200] [--n-scan 40] [--taus 1,2,5,10,20]
    $PY diag_matched.py --plot
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

OUT = os.path.join(HERE, "data", "diag_matched")
DT = D.DT
NS = int(round(S.T_SAMPLE / DT))
GRID = np.sqrt(2.0) ** np.arange(-8, 13)          # x the campaign w*
RSTAR = (20.0, 10.0, 5.0)
RMAX = 200
MAXLAG = 30
PEAK_MIN = 1.25         # cross-correlogram peak / chance below which no lag is reported


def tile(ev, reps):
    ptr, st, ch = ev
    L = np.tile(np.diff(ptr), reps)
    return np.concatenate([[0], np.cumsum(L)]).astype(np.int64), np.tile(st, reps), np.tile(ch, reps)


def batch(model, circs, ev, n, spikes=False):
    """one GPU call: instance (i, k) = circuit i, digit k -> counts (len(circs), n, N), spikes"""
    b, _ = R.binding(model, S.N_LIQ, DT)
    W = len(circs)
    o = engine_gpu.simulate(LS.stack(circs, [b] * W, CI.receptors()), np.repeat(np.arange(W), n).astype(np.int32),
                            np.full(W * n, NS, np.int64), R.y0_for(model, W * n, S.N_LIQ), tile(ev, W), DT,
                            Rec(spikes=tuple(range(W * n)) if spikes else (), rmax=RMAX))
    c = o["counts"].reshape(W, n, -1).astype(float)
    sp = o["spikes"] if spikes else None
    return c, sp, o["compute"]


def hz(c):
    return c.mean(axis=tuple(range(1, c.ndim))) * 1000 / S.T_SAMPLE


def crossing(ws, rs, R_):
    for i in range(1, len(ws)):
        if rs[i - 1] < R_ <= rs[i]:
            if rs[i - 1] <= 0:
                return float(ws[i])
            f = (np.log(R_) - np.log(rs[i - 1])) / (np.log(rs[i]) - np.log(rs[i - 1]))
            return float(np.exp(np.log(ws[i - 1]) + f * (np.log(ws[i]) - np.log(ws[i - 1]))))
    return None


def poisson_events(ev, nu, nin, n, seed=11):
    """input events + independent Poisson trains (rate nu) on liquid channels nin..nin+999,
    same draw for every w and model"""
    rng = np.random.default_rng(seed)
    ptr, st, ch = ev
    P, S_, C = [0], [], []
    for k in range(n):
        m = rng.poisson(nu * S.T_SAMPLE / 1000, S.N_LIQ)
        t = rng.integers(0, NS, m.sum())
        s = np.concatenate([st[ptr[k]:ptr[k + 1]], t]).astype(np.int64)
        c = np.concatenate([ch[ptr[k]:ptr[k + 1]], nin + np.arange(S.N_LIQ).repeat(m)]).astype(np.int32)
        o = np.argsort(s, kind="stable")
        S_.append(s[o]); C.append(c[o]); P.append(P[-1] + len(s))
    return np.array(P, np.int64), np.concatenate(S_), np.concatenate(C)


def binned(sp, n):
    _, spt, spn = sp
    X = np.zeros((n, S.N_LIQ, int(S.T_SAMPLE)), np.float32)
    k, i = np.nonzero(spn)
    for kk, ii in zip(k, i):
        t = (spt[kk, ii, :spn[kk, ii]] * DT).astype(int)
        np.add.at(X[kk, ii], np.minimum(t, int(S.T_SAMPLE) - 1), 1)
    return X


def xcorr(Xt, Xs):
    """sum over (digit, neuron) of the cross-correlogram t_s - t_t, / chance (finite window)"""
    T = Xt.shape[-1]
    L = 2 * T
    H = np.zeros(L)
    nn = 0.0
    for k in range(0, len(Xt), 20):
        a, b = Xt[k:k + 20].reshape(-1, T), Xs[k:k + 20].reshape(-1, T)
        Fa, Fb = np.fft.rfft(a, L), np.fft.rfft(b, L)
        H += np.fft.irfft((np.conj(Fa) * Fb).sum(0), L)
        nn += float((a.sum(1) * b.sum(1)).sum())
    lag = np.concatenate([np.arange(0, T), np.arange(-T, 0)])
    o = np.argsort(lag)
    lag, H = lag[o], H[o]
    chance = nn * (T - np.abs(lag)) / T ** 2
    m = np.abs(lag) <= MAXLAG
    ratio = np.where(chance[m] > 0, H[m] / np.maximum(chance[m], 1e-12), np.nan)
    return lag[m].astype(float), ratio


def agreement(ct, cs, spt, sps, n):
    """ct, cs (n, N) counts; spt, sps spike records -> metrics incl. lag-corrected Gamma"""
    r = dict(rate_t=float(ct.mean() * 4), rate_s=float(cs.mean() * 4), cka=AG.cka(ct, cs),
             count_r=float(np.nanmedian(AG.count_corr(ct, cs))), rate_r=float(np.nanmean(AG.rate_corr_neurons(ct, cs))))
    capped = bool((spt[2] >= RMAX).any() or (sps[2] >= RMAX).any())
    lag, ratio = xcorr(binned(spt, n), binned(sps, n))
    j = int(np.nanargmax(ratio))
    r.update(lag=float(lag[j]), xc_peak=float(ratio[j]), xc_lag=lag.tolist(), xc_ratio=ratio.tolist(), capped=capped)
    g0 = AG.gamma_table(spt[1], spt[2], sps[1], sps[2], DT, 2.0, S.T_SAMPLE)
    sh = sps[1] - int(round(lag[j] / DT))                       # surrogate spikes moved back by the lag
    g1 = AG.gamma_table(spt[1], spt[2], sh, sps[2], DT, 2.0, S.T_SAMPLE - max(lag[j], 0), t0_ms=0.0)
    r.update(gamma=float(np.nanmedian(g0)), gamma_shift=float(np.nanmedian(g1)))
    return r


def raster_dict(sp, ev, k, types):
    _, spt, spn = sp
    li = np.arange(S.N_LIQ).repeat(spn[k])
    lt = np.concatenate([spt[k, i, :spn[k, i]] for i in range(S.N_LIQ)]) * DT
    ptr, st, ch = ev
    return dict(liq_t=lt, liq_i=li, in_t=st[ptr[k]:ptr[k + 1]] * DT, in_i=ch[ptr[k]:ptr[k + 1]], types=types)


def run(z, tau, n, n_scan, data, nin):
    f = os.path.join(OUT, f"{z}_tau{tau:g}.json")
    if os.path.exists(f):
        return json.load(open(f))
    S.TAU_U = tau
    w0 = json.load(open(os.path.join(HERE, "data", "campaign_mnist", "best_w.json")))[f"{z}@fastin"]
    build = lambda w: CI.build(nin, w, DT, np.random.default_rng(100), pulse=R.PULSE_U, input_neurons=False)
    ws = w0 * GRID
    ev_s = R.input_layer(f"{z}@fastin", z, nin, data, 60000 + np.arange(n_scan), "mnist", DT)
    closed = [build(w) for w in ws]
    rt = hz(batch(z, closed, ev_s, n_scan)[0]); rs = hz(batch(f"sur:{z}", closed, ev_s, n_scan)[0])
    Rs = next((x for x in RSTAR if crossing(ws, rt, x) and crossing(ws, rs, x)), None)
    res = dict(model=z, tau_u=tau, w0=w0, scan_w=ws.tolist(), scan_t=rt.tolist(), scan_s=rs.tolist(), R=Rs, n=n, n_scan=n_scan)
    if Rs is None:
        json.dump(res, open(f, "w"), indent=1)
        return res
    wt, wsn = crossing(ws, rt, Rs), crossing(ws, rs, Rs)
    ev_p = poisson_events(ev_s, Rs, nin, n_scan)
    opened = [D.open_loop(c) for c in closed]
    mt = hz(batch(z, opened, ev_p, n_scan)[0]); ms = hz(batch(f"sur:{z}", opened, ev_p, n_scan)[0])
    wt_mf, ws_mf = crossing(ws, mt, Rs), crossing(ws, ms, Rs)
    g, g_mf = wsn / wt, (ws_mf / wt_mf if wt_mf and ws_mf else None)
    res.update(w_t=wt, w_s=wsn, g=g, mf_t=mt.tolist(), mf_s=ms.tolist(), w_t_mf=wt_mf, w_s_mf=ws_mf, g_mf=g_mf)
    ev = R.input_layer(f"{z}@fastin", z, nin, data, 60000 + np.arange(n), "mnist", DT)
    ct, spt, comp_t = batch(z, [build(wt)], ev, n, spikes=True)
    arms = dict(same=wt, matched=wt * g)
    if g_mf:
        arms["predicted"] = wt * g_mf
    res["target"] = dict(w=wt, rate=float(ct.mean() * 4), wall_ms=comp_t["wall_s"] / n * 1e3)
    rasters = {"target": raster_dict(spt, ev, 0, build(wt).types)} if tau == 1.0 else {}
    for name, w in arms.items():
        cs, sps, comp_s = batch(f"sur:{z}", [build(w)], ev, n, spikes=True)
        a = agreement(ct[0], cs[0], spt, sps, n)
        a.update(w=w, wall_ms=comp_s["wall_s"] / n * 1e3)
        res[name] = a
        if tau == 1.0:
            rasters[name] = raster_dict(sps, ev, 0, build(w).types)
        print(f"{z:14s} tau_u {tau:<4g} R* {Rs:g} g {g:.3f} g_mf {g_mf if g_mf is None else round(g_mf, 3)} | {name:9s} "
              f"w {w:8.3f}: tgt {a['rate_t']:6.1f} sur {a['rate_s']:6.1f} Hz CKA {a['cka']:.3f} count r {a['count_r']:.3f} "
              f"rate r {a['rate_r']:.3f} | lag {a['lag']:+.0f} ms peak {a['xc_peak']:.2f} G {a['gamma']:.3f} -> {a['gamma_shift']:.3f}", flush=True)
    if rasters:
        np.savez(os.path.join(OUT, f"{z}_raster.npz"), **{f"{k}__{q}": v for k, d in rasters.items() for q, v in d.items()})
    json.dump(res, open(f, "w"), indent=1)
    S.TAU_U = 1.0
    return res


def plot(models, taus):
    import figs as F
    import matplotlib.pyplot as plt
    L = {(z, t): json.load(open(os.path.join(OUT, f"{z}_tau{t:g}.json"))) for z in models for t in taus
         if os.path.exists(os.path.join(OUT, f"{z}_tau{t:g}.json"))}
    data, y, _ = R.load("mnist")
    fig = plt.figure(figsize=(7.1, 8.6))
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1.05, 1.05], hspace=0.62, wspace=0.42, top=0.955)
    ARMS = (("target", "target"), ("same", "surr., same $w$"), ("matched", "surr., rate-matched"))
    # a: digit + rasters (tau_u = 1 ms)
    ad = fig.add_subplot(gs[0, 0])
    ad.imshow(data[60000].reshape(28, 28), cmap="gray_r"); ad.set_xticks([]); ad.set_yticks([])
    ad.set_title(f"test digit 0 (label {int(y[60000])})", fontsize=7)
    F.label(ad, "a")
    for r_, z in enumerate(models):
        fz = os.path.join(OUT, f"{z}_raster.npz")
        if not os.path.exists(fz):
            continue
        Z = np.load(fz)
        for c_, (k, lab) in enumerate(ARMS):
            if f"{k}__liq_t" not in Z:
                continue
            a = fig.add_subplot(gs[r_, c_ + 1])
            F.raster(a, {q: Z[f"{k}__{q}"] for q in ("liq_t", "liq_i", "in_t", "in_i", "types")}, f"{F.ZLAB[z]} {lab}")
            if r_ < len(models) - 1:
                a.set_xticklabels([])
            else:
                a.set_xlabel("time (ms)")
    # b: rate matching at tau_u = 1 (scan curves, w_t / w_s)
    a = fig.add_subplot(gs[1:3, 0])
    for z in models:
        r = L.get((z, 1.0))
        if r is None:
            continue
        a.plot(r["scan_w"], r["scan_t"], "-", color=F.ZCOL[z], lw=1.0, label=F.ZLAB[z])
        a.plot(r["scan_w"], r["scan_s"], "--", color=F.ZCOL[z], lw=0.9)
        if r.get("w_t"):
            a.plot([r["w_t"]], [r["R"]], "o", color=F.ZCOL[z], ms=3.5)
            a.plot([r["w_s"]], [r["R"]], "o", mfc="w", color=F.ZCOL[z], ms=3.5)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(0.1, 400)
    a.set_xlabel("$w$ (physical)"); a.set_ylabel("liquid rate (Hz)")
    a.set_title("— target  - - surrogate\n● $w_t$  ○ $w_s$ at $R^*$", fontsize=6.5)
    a.legend(fontsize=5.5, frameon=False, loc="lower right"); F.label(a, "b")
    row = [fig.add_subplot(gs[3, i]) for i in range(4)] + [fig.add_subplot(gs[4, i]) for i in range(4)]
    # c: rates (tau 1)
    a = row[0]
    x = 0
    for z in models:
        r = L.get((z, 1.0))
        if r is None or "same" not in r:
            continue
        vals = [r["same"]["rate_t"], r["same"]["rate_s"], r["matched"]["rate_s"]] + ([r["predicted"]["rate_s"]] if "predicted" in r else [])
        for j, v in enumerate(vals):
            a.bar(x + j * 0.22, v, 0.2, color=F.ZCOL[z], alpha=(1.0, 0.35, 0.7, 0.5)[j], hatch=(None, None, None, "///")[j], edgecolor="w", lw=0)
        x += 1
    a.set_xticks(np.arange(x) + 0.33, [F.ZLAB[z] for z in models if (z, 1.0) in L and "same" in L[(z, 1.0)]], fontsize=6.5)
    a.set_ylabel("liquid rate (Hz)"); a.set_title("tgt | same $w$ | matched | predicted", fontsize=6); F.label(a, "c")
    # d: agreement tau 1, same vs matched
    a = row[1]
    keys = (("cka", "CKA"), ("count_r", "count $r$"), ("rate_r", "rate $r$"), ("gamma_shift", "$\\Gamma$ lag-corr."))
    for j, z in enumerate(models):
        r = L.get((z, 1.0))
        if r is None or "same" not in r:
            continue
        xs = np.arange(len(keys)) + (j - 1) * 0.27
        a.bar(xs - 0.065, [r["same"][k] for k, _ in keys], 0.12, color=F.ZCOL[z], alpha=0.35)
        a.bar(xs + 0.065, [r["matched"][k] for k, _ in keys], 0.12, color=F.ZCOL[z])
    a.set_xticks(range(len(keys)), [l for _, l in keys], fontsize=5.5, rotation=20); a.set_ylim(-0.05, 1.02)
    a.axhline(0, color="k", lw=0.4); a.set_title("$\\tau_u$ = 1 ms; pale: same $w$", fontsize=6.5); F.label(a, "d")
    # e: g vs tau, network vs mean-field prediction
    a = row[2]
    for z in models:
        ts = [t for t in taus if (z, t) in L and L[(z, t)].get("g")]
        a.plot(ts, [L[(z, t)]["g"] for t in ts], "-o", ms=3, color=F.ZCOL[z], label=F.ZLAB[z])
        tp = [t for t in ts if L[(z, t)].get("g_mf")]
        a.plot(tp, [L[(z, t)]["g_mf"] for t in tp], "--o", ms=3, mfc="w", color=F.ZCOL[z])
    a.set_xscale("log"); a.set_yscale("log"); a.axhline(1, color="0.7", lw=0.4)
    a.set_xticks(taus, [f"{t:g}" for t in taus]); a.minorticks_off(); a.set_xlabel("$\\tau_u$ (ms)"); a.set_ylabel("gain $g = w_s/w_t$")
    a.legend(fontsize=5, frameon=False, loc="upper right"); a.set_title("— network  - - single-neuron prediction", fontsize=6); F.label(a, "e")
    # f: predicted vs network g (scatter)
    a = row[3]
    P = [(L[k]["g"], L[k]["g_mf"], k[0]) for k in L if L[k].get("g") and L[k].get("g_mf")]
    for g, gm, z in P:
        a.plot(g, gm, "o", ms=3.5, color=F.ZCOL[z])
    if P:
        lo, hi = min(min(p[:2]) for p in P) * 0.8, max(max(p[:2]) for p in P) * 1.25
        a.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls=":")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("network $g$"); a.set_ylabel("predicted $g$")
    a.set_xticks([1, 2, 3], ["1", "2", "3"]); a.set_yticks([1, 2, 3], ["1", "2", "3"]); a.minorticks_off()
    a.set_title("all models and $\\tau_u$", fontsize=6.5); F.label(a, "f")
    # g: count r vs tau (same / matched / predicted)
    a = row[4]
    for z in models:
        ts = [t for t in taus if (z, t) in L and "same" in L[(z, t)]]
        for k, ls, mk in (("same", ":", "s"), ("matched", "-", "o"), ("predicted", "--", "^")):
            tk = [t for t in ts if k in L[(z, t)]]
            a.plot(tk, [L[(z, t)][k]["count_r"] for t in tk], ls=ls, marker=mk, ms=2.8, color=F.ZCOL[z], lw=0.9)
    for k, ls, mk in (("same $w$", ":", "s"), ("rate-matched", "-", "o"), ("predicted", "--", "^")):
        a.plot([], [], ls=ls, marker=mk, ms=2.8, color="k", label=k)
    a.set_xscale("log"); a.set_xticks(taus, [f"{t:g}" for t in taus]); a.minorticks_off(); a.set_ylim(0, 1.02)
    a.set_xlabel("$\\tau_u$ (ms)"); a.set_ylabel("count $r$"); a.legend(fontsize=5, frameon=False); F.label(a, "g")
    # h: cross-correlogram (tau 1, matched)
    a = row[5]
    for z in models:
        r = L.get((z, 1.0))
        if r is None or "matched" not in r:
            continue
        a.plot(r["matched"]["xc_lag"], r["matched"]["xc_ratio"], color=F.ZCOL[z], lw=0.9,
               label=f"{F.ZLAB[z]} lag {r['matched']['lag']:+.0f} ms" if r["matched"]["xc_peak"] >= PEAK_MIN else f"{F.ZLAB[z]} no peak")
    a.axhline(1, color="0.7", lw=0.4); a.axvline(0, color="0.7", lw=0.4)
    a.set_xlabel("lag $t_s - t_t$ (ms)"); a.set_ylabel("coincid. / chance"); a.set_title("$\\tau_u$ = 1 ms, rate-matched", fontsize=6.5)
    a.legend(fontsize=5, frameon=False); F.label(a, "h")
    # i: lag vs tau
    a = row[6]
    for z in models:
        ts = [t for t in taus if (z, t) in L and "matched" in L[(z, t)] and L[(z, t)]["matched"]["xc_peak"] >= PEAK_MIN]
        a.plot(ts, [L[(z, t)]["matched"]["lag"] for t in ts], "-o", ms=3, color=F.ZCOL[z], label=F.ZLAB[z])
    a.set_title(f"rate-matched; peak < {PEAK_MIN:g} x chance omitted", fontsize=6)
    a.set_xscale("log"); a.set_xticks(taus, [f"{t:g}" for t in taus]); a.minorticks_off()
    a.set_xlabel("$\\tau_u$ (ms)"); a.set_ylabel("surrogate lag (ms)"); a.axhline(0, color="0.7", lw=0.4); F.label(a, "i")
    # j: wall per sample (tau 1)
    a = row[7]
    for j, z in enumerate(models):
        r = L.get((z, 1.0))
        if r is None or "matched" not in r:
            continue
        a.bar(j - 0.18, r["target"]["wall_ms"], 0.34, color=F.ZCOL[z])
        a.bar(j + 0.18, r["matched"]["wall_ms"], 0.34, color=F.ZCOL[z], alpha=0.45, hatch="///", edgecolor="w", lw=0)
        a.text(j + 0.18, r["matched"]["wall_ms"], f"{r['target']['wall_ms'] / r['matched']['wall_ms']:.1f}x",
               ha="center", va="bottom", fontsize=5.5)
    a.set_xticks(range(len(models)), [F.ZLAB[z] for z in models], fontsize=6.5)
    a.set_ylabel("GPU wall / sample (ms)"); a.set_title("solid target, hatched surrogate", fontsize=6); F.label(a, "j")
    fig.suptitle("static LSM, identical input, physical weights: target vs surrogate at the same and at the rate-matched weight "
                 f"({L[next(iter(L))]['n']} test digits)", fontsize=7.5, y=0.995)
    os.makedirs(os.path.join(HERE, "figs_diag"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figs_diag", "ivanov_matched.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "figs_diag", "ivanov_matched.png"), dpi=160, bbox_inches="tight")
    print("wrote figs_diag/ivanov_matched.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-scan", type=int, default=40)
    ap.add_argument("--taus", default="1,2,5,10,20")
    ap.add_argument("--models", default=",".join(D.ZOO))
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    global OUT
    OUT = a.out or OUT
    models, taus = a.models.split(","), [float(x) for x in a.taus.split(",")]
    if not a.plot:
        os.makedirs(OUT, exist_ok=True)
        data, _, nin = R.load("mnist")
        for z in models:
            for t in taus:
                run(z, t, a.n, a.n_scan, data, nin)
    plot(models, taus)


if __name__ == "__main__":
    main()
