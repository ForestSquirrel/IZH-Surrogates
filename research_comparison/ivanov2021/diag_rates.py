"""
diag_rates.py -- why do surrogate and target liquid rates differ? (static LSM, MNIST test set)

Both liquids get IDENTICAL input spike trains (the target's '@fastin' input layer, thinned to a
fraction f of its events: rate hypothesis) at the SAME physical synaptic current (the target's
w*, surrogate through the drop-in 'sur:' binding = target current units through the gauge).
For each f:
  closed  the normal recurrent liquid, target vs surrogate
  open    every liquid neuron receives the presynaptic spikes the TARGET network produced
          (input + recurrent, recorded): single-neuron response to the identical network drive,
          no recurrent amplification. The target's own open-loop run must reproduce its
          closed-loop spikes (sanity).
Metrics: liquid rate, CKA, median per-neuron count r, mean rate r over neurons, median Gamma
(2 ms) per neuron.

    $PY diag_rates.py [--n 500] [--f 1,0.5,0.25,0.125] [--out data/diag_rates]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import spec as S
import circuit as CI
import run as R
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec
from lsmkit.metrics import agreement as AG

ZOO = ("hh", "ml_c1", "adex_adapting")
DT = 0.1
RMAX = 512


def thin(ev, f, seed=7):
    """keep each input event with probability f (same draw for both models)"""
    ptr, st, ch = ev
    keep = np.random.default_rng(seed).random(len(st)) < f
    k = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    lens = np.bincount(k[keep], minlength=len(ptr) - 1)
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), st[keep], ch[keep]


def sim(net, n, N, y0, ev, spikes=True):
    o = engine_gpu.simulate(net, np.zeros(n, np.int32), np.full(n, int(round(S.T_SAMPLE / DT)), np.int64),
                            y0, ev, DT, Rec(spikes=tuple(range(n)) if spikes else (), rmax=RMAX))
    if spikes:
        assert (o["spikes"][2] < RMAX).all(), "spike buffer capped"
    return o


def open_loop(circ):
    """the ext circuit with every recurrent edge pre j re-sourced to channel nin + j"""
    src = np.where(circ.src < circ.N, circ.N + circ.nin + circ.src, circ.src).astype(np.int32)
    return LS.Circuit(N=circ.N, nin=circ.nin + circ.N, types=circ.types, src=src, post=circ.post, w=circ.w,
                      dly=circ.dly, rec=circ.rec, U=circ.U, D=circ.D, F=circ.F, stp=circ.stp,
                      meta=dict(circ.meta, open_loop=True))


def replay_events(ev, sp, nin):
    """input events + recorded liquid spikes (spike at step n -> event at step n on channel
    nin + j, as the input layer's spikes) -> CSR sorted by step per instance"""
    ptr, st, ch = ev
    _, spt, spn = sp
    P, S_, C = [0], [], []
    for k in range(len(ptr) - 1):
        j = np.arange(spn.shape[1]).repeat(spn[k])
        t = np.concatenate([spt[k, i, :spn[k, i]] for i in range(spn.shape[1])])
        s = np.concatenate([st[ptr[k]:ptr[k + 1]], t]).astype(np.int64)
        c = np.concatenate([ch[ptr[k]:ptr[k + 1]], nin + j]).astype(np.int32)
        o = np.argsort(s, kind="stable")
        S_.append(s[o]); C.append(c[o]); P.append(P[-1] + len(s))
    return np.array(P, np.int64), np.concatenate(S_), np.concatenate(C)


def metrics(ot, os_, dt=DT):
    ct, cs = ot["counts"].astype(float), os_["counts"].astype(float)
    g = AG.gamma_table(ot["spikes"][1], ot["spikes"][2], os_["spikes"][1], os_["spikes"][2], dt, 2.0, S.T_SAMPLE)
    hz = 1000 / S.T_SAMPLE
    return dict(rate_t=float(ct.mean() * hz), rate_s=float(cs.mean() * hz), cka=AG.cka(ct, cs),
                count_r=float(np.nanmedian(AG.count_corr(ct, cs))), rate_r=float(np.nanmean(AG.rate_corr_neurons(ct, cs))),
                gamma=float(np.nanmedian(g)), silent_t=float((ct.sum(0) == 0).mean()), silent_s=float((cs.sum(0) == 0).mean()))


def run_model(z, n, fs, data, nin, out):
    w = json.load(open(os.path.join(HERE, "data", "campaign_mnist", "best_w.json")))[f"{z}@fastin"]
    idx = 60000 + np.arange(n)
    circ = CI.build(nin, w, DT, np.random.default_rng(100), pulse=R.PULSE_U, input_neurons=False)
    oc = open_loop(circ)
    bt, _ = R.binding(z, S.N_LIQ, DT)
    bs, _ = R.binding(f"sur:{z}", S.N_LIQ, DT)
    nets = {m: LS.stack([c], [b], CI.receptors()) for m, (c, b) in
            dict(tc=(circ, bt), sc=(circ, bs), to=(oc, bt), so=(oc, bs)).items()}
    yt, ys = R.y0_for(z, n, S.N_LIQ), R.y0_for(f"sur:{z}", n, S.N_LIQ)
    ev0 = R.input_layer(f"{z}@fastin", z, nin, data, idx, "mnist", DT)
    rows = []
    for f in fs:
        ev = thin(ev0, f)
        in_hz = len(ev[1]) / n / nin / (S.T_SAMPLE / 1000)
        tc = sim(nets["tc"], n, S.N_LIQ, yt, ev)
        sc = sim(nets["sc"], n, S.N_LIQ, ys, ev)
        rev = replay_events(ev, tc["spikes"], nin)
        to = sim(nets["to"], n, S.N_LIQ, yt, rev)
        so = sim(nets["so"], n, S.N_LIQ, ys, rev)
        sanity = float((to["counts"] == tc["counts"]).mean())
        r = dict(model=z, w=w, f=f, input_hz=in_hz, closed=metrics(tc, sc), open=metrics(tc, so), sanity_open_eq_closed=sanity)
        rows.append(r)
        c, o = r["closed"], r["open"]
        print(f"{z:14s} f {f:<6g} input {in_hz:6.1f} Hz | closed: tgt {c['rate_t']:6.1f} sur {c['rate_s']:7.1f} Hz  CKA {c['cka']:.3f} "
              f"count r {c['count_r']:.3f} rate r {c['rate_r']:.3f} G {c['gamma']:.3f} | open: sur {o['rate_s']:7.1f} Hz  CKA {o['cka']:.3f} "
              f"count r {o['count_r']:.3f} rate r {o['rate_r']:.3f} G {o['gamma']:.3f} | target open==closed {sanity:.4f}", flush=True)
        json.dump(rows, open(os.path.join(out, f"{z}.json"), "w"), indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--f", default="1,0.5,0.25,0.125")
    ap.add_argument("--models", default=",".join(ZOO))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "diag_rates"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data, _, nin = R.load("mnist")
    for z in a.models.split(","):
        run_model(z, a.n, [float(x) for x in a.f.split(",")], data, nin, a.out)


if __name__ == "__main__":
    main()
