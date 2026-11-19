"""
diag_waveforms.py -- one liquid neuron, one test digit, open loop (diag_rates.py): the synaptic
current it receives (identical for target and surrogate: the target network's input and
recurrent spikes, target current units) and the membrane potentials of the target and of its
surrogate, at the paper's tau_u = 1 ms and at 10 ms.

    $PY diag_waveforms.py [--model hh] [--sample 60000] [--neuron auto] [--taus 1,10]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import spec as S
import circuit as CI
import run as R
import diag_rates as D
import figs as F                                    # fonts / rcParams
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec
import matplotlib.pyplot as plt

DT = D.DT


def one(z, tau, sample, data, nin):
    S.TAU_U = tau
    w = json.load(open(os.path.join(HERE, "data", "campaign_mnist", "best_w.json")))[f"{z}@fastin"]
    circ = CI.build(nin, w, DT, np.random.default_rng(100), pulse=R.PULSE_U, input_neurons=False)
    oc = D.open_loop(circ)
    bt, _ = R.binding(z, S.N_LIQ, DT)
    bs, _ = R.binding(f"sur:{z}", S.N_LIQ, DT)
    ev = R.input_layer(f"{z}@fastin", z, nin, data, np.array([sample]), "mnist", DT)
    ns = int(round(S.T_SAMPLE / DT))
    sim = lambda c, b, y0, e: engine_gpu.simulate(LS.stack([c], [b], CI.receptors()), np.zeros(1, np.int32),
                                                  np.array([ns]), y0, e, DT,
                                                  Rec(spikes=(0,), rmax=D.RMAX, v=(0,), v_stride=1))
    tc = sim(circ, bt, R.y0_for(z, 1, S.N_LIQ), ev)
    rev = D.replay_events(ev, tc["spikes"], nin)
    to = sim(oc, bt, R.y0_for(z, 1, S.N_LIQ), rev)
    so = sim(oc, bs, R.y0_for(f"sur:{z}", 1, S.N_LIQ), rev)
    assert (to["counts"] == tc["counts"]).all()
    return dict(oc=oc, rev=rev, to=to, so=so, w=w, iscale=float(bt.iscale[0]), ns=ns)


def current(r, j, tau):
    """synaptic current into neuron j (target units): exp kernel, jump w/tau_u at arrival"""
    oc, (ptr, st, ch) = r["oc"], r["rev"]
    m = oc.post == j
    src, w, dly = oc.src[m] - oc.N, oc.w[m], oc.dly[m]
    wmap = np.zeros(oc.nin); dmap = np.zeros(oc.nin, int)
    np.add.at(wmap, src, w); dmap[src] = dly
    drive = np.zeros(r["ns"] + 2)
    hit = wmap[ch] != 0
    np.add.at(drive, np.minimum(st[hit] + dmap[ch[hit]], r["ns"] + 1), wmap[ch[hit]])
    dec = np.exp(-DT / tau)
    I = np.zeros(r["ns"])
    g = 0.0
    for n in range(r["ns"]):
        g = g * dec + drive[n]
        I[n] = g
    return I * r["iscale"]


def spikes(o, j):
    _, spt, spn = o["spikes"]
    return spt[0, j, :spn[0, j]] * DT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hh")
    ap.add_argument("--sample", type=int, default=60000)
    ap.add_argument("--neuron", default="auto")
    ap.add_argument("--taus", default="1,10")
    ap.add_argument("--zoom", default="auto")
    ap.add_argument("--out", default=os.path.join(HERE, "figs_diag"))
    a = ap.parse_args()
    data, _, nin = R.load("mnist")
    taus = [float(x) for x in a.taus.split(",")]
    res = {t: one(a.model, t, a.sample, data, nin) for t in taus}
    if a.neuron == "auto":                  # fires under every tau, moderate count at the first
        ct = np.array([res[t]["to"]["counts"][0] for t in taus])
        ok = np.nonzero(ct[-1] >= 3)[0]
        ok = ok if len(ok) else np.nonzero((ct > 0).all(0))[0]
        j = int(ok[np.argmin(np.abs(ct[0, ok] - np.median(ct[0, ok])))])
    else:
        j = int(a.neuron)
    t = np.arange(res[taus[0]]["ns"]) * DT
    st0 = spikes(res[taus[0]]["to"], j)
    z0 = float(a.zoom) if a.zoom != "auto" else max(0.0, (st0[len(st0) // 2] if len(st0) else 100.0) - 20)
    zw = (z0, z0 + 40)
    lab = F.ZLAB.get(a.model, a.model)
    fig, ax = plt.subplots(3, 2 * len(taus), figsize=(7.1, 4.2), sharex="col",
                           gridspec_kw=dict(hspace=0.12, wspace=0.32))
    for c, tau in enumerate(taus):
        r = res[tau]
        I = current(r, j, tau)
        vt = r["to"]["v"][1][0, j]
        vs = r["so"]["v"][1][0, j]
        stt, sts = spikes(r["to"], j), spikes(r["so"], j)
        for k, (xl, ttl) in enumerate(((None, f"$\\tau_u$ = {tau:g} ms"), (zw, f"$\\tau_u$ = {tau:g} ms, zoom"))):
            col = 2 * c + k
            m = slice(None) if xl is None else slice(int(xl[0] / DT), int(xl[1] / DT))
            ax[0, col].plot(t[m], I[m], color="0.25", lw=0.5)
            ax[0, col].axhline(0, color="0.6", lw=0.4)
            ax[0, col].axhline(R.calib()[a.model]["gauge"]["i_rh"], color="r", lw=0.5, ls="--")
            ax[1, col].plot(t[m], vt[m], color=F.ZCOL[a.model], lw=0.5)
            ax[2, col].plot(t[m], vs[m], color=F.ZCOL[a.model], lw=0.5, ls="-", alpha=0.85)
            for row, s in ((1, stt), (2, sts)):
                s = s if xl is None else s[(s >= xl[0]) & (s < xl[1])]
                y = ax[row, col].get_ylim()[1]
                ax[row, col].plot(s, np.full(len(s), y), "|", color="k", ms=4, mew=0.6)
            ax[0, col].set_title(ttl, fontsize=7.5)
            if xl is not None:
                ax[2, col].set_xlim(*xl)
            ax[2, col].set_xlabel("time (ms)")
            if k == 0:
                ax[1, col].text(0.99, 0.03, f"{len(stt)} spikes", transform=ax[1, col].transAxes, ha="right", va="bottom", fontsize=6.5)
                ax[2, col].text(0.99, 0.03, f"{len(sts)} spikes", transform=ax[2, col].transAxes, ha="right", va="bottom", fontsize=6.5)
        ax[0, 2 * c].set_ylabel("$I_\\mathrm{syn}$ (target units)")
        ax[1, 2 * c].set_ylabel(f"{lab}: $V$ (mV)")
        ax[2, 2 * c].set_ylabel("surrogate: $v$")
    for i, x in enumerate(ax[0]):
        x.text(-0.18, 1.08, "abcdefgh"[i], transform=x.transAxes, fontweight="bold", fontsize=9)
    fig.suptitle(f"{lab}, liquid neuron {j}, test digit {a.sample - 60000} (label {int(R.load('mnist')[1][a.sample])}), "
                 f"open loop, w = {res[taus[0]]['w']:g}", fontsize=8, y=0.995)
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"waveforms_{a.model}_n{j}")
    fig.savefig(f + ".pdf", bbox_inches="tight"); fig.savefig(f + ".png", dpi=160, bbox_inches="tight")
    print("wrote", f + ".pdf", "neuron", j)


if __name__ == "__main__":
    main()
