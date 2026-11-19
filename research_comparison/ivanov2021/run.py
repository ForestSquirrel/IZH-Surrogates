"""
run.py -- Ivanov 2021 baseline LSM on MNIST / N-MNIST with a swappable neuron model (GPU).

For each network seed and liquid weight w: build the circuit (input neurons + 1000 liquid
neurons, all of the chosen model), encode every sample, simulate 250 ms per sample, take the
liquid spike counts, train the readout (train / val / test = 50k / 10k / 10k) and report the
test accuracy at the best validation epoch.

    PY=python
    $PY run.py --dataset mnist --model lif --bench
    $PY run.py --dataset mnist --model lif --w 0.4,0.6,0.8,1.0,1.2 --nets 1 --subset 10000,2000,2000 --tag sweep
    $PY run.py --dataset mnist --model lif --w 0.8 --nets 10 --tag full
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import spec as S
import circuit as CI
import encode as EN
import readout as RO
from lsmkit import spec as LS, datasets as DS, engine_gpu
from lsmkit.engine_cpu import Rec
from lsmkit.neurons import lif_euler, zoo, izh, MODELS


def load(dataset):
    """-> (samples, labels) for train(60k) + test(10k): samples are images or event tuples"""
    if dataset == "mnist":
        Xtr, ytr = DS.mnist("train"); Xte, yte = DS.mnist("test")
        return np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte]), 784
    p1, t1, c1, y1 = DS.nmnist("train"); p2, t2, c2, y2 = DS.nmnist("test")
    ev = [(t1[p1[i]:p1[i + 1]], c1[p1[i]:p1[i + 1]]) for i in range(len(y1))] + \
         [(t2[p2[i]:p2[i + 1]], c2[p2[i]:p2[i + 1]]) for i in range(len(y2))]
    return ev, np.concatenate([y1, y2]), 2312


def split_idx(subset):
    """indices into the concatenated train+test arrays: train = first n_tr of 0..49999,
    val = first n_va of 50000..59999, test = first n_te of the official test set"""
    ntr, nva, nte = subset
    return np.arange(ntr), 50000 + np.arange(nva), 60000 + np.arange(nte)


CAL = None


def calib():
    global CAL
    if CAL is None:
        CAL = json.load(open(os.path.join(HERE, "data", "calib.json")))
    return CAL


def binding(model, N, dt):
    """model: 'lif' | zoo key ('hh', 'ml_c1', 'adex_adapting') | 'sur:<key>' (drop-in:
    the target's currents through the gauge) | 'sur*:<key>' (self-calibrated surrogate:
    its own 1 ms-pulse threshold as the current unit). -> (Binding, input pulse amplitude u)"""
    f = np.float32
    if model == "lif":
        return LS.Binding(model="lif_euler", P=lif_euler.params(N, dt, S.TAU_V, S.THETA, S.T_REF),
                          tref=np.zeros(N, np.int32), clamp=np.zeros(N, np.int32),
                          bias=np.full(N, S.BIAS, f), iscale=np.ones(N, f), va=np.ones(N, f),
                          vb=np.zeros(N, f), dim=2, meta=dict(model="LIF (Ivanov, Euler 1 ms)")), S.I_PULSE
    kind, _, key = model.rpartition(":")
    c = calib()[key]
    if not kind:
        sp = zoo.SPECS[key]
        return LS.Binding(model=sp["model"], P=zoo.params(key, N, dt), tref=np.zeros(N, np.int32),
                          clamp=np.zeros(N, np.int32), bias=np.zeros(N, f),
                          iscale=np.full(N, c["iscale"], f), va=np.ones(N, f), vb=np.zeros(N, f),
                          dim=MODELS[sp["model"]].DIM, meta=dict(model=key, unit_thr=c["thr"])), PULSE_U
    g = c["gauge"]
    unit = c["thr"] if kind == "sur" else c["sur_thr_target_units"]
    return LS.Binding(model="izh", P=izh.params(N, g["theta"], dt, g["tau"]), tref=np.zeros(N, np.int32),
                      clamp=np.zeros(N, np.int32), bias=np.zeros(N, f),
                      iscale=np.full(N, unit / 20.0 / g["k_I"], f), va=np.ones(N, f), vb=np.zeros(N, f),
                      dim=2, meta=dict(model=model, gauge=g, unit_target_units=unit)), PULSE_U


def y0_for(model, n, N):
    if model == "lif":
        return np.zeros((n, N, 2), np.float32)
    kind, _, key = model.rpartition(":")
    c = calib()[key]
    r = c["rest"] if not kind else c["sur_rest"]
    return np.broadcast_to(np.asarray(r, np.float32), (n, N, len(r))).copy()


def fastin(key, b, nin, dt, n_input=None):
    """the first nin neurons of binding b become time-scaled input neurons of zoo model key:
    S.FASTIN_SPEED[key] x faster (P rows at dt * s); pulse A = max(2 thr_s, 2 i_rh) in model
    units -> returned in u units via the binding's iscale"""
    sp = S.FASTIN_SPEED[key]
    js = json.load(open(os.path.join(HERE, "data", "input_speed.json")))[key][str(sp)]
    c = calib()[key]
    A = max(2 * js["thr"], 2 * c["gauge"]["i_rh"])
    b.P[:nin] = zoo.params(key, nin, dt * sp)
    b.meta.update(fastin_speed=sp, fastin_pulse_model_units=A)
    return b, A / c["iscale"]


PULSE_U = 40.0          # continuous models: 2 x the 1 ms-pulse threshold (= 2 x 20 units)


ENC_SEED = 1000           # one stimulus realisation for every network and condition


def stimulus(data, i, dataset, enc_seed=ENC_SEED):
    """stimulus events of sample i: deterministic per (seed, sample), independent of batching"""
    if dataset == "mnist":
        return EN.mnist_events(data[i], np.random.default_rng([enc_seed, int(i)]))
    return EN.nmnist_events(*data[i])


CACHE_BLOCK = 1000
FAST_RMAX = 192


def input_layer(model, key, nin, data, idx, dataset, dt):
    """spike trains of the time-scaled input layer ('@fastin') for samples idx, from the disk
    cache data/input_cache/<dataset>_<model>/b<block>.npz (blocks of CACHE_BLOCK consecutive
    sample indices, simulated once with the input neurons alone: they receive no liquid input).
    -> event CSR (ev_ptr, ev_step, ev_src) for the liquid network (spike at step n -> event at
    step n -> arrival n + 1, as in the coupled network)"""
    d = os.path.join(HERE, "data", "input_cache", f"{dataset}_{model.replace('@', '_')}")
    if key.startswith("sur"):                # the surrogate layer depends on the zoo key and the fitted
        import fastin_sur                    # pulse only, not on the liquid's current unit (sur / sur*)
        zk = key.rpartition(":")[2]
        d = os.path.join(HERE, "data", "input_cache",
                         f"{dataset}_sur*:{zk}_fastin_p{fastin_sur.input_binding(zk, 1, dt)[1]:.6g}")
    os.makedirs(d, exist_ok=True)
    out = {}
    for blk in sorted(set(int(i) // CACHE_BLOCK for i in idx)):
        f = os.path.join(d, f"b{blk}.npz")
        if not os.path.exists(f):
            ids = np.arange(blk * CACHE_BLOCK, (blk + 1) * CACHE_BLOCK)
            ids = ids[ids < len(data)]
            if key.startswith("sur"):      # surrogate of the time-scaled input neuron (fastin_sur.py)
                import fastin_sur
                zk = key.rpartition(":")[2]
                b, pulse = fastin_sur.input_binding(zk, nin, dt)
                y0 = fastin_sur.y0(zk, len(ids), nin)
            else:
                b, pulse = binding(key, nin, dt)
                b, pulse = fastin(key, b, nin, dt, n_input=nin)
                y0 = y0_for(key, len(ids), nin)
            circ = CI.build_inputs_only(nin, dt, pulse)
            net = LS.stack([circ], [b], CI.receptors())
            evs = LS.events([stimulus(data, i, dataset) for i in ids], dt, nin)
            o = engine_gpu.simulate(net, np.zeros(len(ids), np.int32),
                                    np.full(len(ids), int(round(S.T_SAMPLE / dt)), np.int64),
                                    y0, evs, dt,
                                    Rec(spikes=tuple(range(len(ids))), rmax=FAST_RMAX))
            _, spt, spn = o["spikes"]
            assert (spn < FAST_RMAX).all(), "input-layer spike buffer capped; raise FAST_RMAX"
            ptr = np.concatenate([[0], np.cumsum(spn.sum(1))]).astype(np.int64)
            st = np.concatenate([spt[k][np.arange(nin).repeat(spn[k]), np.concatenate([np.arange(c) for c in spn[k]])]
                                 for k in range(len(ids))]).astype(np.int32)
            ch = np.concatenate([np.arange(nin).repeat(spn[k]) for k in range(len(ids))]).astype(np.int16)
            np.savez(f, ids=ids, ptr=ptr, step=st, ch=ch, dt=dt)
    return _csr(d, idx)


def _csr(d, idx):
    """event CSR (ev_ptr, ev_step, ev_src) for samples idx, gathered from the cache blocks
    without a per-sample Python loop (steps are stored exactly; no float round trip)"""
    idx = np.asarray(idx, np.int64)
    lens = np.zeros(len(idx), np.int64)
    parts_s, parts_c, order = [], [], []
    for blk in np.unique(idx // CACHE_BLOCK):
        z = np.load(os.path.join(d, f"b{blk}.npz"))
        sel = np.nonzero(idx // CACHE_BLOCK == blk)[0]
        loc = idx[sel] - int(z["ids"][0])
        ptr = z["ptr"]
        st, ln = ptr[loc], ptr[loc + 1] - ptr[loc]
        rag = np.repeat(st - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(ln.sum())
        parts_s.append(z["step"][rag]); parts_c.append(z["ch"][rag])
        lens[sel] = ln
        order.append(sel)
    # blocks are visited in ascending order and idx is ascending inside the campaign, so the
    # concatenation is already in idx order; assert rather than reorder
    assert np.all(np.concatenate(order) == np.arange(len(idx))), "idx must be sorted"
    ev_ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    step = np.concatenate(parts_s).astype(np.int64)
    ch = np.concatenate(parts_c).astype(np.int32)
    # the engine consumes each sample's events in time order (the cache groups them by neuron)
    o = np.lexsort((step, np.repeat(np.arange(len(idx)), lens)))
    return ev_ptr, step[o], ch[o]


def counts_for(model, w, nin, data, idx, dataset, dt, net_seed, enc_seed=ENC_SEED, chunk=2000):
    """simulate samples `idx` -> liquid spike counts (n, 1000) uint16, compute record.
    '<model>@lifin': the liquid of <model> driven by the paper's LIF input layer;
    '<model>@fastin': driven by <model>'s own input layer, time-scaled (cached)."""
    base, _, mod = model.partition("@")
    ext = mod in ("lifin", "fastin")              # input layer computed outside the network
    b, pulse = binding(base, S.N_LIQ if ext else nin + S.N_LIQ, dt)
    circ = CI.build(nin, w, dt, np.random.default_rng(net_seed), pulse=pulse, input_neurons=not ext)
    net = LS.stack([circ], [b], CI.receptors())
    ns = int(round(S.T_SAMPLE / dt))
    out, comp = [], []
    for c0 in range(0, len(idx), chunk):
        ids = idx[c0:c0 + chunk]
        if mod == "fastin":
            evs = input_layer(model, base, nin, data, ids, dataset, dt)
        else:
            ev = [stimulus(data, i, dataset, enc_seed) for i in ids]
            if mod == "lifin":
                ev = [EN.lif_input_spikes(e, nin) for e in ev]
            evs = LS.events(ev, dt, circ.nin)
        o = engine_gpu.simulate(net, np.zeros(len(ids), np.int32), np.full(len(ids), ns, np.int64),
                                y0_for(base, len(ids), circ.N), evs, dt, Rec())
        out.append(o["counts"][:, 0 if ext else circ.nin:].astype(np.uint16))
        comp.append(o["compute"])
    from lsmkit.metrics.compute import merge
    return np.concatenate(out), merge(comp), circ


def dt_for(model):
    return S.DT_LIF if model.partition("@")[0] == "lif" else 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=("mnist", "nmnist"))
    ap.add_argument("--model", default="lif")
    ap.add_argument("--w", default="0.8")
    ap.add_argument("--nets", type=int, default=1)
    ap.add_argument("--subset", default="50000,10000,10000")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    dt = a.dt or dt_for(a.model)
    data, y, nin = load(a.dataset)
    subset = tuple(int(x) for x in a.subset.split(","))
    tr, va, te = split_idx(subset)
    if a.bench:
        idx = np.arange(a.chunk)
        w0 = float(a.w.split(",")[0])
        counts_for(a.model, w0, nin, data, idx[:64], a.dataset, dt, 0, chunk=a.chunk)      # compile
        cnt, comp, circ = counts_for(a.model, w0, nin, data, idx, a.dataset, dt, 0, chunk=a.chunk)
        per = comp["wall_s"] / len(idx)
        print(f"{a.model} {a.dataset}: {comp['wall_s']:.2f} s / {len(idx)} samples ({per * 1e3:.2f} ms each) "
              f"-> {per * 70000 / 60:.1f} min per (net, w) on 70k; rate {cnt.mean() / 0.25:.1f} Hz, "
              f"silent {(cnt.sum(0) == 0).mean():.2f}; substeps {comp['model_substeps']}; edges LL {circ.meta['n_ll']}, IL {circ.meta['n_il']}")
        return
    out_dir = os.path.join(HERE, "data", "runs", f"{a.dataset}_{a.model}_{a.tag or time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    idx = np.concatenate([tr, va, te])
    for net in range(a.nets):
        for w in [float(x) for x in a.w.split(",")]:
            f = os.path.join(out_dir, f"net{net}_w{w:g}.json")
            if os.path.exists(f):
                rows.append(json.load(open(f))); continue
            t0 = time.time()
            cnt, comp, circ = counts_for(a.model, w, nin, data, idx, a.dataset, dt, 100 + net,
                                         chunk=a.chunk)
            n1, n2 = len(tr), len(tr) + len(va)
            r = RO.train(cnt[:n1], y[tr], cnt[n1:n2], y[va], cnt[n2:], y[te], seed=net)
            r.update(net=net, w=w, rate_hz=float(cnt.mean() / (S.T_SAMPLE / 1000)),
                     silent=float((cnt.sum(0) == 0).mean()), compute=comp, subset=subset,
                     dataset=a.dataset, model=a.model, dt=dt)
            json.dump(r, open(f, "w"), indent=1)
            rows.append(r)
            print(f"net {net} w {w:g}: test {r['test'] * 100:.2f}% (val {r['val'] * 100:.2f}, "
                  f"epoch {r['best_epoch']}), rate {r['rate_hz']:.1f} Hz, sim {comp['wall_s']:.0f}s, "
                  f"total {time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
