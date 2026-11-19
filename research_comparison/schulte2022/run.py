"""
run.py -- Schulte 2022 data-based circuit, spike-pattern tasks: HH target vs Izhikevich
surrogate on identical circuits and inputs (GPU).

One instance per run (circuit seed 3000 + r, input seed 4000 + r), all runs in one launch
per model; one continuous simulation of DISCARD + 1800 trials (8.1 M steps at dt 0.1 ms).
Readout states: spike trains filtered with tau 15 ms (paper) at each trial end; readouts
L23 / L5 sample presynaptic neurons like an E23 / E5 neuron would (Bernoulli with the
Fig 1 probabilities from every population; inhibitory sources enter with sign -1) and are
trained by non-negative least squares on 1500 trials, tested on 300, threshold 0.5, kappa.

    PY=python
    $PY run.py --runs 1 --trials 60 --out <scratch>       # smoke
    $PY run.py --bench                                    # ETA of --runs 20
    $PY run.py --runs 20 --tag noise                      # Table 6 conditions
    $PY run.py --runs 20 --no-noise --tag nonoise

Recording: L0 = trial-end states, spike counts, compute (always); L1 = spike times of
--rec-spikes runs from --rec-from trial on (capped at --rmax per neuron); L2 = voltage
(paper mV; surrogate through its voltage map) of --rec-v runs over --v-trials trials
from --rec-from, every --v-stride steps.
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import spec as S
import tasks
import circuit as CI
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec
from lsmkit.metrics import task as MT, agreement as AG, compute as CO

SEG = int(round(S.T_SEG / S.DT))
TRIAL = S.N_SEG * SEG


def readout_index(rng):
    """-> {readout: (neuron ids, signs)} sampled once per run (shared by all models)"""
    out = {}
    for ro in S.READOUTS:
        ids, sg = [], []
        for src, row in S.CONN.items():
            p = row.get(ro, (0, 0))[1]
            if p <= 0:
                continue
            sel = CI.IDX[src][rng.random(len(CI.IDX[src])) <= p]
            ids.append(sel)
            sg.append(np.full(len(sel), 1.0 if S.PTYPE[src] == 0 else -1.0))
        out[ro] = (np.concatenate(ids), np.concatenate(sg))
    return out


def evaluate(traj, y, ro_idx, ntr):
    """traj: (trials, N) states of one run -> {task L23/L5: kappa}"""
    res = {}
    for ro, (ids, sg) in ro_idx.items():
        X = traj[:, ids] * sg
        tag = "L23" if ro == "E23" else "L5"
        for task, t in y.items():
            if task == "XOR" or task.startswith("tcl"):
                w, _ = nnls(X[:ntr].astype(float), t[:ntr].astype(float))
                pred = (X[ntr:] @ w > S.THRESH).astype(int)
                res[f"{task} {tag}"] = MT.kappa(pred, t[ntr:], 2)
    return res


def syn_gain(view):
    """per-type synaptic gain that matches the single-EPSC firing threshold of surrogate and
    target (sd_check.py): factor = g_thr(surrogate) / g_thr(HH)"""
    js = json.load(open(os.path.join(HERE, "data", f"sd_check_{view}.json")))["cells"]
    return {t: js[f"sur-{t}"]["g_single_nS"] / js[f"HH-{t}"]["g_single_nS"] for t in "EI"}


def run_model(name, circs, inputs, V0, rec, noise, seed):
    """name: 'target', a posterior view ('best', 'mean'), '<view>@gain' (surrogate with its
    synaptic input scaled per type by syn_gain(view)), '<view>@x<fE>:<fI>' (explicit, all input
    current) or '<view>@xe<fE>:<fI>' (excitatory synaptic conductances onto E / I cells: the
    predicted gain of gain.py)"""
    if name == "target":
        b = CI.bind_target()
        Y0 = CI.y0_target(V0)
    else:
        view, _, mod = name.partition("@")
        g = CI.gauges(view)
        b = CI.bind_surrogate(g)
        if mod.startswith("xe"):
            import gain as GN
            fE, fI = map(float, mod[2:].split(":"))
            circs = [GN.scale_exc(c, fE, fI) for c in circs]
        elif mod:
            # '@gain' -> threshold-matched (syn_gain); '@x<fE>:<fI>' -> explicit factors
            f = syn_gain(view) if mod == "gain" else dict(zip("EI", map(float, mod[1:].split(":"))))
            b.iscale = (b.iscale * np.where(CI.TYPES > 0, f["E"], f["I"])).astype(np.float32)
            b.meta["syn_gain"] = f
        Y0 = CI.y0_surrogate(V0, g)
    net = LS.stack(circs, [b] * len(circs), CI.receptors(noise))
    ev = LS.events(inputs, S.DT, 2 * S.N_CH)
    ns = np.full(len(circs), int(np.ceil(T_SIM / S.DT)), np.int64)
    return engine_gpu.simulate(net, np.arange(len(circs)), ns, Y0, ev, S.DT, rec, seed=seed,
                               progress=PROGRESS)


def pop_rates(counts, T_s):
    return {k: float(counts[..., v].sum(-1).mean() / len(v) / T_s) for k, v in CI.IDX.items()}


def main():
    global T_SIM, PROGRESS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=S.N_RUNS)
    ap.add_argument("--trials", type=int, default=S.N_TRAIN + S.N_TEST)
    ap.add_argument("--models", default="target,best,mean")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--rec-spikes", type=int, default=2)
    ap.add_argument("--rmax", type=int, default=512)
    ap.add_argument("--gamma-trials", type=int, default=2, help="Gamma window (trials from --rec-from)")
    ap.add_argument("--rec-v", type=int, default=1)
    ap.add_argument("--v-trials", type=int, default=1)
    ap.add_argument("--v-stride", type=int, default=1)
    ap.add_argument("--rec-from", type=int, default=100, help="trial index for L1/L2 windows")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    PROGRESS = a.progress
    models = a.models.split(",")
    noise = not a.no_noise
    if a.bench:
        a.trials, a.runs = 10, min(a.runs, 20)
    ntr = int(round(a.trials * S.N_TRAIN / (S.N_TRAIN + S.N_TEST)))
    circs, inputs, ys, ros, V0 = [], [], [], [], []
    for r in range(a.runs):
        rng = np.random.default_rng(3000 + r)
        circs.append(CI.build(rng))
        ros.append(readout_index(rng))
        V0.append(rng.uniform(*S.V0, CI.N))
        inp, y, T_SIM = tasks.make(np.random.default_rng(4000 + r), a.trials)
        inputs.append(inp)
        ys.append(y)
    V0 = np.stack(V0)
    rec_from = min(a.rec_from, a.trials - 1)
    n0 = S.DISCARD_SEG * SEG + rec_from * TRIAL
    rec = Rec(ro_tau=S.TAU_FILTER, ro_every=TRIAL, ro_offset=S.DISCARD_SEG * SEG,
              spikes=tuple(range(min(a.rec_spikes, a.runs))), rmax=a.rmax, sp_from=n0,
              v=tuple(range(min(a.rec_v, a.runs))), v_stride=a.v_stride, v_from=n0,
              v_to=n0 + a.v_trials * TRIAL)
    if a.bench:
        for m in models:
            o = run_model(m, circs, inputs, V0, rec, noise, 0)
            c = o["compute"]
            full = c["wall_s"] * (S.N_TRAIN + S.N_TEST) / a.trials
            print(f"{m:7s}: {c['wall_s']:.1f} s for {a.runs} runs x {a.trials} trials -> "
                  f"~{full / 60:.1f} min for 1800 trials; substeps {c['model_substeps']}, "
                  f"{c['neuron_steps_per_s']:.3g} neuron-steps/s; rates "
                  + " ".join(f"{k} {v:.1f}" for k, v in pop_rates(o["counts"], T_SIM / 1000).items()))
        return
    out_dir = a.out or os.path.join(HERE, "data", "runs", a.tag or time.strftime("%Y%m%d_%H%M%S"))
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        raise SystemExit(f"{out_dir} already holds a summary; use a new --tag/--out")
    os.makedirs(out_dir, exist_ok=True)
    raw, res = {}, {}
    for i, m in enumerate(models):
        t0 = time.time()
        o = run_model(m, circs, inputs, V0, rec, noise, seed=17 + i)
        traj = o["traj"][:, :a.trials]
        per_run = [evaluate(traj[r], ys[r], ros[r], ntr) for r in range(a.runs)]
        keys = per_run[0].keys()
        res[m] = dict(kappa={k: float(np.mean([p[k] for p in per_run])) for k in keys},
                      kappa_sd={k: float(np.std([p[k] for p in per_run])) for k in keys},
                      kappa_runs=per_run, rates=pop_rates(o["counts"], T_SIM / 1000),
                      compute=o["compute"])
        raw[m] = o
        np.savez_compressed(os.path.join(out_dir, f"{m}.npz"), traj=traj, counts=o["counts"],
                            **({"sp_t": o["spikes"][1], "sp_n": o["spikes"][2]} if "spikes" in o else {}),
                            **({"v": o["v"][1]} if "v" in o else {}))
        print(f"{m}: {time.time() - t0:.0f}s  " + "  ".join(f"{k} {v:.3f}" for k, v in res[m]["kappa"].items()), flush=True)
    if "target" in raw:
        t = raw["target"]
        for m in models:
            if m == "target":
                continue
            s = raw[m]
            ag = dict(cka_states=float(np.mean([AG.cka(t["traj"][r, :a.trials], s["traj"][r, :a.trials])
                                                for r in range(a.runs)])),
                      rate_corr_neurons=float(np.nanmean(AG.rate_corr_neurons(t["counts"], s["counts"]))),
                      rate_ratio={k: res[m]["rates"][k] / max(res["target"]["rates"][k], 1e-9)
                                  for k in CI.IDX},
                      compute_ratio=CO.ratio(s["compute"], t["compute"]))
            if "spikes" in t:
                _, tt, tn = t["spikes"]
                _, st, sn = s["spikes"]
                # common window: --gamma-trials trials from the L1 start; check no buffer capped
                n1 = n0 + a.gamma_trials * TRIAL
                capped = bool(any(((n >= a.rmax) & (spt[..., -1] < n1)).any()
                                  for spt, n in ((tt, tn), (st, sn))))
                G = AG.gamma_table(tt, tn, st, sn, S.DT, 2.0, a.gamma_trials * TRIAL * S.DT,
                                   t0_ms=n0 * S.DT)
                ag["gamma_median"] = float(np.nanmedian(G)) if np.isfinite(G).any() else None
                ag["gamma_window_ms"] = [n0 * S.DT, (n0 + a.gamma_trials * TRIAL) * S.DT]
                ag["gamma_l1_capped"] = capped
            res[m]["agreement"] = ag
    summ = dict(paper="Schulte to Brinke 2022, data-based circuit, spike-pattern tasks",
                ref_kappa=S.REF, noise=noise, args=vars(a), results=res)
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({m: dict(kappa=r["kappa"], rates=r["rates"]) for m, r in res.items()}, indent=1))


T_SIM, PROGRESS = 0.0, False

if __name__ == "__main__":
    main()
