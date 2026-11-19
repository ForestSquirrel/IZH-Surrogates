"""
gain.py -- synaptic gain of a surrogate liquid, predicted without simulating the network.

The gauge (tau, k_I) maps the target's response to step currents onto the surrogate. Synaptic
drive is not a step: it fluctuates on the time scale tau_u of the PSC kernel, and a target that
fires on fluctuations (HH at tau_u = 1 ms) needs less mean current for the same rate than its
mean-driven surrogate. The missing factor is the synaptic gain g: the surrogate liquid at
w_s = g w_t (physical current units, 'sur:' binding) runs at the target's operating rate.

Prediction (self-consistent single-neuron rate, no recurrent network):
  1. each liquid neuron of the target alone, at w_t: its input edges driven by the target's input
     layer, its recurrent in-edges by independent Poisson trains at rate nu. The stable fixed
     point nu* of  <r_out>(nu) = nu  is the predicted operating rate of the target liquid
  2. the same for the surrogate (its own input layer), at the fixed rate nu*, as a function of w:
     w_s solves <r_out>(w) = nu*;  g = w_s / w_t
Checked against the network (diag_matched.py): within 4 % for HH and ML C1 at tau_u = 1..20 ms,
<= 9 % low for AdEx at tau_u <= 5 ms.

    $PY gain.py --model hh [--w 12.8] [--n 40]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import spec as S
import circuit as CI
import run as R
import diag_rates as D
import diag_matched as M
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec

DT = D.DT
NU = np.geomspace(0.5, 300.0, 28)
WGRID = np.sqrt(2.0) ** np.arange(-6, 15)


def _poisson_batch(ev, nus, nin, n, seed=11):
    """instances (nu_i, digit k), nu-major: input events of digit k + Poisson trains at nu_i
    (the same seed for every nu)"""
    parts = [M.poisson_events(ev, nu, nin, n, seed=seed) for nu in nus]
    P, S_, C = [0], [], []
    for p, s, c in parts:
        P.extend((P[-1] + p[1:]).tolist()); S_.append(s); C.append(c)
    return np.array(P, np.int64), np.concatenate(S_), np.concatenate(C)


def _sim(model, circs, cidx, ev, K):
    b, _ = R.binding(model, S.N_LIQ, DT)
    o = engine_gpu.simulate(LS.stack(circs, [b] * len(circs), CI.receptors()), cidx.astype(np.int32),
                            np.full(K, M.NS, np.int64), R.y0_for(model, K, S.N_LIQ), ev, DT, Rec())
    return o["counts"].astype(float)


def stable_fixed_point(nus, r):
    """first downward crossing of r(nu) - nu (log-interpolated); None if r > nu everywhere"""
    f = np.log(np.maximum(r, 1e-6)) - np.log(nus)
    for i in range(1, len(nus)):
        if f[i - 1] > 0 >= f[i]:
            x = f[i - 1] / (f[i - 1] - f[i])
            return float(np.exp(np.log(nus[i - 1]) + x * (np.log(nus[i]) - np.log(nus[i - 1]))))
    return None


def predict(z, w_t, n=40, net_seed=100, data=None, nin=784):
    """-> dict(nu, w_t, w_s, g, curves); tau_u = spec.TAU_U at call time"""
    if data is None:
        data, _, nin = R.load("mnist")
    idx = 60000 + np.arange(n)
    build = lambda w: D.open_loop(CI.build(nin, w, DT, np.random.default_rng(net_seed), pulse=R.PULSE_U,
                                           input_neurons=False))
    ev_t = R.input_layer(f"{z}@fastin", z, nin, data, idx, "mnist", DT)
    ev_s = R.input_layer(f"sur:{z}@fastin", f"sur:{z}", nin, data, idx, "mnist", DT)
    # 1. target fixed point at w_t
    ct = _sim(z, [build(w_t)], np.zeros(len(NU) * n), _poisson_batch(ev_t, NU, nin, n), len(NU) * n)
    rt = ct.reshape(len(NU), n, -1).mean((1, 2)) * 1000 / S.T_SAMPLE
    nu = stable_fixed_point(NU, rt)
    out = dict(model=z, w_t=float(w_t), tau_u=float(S.TAU_U), n=n, nu_grid=NU.tolist(), r_t_nu=rt.tolist(), nu=nu)
    if nu is None:
        return out
    # 2. surrogate weight at the fixed rate nu*
    ws = w_t * WGRID
    ev = M.tile(M.poisson_events(ev_s, nu, nin, n), len(ws))
    cs = _sim(f"sur:{z}", [build(w) for w in ws], np.repeat(np.arange(len(ws)), n), ev, len(ws) * n)
    rs = cs.reshape(len(ws), n, -1).mean((1, 2)) * 1000 / S.T_SAMPLE
    w_s = M.crossing(ws, rs, nu)
    out.update(w_grid=ws.tolist(), r_s_w=rs.tolist(), w_s=w_s, g=(w_s / w_t if w_s else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hh")
    ap.add_argument("--w", type=float, default=None, help="target weight (default: campaign w*)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tau-u", type=float, default=S.TAU_U)
    a = ap.parse_args()
    S.TAU_U = a.tau_u
    w = a.w or json.load(open(os.path.join(HERE, "data", "campaign_mnist", "best_w.json")))[f"{a.model}@fastin"]
    p = predict(a.model, w, a.n)
    print(f"{a.model}: w_t {w:g}  predicted rate nu* {p['nu']}  w_s {p.get('w_s')}  g {p.get('g')}")


if __name__ == "__main__":
    main()
