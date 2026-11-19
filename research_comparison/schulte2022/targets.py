"""
targets.py -- Schulte to Brinke 2022 HH neurons as pipeline continuous targets.

Two targets: E (gM = 100 nS) and I (gM = 0), both with the mean intrinsic noise
conductances (gE0 12 nS, gI0 57 nS) and no fluctuation (sigma = 0, the deterministic
operating point). Current in pA (I_RH_MIN-safe: rheobase is ~10^2-10^3 pA).

Voltage map: V_HH = va * v_izh + vb with
  rest      HH steady state at I = 0            <->  Izhikevich stable node at I = 0
  threshold HH dV/dt = 10 mV/ms on the first spike at 1.2 i_rh  <->  Izhikevich saddle at I = 0
(the voltage where the cell's own spike becomes self-sustaining, on each side).

    python targets.py [E|I]      # -> targets/hh_E.json, targets/hh_I.json (refuse to overwrite)
"""
import os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from lsmkit import infer
from lsmkit.neurons import hh_dp as H

GM = {"E": H.G_M_E, "I": H.G_M_I}


def _rk4(gM, y, I, T, h=0.005, rec=False):
    pr = np.array([gM, H.G_E0, H.G_I0, 0.0])
    k = [np.zeros(5) for _ in range(4)]
    vs = []
    for _ in range(int(round(T / h))):
        H.rhs_pipeline(0, y, pr, I, k[0])
        H.rhs_pipeline(0, y + h / 2 * k[0], pr, I, k[1])
        H.rhs_pipeline(0, y + h / 2 * k[1], pr, I, k[2])
        H.rhs_pipeline(0, y + h * k[2], pr, I, k[3])
        y = y + h / 6 * (k[0] + 2 * k[1] + 2 * k[2] + k[3])
        if rec:
            vs.append(y[0])
    return y, np.array(vs)


def rest_state(gM):
    y0 = np.array([-65.0, *H.gates_at(-65.0)])
    return _rk4(gM, y0, 0.0, 3000.0)[0]


def hh_threshold(gM, i_rh, h=0.005):
    y, v = _rk4(gM, rest_state(gM), 1.2 * i_rh, 300.0, h, rec=True)
    dv = np.diff(v) / h
    k = int(np.argmax(dv > 10.0))
    return float(v[k])


def izh_rest_saddle(theta):
    a, b, c, d = theta
    disc = (5.0 - b) ** 2 - 4 * 0.04 * 140.0
    if disc <= 0:
        return None
    r = math.sqrt(disc)
    return (-(5.0 - b) - r) / 0.08, (-(5.0 - b) + r) / 0.08


def vmap_for(gM, V_rest):
    def f(res):
        th = hh_threshold(gM, res["i_rh"])
        rs = izh_rest_saddle(res["best"]["theta"])
        if rs is None:
            return dict(va=1.0, vb=V_rest + 65.0, note="surrogate has no rest at I=0: shift "
                        "only (-65 -> HH rest)", hh_rest=V_rest, hh_thr=th)
        vr, vs = rs
        va = (th - V_rest) / (vs - vr)
        return dict(va=va, vb=V_rest - va * vr, hh_rest=V_rest, hh_thr=th, izh_rest=vr,
                    izh_saddle=vs, note="rest<->rest (I=0), threshold (10 mV/ms at 1.2 i_rh) "
                    "<-> Izhikevich saddle (I=0); best view")
    return f
# (targets/*.json also carry vmap["views"][view]: the same rule applied to each posterior
#  view's theta; surrogate.Gauge.load picks the view's own map)


def main():
    from v3_an_features import V3Spec       # infer.run puts the pipeline on sys.path
    for which in (sys.argv[1:] or ["E", "I"]):
        gM = GM[which]
        y0 = rest_state(gM)
        spec = V3Spec(rhs=H.rhs_pipeline, y0=y0, params=np.array([gM, H.G_E0, H.G_I0, 0.0]),
                      v_th=H.V_THRESH, rheo_lo=0.0, rheo_hi=5000.0, h=0.01,
                      boot_t_trans=200.0, boot_t_win=2000.0, boot_t_disc=200.0,
                      name=f"Schulte HH-{which}")
        infer.run("continuous", spec, os.path.join(HERE, "targets", f"hh_{which}.json"),
                  vmap_for(gM, float(y0[0])), name=f"Schulte HH-{which}",
                  target_info=dict(model="Destexhe-Pare HH (Schulte 2022 code values)",
                                   gM=gM, gE0=H.G_E0, gI0=H.G_I0, noise="mean only",
                                   y0=y0.tolist(), v_th=H.V_THRESH, units="mV ms pF nS pA"))


if __name__ == "__main__":
    infer._pipeline()
    main()
