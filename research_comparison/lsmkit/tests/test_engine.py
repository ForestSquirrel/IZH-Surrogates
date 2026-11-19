"""
test_engine.py -- analytic / reference checks of the CPU engine.

    python -m pytest -q lsmkit/tests   (from research_comparison)
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from lsmkit import spec as LS, engine_cpu
from lsmkit.neurons import lif, izh

DT = 0.1


def _net(model, P, dim, edges, nin=1, N=1, tref=0, clamp=1, bias=0.0, tau_s=3.0, iscale=1.0):
    c = LS.Circuit(N=N, nin=nin, types=np.ones(N, np.int8), meta={}, **edges)
    f = np.float32
    b = LS.Binding(model=model, P=P, tref=np.full(N, tref, np.int32),
                   clamp=np.full(N, clamp, np.int32), bias=np.full(N, bias, f),
                   iscale=np.full(N, iscale, f), va=np.ones(N, f), vb=np.zeros(N, f), dim=dim)
    rec = LS.Receptors(tau=np.array([tau_s]), cond=np.zeros(1, np.int32), E=np.zeros(1, np.float32))
    return LS.stack([c], [b], rec)


def _edges(src, post, w, dly, stp=0, U=0.5, D=1100.0, F=50.0):
    n = len(src)
    a = lambda x, t: np.asarray(x, t) if np.ndim(x) else np.full(n, x, t)
    return dict(src=a(src, np.int32), post=a(post, np.int32), w=a(w, float), dly=a(dly, np.int32),
                rec=np.zeros(n, np.int32), U=a(U, float), D=a(D, float), F=a(F, float),
                stp=a(stp, np.int32))


def test_lif_psp_matches_analytic():
    """one static input spike -> LIF PSP = R w ts/(tm-ts) (e^-t/tm - e^-t/ts)"""
    tm, ts, w = 30.0, 3.0, 18.0
    net = _net("lif", lif.params(1, DT, tm, 1.0, 1e9, 0.0), 2, _edges([1], [0], w, 1), tau_s=ts)
    ev = LS.events([(np.array([0.0]), np.array([0]))], DT, 1)
    n = 1000
    o = engine_cpu.simulate(net, [0], [n], np.zeros((1, 1, 2)), ev, DT,
                            engine_cpu.Rec(v=(0,), v_stride=1))
    v = o["v"][1][0, 0, :n]
    t = (np.arange(n) - 1) * DT                    # arrival at step 1, acts from step 1 on
    ref = np.where(t >= 0, w * ts / (tm - ts) * (np.exp(-t / tm) - np.exp(-t / ts)), 0.0)
    assert abs(v.max() - ref.max()) / ref.max() < 0.03, (v.max(), ref.max())
    assert abs(np.argmax(v) - np.argmax(ref)) <= 3


def test_lif_refractory_isi():
    """constant suprathreshold drive: ISI = t_ref + tm ln((RI - Vr)/(RI - Vth))"""
    tm, I, vth, vr, tref = 30.0, 20.0, 15.0, 13.5, 3.0
    net = _net("lif", lif.params(1, DT, tm, 1.0, vth, vr), 2, _edges([], [], 0.0, 1),
               tref=int(tref / DT), bias=I, nin=1)
    ev = LS.events([(np.zeros(0), np.zeros(0, int))], DT, 1)
    o = engine_cpu.simulate(net, [0], [20000], np.zeros((1, 1, 2)), ev, DT,
                            engine_cpu.Rec(spikes=(0,)))
    st = o["spikes"][1][0, 0, :o["spikes"][2][0, 0]] * DT
    isi = np.diff(st[2:]).mean()
    ref = tref + tm * np.log((I - vr) / (I - vth))
    assert abs(isi - ref) < 2 * DT, (isi, ref)


def test_tsodyks_amplitudes():
    """regular input train -> PSC jumps follow Markram 1998 / Maass 2002 recursion"""
    U, D, F, w, isi = 0.5, 1100.0, 50.0, 1.0, 20.0
    nsp = 6
    times = np.arange(nsp) * isi
    # huge tau_s so the receptor integrates the jumps; LIF state unused
    net = _net("lif", lif.params(1, DT, 30.0, 0.0, 1e9, 0.0), 2,
               _edges([1], [0], w, 1, stp=1, U=U, D=D, F=F), tau_s=1e12)
    ev = LS.events([(times, np.zeros(nsp, int))], DT, 1)
    u, R, amps = U, 1.0, []
    for k in range(nsp):
        if k:
            u_n = U + u * (1 - U) * np.exp(-isi / F)
            R = 1 + (R - u * R - 1) * np.exp(-isi / D)
            u = u_n
        amps.append(w * u * R)
    # probe: LIF with decay 0 and R = 1 reads V = I (the receptor sum) every step
    net.P[0, 0, 0] = 0.0
    net.P[0, 0, 1] = 1.0
    o = engine_cpu.simulate(net, [0], [int(times[-1] / DT) + 5], np.zeros((1, 1, 2)), ev, DT,
                            engine_cpu.Rec(v=(0,), v_stride=1))
    v = o["v"][1][0, 0]
    got = [v[int(t / DT) + 1] for t in times]      # after arrival (delay 1) the step's I
    inc = np.diff(np.concatenate([[0.0], got]))
    assert np.allclose(inc, amps, rtol=1e-4), (inc, amps)


def test_izh_matches_pipeline_integrator():
    """surrogate step (tau = 1, h = 0.01) reproduces the pipeline's spike train"""
    sys.path.insert(0, os.path.join(ROOT, "..", "pipeline"))
    from v3_core import integrate_v3, STIM_STEP
    from v3_models import izh_rhs_v3
    th = (0.02, 0.2, -65.0, 8.0)
    I, T = 10.0, 500.0
    st = np.zeros(1000)
    y = np.array([-65.0, 0.2 * -65.0])
    n, _, _ = integrate_v3(izh_rhs_v3, STIM_STEP, np.array([0.0, I]), 1.0, y, 0.0, T, 0.01,
                           st, th[2], th[3], np.array(th, float))
    ref = st[:n]
    net = _net("izh", izh.params(1, th, DT, 1.0), 2, _edges([], [], 0.0, 1), bias=I)
    ev = LS.events([(np.zeros(0), np.zeros(0, int))], DT, 1)
    Y0 = np.array([[[-65.0, -13.0]]])
    o = engine_cpu.simulate(net, [0], [int(T / DT)], Y0, ev, DT, engine_cpu.Rec(spikes=(0,)))
    got = o["spikes"][1][0, 0, :o["spikes"][2][0, 0]] * DT
    assert len(got) == len(ref), (len(got), len(ref))
    assert np.abs(got - ref).max() <= DT + 1e-9


def test_gauge_time_scaling():
    """tau = 2 halves the firing rate of the same cell in paper time"""
    th, I = (0.02, 0.2, -65.0, 8.0), 10.0
    rates = []
    for tau in (1.0, 2.0):
        net = _net("izh", izh.params(1, th, DT, tau), 2, _edges([], [], 0.0, 1), bias=I)
        ev = LS.events([(np.zeros(0), np.zeros(0, int))], DT, 1)
        o = engine_cpu.simulate(net, [0], [40000], np.array([[[-65.0, -13.0]]]), ev, DT)
        rates.append(o["counts"][0, 0])
    assert abs(rates[0] / rates[1] - 2.0) < 0.05, rates


def test_hh_dp_f32_step_matches_f64_reference():
    """HH-DP float32 network step (dt 0.1 = 4 x 0.025 RK4) vs float64 RK4 at h = 0.005:
    spike count over 1 s at 1000 pA (E, gM 100) and mean ISI within 1 %"""
    from lsmkit.neurons import hh_dp as H
    gM, I, T = H.G_M_E, 1000.0, 1000.0
    y = np.array([-65.0, *H.gates_at(-65.0)])
    pr = np.array([gM, H.G_E0, H.G_I0, 0.0])
    k = [np.zeros(5) for _ in range(4)]
    h, ref = 0.005, []
    for s in range(int(T / h)):
        H.rhs_pipeline(0, y, pr, I, k[0]); H.rhs_pipeline(0, y + h / 2 * k[0], pr, I, k[1])
        H.rhs_pipeline(0, y + h / 2 * k[1], pr, I, k[2]); H.rhs_pipeline(0, y + h * k[2], pr, I, k[3])
        yn = y + h / 6 * (k[0] + 2 * k[1] + 2 * k[2] + k[3])
        if yn[0] < y[0] and y[0] > H.V_THRESH and (not ref or s * h - ref[-1] > H.T_REF):
            ref.append(s * h)
        y = yn
    net = _net("hh_dp", H.params(1, DT, gM), 5, _edges([], [], 0.0, 1), bias=I,
               tref=int(H.T_REF / DT), clamp=0)
    ev = LS.events([(np.zeros(0), np.zeros(0, int))], DT, 1)
    Y0 = np.array([[[-65.0, *H.gates_at(-65.0)]]])
    o = engine_cpu.simulate(net, [0], [int(T / DT)], Y0, ev, DT, engine_cpu.Rec(spikes=(0,)))
    got = o["spikes"][1][0, 0, :o["spikes"][2][0, 0]] * DT
    assert abs(len(got) - len(ref)) <= 1, (len(got), len(ref))
    assert abs(np.diff(got).mean() / np.diff(ref).mean() - 1) < 0.01


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
