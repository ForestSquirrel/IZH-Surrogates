"""
engine_cpu.py -- event-driven LSM simulator on the CPU (numba, parallel over instances).

Same algorithm and data layout as the GPU engine; this one is the
reference for parity tests and runs the small papers. One instance = one circuit driven by
one input sample. Per step n (dt fixed):
  1. receptor states take this step's arrivals from the delay ring;
  2. each reservoir neuron: I = bias + sum_r s_r (current receptor) or s_r (E_r - V)
     (conductance receptor, V = va*y[0] + vb in paper units), times iscale into model units;
     refractory neurons count down (state frozen if clamp) and cannot spike;
     on a spike the neuron pushes its out-edges: Tsodyks u, R updated analytically from the
     time since its previous spike, amplitude w*u*R added to ring[(n + dly) % dmax, post, rec];
  3. input channels spiking at step n push their out-edges the same way;
  4. receptor states and the readout filters decay by exp(-dt/tau).
All state is float32.

Recording (Rec): always the final readout-filter state, per-neuron spike counts and the
first spike step; optionally the filter state every ro_every steps (L0 trajectory), all
spike steps for selected instances (L1, capped at rmax per neuron) and model voltage in
paper units every v_stride steps for selected instances (L2).
"""
import time
from dataclasses import dataclass, field
import numpy as np
from numba import njit, prange

from .neurons import MODELS, NSUB_COL

F32 = np.float32
_RUNNERS = {}


@dataclass
class Rec:
    ro_tau: float = 30.0             # readout low-pass filter (ms)
    ro_every: int = 0                # >0: filter state every ro_every steps (L0 trajectory)
    spikes: tuple = ()               # instance indices with full spike record (L1)
    rmax: int = 512
    v: tuple = ()                    # instance indices with voltage record (L2)
    v_stride: int = 10
    ro_offset: int = 0               # trajectory samples at n + 1 = ro_offset + k * ro_every
    sp_from: int = 0                 # L1 records spikes from this step on
    v_from: int = 0                  # L2 window [v_from, v_to)
    v_to: int = -1                   # -1 = to the end

    def windows(self, nmax):
        """-> int64 [ro_offset, sp_from, v_from, v_to] with v_to resolved"""
        vt = nmax if self.v_to < 0 else min(self.v_to, nmax)
        return np.array([self.ro_offset, self.sp_from, self.v_from, vt], np.int64)

    def n_traj(self, nmax):
        return max(0, (nmax - self.ro_offset) // self.ro_every) if self.ro_every > 0 else 0

    def n_v(self, nmax):
        w = self.windows(nmax)
        return max(1, (w[3] - w[2] - 1) // self.v_stride + 1) if self.v else 1


@njit(inline="always")
def _push(src, n, c, dt, out_ptr, e0, last, uu, rr, e_post, e_w, e_dly, e_rec,
          e_U, e_D, e_F, e_stp, ring, dmax):
    a = out_ptr[c, src]
    b = out_ptr[c, src + 1]
    lst = last[src]
    gap = np.float32((n - lst) * dt)
    for e in range(a, b):
        A = e_w[e]
        if e_stp[e]:
            j = e - e0
            U = e_U[e]
            if lst < 0:
                u = U
                R = np.float32(1.0)
            else:
                u = U + uu[j] * (np.float32(1.0) - U) * np.exp(-gap / e_F[e])
                R = np.float32(1.0) + (rr[j] - uu[j] * rr[j] - np.float32(1.0)) * np.exp(-gap / e_D[e])
            uu[j] = u
            rr[j] = R
            A = A * u * R
        ring[(n + e_dly[e]) % dmax, e_post[e], e_rec[e]] += A
    last[src] = n
    return b - a


def _make(step):
    @njit(parallel=True)
    def run(circ, nsteps, Y0, P, tref, clamp, bias, iscale, va, vb,
            out_ptr, e_post, e_w, e_dly, e_rec, e_U, e_D, e_F, e_stp,
            rdec, rcond, rE, dmax, ev_ptr, ev_step, ev_src, ro_dec, ro_every, dt,
            rnoise, win, traj, final, counts, first, nev, sp_slot, sp_t, sp_n, v_slot, v_stride, v_tr):
        n_inst = circ.shape[0]
        N = P.shape[1]
        R = rdec.shape[0]
        nsrc = out_ptr.shape[1] - 1
        for k in prange(n_inst):
            c = circ[k]
            y = Y0[k].copy()
            refc = np.zeros(N, np.int32)
            syn = np.zeros((N, R), np.float32)
            ring = np.zeros((dmax, N, R), np.float32)
            e0 = out_ptr[c, 0]
            ne_c = out_ptr[c, nsrc] - e0
            uu = np.zeros(ne_c, np.float32)
            rr = np.ones(ne_c, np.float32)
            last = np.full(nsrc, -1, np.int64)
            filt = np.zeros(N, np.float32)
            ev = ev_ptr[k]
            ev_end = ev_ptr[k + 1]
            ss = sp_slot[k]
            vs = v_slot[k]
            nro = 0
            ne = 0
            for n in range(nsteps[k]):
                slot = n % dmax
                for i in range(N):
                    for r in range(R):
                        syn[i, r] += ring[slot, i, r]
                        ring[slot, i, r] = np.float32(0.0)
                for i in range(N):
                    I = bias[c, i]
                    vt = va[c, i] * y[i, 0] + vb[c, i]
                    for r in range(R):
                        if rcond[r]:
                            I += syn[i, r] * (rE[r] - vt)
                        else:
                            I += syn[i, r]
                    I = I * iscale[c, i]
                    if refc[i] > 0:
                        refc[i] -= 1
                        if clamp[c, i] == 0:
                            step(y[i], P[c, i], I)
                        continue
                    if step(y[i], P[c, i], I):
                        refc[i] = tref[c, i]
                        counts[k, i] += 1
                        if first[k, i] < 0:
                            first[k, i] = n
                        filt[i] += np.float32(1.0)
                        if ss >= 0 and n >= win[1] and sp_n[ss, i] < sp_t.shape[2]:
                            sp_t[ss, i, sp_n[ss, i]] = n
                            sp_n[ss, i] += 1
                        ne += _push(i, n, c, dt, out_ptr, e0, last, uu, rr, e_post, e_w,
                                    e_dly, e_rec, e_U, e_D, e_F, e_stp, ring, dmax)
                while ev < ev_end and ev_step[ev] == n:
                    ne += _push(N + ev_src[ev], n, c, dt, out_ptr, e0, last, uu, rr, e_post,
                                e_w, e_dly, e_rec, e_U, e_D, e_F, e_stp, ring, dmax)
                    ev += 1
                for i in range(N):
                    for r in range(R):
                        syn[i, r] *= rdec[r]
                        if rnoise[r] > 0:
                            syn[i, r] += rnoise[r] * np.float32(np.random.standard_normal())
                    filt[i] *= ro_dec
                if vs >= 0 and n >= win[2] and n < win[3] and (n - win[2]) % v_stride == 0:
                    m = (n - win[2]) // v_stride
                    if m < v_tr.shape[2]:
                        for i in range(N):
                            v_tr[vs, i, m] = va[c, i] * y[i, 0] + vb[c, i]
                if ro_every > 0 and n + 1 >= win[0] + ro_every and (n + 1 - win[0]) % ro_every == 0 \
                        and nro < traj.shape[1]:
                    for i in range(N):
                        traj[k, nro, i] = filt[i]
                    nro += 1
            for i in range(N):
                final[k, i] = filt[i]
            nev[k] = ne
    return run


def runner(model):
    if model not in _RUNNERS:
        _RUNNERS[model] = _make(MODELS[model].step)
    return _RUNNERS[model]


def simulate(net, circ, nsteps, Y0, ev, dt, rec=Rec()):
    """net: spec.Network; circ: (K,) circuit index per instance; nsteps: (K,) steps per
    instance; Y0: (K, N, dim) initial model state; ev: spec.events(...) output.
    -> dict of numpy arrays + compute record."""
    circ = np.ascontiguousarray(circ, np.int32)
    nsteps = np.ascontiguousarray(nsteps, np.int64)
    K, N = len(circ), net.N
    Y0 = np.ascontiguousarray(Y0, F32)
    assert Y0.shape == (K, N, net.dim)
    ev_ptr, ev_step, ev_src = ev
    nmax = int(nsteps.max())
    nro = rec.n_traj(nmax)
    traj = np.zeros((K if nro else 1, max(nro, 1), N), F32)
    final = np.zeros((K, N), F32)
    counts = np.zeros((K, N), np.int32)
    first = np.full((K, N), -1, np.int64)
    nev = np.zeros(K, np.int64)
    sp_slot = np.full(K, -1, np.int32)
    for j, k in enumerate(rec.spikes):
        sp_slot[k] = j
    sp_t = np.zeros((max(len(rec.spikes), 1), N, rec.rmax), np.int32)
    sp_n = np.zeros((max(len(rec.spikes), 1), N), np.int32)
    v_slot = np.full(K, -1, np.int32)
    for j, k in enumerate(rec.v):
        v_slot[k] = j
    nv = rec.n_v(nmax)
    v_tr = np.zeros((max(len(rec.v), 1), N, nv), F32)
    run = runner(net.model)
    t0 = time.perf_counter()
    run(circ, nsteps, Y0, net.P, net.tref, net.clamp, net.bias, net.iscale, net.va, net.vb,
        net.out_ptr, net.e_post, net.e_w, net.e_dly, net.e_rec, net.e_U, net.e_D, net.e_F,
        net.e_stp, net.rec.decay(dt), np.asarray(net.rec.cond, np.int32),
        np.asarray(net.rec.E, F32), net.dmax, ev_ptr, ev_step, ev_src,
        F32(np.exp(-dt / rec.ro_tau)), rec.ro_every, dt, net.rec.noise_amp(dt),
        rec.windows(nmax), traj, final, counts, first, nev, sp_slot, sp_t, sp_n, v_slot, rec.v_stride, v_tr)
    wall = time.perf_counter() - t0
    col = NSUB_COL.get(net.model)
    nsub = int(net.P[0, :, col].max()) if col is not None else 1
    comp = dict(backend="cpu", wall_s=wall, instances=K, N=N, dt=dt,
                steps=int(nsteps.sum()), neuron_steps=int(nsteps.sum()) * N,
                model_substeps=nsub, events=int(nev.sum()))
    comp["neuron_steps_per_s"] = comp["neuron_steps"] / wall if wall > 0 else float("nan")
    out = dict(final=final, counts=counts, first=first, nev=nev, compute=comp)
    if nro:
        out["traj"] = traj
    if rec.spikes:
        out["spikes"] = (np.asarray(rec.spikes), sp_t, sp_n)
    if rec.v:
        out["v"] = (np.asarray(rec.v), v_tr, rec.v_stride)
    return out

