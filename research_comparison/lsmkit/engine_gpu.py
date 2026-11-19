"""
engine_gpu.py -- event-driven LSM simulator on the GPU (numba.cuda, float32 only).

Same algorithm, inputs and outputs as engine_cpu (the reference). Layout, replacing a dense
block-resident kernel:
  * grid = instances, one block each; the block's threads stride over the N neurons, so N
    is not limited by the block size;
  * all state lives in global memory, neuron-minor ([instance][var][neuron]), and persists
    across launches: the time loop is cut into chunks of `chunk` steps (progress, no
    display-watchdog risk under WSL);
  * spikes are pushed: the owning thread walks its neuron's out-edges, updates the per-edge
    Tsodyks state (owned by that source, so race-free) and atomically adds the amplitude
    into the delay ring [instance][slot][receptor][neuron]; input channels are pushed by
    thread 0 when synapses are dynamic (it owns their STP state), and split across the
    block's threads when the circuit is static;
  * one __syncthreads after the arrival/update phase and one after the push phase per step:
    delays are >= 1 step, so a slot is never written while it is being read.
Differences from the CPU engine: atomic adds commute only up to float32 rounding, so runs
are not bit-identical to the CPU engine (nor to each other); compare statistics.
"""
import math, time
import numpy as np
from numba import cuda, float32, int32, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_next

from .neurons import MODELS, NSUB_COL
from .engine_cpu import Rec

F32 = np.float32
_KERNELS = {}
ONE = float32(1.0)
ZERO = float32(0.0)
INV24 = float32(1.0 / 16777216.0)
TWO = float32(2.0)
M2 = float32(-2.0)


@cuda.jit(device=True, inline=True)
def normal32(rng, idx):
    """standard normal in float32 only: Marsaglia polar method on 24-bit uniforms from
    xoroshiro128p_next (numba's xoroshiro128p_normal_float32 and libdevice cosf both
    emit float64 instructions)"""
    while True:
        x = float32(xoroshiro128p_next(rng, idx) >> 40) * INV24 * TWO - ONE
        y = float32(xoroshiro128p_next(rng, idx) >> 40) * INV24 * TWO - ONE
        q = x * x + y * y
        if q > ZERO and q < ONE:
            return x * math.sqrt(M2 * math.log(q) / q)


def _make(step, DIM):
    @cuda.jit(device=True, inline=True)
    def push(k, src, n, c, dt, out_ptr, e0, last, uu, rr, e_post, e_w, e_dly, e_rec,
             e_U, e_D, e_F, e_stp, ring, dmax, use_stp):
        a = out_ptr[c, src]
        b = out_ptr[c, src + 1]
        if not use_stp:          # static circuit: no per-edge / per-source state at all
            for e in range(a, b):
                cuda.atomic.add(ring, (k, (n + e_dly[e]) % dmax, e_rec[e], e_post[e]), e_w[e])
            return b - a
        lst = last[k, src]
        gap = float32(n - lst) * dt
        for e in range(a, b):
            A = e_w[e]
            if e_stp[e]:
                j = e - e0
                U = e_U[e]
                if lst < 0:
                    u = U
                    R = ONE
                else:
                    u = U + uu[k, j] * (ONE - U) * math.exp(-gap / e_F[e])
                    R = ONE + (rr[k, j] - uu[k, j] * rr[k, j] - ONE) * math.exp(-gap / e_D[e])
                uu[k, j] = u
                rr[k, j] = R
                A = A * u * R
            cuda.atomic.add(ring, (k, (n + e_dly[e]) % dmax, e_rec[e], e_post[e]), A)
        last[k, src] = n
        return b - a

    @cuda.jit
    def run(n0, n1, circ, nsteps, Y, P, tref, clamp, bias, iscale, va, vb,
            out_ptr, e_post, e_w, e_dly, e_rec, e_U, e_D, e_F, e_stp,
            rdec, rcond, rE, dmax, ev_ptr, ev_step, ev_src, ev_pos, ro_dec, ro_every, dt,
            rnoise, win, rng, use_stp, refc, syn, ring, uu, rr, last, filt,
            traj, final, counts, first, nev, sp_slot, sp_t, sp_n, v_slot, v_stride, v_tr):
        k = cuda.blockIdx.x
        t = cuda.threadIdx.x
        nt = cuda.blockDim.x
        c = circ[k]
        N = P.shape[1]
        R = rdec.shape[0]
        e0 = out_ptr[c, 0]
        ss = sp_slot[k]
        vs = v_slot[k]
        nend = nsteps[k]               # builtin min() does not compile here (numba 0.66)
        if n1 < nend:
            nend = n1
        y = cuda.local.array(DIM, float32)
        ev_hi = cuda.shared.array(1, int64)
        ne = int64(0)
        for n in range(n0, nend):
            slot = n % dmax
            for i in range(t, N, nt):
                for r in range(R):
                    syn[k, r, i] += ring[k, slot, r, i]
                    ring[k, slot, r, i] = ZERO
                for d in range(DIM):
                    y[d] = Y[k, d, i]
                I = bias[c, i]
                vt = va[c, i] * y[0] + vb[c, i]
                for r in range(R):
                    if rcond[r]:
                        I += syn[k, r, i] * (rE[r] - vt)
                    else:
                        I += syn[k, r, i]
                I = I * iscale[c, i]
                spk = 0
                if refc[k, i] > 0:
                    refc[k, i] -= 1
                    if clamp[c, i] == 0:
                        step(y, P[c, i], I)
                else:
                    spk = step(y, P[c, i], I)
                for d in range(DIM):
                    Y[k, d, i] = y[d]
                if spk:
                    refc[k, i] = tref[c, i]
                    counts[k, i] += 1
                    if first[k, i] < 0:
                        first[k, i] = n
                    filt[k, i] += ONE
                    if ss >= 0 and n >= win[1] and sp_n[ss, i] < sp_t.shape[2]:
                        sp_t[ss, i, sp_n[ss, i]] = n
                        sp_n[ss, i] += 1
                    ne += push(k, i, n, c, dt, out_ptr, e0, last, uu, rr, e_post, e_w,
                               e_dly, e_rec, e_U, e_D, e_F, e_stp, ring, dmax, use_stp)
            if use_stp:
                # dynamic synapses: one thread owns the per-source STP state
                if t == 0:
                    ev = ev_pos[k]
                    end = ev_ptr[k + 1]
                    while ev < end and ev_step[ev] == n:
                        ne += push(k, N + ev_src[ev], n, c, dt, out_ptr, e0, last, uu, rr, e_post,
                                   e_w, e_dly, e_rec, e_U, e_D, e_F, e_stp, ring, dmax, use_stp)
                        ev += 1
                    ev_pos[k] = ev
            else:
                # static synapses: no per-source state, so this step's input events are split
                # across the block's threads (dense input layers: 10^4 events x 150 edges)
                if t == 0:
                    ev = ev_pos[k]
                    end = ev_ptr[k + 1]
                    while ev < end and ev_step[ev] == n:
                        ev += 1
                    ev_hi[0] = ev
                cuda.syncthreads()
                for ev in range(ev_pos[k] + t, ev_hi[0], nt):
                    ne += push(k, N + ev_src[ev], n, c, dt, out_ptr, e0, last, uu, rr, e_post,
                               e_w, e_dly, e_rec, e_U, e_D, e_F, e_stp, ring, dmax, use_stp)
                cuda.syncthreads()
                if t == 0:
                    ev_pos[k] = ev_hi[0]
            for i in range(t, N, nt):
                for r in range(R):
                    syn[k, r, i] *= rdec[r]
                    if rnoise[r] > ZERO:
                        syn[k, r, i] += rnoise[r] * normal32(rng, k * N + i)
                filt[k, i] *= ro_dec
                if vs >= 0 and n >= win[2] and n < win[3] and (n - win[2]) % v_stride == 0:
                    m = (n - win[2]) // v_stride
                    if m < v_tr.shape[2]:
                        v_tr[vs, i, m] = va[c, i] * Y[k, 0, i] + vb[c, i]
                if ro_every > 0 and n + 1 >= win[0] + ro_every and (n + 1 - win[0]) % ro_every == 0:
                    m = (n + 1 - win[0]) // ro_every - 1
                    if m < traj.shape[1]:
                        traj[k, m, i] = filt[k, i]
                if n == nsteps[k] - 1:
                    final[k, i] = filt[k, i]
            cuda.syncthreads()
        cuda.atomic.add(nev, k, ne)
    return run


def kernel(model):
    if model not in _KERNELS:
        m = MODELS[model]
        _KERNELS[model] = _make(m.step_cuda, max(m.DIM, 1))
    return _KERNELS[model]


def simulate(net, circ, nsteps, Y0, ev, dt, rec=Rec(), chunk=2000, threads=None,
             progress=False, seed=0):
    """Same contract as engine_cpu.simulate (Y0: (K, N, dim)); returns the same dict."""
    circ = np.ascontiguousarray(circ, np.int32)
    nsteps = np.ascontiguousarray(nsteps, np.int64)
    K, N = len(circ), net.N
    DIM = max(MODELS[net.model].DIM, 1)
    Y0 = np.asarray(Y0, F32)
    assert Y0.shape[:2] == (K, N) and Y0.shape[2] >= DIM
    R = len(net.rec.tau)
    nsrc = net.out_ptr.shape[1] - 1
    use_stp = int(bool(net.e_stp.any()))
    # per-instance plasticity state only when some synapse is dynamic (static circuits with
    # many instances would otherwise allocate K x edges floats)
    emax = int((net.out_ptr[:, nsrc] - net.out_ptr[:, 0]).max()) if use_stp else 1
    ev_ptr, ev_step, ev_src = ev
    d = cuda.to_device
    f32z = lambda *s: cuda.to_device(np.zeros(s, F32))
    nmax = int(nsteps.max())
    nro = rec.n_traj(nmax)
    sp_slot = np.full(K, -1, np.int32)
    for j, k in enumerate(rec.spikes):
        sp_slot[k] = j
    v_slot = np.full(K, -1, np.int32)
    for j, k in enumerate(rec.v):
        v_slot[k] = j
    nv = rec.n_v(nmax)
    a = dict(
        circ=d(circ), nsteps=d(nsteps),
        Y=d(np.ascontiguousarray(Y0[:, :, :DIM].transpose(0, 2, 1))),
        P=d(net.P), tref=d(net.tref), clamp=d(net.clamp), bias=d(net.bias),
        iscale=d(net.iscale), va=d(net.va), vb=d(net.vb), out_ptr=d(net.out_ptr),
        e_post=d(net.e_post), e_w=d(net.e_w), e_dly=d(net.e_dly), e_rec=d(net.e_rec),
        e_U=d(net.e_U), e_D=d(net.e_D), e_F=d(net.e_F), e_stp=d(net.e_stp),
        rdec=d(net.rec.decay(dt)), rcond=d(np.asarray(net.rec.cond, np.int32)),
        rE=d(np.asarray(net.rec.E, F32)), dmax=net.dmax,
        ev_ptr=d(ev_ptr), ev_step=d(ev_step if len(ev_step) else np.zeros(1, np.int64)),
        ev_src=d(ev_src if len(ev_src) else np.zeros(1, np.int32)),
        ev_pos=d(np.ascontiguousarray(ev_ptr[:-1])), ro_dec=F32(np.exp(-dt / rec.ro_tau)),
        ro_every=rec.ro_every, dt=F32(dt), rnoise=d(net.rec.noise_amp(dt)),
        win=d(rec.windows(nmax)),
        rng=create_xoroshiro128p_states(K * N if (net.rec.noise_amp(dt) > 0).any() else 1, seed=seed),
        use_stp=use_stp,
        refc=d(np.zeros((K, N), np.int32)), syn=f32z(K, R, N), ring=f32z(K, net.dmax, R, N),
        uu=f32z(K, max(emax, 1)), rr=d(np.ones((K, max(emax, 1)), F32)),
        last=d(np.full((K, nsrc) if use_stp else (1, 1), -1, np.int64)), filt=f32z(K, N),
        traj=f32z(K if nro else 1, max(nro, 1), N), final=f32z(K, N),
        counts=d(np.zeros((K, N), np.int32)), first=d(np.full((K, N), -1, np.int64)),
        nev=d(np.zeros(K, np.int64)), sp_slot=d(sp_slot),
        sp_t=d(np.zeros((max(len(rec.spikes), 1), N, rec.rmax), np.int32)),
        sp_n=d(np.zeros((max(len(rec.spikes), 1), N), np.int32)), v_slot=d(v_slot),
        v_stride=rec.v_stride, v_tr=f32z(max(len(rec.v), 1), N, nv))
    order = ("circ nsteps Y P tref clamp bias iscale va vb out_ptr e_post e_w e_dly e_rec e_U "
             "e_D e_F e_stp rdec rcond rE dmax ev_ptr ev_step ev_src ev_pos ro_dec ro_every dt "
             "rnoise win rng use_stp refc syn ring uu rr last filt traj final counts first nev sp_slot sp_t sp_n "
             "v_slot v_stride v_tr").split()
    args = [a[x] for x in order]
    run = kernel(net.model)
    T = int(nsteps.max())
    # block size: one thread per neuron if the kernel's registers allow it, else the largest
    # balanced split (ceil(N / j) neurons per thread, rounded up to a warp) that launches;
    # a launch refused for resources does not run, so retrying is safe
    cands = [threads] if threads else sorted({min(1024, 32 * math.ceil(math.ceil(N / j) / 32))
                                              for j in range(1, 9)}, reverse=True)
    cuda.synchronize()
    t0 = time.perf_counter()
    nt = None
    for c in cands:
        try:
            run[K, c](0, min(chunk, T), *args)
            nt = c
            break
        except cuda.cudadrv.driver.CudaAPIError as e:
            if e.code != 701 or c == cands[-1]:
                raise
    for n0 in range(chunk, T, chunk):
        run[K, nt](n0, min(n0 + chunk, T), *args)
        if progress:
            cuda.synchronize()
            print(f"  step {min(n0 + chunk, T)}/{T}  {time.perf_counter() - t0:.1f}s", flush=True)
    cuda.synchronize()
    wall = time.perf_counter() - t0
    h = lambda x: a[x].copy_to_host()
    nev = h("nev")
    col = NSUB_COL.get(net.model)
    nsub = int(net.P[0, :, col].max()) if col is not None else 1
    comp = dict(backend="gpu", wall_s=wall, instances=K, N=N, dt=dt, threads=nt, chunk=chunk,
                steps=int(nsteps.sum()), neuron_steps=int(nsteps.sum()) * N,
                model_substeps=nsub, events=int(nev.sum()))
    comp["neuron_steps_per_s"] = comp["neuron_steps"] / wall if wall > 0 else float("nan")
    out = dict(final=h("final"), counts=h("counts"), first=h("first"), nev=nev, compute=comp)
    if nro:
        out["traj"] = h("traj")
    if rec.spikes:
        out["spikes"] = (np.asarray(rec.spikes), h("sp_t"), h("sp_n"))
    if rec.v:
        out["v"] = (np.asarray(rec.v), h("v_tr"), rec.v_stride)
    return out
