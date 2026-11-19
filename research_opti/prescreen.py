"""
prescreen.py -- a cost-bounded PASS 0, so one pathological candidate cannot stall a batch.

THE PROBLEM. `v3_gpu.landmark_kernel` measures the rheobase twice: once in an absolute
500 ms bootstrap window (bounded, uniform, the same for every cell) and then again on the
cell's OWN scale, over `K_WIN = 100` cycle times. A candidate whose provisional cycle time
is 500 ms therefore has its 40-step bisection run over a 50 s window -- about 2e8
integration steps in a single GPU lane, tens of seconds -- and every other lane in the
launch waits for it. This is not rare: the atlas's own cycle times have p90 = 13 ms,
p99 = 73 ms and a maximum of 1586 ms, so a random batch of 64 cells contains about one
such candidate, and a measured batch of 64 took over a minute.

The atlas build never noticed because it ran 200k-cell chunks, where warp binning puts the
slow cells together and the other warps keep working. An optimizer generation is 32 cells.

WHAT THIS DOES. The kernel below is `v3_gpu.landmark_kernel` with one early exit: after
round 1 -- whose cost is bounded and identical for every cell -- a provisional cycle time
above `tc_max` returns immediately instead of entering the cell-relative round 2. It calls
the pipeline's own device functions (`_rheobase`, `_rheobase_seeded`, `_t_cycle`, `_l_inf`)
in the pipeline's own order, so for every candidate it does NOT reject it returns exactly
what `landmarks_gpu` returns; `check_prescreen` asserts that bit for bit.

WHAT IT COSTS SCIENTIFICALLY. A rejected candidate is charged the infeasibility barrier,
so the optimizers search a strict subset of the box the cINN was trained on. That can only
handicap the optimizer arm, never flatter it, which is the safe direction for a comparison
whose question is whether the optimizers keep up. The excluded fraction of the atlas is
reported, and it is small: 1.8% of atlas cells have a cycle time above 50 ms, and every
surrogate any method has ever returned for these targets is far below that, because the
gauge makes a surrogate's cycle time the target's divided by tau.

    lm, kept = landmarks_capped(theta, tc_max=50.0)
"""
import numpy as np
from numba import cuda

import common as K
import v3_config as C
import v3_gpu as G

# module-scope copies, so the device code closes over plain floats exactly as v3_gpu does
_SENT = G.SENT
_SENT_CUT = G.SENT_CUT
_BOOT_T, _BOOT_W, _BOOT_D = G.BOOT_T_TRANS, G.BOOT_T_WIN, G.BOOT_T_DISC
_KT, _KW, _KD = G.K_TRANS, G.K_WIN, G.K_DISCARD
_LK = G.LINF_SEARCH_K
_NSP = G.GPU_MAX_SPIKES
_REAL = G.REAL

# Column 5 (beyond landmarks_gpu's five) flags WHY a cell was dropped, so the graded
# infeasibility penalty and the cost accounting can tell "too slow to afford" from
# "never fires".
DROP_NONE, DROP_NO_RHEO, DROP_CAPPED, DROP_SLOW = 0.0, 1.0, 2.0, 3.0


@cuda.jit(cache=True)
def landmark_kernel_capped(params, tc_max, out_lm):
    """PASS 0 with a cost ceiling. out_lm[i] = (i_rh, T_cycle, L_inf, p_hat, capped, drop)."""
    i = cuda.grid(1)
    if i >= params.shape[0]:
        return
    a = params[i, 0]; b = params[i, 1]; c = params[i, 2]; d = params[i, 3]
    st = cuda.local.array(_NSP, dtype=_REAL)
    isi = cuda.local.array(_NSP, dtype=_REAL)

    out_lm[i, 0] = _SENT; out_lm[i, 1] = _SENT; out_lm[i, 2] = _SENT
    out_lm[i, 3] = 0.0;   out_lm[i, 4] = 0.0;   out_lm[i, 5] = DROP_NONE

    i_rh = G._rheobase(a, b, c, d, _BOOT_T, _BOOT_W, st)
    if i_rh < _SENT_CUT:
        out_lm[i, 5] = DROP_NO_RHEO
        return
    tc, p, capped = G._t_cycle(a, b, c, d, i_rh, _BOOT_T, _BOOT_W, _BOOT_D, st, isi)
    if capped == 1 or tc < _SENT_CUT or tc <= 0.0:
        out_lm[i, 0] = i_rh
        out_lm[i, 4] = float(capped)
        out_lm[i, 5] = DROP_CAPPED
        return

    # the one addition: everything past this point scales with tc, so a cell too slow to
    # afford is dropped here, before the cell-relative windows are built
    if tc > tc_max:
        out_lm[i, 0] = i_rh
        out_lm[i, 1] = tc                      # the provisional value, for the record
        out_lm[i, 3] = float(p)
        out_lm[i, 5] = DROP_SLOW
        return

    i_rh = G._rheobase_seeded(a, b, c, d, _KT * tc, _KW * tc, i_rh, st)
    if i_rh < _SENT_CUT:
        out_lm[i, 5] = DROP_NO_RHEO
        return
    tc2, p2, cap2 = G._t_cycle(a, b, c, d, i_rh, _KT * tc, _KW * tc, _KD * tc, st, isi)
    if cap2 == 1:
        out_lm[i, 0] = i_rh
        out_lm[i, 4] = 1.0
        out_lm[i, 5] = DROP_CAPPED
        return
    if tc2 > _SENT_CUT and tc2 > 0.0:
        tc = tc2
        p = p2
    l0 = G._l_inf(a, b, c, d, i_rh, _BOOT_W)
    li = _SENT
    if l0 > _SENT_CUT:
        l1 = G._l_inf(a, b, c, d, i_rh, _LK * l0)
        li = l1 if l1 > _SENT_CUT else l0
    out_lm[i, 0] = i_rh
    out_lm[i, 1] = tc
    out_lm[i, 2] = li
    out_lm[i, 3] = float(p)
    out_lm[i, 4] = 0.0


def landmarks_capped(theta, tc_max, tpb=64):
    """(lm (n,6), ok (n,)) -- ok means PASS 0 completed and the cell is affordable."""
    th = np.ascontiguousarray(theta, dtype=G.NP_REAL)
    n = th.shape[0]
    d_th = cuda.to_device(th)
    d_lm = cuda.device_array((n, 6), dtype=G.NP_REAL)
    blocks = (n + tpb - 1) // tpb
    landmark_kernel_capped[blocks, tpb](d_th, G.NP_REAL(tc_max), d_lm)
    cuda.synchronize()
    lm = d_lm.copy_to_host().astype(np.float64)
    ok = lm[:, 5] == DROP_NONE
    return lm, ok


def check_prescreen(n=512, seed=7, tc_max=50.0, tpb=64):
    """The capped kernel must equal landmarks_gpu wherever it does not reject.

    Same device functions, same order, same arithmetic -- so this should be exact, and if
    it ever is not, the prescreen has changed the measurement rather than only its cost.
    """
    import objective as OB
    from scipy.stats import qmc
    u = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    th = OB.Space().to_theta(u)
    lmc, ok = landmarks_capped(th, tc_max, tpb=tpb)
    keep = np.flatnonzero(ok)
    ref = G.landmarks_gpu(np.ascontiguousarray(th[keep]), tpb=tpb).astype(np.float64)
    diff = np.abs(ref - lmc[keep, :5])
    return dict(n=n, tc_max=tc_max, n_kept=int(ok.sum()),
                frac_kept=float(ok.mean()),
                drops={"no_rheobase": int((lmc[:, 5] == DROP_NO_RHEO).sum()),
                       "capped": int((lmc[:, 5] == DROP_CAPPED).sum()),
                       "too_slow": int((lmc[:, 5] == DROP_SLOW).sum())},
                max_abs_diff=float(diff.max()) if diff.size else 0.0,
                exact=bool(diff.size == 0 or diff.max() == 0.0))


def atlas_excluded(tc_max):
    """What fraction of the atlas this ceiling would exclude -- the price, measured."""
    import os
    import struct
    p = os.path.join(K.ATLAS, "izh_landmarks.bin")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        fh.read(4)
        n, k = struct.unpack("<qI", fh.read(12))
    a = np.memmap(p, dtype=np.float32, mode="r", offset=16, shape=(n, k))
    tc = np.asarray(a[:, 1], dtype=np.float64)
    d = tc[tc > C.SENT_CUT]
    return dict(n_atlas=int(n), n_anchored=int(d.size),
                frac_above=float((d > tc_max).mean()),
                n_above=int((d > tc_max).sum()),
                quantiles={f"p{q}": float(np.percentile(d, q))
                           for q in (50, 90, 99, 99.9, 100)})


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc-max", type=float, default=50.0)
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    K.ensure_dirs()
    ex = atlas_excluded(a.tc_max)
    print(f"atlas cycle times: " +
          ", ".join(f"{k} {v:.4g} ms" for k, v in ex["quantiles"].items()))
    print(f"ceiling {a.tc_max:g} ms excludes {100*ex['frac_above']:.3f}% of the atlas "
          f"({ex['n_above']} of {ex['n_anchored']} anchored cells)")
    print(f"\nchecking the capped kernel against landmarks_gpu on {a.n} cells ...",
          flush=True)
    r = check_prescreen(a.n, tc_max=a.tc_max)
    print(f"kept {r['n_kept']}/{r['n']} ({100*r['frac_kept']:.1f}%), drops {r['drops']}")
    print(f"max |difference| against landmarks_gpu on kept cells: {r['max_abs_diff']:g} "
          f"-> {'EXACT' if r['exact'] else 'NOT EXACT'}")
    K.jwrite(os.path.join(K.DATA, "check_prescreen.json"), dict(check=r, atlas=ex))
    print("wrote data/check_prescreen.json")
