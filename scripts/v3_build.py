"""
v3_build.py -- build the v3 atlas on the GPU.

    python scripts/v3_build.py --out /scratch/atlas_new

    --out MUST be a new directory. Pointing it at an existing atlas
    truncates and overwrites that atlas; the guard in main() refuses
    unless --force is given. See README.md section 2.

Resumable by construction. A 31-hour run must survive a crash, a reboot or a
Ctrl-C, so the payload is appended chunk by chunk to headerless `.part` files
and `progress.json` records how many candidate rows have been consumed. Re-run
the same command to continue; the headers are written only at the end, when the
row count is finally known.

Two passes per chunk, as spec §6:

  PASS 0  landmarks (i_rh, T_cycle, L_inf), then the B5 resolution gate
  PASS 1  full suite, launched on cells SORTED BY T_cycle so a warp does not run
          at its slowest lane

The gate DROPS rows rather than sentinelling them (B5), so the number of
candidates is raised by the measured gate rate to land on the requested size.
"""
import os, sys, json, time, struct, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # pipeline_final/scripts
ROOT = os.path.dirname(HERE)                       # pipeline_final
PKG     = os.path.join(ROOT, "pipeline")
DATA    = os.path.join(ROOT, "data")
MODELS  = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
PLOTS   = os.path.join(ROOT, "plots")
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)   # sibling runners (v3_plot_pair -> v3_plot_family)

import v3_config as C
import v3_gpu as G


def lhs(n, lo, hi, seed):
    """Latin hypercube, matching v2's `sampling: lhs`.

    One permuted stratum per dimension: at 1e7 in 4-D this is barely different
    from uniform, but it is what v2 recorded and the two atlases should differ
    by design decisions, not by sampler.
    """
    rng = np.random.default_rng(seed)
    d = len(lo)
    out = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        strata = (rng.permutation(n) + rng.random(n)) / n
        out[:, j] = lo[j] + strata * (hi[j] - lo[j])
    return out


def _finalize(part, final, magic, n, k):
    with open(final, "wb") as fo:
        fo.write(magic)
        fo.write(struct.pack("<qI", n, k))
        with open(part, "rb") as fi:
            while True:
                buf = fi.read(1 << 26)
                if not buf:
                    break
                fo.write(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=10_000_000)
    ap.add_argument("--n-val", type=int, default=250_000)
    ap.add_argument("--gate-headroom", type=float, default=0.02,
                    help="oversample fraction; the B5 gate drops ~0.7%% of candidates")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--tpb", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--force", action="store_true",
                    help="permit writing into a directory that already holds an "
                         "atlas or a resumable build. Required, because the "
                         "default behaviour is destructive -- see below.")
    args = ap.parse_args()

    # ---- destructive-write guard -------------------------------------------
    # Without this, pointing --out at an existing atlas is silent and fatal:
    # makedirs(exist_ok=True) accepts it, a present progress.json sends the run
    # down the RESUME branch which TRUNCATES the .part files, and on completion
    # the final izh_*.bin are rewritten. A shipped atlas directory contains a
    # progress.json, so a mistyped --out does not error -- it starts eating the
    # atlas. This one represents ~29 h of GPU time.
    _guard = [n for n in ("izh_features.bin", "izh_params.bin",
                          "izh_landmarks.bin", "progress.json")
              if os.path.exists(os.path.join(args.out, n))]
    if _guard and not args.force:
        raise SystemExit(
            f"refusing to build into {args.out!r}: it already contains "
            f"{', '.join(_guard)}.\n"
            f"That directory holds an atlas or a resumable build, and this "
            f"script would truncate and overwrite it.\n"
            f"Point --out at a NEW directory, or pass --force if you really "
            f"mean to resume/overwrite this one.")

    os.makedirs(args.out, exist_ok=True)
    n_keep = args.n_train + args.n_val
    n_cand = int(n_keep * (1.0 + args.gate_headroom))

    lo = np.array([0.001, -2.2, -80.0, -2.0])
    hi = np.array([0.3, 1.1, -25.0, 14.0])

    fnames = list(C.FEATURE_NAMES) + [f"_reserved_{i:02d}"
                                      for i in range(C.N_COLS - C.N_ACTIVE)]
    assert len(fnames) == C.N_COLS

    f_part = os.path.join(args.out, "izh_features.part")
    p_part = os.path.join(args.out, "izh_params.part")
    l_part = os.path.join(args.out, "izh_landmarks.part")
    prog_p = os.path.join(args.out, "progress.json")

    prog = {"consumed": 0, "kept": 0, "gated": 0, "seconds": 0.0}
    if os.path.exists(prog_p):
        with open(prog_p) as fh:
            prog = json.load(fh)
        # trust the files, not the counter: truncate to the recorded row count
        for path, k in ((f_part, C.N_COLS), (p_part, 4), (l_part, 5)):
            want = prog["kept"] * k * 4
            if os.path.getsize(path) != want:
                with open(path, "r+b") as fh:
                    fh.truncate(want)
        print(f"[resume] {prog['consumed']:,}/{n_cand:,} candidates consumed, "
              f"{prog['kept']:,} rows kept")
    else:
        for path in (f_part, p_part, l_part):
            open(path, "wb").close()

    print(f"[build] target {n_keep:,} rows ({args.n_train:,} train + "
          f"{args.n_val:,} val), sampling {n_cand:,} candidates")
    print(f"[build] {C.N_ACTIVE} active / {C.N_COLS} on disk, arch {G.GPU_ARCH}, "
          f"DT {C.DT}, chunk {args.chunk:,}")

    theta_all = lhs(n_cand, lo, hi, args.seed)
    t_start = time.time()
    i = prog["consumed"]
    while i < n_cand and prog["kept"] < n_keep:
        j = min(i + args.chunk, n_cand)
        th = np.ascontiguousarray(theta_all[i:j])
        t0 = time.time()

        lm = G.landmarks_gpu(th, tpb=args.tpb)
        F, _ = G.features_gpu(th, lm=lm, tpb=args.tpb, sort=True)

        # B5: a gated cell is DROPPED. Both conditions are gate conditions --
        # `capped` means the anchor was measured on a bound record (B3), and
        # T_cycle < 10*DT is the resolution floor (§34, §35).
        tc = lm[:, 1].astype(np.float64)
        capped = lm[:, 4] != 0.0
        gated = capped | ((tc > C.SENT_CUT) & (tc < C.GATE_K_DT * C.DT))
        keep = ~gated
        room = n_keep - prog["kept"]
        idx = np.flatnonzero(keep)[:room]

        with open(f_part, "ab") as fh:
            fh.write(np.ascontiguousarray(F[idx], dtype=np.float32).tobytes())
        with open(p_part, "ab") as fh:
            fh.write(np.ascontiguousarray(th[idx], dtype=np.float32).tobytes())
        with open(l_part, "ab") as fh:
            fh.write(np.ascontiguousarray(lm[idx], dtype=np.float32).tobytes())

        prog["consumed"] = j
        prog["kept"] += len(idx)
        prog["gated"] += int(gated.sum())
        prog["seconds"] += time.time() - t0
        with open(prog_p, "w") as fh:
            json.dump(prog, fh)

        done = prog["kept"] / n_keep
        rate = prog["seconds"] / max(prog["kept"], 1)
        eta = rate * (n_keep - prog["kept"])
        print(f"[build] {prog['kept']:>9,}/{n_keep:,} ({done*100:5.1f}%)  "
              f"gate {prog['gated']/max(prog['consumed'],1)*100:4.2f}%  "
              f"{rate*1e3:5.2f} ms/cell  elapsed {prog['seconds']/3600:5.2f}h  "
              f"ETA {eta/3600:5.2f}h", flush=True)
        i = j

    n = prog["kept"]
    print(f"[build] finalizing {n:,} rows")
    _finalize(f_part, os.path.join(args.out, "izh_features.bin"), b"IZHF", n, C.N_COLS)
    _finalize(p_part, os.path.join(args.out, "izh_params.bin"), b"IZHP", n, 4)
    _finalize(l_part, os.path.join(args.out, "izh_landmarks.bin"), b"IZHL", n, 5)

    meta = dict(
        n_samples=n, n_features=C.N_COLS, dtype="float32",
        param_names=["a", "b", "c", "d"],
        param_lo=lo.tolist(), param_hi=hi.tolist(), sampling="lhs",
        feature_names=fnames,
        active_features=list(C.FEATURE_NAMES),
        nan_sentinel=C.NAN_SENTINEL, arch=G.GPU_ARCH, noinh=False,
        inhibition_features=[],
        inhibition_note="v3 removes inhibition entirely -- protocols, columns "
                        "and reserved slots (spec §4.6). noinh is False because "
                        "there is nothing to drop.",
        schema="v3",
        config={"DT": C.DT, "K_WIN": C.K_WIN, "K_WIN_LONG": C.K_WIN_LONG,
                "K_TRANS": C.K_TRANS, "K_DISCARD": C.K_DISCARD,
                "N_FI": C.N_FI, "GPU_MAX_SPIKES": G.GPU_MAX_SPIKES,
                "RHEO_BISECT_ITERS": C.RHEO_BISECT_ITERS,
                "RHEO_SEED_ROUND2": C.RHEO_SEED_ROUND2,
                "GATE_K_DT": C.GATE_K_DT},
        geometry_note="Every protocol window is a multiple of the cell's own "
                      "T_cycle (P8': of L_inf). No absolute time constant "
                      "reaches a protocol; the bootstrap windows exist only to "
                      "measure the landmarks (B7).",
        gate_note=f"B5 resolution gate: cells with T_cycle < {C.GATE_K_DT}*DT, "
                  f"or whose anchor was measured on a buffer-bound record, are "
                  f"DROPPED. {prog['gated']:,} of {prog['consumed']:,} "
                  f"candidates ({prog['gated']/max(prog['consumed'],1)*100:.3f}%).",
        generator="pipeline_final/scripts/v3_build.py",
        subset_n_train=args.n_train, subset_n_val=args.n_val,
        seed=args.seed,
        candidates=prog["consumed"], gated=prog["gated"],
        seconds=round(prog["seconds"], 1),
    )
    with open(os.path.join(args.out, "izh_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    for path in (f_part, p_part, l_part):
        os.remove(path)
    print(f"[build] done: {n:,} rows in {prog['seconds']/3600:.2f}h -> {args.out}")


if __name__ == "__main__":
    main()
