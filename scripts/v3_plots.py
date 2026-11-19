"""
v3_plots.py -- per-protocol target-vs-surrogate figures for the v3 surrogate.

One figure per protocol per target. Each shows the TARGET's voltage trace above
the SURROGATE's, with the injected current overlaid on both.

THE X AXIS IS EACH SIDE'S OWN T_cycle, NOT MILLISECONDS. This is not cosmetic.
Every v3 protocol window is a multiple of the cell's own characteristic cycle
(B7), and the two sides have different tau by construction -- for an
out-of-family target they may not even share time units (Hodgkin-Huxley ms vs
Hindmarsh-Rose dimensionless). Plotting against absolute time would show the
scale mismatch and hide the spike-train comparison, which is the thing being
judged. Normalizing each side by its own anchor makes the stimulus events align
by construction, because they are DEFINED in cycle units.

The seven panels follow the v3 protocol set, so two differ from v2's:
  * `ramp` is a SINGLE ramp with the hold retained (P5 -> gamma_ramp), not a
    slope grid -- the grid was removed in §3.2
  * `recovery` is new (P8' -> A_rec) and its geometry is in L_inf units, not
    T_cycle, matching §39.1
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from v3_core import alloc_buffers
from v3_landmarks import landmarks as izh_landmarks
from v3_models import izh_rhs_v3

N_CYC = 12.0          # cycles of steady behaviour to show


def protocols():
    """key, title, stim(i_rh, tc, linf, t_on) -> callable, span in cycles."""
    G = C.GAMMA_STEP

    def step(g):
        return lambda irh, tc, li, ton: (lambda t: g * irh if t >= ton else 0.0)

    def zero():
        return lambda irh, tc, li, ton: (lambda t: 0.0)

    def pulsepair():
        def mk(irh, tc, li, ton):
            amp = C.GAMMA_PULSE * irh
            w = C.K_PULSE_WIDTH * tc
            t1, t2 = ton, ton + 3.0 * tc
            return lambda t: amp if (t1 <= t < t1 + w or t2 <= t < t2 + w) else 0.0
        return mk

    def ramp():
        def mk(irh, tc, li, ton):
            amp = C.RAMP_GAMMA_F * irh
            dur = C.K_RAMP * tc
            def s(t):
                if t < ton:
                    return 0.0
                if t >= ton + dur:
                    return amp          # HOLD retained -- §3.3
                return amp * (t - ton) / dur
            return s
        return mk

    def hyst():
        def mk(irh, tc, li, ton):
            amp = C.HYST_GAMMA_MAX * irh
            total = 20.0 * tc
            half = total * 0.5
            def s(t):
                if t < ton:
                    return 0.0
                tt = t - ton
                if tt >= total:
                    return 0.0
                return amp * (tt / half) if tt < half else amp * ((total - tt) / half)
            return s
        return mk

    def recovery():
        """P8': conditioning pulse, then a sustained test step at 1.2*i_rh.
        Geometry in L_inf units (§39.1), which is why `li` is passed at all."""
        def mk(irh, tc, li, ton):
            ac = C.GAMMA_PULSE * irh
            at = C.GAMMA_LINF * irh
            w = max(C.REC_W_COND * li, 5.0 * C.DT)
            delay = 2.0 * li
            t2 = ton + delay
            def s(t):
                if ton <= t < ton + w:
                    return ac
                if t >= t2:
                    return at
                return 0.0
            return s
        return mk

    return [
        dict(key="rheobase",  title="Rheobase step  γ=1  (P0)",      make=step(1.0),  span=N_CYC),
        dict(key="refstep",   title=f"Reference step  γ={G:g}  (P1/P3)", make=step(G), span=N_CYC),
        dict(key="pulsepair", title="Pulse pair  (P4 → δ*, ϱ_res)",  make=pulsepair(), span=8.0),
        dict(key="ramp",      title="Single ramp + hold  (P5 → γ_ramp)", make=ramp(), span=N_CYC),
        dict(key="hysteresis",title="Hysteresis triangle  (P7 → η)", make=hyst(),     span=21.0),
        dict(key="recovery",  title="Paired-pulse recovery  (P8′ → A_rec)", make=recovery(), span=8.0),
        dict(key="spont",     title="Spontaneous  I=0  (P10 → ν_spont)", make=zero(), span=N_CYC),
    ]


def trace(side, proto, t_on_cyc=1.0):
    """Simulate one side under one protocol. Returns (x_cycles, v, I, spikes)."""
    from run_surrogate import simulate_trace
    tc, irh, li = side["tc"], side["i_rh"], side["l_inf"]
    ton = t_on_cyc * tc
    stim = proto["make"](irh, tc, li if li > C.SENT_CUT else tc, ton)
    t_end = ton + proto["span"] * tc
    ts, vs, is_, sp = simulate_trace(
        side["rhs"], side["params"], side["y0"], stim, side["v_th"], side["h"],
        t_end, reset_v=side.get("reset_v"), reset_du=side.get("reset_du"))
    return (ts - ton) / tc, vs, is_, (sp - ton) / tc


def _theta_str(th):
    a, b, c, d = (float(x) for x in th)
    return f"a={a:.4g}  b={b:.4g}  c={c:.4g}  d={d:.4g}"


def render(name, proto, model, surr, out_path, subtitle=""):
    xm, vm, im, spm = trace(model, proto)
    xs, vs, is_, sps = trace(surr, proto)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True)
    labs = [f"target · {model['label']}",
            f"surrogate · {surr['label']}   θ = {_theta_str(surr['params'][:4])}"]
    for ax, (x, v, I, sp, lab, col, vth) in zip(
            axes, [(xm, vm, im, spm, labs[0], "#1f77b4", model["v_th"]),
                   (xs, vs, is_, sps, labs[1], "#2ca02c", surr["v_th"])]):
        ax.plot(x, v, color=col, lw=0.9)
        # spike threshold: each side detects on its OWN v_th, and for an
        # out-of-family target that value is a modelling choice, not a constant.
        ax.axhline(vth, color="#d62728", lw=0.8, ls="--", alpha=0.7)
        ax.text(0.995, vth, f" v_th={vth:g}", color="#d62728", fontsize=7,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_ylabel("V")
        ax2 = ax.twinx()
        ax2.plot(x, I, color="#999999", lw=0.7, alpha=0.55)
        ax2.set_ylabel("I", color="#999999", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7, colors="#999999")
        for s in sp:
            ax.axvline(s, color=col, alpha=0.16, lw=0.6)
        ax.set_title(f"{lab}   ({len(sp)} spikes)", fontsize=9, loc="left")
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("time  (units of each side's own $T_{cycle}$)")
    fig.suptitle(f"{name} — {proto['title']}" + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def izh_side(theta, label, bufs):
    th = np.ascontiguousarray(np.asarray(theta, float))
    lm = izh_landmarks(izh_rhs_v3, th, bufs)
    if lm["T_cycle"] < C.SENT_CUT or lm["i_rh"] < C.SENT_CUT:
        return None
    return dict(rhs=izh_rhs_v3, params=th, y0=np.array([C.V_REST, th[1]*C.V_REST]),
                v_th=C.V_TH, h=C.DT, tc=lm["T_cycle"], i_rh=lm["i_rh"],
                l_inf=lm["L_inf"], reset_v=th[2], reset_du=th[3], label=label)


def slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["infamily", "offfamily", "both"], default="both")
    ap.add_argument("--out", default=PLOTS)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    bufs = alloc_buffers()
    protos = protocols()
    n = 0

    if args.which in ("infamily", "both"):
        rows = json.load(open(os.path.join(RESULTS, "v3_infamily.json")))
        for r in rows:
            if r.get("status") != "ok":
                continue
            if args.only and args.only.lower() not in r["name"].lower():
                continue
            m = izh_side(r["theta_true"], "θ_true", bufs)
            s = izh_side(r["theta_best"], "best draw", bufs)
            if m is None or s is None:
                print(f"  skip {r['name']} (no anchor)"); continue
            sub = (f"d²={r['d2']:.3g} ({r['n_dims']}d)   τ={r['tau']:.3f}   "
                   f"k_I={r['k_I']:.3f}   w₁_log={r['w1_log']:.3g}")
            d = os.path.join(args.out, "infamily", slug(r["name"]))
            for p in protos:
                render(r["name"], p, m, s,
                       os.path.join(d, f"v3_{slug(r['name'])}_{p['key']}.png"), sub)
                n += 1
            print(f"  [infamily] {r['name']}  -> {d}", flush=True)

    if args.which in ("offfamily", "both"):
        from v3_an_zoo import targets as zoo, reset_targets
        specs = dict(zoo())
        # AdEx is a RESET model: 2-D state, threshold 30, reset V -> params[2],
        # w -> w + params[3]. Its trace side is built like the Izhikevich
        # surrogate's, not like a reset-free target's -- passing it through the
        # reset-free branch would integrate it without the reset and diverge.
        resets = {n: (rhs, pr, rh) for n, rhs, pr, rh in reset_targets()}
        rows = json.load(open(os.path.join(RESULTS, "v3_offfamily.json")))
        for r in rows:
            if r.get("status") != "ok":
                continue
            if args.only and args.only.lower() not in r["name"].lower():
                continue
            if r["name"] in resets:
                rhs_m, pr_m, _ = resets[r["name"]]
                m = dict(rhs=rhs_m, params=np.asarray(pr_m, float),
                         y0=np.array([C.V_REST, pr_m[1] * C.V_REST]),
                         v_th=C.V_TH, h=C.DT, reset_v=pr_m[2], reset_du=pr_m[3],
                         tc=r["T_cycle_model"], i_rh=r["i_rh_model"],
                         l_inf=C.NAN_SENTINEL, label=r["name"])
            elif r["name"] in specs:
                sp = specs[r["name"]]
                m = dict(rhs=sp.rhs, params=sp.params, y0=sp.y0, v_th=sp.v_th,
                         h=sp.h, tc=r["T_cycle_model"], i_rh=r["i_rh_model"],
                         l_inf=C.NAN_SENTINEL, label=r["name"])
            else:
                print(f"  skip {r['name']} (no spec)"); continue
            b = r["best"]
            s = izh_side(b["theta"], "best draw", bufs)
            if s is None:
                print(f"  skip {r['name']} (surrogate has no anchor)"); continue
            m["l_inf"] = s["l_inf"]
            w1s = b.get("w1_shape")
            w1txt = "n/a" if w1s is None or not np.isfinite(w1s) else f"{w1s:.3g}"
            sub = (f"d²={b['d2']:.3g} ({b['n_dims']}d)   τ={b['tau']:.3f}   "
                   f"k_I={b['k_I']:.3f}   w₁_shape={w1txt}   "
                   f"support={r['support']:.3f}")
            d = os.path.join(args.out, "offfamily", slug(r["name"]))
            for p in protos:
                render(r["name"], p, m, s,
                       os.path.join(d, f"v3_{slug(r['name'])}_{p['key']}.png"), sub)
                n += 1
            print(f"  [offfamily] {r['name']}  -> {d}", flush=True)

    print(f"\nwrote {n} figures under {args.out}")


if __name__ == "__main__":
    main()
