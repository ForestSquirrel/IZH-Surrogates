"""
figs.py -- manuscript figures for the Schulte 2022 reproduction (tex/figs/).

    $PY figs.py            # -> tex/figs/schulte_{noise,nonoise}.pdf, schulte_cell.pdf (+ .png)

Reads data/runs/{noise,nonoise}/ (run.py), data/sd_check_best.json (sd_check.py),
data/runs/gain_ctrl/ (gain control), data/gain.json (gain.py),
data/integration_cost.json (bench_integration.py). Fonts: Tinos (Times New Roman metrics); panels a, b, ...
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import spec as S
import bench_integration as BI
import tasks
import circuit as CI
import sd_check as SD

import texdirs as TD
OUT = TD.figs("schulte")
# Tinos: TrueType, metric-compatible with Times New Roman (embeds cleanly as Type 42;
# the TeX Gyre Termes OTF is CFF-based and produced font-type mismatches in the PDF)
FONT_DIR = "/usr/share/fonts/truetype/croscore"
for f in ("Regular", "Bold", "Italic", "BoldItalic"):
    font_manager.fontManager.addfont(os.path.join(FONT_DIR, f"Tinos-{f}.ttf"))
plt.rcParams.update({
    "font.family": "Tinos", "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 300})

POP_COL = {"E23": "#1f3b73", "E4": "#3f7fbf", "E5": "#7fb2e0",
           "I23": "#9e1b1b", "I4": "#d9452b", "I5": "#f29e80"}
POP_LAB = {"E23": "L2/3-E", "I23": "L2/3-I", "E4": "L4-E", "I4": "L4-I", "E5": "L5-E", "I5": "L5-I"}
C_T, C_S, C_M, C_P = "#222222", "#e07b00", "#b5651d", "#9a9a9a"
SEG = int(round(S.T_SEG / S.DT))
TRIAL = S.N_SEG * SEG
TASKS = ["tcl1 L23", "tcl2 L23", "tcl1(t-1) L23", "tcl2(t-1) L23", "XOR L23",
         "tcl1 L5", "tcl2 L5", "tcl1(t-1) L5", "tcl2(t-1) L5", "XOR L5"]


def task_label(k):
    t, ro = k.rsplit(" ", 1)
    t = t.replace("tcl1(t-1)", r"$tcl_1(t\!-\!\Delta t)$").replace("tcl2(t-1)", r"$tcl_2(t\!-\!\Delta t)$")
    t = t.replace("tcl1", r"$tcl_1(t)$").replace("tcl2", r"$tcl_2(t)$")
    return f"{t}\n{ro}"


def label(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom", ha="left")


# gain-matched surrogates (gain.py, '<view>@xe<g_E>:<g_I>') in data/runs/<cond>_matched; the
# untuned version (data/runs/<cond>) is kept with tex/archive/v1_untuned
RUN_SUFFIX = "_matched"


def alias(m):
    """'best@xe1.17:1.26' -> 'best' (one surrogate per view in a run)"""
    return m.partition("@")[0]


def results(cond):
    R = json.load(open(os.path.join(HERE, "data", "runs", cond + RUN_SUFFIX, "summary.json")))["results"]
    return {alias(m): r for m, r in R.items()}


def load(cond):
    d = os.path.join(HERE, "data", "runs", cond + RUN_SUFFIX)
    summ = json.load(open(os.path.join(d, "summary.json")))
    npz = {alias(m): np.load(os.path.join(d, f"{m}.npz")) for m in summ["results"]}
    summ["results"] = {alias(m): r for m, r in summ["results"].items()}
    return summ, npz


def spikes(z, run, t0, t1):
    """L1 record -> (times ms, neuron ids) inside [t0, t1) ms"""
    T, I = [], []
    for i in range(CI.N):
        t = z["sp_t"][run, i, :z["sp_n"][run, i]] * S.DT
        t = t[(t >= t0) & (t < t1)]
        T.append(t); I.append(np.full(len(t), i))
    return np.concatenate(T), np.concatenate(I)


def raster(ax, t, i, t0, title):
    col = np.empty(len(i), object)
    for p, idx in CI.IDX.items():
        col[np.isin(i, idx)] = POP_COL[p]
    ax.scatter(t - t0, i, s=0.5, c=list(col), marker="|", linewidths=0.4, rasterized=True)
    for k in range(1, S.N_SEG):
        ax.axvline(k * S.T_SEG, color="0.85", lw=0.4, zorder=0)
    ax.set_xlim(0, S.N_SEG * S.T_SEG)
    ax.set_ylim(CI.N, 0)
    ticks = [int(np.mean(v)) for v in CI.IDX.values()]
    ax.set_yticks(ticks, [POP_LAB[p] for p in CI.IDX])
    for tl, p in zip(ax.get_yticklabels(), CI.IDX):
        tl.set_color(POP_COL[p])
    ax.set_title(title, loc="left", fontsize=8)
    ax.tick_params(axis="y", length=0)


def fig_condition(cond):
    summ, npz = load(cond)
    R = summ["results"]
    a = summ["args"]
    t0 = (S.DISCARD_SEG * SEG + a["rec_from"] * TRIAL) * S.DT
    t1 = t0 + S.N_SEG * S.T_SEG
    fig = plt.figure(figsize=(7.1, 8.6))
    gs = GridSpec(5, 6, figure=fig, height_ratios=[0.5, 1.3, 0.85, 1.0, 1.0], hspace=0.55,
                  wspace=1.1, left=0.09, right=0.985, top=0.965, bottom=0.07)
    # (a) input
    ax = fig.add_subplot(gs[0, :])
    (ti, ci), y, _, (b1, b2) = tasks.make(np.random.default_rng(4000), S.N_TRAIN + S.N_TEST, bits=True)
    s0 = S.DISCARD_SEG + S.N_SEG * a["rec_from"]
    bits1 = (b1[s0:s0 + S.N_SEG], b2[s0:s0 + S.N_SEG])
    m = (ti >= t0) & (ti < t1)
    ax.scatter(ti[m] - t0, ci[m], s=0.6, c=np.where(ci[m] < S.N_CH, "#555555", "#2a8a5a"),
               marker="|", linewidths=0.5, rasterized=True)
    for k in range(1, S.N_SEG):
        ax.axvline(k * S.T_SEG, color="0.85", lw=0.4, zorder=0)
    for k in range(S.N_SEG):
        ax.text((k + 0.5) * S.T_SEG, -6, str(bits1[0][k]), ha="center", va="bottom", fontsize=6)
        ax.text((k + 0.5) * S.T_SEG, 2 * S.N_CH + 2, str(bits1[1][k]), ha="center", va="top", fontsize=6, color="#2a8a5a")
    ax.set_xlim(0, S.N_SEG * S.T_SEG); ax.set_ylim(2 * S.N_CH + 13, -12)
    ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
    ax.set_yticks([S.N_CH / 2, 1.5 * S.N_CH], ["stream 1", "stream 2"])
    ax.get_yticklabels()[1].set_color("#2a8a5a")
    ax.tick_params(axis="y", length=0)
    ax.set_xticklabels([])
    ax.set_title(f"input, trial {a['rec_from']} of run 0 (bits = template per 30 ms segment)", loc="left", fontsize=8)
    label(ax, "a", x=-0.075)
    # (b, c) rasters
    for j, (mname, ttl, lab) in enumerate((("target", "HH target network", "b"),
                                           ("best", "Izhikevich surrogate network", "c"))):
        ax = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
        t, i = spikes(npz[mname], 0, t0, t1)
        raster(ax, t, i, t0, ttl)
        ax.set_xlabel("time in trial (ms)")
        if j:
            ax.set_yticklabels([])
        label(ax, lab, x=-0.2 if j == 0 else -0.06)
    # (d) population rates
    ax = fig.add_subplot(gs[2, 0:2])
    bins = np.arange(0, S.N_SEG * S.T_SEG + 1, 10.0)
    for mname, ls, c in (("target", "-", None), ("best", "--", None)):
        t, i = spikes(npz[mname], 0, t0, t1)
        for p in ("E23", "E4", "E5"):
            sel = np.isin(i, CI.IDX[p])
            h, _ = np.histogram(t[sel] - t0, bins)
            ax.plot(bins[:-1] + 5, h / len(CI.IDX[p]) / 0.01, ls, color=POP_COL[p], lw=0.8)
    ax.plot([], [], "-", color="k", lw=0.8, label="HH"); ax.plot([], [], "--", color="k", lw=0.8, label="surrogate")
    ax.set_xlabel("time in trial (ms)"); ax.set_ylabel("rate (spikes/s)")
    ax.set_xlim(0, S.N_SEG * S.T_SEG); ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(frameon=False, loc="lower left", ncol=2, handlelength=1.6, bbox_to_anchor=(0, 0.98), borderaxespad=0)
    for j, p in enumerate(("E23", "E4", "E5")):
        ax.text(1.0, 0.98 - 0.09 * j, POP_LAB[p], color=POP_COL[p], transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    label(ax, "d", x=-0.3)
    # (e) voltage traces
    ax = fig.add_subplot(gs[2, 2:4])
    vt, vs = npz["target"]["v"][0], npz["best"]["v"][0]
    tt = np.arange(vt.shape[1]) * S.DT
    w = (tt >= 150) & (tt < 300)
    nE = CI.IDX["E23"][int(np.argmax([(vt[i] > -30).sum() for i in CI.IDX["E23"][:60]]))]
    ax.plot(tt[w], vt[nE, w], color=C_T, lw=0.6, label="HH")
    ax.plot(tt[w], vs[nE, w], color=C_S, lw=0.6, label="surrogate (mapped V)")
    ax.set_xlabel("time in trial (ms)"); ax.set_ylabel("V (mV)")
    ax.set_title(f"L2/3-E #{nE}: HH / surrogate", loc="left", fontsize=7)
    label(ax, "e", x=-0.3)
    # (f) per-neuron rate scatter (all runs)
    ax = fig.add_subplot(gs[2, 4:6])
    T_s = (S.DISCARD_SEG + S.N_SEG * a["trials"]) * S.T_SEG / 1000
    rt, rs = npz["target"]["counts"] / T_s, npz["best"]["counts"] / T_s
    for p, idx in CI.IDX.items():
        ax.scatter(rt[:, idx].ravel() + 0.1, rs[:, idx].ravel() + 0.1, s=1.2, color=POP_COL[p],
                   alpha=0.5, lw=0, rasterized=True, label=POP_LAB[p])
    lim = [0.1, 400]
    ax.plot(lim, lim, color="0.5", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    r = R["best"]["agreement"]["rate_corr_neurons"]
    ax.set_xlabel("HH rate + 0.1 (spikes/s)"); ax.set_ylabel("surrogate rate + 0.1")
    ax.text(0.04, 0.96, f"$r$ = {r:.3f}", transform=ax.transAxes, va="top", fontsize=7)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, 0.88), markerscale=3,
              handletextpad=0.1, borderaxespad=0.1, labelspacing=0.15)
    label(ax, "f", x=-0.32)
    # (g) kappa per task
    ax = fig.add_subplot(gs[3, :])
    x = np.arange(len(TASKS))
    bars = [("paper (Table 6)", C_P, {k: S.REF[k] for k in TASKS} if cond == "noise" else None, None),
            ("HH target", C_T, R["target"]["kappa"], R["target"]["kappa_sd"]),
            ("surrogate, best", C_S, R["best"]["kappa"], R["best"]["kappa_sd"]),
            ("surrogate, mean", C_M, R["mean"]["kappa"], R["mean"]["kappa_sd"])]
    bars = [b for b in bars if b[2] is not None]
    wdt = 0.8 / len(bars)
    for j, (lab, c, kv, sd) in enumerate(bars):
        off = (j - (len(bars) - 1) / 2) * wdt
        ax.bar(x + off, [kv[k] for k in TASKS], wdt, color=c, label=lab,
               yerr=None if sd is None else [sd[k] for k in TASKS], error_kw=dict(lw=0.5, capsize=0))
    ax.set_xticks(x, [task_label(k) for k in TASKS], fontsize=6.5)
    ax.set_ylabel(r"$\kappa$ (test)"); ax.set_ylim(0, 1.05)
    ax.axvline(4.5, color="0.7", lw=0.5)
    ax.legend(frameon=False, ncol=len(bars), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    label(ax, "g", x=-0.075)
    # (h) paired differences
    ax = fig.add_subplot(gs[4, 0:4])
    kr = lambda mm: np.array([[r_[k] for k in TASKS] for r_ in R[mm]["kappa_runs"]])
    for j, (mm, c) in enumerate((("best", C_S), ("mean", C_M))):
        d = kr(mm) - kr("target")
        ax.errorbar(x + (j - 0.5) * 0.25, d.mean(0), yerr=d.std(0, ddof=1) / np.sqrt(len(d)), fmt="o",
                    ms=3, color=c, lw=0.8, capsize=1.5, label=f"surrogate {mm} − HH")
    ax.axhline(0, color="0.5", lw=0.5); ax.axvline(4.5, color="0.7", lw=0.5)
    ax.set_xticks(x, [task_label(k) for k in TASKS], fontsize=6.5)
    ax.set_ylabel(r"$\Delta\kappa$ (paired, mean $\pm$ SE)")
    ax.legend(frameon=False, loc="lower left", fontsize=6.5, ncol=2, bbox_to_anchor=(0, 0.98), borderaxespad=0)
    label(ax, "h", x=-0.11)
    # (i) population rate ratio + compute
    ax = fig.add_subplot(gs[4, 4:6])
    pops = list(CI.IDX)
    rr = R["best"]["agreement"]["rate_ratio"]
    ax.bar(np.arange(6), [rr[p] for p in pops], color=[POP_COL[p] for p in pops])
    ax.axhline(1, color="0.5", lw=0.5)
    ax.set_xticks(np.arange(6), [POP_LAB[p] for p in pops], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("rate ratio (surrogate / HH)"); ax.set_ylim(0, 1.4)
    ct, cs = BI.cost("target"), BI.cost("best")
    ax.set_title(f"integration*: {ct / cs:.1f}× cheaper", loc="left", fontsize=6.5)
    label(ax, "i", x=-0.35)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"schulte_{cond}.{ext}"))
    plt.close(fig)


def fig_cell():
    js = json.load(open(os.path.join(HERE, "data", "sd_check_best.json")))
    C = js["cells"]
    W = np.array(js["widths_ms"]); seps = np.array(js["seps_ms"])
    fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.4))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.1, wspace=0.42, hspace=0.6)
    # (a) strength-duration
    ax = axs[0, 0]
    for t, ls in (("E", "-"), ("I", ":")):
        ax.loglog(W, C[f"HH-{t}"]["sd_pA"], ls, color=C_T, lw=0.9, label=f"HH-{t}")
        ax.loglog(W, C[f"sur-{t}"]["sd_pA"], ls, color=C_S, lw=0.9, label=f"surrogate-{t}")
    ax.set_xlabel("pulse width (ms)"); ax.set_ylabel("threshold current (pA)")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("strength–duration from rest", loc="left")
    label(ax, "a", x=-0.3)
    # (b) PSP waveforms
    ax = axs[0, 1]
    cells = SD.cells(js["view"])
    for t, ls in (("E", "-"), ("I", ":")):
        for name, c in ((f"HH-{t}", C_T), (f"sur-{t}", C_S)):
            ce = cells[name]
            g = 0.3 * C[name]["g_single_nS"]
            _, v = ce["sim"](ce["y0"], ce["P"], ce["va"], ce["vb"], ce["iscale"], 60.0, 0., 0., 0.,
                             np.array([5.0]), np.float32(g), np.float32(3.0), np.float32(0.0), True)
            tt = np.arange(len(v)) * SD.DT
            ax.plot(tt - 5, v - v[int(4 / SD.DT)], ls, color=c, lw=0.9)
    ax.set_xlim(-2, 40); ax.set_xlabel("time after EPSC onset (ms)"); ax.set_ylabel(r"$\Delta V$ (mV)")
    ax.set_title(r"PSP, $g = 0.3\,g_{\mathrm{thr}}$", loc="left")
    label(ax, "b", x=-0.3)
    # (c) pair coincidence
    ax = axs[0, 2]
    for t, ls in (("E", "-"), ("I", ":")):
        for name, c in ((f"HH-{t}", C_T), (f"sur-{t}", C_S)):
            g = np.array(C[name]["g_pair_nS"])
            ax.plot(seps, g / g[0], ls, color=c, lw=0.9, marker="o", ms=2)
    ax.axhline(1.5, color="0.7", lw=0.5)
    ax.set_xlabel("EPSC pair separation (ms)"); ax.set_ylabel("pair threshold / at 0 ms")
    ax.set_title("coincidence window", loc="left")
    label(ax, "c", x=-0.3)
    # (d) threshold table as bars
    ax = axs[1, 0]
    q = [("rheobase_pA", "rheobase\n(×100 pA)", 100), ("chronaxie_ms", "chronaxie\n(ms)", 1),
         ("g_single_nS", "1-EPSC thr.\n(×10 nS)", 10)]
    x = np.arange(len(q))
    for j, (name, c) in enumerate((("HH-E", C_T), ("sur-E", C_S), ("HH-I", "#666666"), ("sur-I", "#f0b060"))):
        ax.bar(x + (j - 1.5) * 0.2, [C[name][k] / s for k, _, s in q], 0.2, color=c, label=name.replace("sur", "surr."))
    ax.set_xticks(x, [l for _, l, _ in q], fontsize=6.5)
    ax.set_ylim(0, 11)
    ax.legend(frameon=False, fontsize=5.5, ncol=4, loc="upper center", columnspacing=0.8, handlelength=1.2)
    ax.set_title("single-cell summary", loc="left")
    label(ax, "d", x=-0.3)
    # (e) gain control: XOR change against the excitatory gain on E cells, two scalings
    ax = axs[1, 1]
    full = lambda c: json.load(open(os.path.join(HERE, "data", "runs", c, "summary.json")))["results"]
    ctl = {f: json.load(open(os.path.join(HERE, "data", "runs", "gain_ctrl", f"summary_{f}.json")))["results"]
           for f in ("gain113", "gain146")}
    A = [(1.0, ctl["gain146"], "best"), (1.13, ctl["gain113"], "best@x1.13:1.0"), (1.46, ctl["gain146"], "best@gain")]
    gE = json.load(open(os.path.join(HERE, "data", "gain.json")))["best|noise"]["g_E"]
    m = [k for k in full("noise_matched") if k.startswith("best")][0]
    B = [(1.0, full("noise"), "best"), (gE, full("noise_matched"), m)]
    for (pts, col, mk, lab) in ((A, "#999999", "o", "all input, E cells only"), (B, C_S, "*", "predicted: excitatory, E and I cells")):
        for t, ls in (("XOR L23", "-"), ("XOR L5", "--")):
            ax.plot([g for g, _, _ in pts], [r[k]["kappa"][t] - r["target"]["kappa"][t] for g, r, k in pts], ls,
                    marker=mk, color=col, ms=4 if mk == "*" else 3, lw=0.9, label=f"{lab}" if t == "XOR L23" else None)
    ax.axhline(0, color=C_T, lw=0.6)
    ax.set_xlabel("synaptic gain onto E cells"); ax.set_ylabel(r"$\Delta\kappa$ XOR (surrogate $-$ HH)")
    ax.set_ylim(-0.25, 0.1)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    label(ax, "e", x=-0.3)
    # (f) agreement & compute across conditions
    ax = axs[1, 2]
    rows = []
    for cond in ("noise", "nonoise"):
        R = results(cond)
        ag = R["best"]["agreement"]
        rows.append((cond, ag["cka_states"], ag["rate_corr_neurons"]))
    x = np.arange(2)
    for j, (cond, *v) in enumerate(rows):
        ax.bar(x + (j - 0.5) * 0.35, v, 0.35, color=["#555555", "#aaaaaa"][j], label=cond.replace("nonoise", "no noise"))
    ax.set_xticks(x, ["state CKA", "rate corr.\n(neurons)"], fontsize=6.5)
    ax.set_ylim(0, 1.1); ax.legend(frameon=False, fontsize=6)
    ax.set_title("surrogate vs HH network", loc="left")
    label(ax, "f", x=-0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"schulte_cell.{ext}"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_condition("noise")
    fig_condition("nonoise")
    fig_cell()
    print("wrote", OUT)
