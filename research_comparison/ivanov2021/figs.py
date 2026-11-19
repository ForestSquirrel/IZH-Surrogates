"""
figs.py -- manuscript figures for the Ivanov 2021 zoo comparison (MNIST): the paper's LIF, each zoo
neuron with its time-scaled input layer (<zoo>@fastin), and its surrogate with the surrogate
input layer; identical topology and input code throughout. Static LSM: gain-matched surrogates
(sur:<zoo>@fastin, w_s = g w*_target, gain.py).

    ivanov_static.pdf   static LSM (baseline wiring of the paper)
        a  one test digit and its spike rasters (input layer grey, liquid E blue / I red), network 0
        b  test accuracy (3 networks), c  liquid rates, d  surrogate-target agreement on the test
        set (state CKA, per-neuron count correlation, per-sample rate correlation over neurons),
        e  compute (GPU wall per sample), f  confusion matrices (row-normalised errors)
    $PY figs.py [--campaign DIR] [--out DIR]     # default: data/..., tex/figs/
Missing conditions are skipped (smoke data). Fonts: Tinos (Times metrics); panels a, b, ...
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import spec as S
import showcase as SC
from lsmkit.metrics import agreement as AG

FONT_DIR = "/usr/share/fonts/truetype/croscore"
for f in ("Regular", "Bold", "Italic", "BoldItalic"):
    font_manager.fontManager.addfont(os.path.join(FONT_DIR, f"Tinos-{f}.ttf"))
plt.rcParams.update({
    "font.family": "Tinos", "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 300})

ZOO = ("hh", "ml_c1", "adex_adapting")
ZLAB = {"hh": "HH", "ml_c1": "ML C1", "adex_adapting": "AdEx"}
ZCOL = {"lif": "#222222", "hh": "#1f5fa8", "ml_c1": "#2a8a5a", "adex_adapting": "#b5651d"}


STATIC_FINAL = ["lif"] + [f"{z}@fastin" for z in ZOO] + [f"sur:{z}@fastin" for z in ZOO]  # gain-matched (gain.py)
C_IN, C_E, C_I = "#9a9a9a", "#1f3b73", "#c0392b"
NETS = 3


def zoo_of(c):
    return "lif" if c == "lif" else c.split(":")[-1].split("@")[0]


def is_sur(c):
    return c.startswith("sur")


def clab(c, short=False):
    if c == "lif":
        return "LIF"
    z = ZLAB[zoo_of(c)]
    return (f"{z} surr." if short else f"{z} surrogate") if is_sur(c) else z


def ccol(c):
    return ZCOL[zoo_of(c)]


def label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom", ha="left")


def safe(m):
    return m.replace(":", "_").replace("*", "x").replace("@", "_at_")


# ---------------------------------------------------------------- loading
def static_runs(cdir, cond):
    """-> list of (result dict with 'confusion', test counts npz) for the networks on disk"""
    import campaign as CA
    w = CA.cond_w(cdir, cond)
    if w is None:
        return []
    d = CA.full_dir(cdir, cond, w)
    out = []
    for k in range(NETS):
        f = os.path.join(d, f"net{k}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        if "confusion" not in r and os.path.exists(os.path.join(d, f"net{k}_pred.json")):
            r["confusion"] = json.load(open(os.path.join(d, f"net{k}_pred.json")))["confusion"]
        out.append((r, np.load(os.path.join(d, f"net{k}_test_counts.npz"))))
    return out


# ---------------------------------------------------------------- panels
def raster(ax, sc, title):
    ax.scatter(sc["liq_t"], sc["liq_i"], s=0.15, c=np.where(sc["types"][sc["liq_i"]] > 0, C_E, C_I),
               marker="|", linewidths=0.25, rasterized=True)
    ax.scatter(sc["in_t"], 1030 + 0.4 * sc["in_i"], s=0.15, c=C_IN, marker="|", linewidths=0.25, rasterized=True)
    ax.set_xlim(0, S.T_SAMPLE); ax.set_ylim(-10, 1030 + 0.4 * 784 + 10)
    ax.set_yticks([500, 1030 + 0.4 * 392], ["liquid", "input"], fontsize=6)
    ax.tick_params(axis="y", length=0)
    nr = len(sc["liq_t"]) / 1000 / (S.T_SAMPLE / 1000)
    ax.set_title(f"{title}  ({nr:.0f} Hz)", loc="left", fontsize=7, color=title_col(title))


def title_col(t):
    for c in STATIC_FINAL:
        if clab(c) == t:
            return ccol(c)
    return "k"


def sur_key(runs, z):
    """the surrogate condition of zoo model z in runs (sur:<z>@fastin gain-matched or sur*:<z>@fastin)"""
    return next((c for c in runs if is_sur(c) and zoo_of(c) == z), None)


def bars(ax, runs, key, scale=100.0, ylabel="", ref=None):
    C = list(runs)
    x = np.arange(len(C))
    for j, c in enumerate(C):
        v = np.array([r[key] for r, _ in runs.get(c, [])]) * scale
        if not len(v):
            continue
        ax.bar(j, v.mean(), 0.7, color=ccol(c), alpha=0.45 if is_sur(c) else 0.9,
               hatch="////" if is_sur(c) else None, edgecolor=ccol(c), linewidth=0.5)
        ax.errorbar(j, v.mean(), v.std(), color="k", lw=0.6, capsize=1.5)
        ax.scatter(np.full(len(v), j) + np.linspace(-0.15, 0.15, len(v)), v, s=3, c="k", zorder=3)
    if ref is not None:
        ax.axhline(ref, color="k", lw=0.5, ls=":")
    ax.set_xticks(x, [clab(c, True) for c in C], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel(ylabel)


def rates(ax, runs):
    data, pos, cols = [], [], []
    C = list(runs)
    for j, c in enumerate(C):
        if c in runs and runs[c]:
            r = runs[c][0][1]["counts"].astype(float).mean(0) / (S.T_SAMPLE / 1000)   # per-neuron mean rate, net 0
            data.append(np.log10(np.maximum(r, 0.1))); pos.append(j); cols.append(ccol(c))
    if data:
        vp = ax.violinplot(data, pos, widths=0.8, showmedians=True, showextrema=False)
        for b, col in zip(vp["bodies"], cols):
            b.set_facecolor(col); b.set_alpha(0.5); b.set_edgecolor("none")
        vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(0.6)
    ax.set_xticks(np.arange(len(C)), [clab(c, True) for c in C], rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks([-1, 0, 1, 2], ["≤0.1", "1", "10", "100"])
    ax.set_ylabel("neuron rate (Hz, test mean)")


def agreement(runs):
    """per zoo model: CKA, median per-neuron count r, mean per-sample rate r over neurons (nets averaged)"""
    out = {}
    for z in ZOO:
        t, s = runs.get(f"{z}@fastin", []), runs.get(sur_key(runs, z), [])
        vals = []
        for (_, zt), (_, zs) in zip(t, s):
            ct, cs = zt["counts"].astype(float), zs["counts"].astype(float)
            vals.append((AG.cka(ct, cs), np.nanmedian(AG.count_corr(ct, cs)), np.nanmean(AG.rate_corr_neurons(ct, cs))))
        if vals:
            out[z] = np.array(vals)
    return out


def agree_panel(ax, runs):
    ag = agreement(runs)
    names = ["CKA", "count $r$", "rate $r$"]
    for j, z in enumerate(ZOO):
        if z not in ag:
            continue
        v = ag[z]
        x = np.arange(3) + (j - 1) * 0.26
        ax.bar(x, v.mean(0), 0.24, color=ZCOL[z], alpha=0.8, label=ZLAB[z])
        ax.errorbar(x, v.mean(0), v.std(0), fmt="none", color="k", lw=0.6, capsize=1.2)
    ax.set_xticks(np.arange(3), names, fontsize=6.5)
    ax.set_ylim(-0.05, 1.05); ax.axhline(0, color="k", lw=0.4); ax.set_ylabel("surrogate vs target")
    ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12), handlelength=1,
              columnspacing=0.7)


def compute_panel(ax, runs):
    """neuron-integration cost of each liquid (bench_integration.py: no synapses, no input); the
    LSM wall time adds network overhead, which is not shown (footnote of the figure)"""
    import bench_integration as BI
    C = list(runs)
    cost = {c: BI.cost(c) for c in C if runs.get(c)}
    for j, c in enumerate(C):
        if c in cost:
            ax.bar(j, cost[c], 0.7, color=ccol(c), alpha=0.45 if is_sur(c) else 0.9,
                   hatch="////" if is_sur(c) else None, edgecolor=ccol(c), linewidth=0.5)
    for z in ZOO:
        t, s = cost.get(f"{z}@fastin"), cost.get(sur_key(runs, z))
        if t and s:
            ax.text(C.index(sur_key(runs, z)), s + 0.04 * max(cost.values()), f"{t / s:.1f}×", ha="center", va="bottom",
                    fontsize=5.5, rotation=90)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_xticks(np.arange(len(C)), [clab(c, True) for c in C], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("integration* (GPU ms / sample)")


def confusions(fig, spec, runs):
    C = list(runs)
    g = GridSpecFromSubplotSpec(1, len(C), subplot_spec=spec, wspace=0.12)
    ims = None
    for j, c in enumerate(C):
        ax = fig.add_subplot(g[0, j])
        cm = [np.asarray(r["confusion"], float) for r, _ in runs.get(c, []) if "confusion" in r]
        if cm:
            m = np.sum(cm, 0)
            m = m / np.maximum(m.sum(1, keepdims=True), 1)
            np.fill_diagonal(m, np.nan)                         # errors only
            ims = ax.imshow(m * 100, cmap="magma_r", vmin=0, vmax=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(clab(c, True), fontsize=6.5, color=ccol(c))
        if j == 0:
            ax.set_ylabel("true digit", fontsize=6.5); ax.set_xlabel("predicted", fontsize=6.5)
    return ims


def fig_family(family, runs, sc, out, digit):
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2.3, 1.25, 0.62], hspace=0.62)
    top = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0], wspace=0.3, hspace=0.32)
    order = [["lif"] + [f"{z}@fastin" for z in ZOO], [None] + [sur_key(runs, z) for z in ZOO]]
    first = True
    for i, row in enumerate(order):
        for j, c in enumerate(row):
            ax = fig.add_subplot(top[i, j])
            if c is None:
                ax.imshow(digit.reshape(28, 28), cmap="Greys"); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"test image {SC.SAMPLE - 60000} (digit {digit_label})", loc="left", fontsize=7)
                for sp in ax.spines.values():
                    sp.set_visible(True)
                continue
            if c in sc:
                raster(ax, sc[c], clab(c))
            else:
                ax.text(0.5, 0.5, "not run", ha="center", transform=ax.transAxes); ax.set_title(clab(c), loc="left", fontsize=7)
            if i == 1:
                ax.set_xlabel("time (ms)")
            else:
                ax.set_xticklabels([])
            if first:
                label(ax, "a", x=-0.3); first = False
    mid = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.55, width_ratios=[1.1, 1.1, 1.0, 1.1])
    ax = fig.add_subplot(mid[0]); label(ax, "b", x=-0.32)
    ref = S.REF["mnist"]["top"]
    bars(ax, runs, "test", ylabel="test accuracy (%)", ref=ref)
    lo = min([r["test"] * 100 for v in runs.values() for r, _ in v] + [ref]) - 1.5
    ax.set_ylim(max(lo, 0), 100)
    ax = fig.add_subplot(mid[1]); label(ax, "c", x=-0.32); rates(ax, runs)
    ax = fig.add_subplot(mid[2]); label(ax, "d", x=-0.3); agree_panel(ax, runs)
    ax = fig.add_subplot(mid[3]); label(ax, "e", x=-0.32); compute_panel(ax, runs)
    ims = confusions(fig, gs[2], runs)
    fig.text(0.075, 0.205, "f", fontweight="bold", fontsize=10)
    if ims is not None:
        cax = fig.add_axes([0.93, 0.07, 0.008, 0.1])
        cb = fig.colorbar(ims, cax=cax); cb.set_label("% of row", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    fig.text(0.08, 0.015, "* Neuron integration only (liquid without synapses or input; × = target / surrogate). "
             "Wall-clock time of the LSM may vary with the network overhead\n   of synapses, signal propagation and the code base.",
             fontsize=5.5, color="0.3", va="bottom")
    save(fig, out, f"ivanov_{family}")


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".pdf"))


CONF_COL = ["#bdbdbd", "#8c8c8c", "#525252", "#d95f02"]


def main():
    global digit_label
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=os.path.join(HERE, "data", "campaign_mnist"))
    ap.add_argument("--out", default=None, help="all figures into one dir (default: tex_comparison/<section>/figs)")
    a = ap.parse_args()
    import texdirs as TD
    out_st = a.out or TD.figs("ivanov_static")
    cache = a.out or os.path.join(HERE, "data")          # showcase cache: <cache>/showcase/*.npz
    import run as R
    data, y, nin = R.load("mnist")
    digit_label = int(y[SC.SAMPLE])
    digit = data[SC.SAMPLE]
    st_runs = {c: static_runs(a.campaign, c) for c in STATIC_FINAL}
    st_sc = {c: SC.static(a.campaign, c, cache, data, nin) for c in STATIC_FINAL if st_runs[c]}
    fig_family("static", st_runs, st_sc, out_st, digit)


if __name__ == "__main__":
    main()
