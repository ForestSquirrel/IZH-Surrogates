"""
style.py -- house figure style (import for the side effect): Times-like serif + STIX math,
8.5 pt base, 7.5 pt ticks and legend.

Times New Roman is used when matplotlib can find it. To point it at a font folder that holds
times.ttf / timesbd.ttf / timesi.ttf / timesbi.ttf, set FONT_DIR; otherwise the fallbacks in
font.serif apply (Nimbus Roman, TeX Gyre Termes, STIX).
"""
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

_FONT_DIR = os.environ.get("FONT_DIR", "")
if _FONT_DIR:
    for f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        if os.path.exists(os.path.join(_FONT_DIR, f)):
            font_manager.fontManager.addfont(os.path.join(_FONT_DIR, f))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "STIX"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
})
