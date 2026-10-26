"""Shared figure style.

* Categorical colours: a colour-vision-deficiency-checked 8-hue palette, always
  assigned in the same fixed order (colour follows the entity, never its rank).
* Sequential data (e.g. heatmaps): a single-hue light-to-dark ramp.
* One y-axis per panel (no dual axes); recessive grid; labelled axes with units.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
SEQ_CMAP = "Blues"

# Fixed colour per habitat / status group so every figure agrees.
HABITAT_COLORS = dict(zip(["Dry forest", "Wet forest", "Agricultural", "Peri-urban",
                           "Urban", "Mangrove", "Beach"], PALETTE))
STATUS_COLORS = dict(zip(["Endemic", "Native", "Exotic", "Migrant", "Seabird"], PALETTE))


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.labelcolor": INK2, "axes.edgecolor": "#c8c7c2",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def savefig(fig, name):
    """Save to outputs/figures/<name>.png and return the path."""
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
