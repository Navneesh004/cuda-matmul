"""Shared chart style: one colour per kernel, used identically in every chart."""
import matplotlib as mpl

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"   # roofs, reference lines

# Fixed categorical order: a kernel keeps its colour in every chart.
KERNEL_COLORS = {
    "naive": "#2a78d6",
    "coalesced": "#eb6834",
    "shared": "#1baf7a",
    "reg1d": "#eda100",
    "reg2d": "#e87ba4",
    "vectorized": "#008300",
    "tuned": "#6250d6",
    "cublas": "#e34948",
}
KERNEL_ORDER = list(KERNEL_COLORS)
LABELS = {
    "cpu": "CPU (1 thread)",
    "naive": "1 naive",
    "coalesced": "2 coalesced",
    "shared": "3 shared tiling",
    "reg1d": "4 1D reg tiling",
    "reg2d": "5 2D reg tiling",
    "vectorized": "6 vectorised",
    "tuned": "7 tuned",
    "cublas": "cuBLAS",
}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "text.color": TEXT,
        "legend.frameon": False,
        "font.size": 10,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })
