"""Figure style for the progress slides: large fonts, few colours, no titles, editable SVG text."""
import matplotlib as mpl

FULL, HALF = (10.0, 4.2), (5.0, 3.8)          # inches, for a 16:9 slide

INK, INK2, GRID = "#1a1a1a", "#555555", "#e2e2e2"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY, LIGHT = "#9a9994", "#d3d2cd"

# synchronization classes of the collectives
CLASSES = ["mbarrier-only", "single-exchange", "round-synchronous"]
CLASS_COL = dict(zip(CLASSES, [BLUE, ORANGE, AQUA]))

CYC_PER_NS = 1.635                 # locked SM clock: cycles per ns
GBS = lambda bpc: bpc * CYC_PER_NS  # B/cycle -> GB/s


def apply():
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.transparent": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 10.5, "legend.frameon": False, "legend.handlelength": 1.2,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "patch.linewidth": 0, "lines.linewidth": 1.6,
        "xtick.major.size": 0, "ytick.major.size": 3,
    })


apply()
