"""Shared figure style for the report: paper-ready, light surface, few colours.

Rules used by every figure function:
- widths: COL (one column, 3.4 in) or FULL (two columns, 7.0 in);
- categorical colours in the fixed order of PAL, never cycled; context data in GREY;
- no titles inside figures (the notebook text introduces them), panel letters via `panel()`;
- cycles on the left axis, nanoseconds as a secondary axis via `ns_axis()` (same quantity, 1635 MHz);
- reference levels as dotted lines via `ref_line()`, labelled directly.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

SM_MHZ = 1635.0           # locked SM clock: cycles / 1.635 = ns
CYC_PER_NS = SM_MHZ / 1e3

COL, FULL = 3.4, 7.0      # figure widths in inches

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = PAL
GREY, LIGHTGREY = "#9a9994", "#d3d2cd"


def apply():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.titlesize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "lines.markersize": 4, "patch.linewidth": 0,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.3,
        "axes.prop_cycle": mpl.cycler(color=PAL),
        "image.cmap": "Blues",
    })


def fig(ncols=1, nrows=1, width=COL, height=2.3, **kw):
    f, ax = plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True, **kw)
    return f, ax


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, fontweight="bold", color=INK)


def ns_axis(ax, which="y", label="ns"):
    """Secondary axis in ns for an axis in SM cycles (same quantity, not a second measure)."""
    fwd, inv = (lambda c: c / CYC_PER_NS), (lambda n: n * CYC_PER_NS)
    sec = ax.secondary_yaxis("right", functions=(fwd, inv)) if which == "y" \
        else ax.secondary_xaxis("top", functions=(fwd, inv))
    sec.set_ylabel(label) if which == "y" else sec.set_xlabel(label)
    sec.tick_params(colors=INK2)
    sec.spines[("right" if which == "y" else "top")].set_visible(True)
    sec.spines[("right" if which == "y" else "top")].set_color(INK2)
    return sec


def ref_line(ax, v, label, orient="h", color=INK2, pos=0.99, va="bottom", ha=None, fs=7):
    """Dotted reference level with a direct label (axes-fraction position along the line)."""
    if orient == "h":
        ax.axhline(v, ls=":", lw=1.0, color=color, zorder=1)
        ax.text(pos, v, label, transform=ax.get_yaxis_transform(), ha=ha or "right", va=va,
                fontsize=fs, color=INK2)
    else:
        ax.axvline(v, ls=":", lw=1.0, color=color, zorder=1)
        ax.text(v, pos, label, transform=ax.get_xaxis_transform(), ha=ha or "left", va="top",
                fontsize=fs, color=INK2, rotation=90)


def kib_formatter(x, _=None):
    if x >= 1 << 30:
        return f"{x / (1 << 30):g} GiB"
    if x >= 1 << 20:
        return f"{x / (1 << 20):g} MiB"
    if x >= 1 << 10:
        return f"{x / (1 << 10):g} KiB"
    return f"{x:g} B"


def bytes_axis(ax, which="x"):
    a = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log", base=2)
    a.set_major_formatter(mpl.ticker.FuncFormatter(kib_formatter))


apply()
