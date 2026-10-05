"""Plot piecewise-linear approximations of activation functions against the true functions.

Using the same parameter generation as rewrite_pwl.py (make_pwl_params), overlays the true
shapes of SiLU / GELU with their piecewise-linear approximations for each number of knots K,
with the knot positions marked by dots.

No trained model is needed (only the function shapes are compared).

Usage:
    cd src
    python plot_pwl_activation.py
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # also works without a GUI (e.g., on a server)
import matplotlib.pyplot as plt

from rewrite_pwl import (
    DEFAULT_KNOT_METHOD,
    DEFAULT_KNOT_RANGE,
    activation_values,
    make_pwl_params,
)

# ===== Settings (edit here to match the comparison setup) =======
OUTPUT_DIR = "../plots/figures/pwl"     # output directory for figures
ACT_FNS = ["silu", "gelu"]              # activation functions to plot
KNOTS_LIST = [8]                        # knots K to plot (experiment setting; K+1 pieces)
KNOT_METHOD = DEFAULT_KNOT_METHOD       # knot placement method (curvature / uniform)
KNOT_RANGE = DEFAULT_KNOT_RANGE         # approximation range (rewrite_pwl.py default)
PLOT_RANGE = (-6.0, 6.0)                # x range to plot (for shape comparison)
NUM_POINTS = 2001                       # grid points for plotting and error evaluation
FIG_SIZE = (7, 5.0)                     # figure size (widened to fit the larger fonts)

# Font sizes (adjust all of them here for readability in the paper)
FONT_SIZE_BASE = 16                     # default font size
FONT_SIZE_LABEL = 20                    # axis labels
FONT_SIZE_TICK = 16                     # tick labels
FONT_SIZE_LEGEND = 15                   # legend
# ================================================================

ACT_LABELS = {"silu": "SiLU", "gelu": "GELU"}

# Apply common font sizes to the whole figure
plt.rcParams.update(
    {
        "font.size": FONT_SIZE_BASE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
    }
)


def pwl_values(x: np.ndarray, knots, slopes, intercepts) -> np.ndarray:
    """Evaluate the PWL approximation (left-closed, right-open intervals, as in si4onnx)."""
    index = np.searchsorted(knots, x, side="right")
    return slopes[index] * x + intercepts[index]


def plot_activation(act_fn: str) -> None:
    """Save a shape comparison of one activation function and its PWL approximation."""
    x = np.linspace(*PLOT_RANGE, NUM_POINTS)
    y_true = activation_values(act_fn, x)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(x, y_true, color="black", linewidth=2, label=ACT_LABELS[act_fn])
    for num_knots in KNOTS_LIST:
        knots, slopes, intercepts = make_pwl_params(
            act_fn, num_knots, KNOT_RANGE, KNOT_METHOD
        )
        y_pwl = pwl_values(x, knots, slopes, intercepts)
        (line,) = ax.plot(
            x,
            y_pwl,
            linewidth=1.2,
            linestyle="--",
            label=f"Piecewise-linear approximation ({num_knots} knots)",
        )
        # Mark knots with dots (they cluster near the origin, where curvature is high)
        in_range = (knots >= PLOT_RANGE[0]) & (knots <= PLOT_RANGE[1])
        knot_y = pwl_values(knots[in_range], knots, slopes, intercepts)
        ax.scatter(knots[in_range], knot_y, s=30, color=line.get_color(), zorder=3)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # fig.suptitle(f"Piecewise-linear approximation of {ACT_LABELS[act_fn]}")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"activation_{act_fn}_{KNOT_METHOD}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[{act_fn}] Saved: {out_path}")


if __name__ == "__main__":
    for act_fn in ACT_FNS:
        plot_activation(act_fn)
