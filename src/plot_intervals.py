"""Plot the number of intervals (interval count) for each image size.

Loads results collected under the design "one p-value (= one interval computation) = one
independently trained model" and draws a line plot of interval count (y-axis) against
image size (x-axis).

Results are collected by scripts/submit_intervals.sh, which trains a throwaway model per
iteration in parallel (one core = one model) and computes its intervals. Each (category,
size) is split into chunk jobs of at most 96 iterations, and the results of chunk c are
stored in a single pickle:
    ../results/intervals/{iid|corr}/fpr/{cat}_size{S}_signal0_seed{c}.pickle

For each (category, size), the extra iterations run (size 8: 300, others: 120) are merged
in chunk order, and the interval counts of the first N_ADOPT (=100) successful models
(one per model) are used.

Usage (run from a top-level subdirectory of the repo, where the result pickles are visible):
    cd src
    python plot_intervals.py
"""

import os
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")  # also works without a GUI (e.g., on a server)
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator, ScalarFormatter

# ===== Settings (edit here to match the sweep setup) ============
RESULTS_BASE = "../results/intervals"       # root directory of the exhaustive results
OUTPUT_DIR = "../plots/figures/intervals"   # output directory for figures
SIZES = [8, 16, 32, 64]                     # image sizes on the x-axis

N_ADOPT = 100         # number of models used (first N successful iterations)
# Chunk numbers to scan (= data seeds; match the split in submit_intervals.sh.
# size 8 has 4 chunks, the others 2. Missing chunks are skipped.)
DATA_SEEDS = [0, 1, 2, 3]

SIGNAL = 0.0                                # synthetic data collected for FPR (signal=0)

# Categories to plot (only those with existing pickles are drawn)
CATEGORIES = ["iid", "corr"]

# Metrics to plot. One figure is produced per metric.
#   search_count : number of searched intervals
#   time         : computation time of parametric SI [s] (not the exhaustive search time)
# (detect_count, the number of detected intervals, is also saved and can be added if needed)
METRICS = ["search_count", "time"]

# y-axis labels for the paper (avoid showing raw variable names)
METRIC_LABELS = {
    "search_count": "Number of intervals",
    "detect_count": "Number of detected intervals",
    "time": "Calculation time [s]",
}

# Output file names (unlisted metrics default to intervals_{metric}.pdf)
METRIC_FILENAMES = {
    "search_count": "num_of_intervals.pdf",
}

# Match the existing calculation_time figure (plots/figures/syn/calculation_time.pdf)
LOG_X = False                # linear x-axis (number of pixels)
LOG_Y_METRICS = {"time"}     # metrics on a log y-axis (time only; interval counts linear)
ERROR_BAND = None            # band for the spread across models: "std" | "sem" | None
SHOW_POINTS = False          # overlay each model's raw interval count as faint points
# ================================================================


def result_path(category: str, size: int, chunk_seed: int) -> str:
    """Return the pickle path following the exhaustive saving rule of experiment.py.

    One chunk job = one pickle (data seed = chunk number; see submit_intervals.sh).
    """
    error = "fpr" if SIGNAL == 0 else "power"
    return (
        f"{RESULTS_BASE}/{category}/{error}/"
        f"{category}_size{size}_signal{int(SIGNAL)}_seed{chunk_seed}.pickle"
    )


def load_counts(category: str, size: int, metric: str) -> np.ndarray | None:
    """Return the metric for (category, size) over the first N_ADOPT successful models.

    result[metric] is a list over successful iterations (one iteration = one independent
    model = one interval count). Chunks are merged in order and the first N_ADOPT are kept.
    """
    counts: list[float] = []
    for chunk_seed in DATA_SEEDS:
        if len(counts) >= N_ADOPT:
            break
        path = result_path(category, size, chunk_seed)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            result = pickle.load(f)
        if metric not in result:
            print(f"  Warning: {os.path.basename(path)} has no '{metric}' (check it was saved with --exhaustive)")
            continue
        counts.extend(c for c in result[metric] if c is not None)
    if not counts:
        return None
    if len(counts) < N_ADOPT:
        print(
            f"  Warning: {category} size={size}: only {len(counts)}/{N_ADOPT} models succeeded. "
            "For the shortfall, check for missing chunks and the job logs (eo/)"
        )
    return np.asarray(counts[:N_ADOPT], dtype=float)


def collect(category: str, metric: str):
    """Aggregate (size, mean, spread, raw data) per size; returns only sizes with data."""
    xs, means, spreads, raw = [], [], [], []
    for size in SIZES:
        counts = load_counts(category, size, metric)
        if counts is None or counts.size == 0:
            continue
        xs.append(size)
        means.append(counts.mean())
        if ERROR_BAND == "std":
            spreads.append(counts.std())
        elif ERROR_BAND == "sem":
            spreads.append(counts.std() / np.sqrt(counts.size))
        else:
            spreads.append(0.0)
        raw.append(counts)
    return np.array(xs), np.array(means), np.array(spreads), raw


def plot_metric(metric: str) -> None:
    """Plot one metric with one line per category in a single figure and save it.

    Styling follows the existing calculation_time figure:
    x-axis = number of pixels (size^2, linear), "x" markers; the y-axis is log scale for the
    metrics in LOG_Y_METRICS (time) and linear for the interval counts.
    """
    plt.rcParams["font.size"] = 12
    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    plotted = False
    for category in CATEGORIES:
        xs, means, spreads, raw = collect(category, metric)
        if xs.size == 0:
            continue
        plotted = True
        pixels = xs**2  # x-axis is the number of pixels
        (line,) = ax.plot(pixels, means, "x-", label=category, zorder=6)
        if ERROR_BAND:
            ax.fill_between(
                pixels, means - spreads, means + spreads, alpha=0.2, color=line.get_color()
            )
        if SHOW_POINTS:
            for x, counts in zip(pixels, raw):
                ax.scatter(
                    [x] * counts.size, counts, s=6, alpha=0.15, color=line.get_color()
                )

    if not plotted:
        print(f"[{metric}] No data found (check {RESULTS_BASE})")
        plt.close(fig)
        return

    if LOG_X:
        ax.set_xscale("log")
    if metric in LOG_Y_METRICS:
        ax.set_yscale("log")
        # Put major ticks at 1, 2, 5 x 10^k: with powers of 10 only, the visible range may
        # contain no major tick and lose its labels (e.g., interval counts from 300 to 900)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        # Log-spaced minor ticks (no labels; marks only, so the axis reads as logarithmic)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(0.1, 1.0, 0.1))))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", which="minor", length=3)
        ax.tick_params(axis="y", which="major", length=6)
    ax.set_xticks([size**2 for size in SIZES])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Image size")
    ax.set_ylabel(METRIC_LABELS.get(metric, "Number of intervals"))
    ax.legend(loc="upper left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = METRIC_FILENAMES.get(metric, f"intervals_{metric}.pdf")
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"[{metric}] Saved: {out_path}")


if __name__ == "__main__":
    for metric in METRICS:
        plot_metric(metric)
