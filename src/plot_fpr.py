"""Plot the FPR (type I error rate) for each image size.

Loads the FPR results of experiment.py (pickles in ../results/.../{category}/fpr/) and
draws a line plot of FPR (y-axis) against image size in pixels (x-axis).
Aggregation and styling follow plot_fpr.ipynb, which produced the existing paper figure
(truncated to the first 100 iterations per seed and 1000 in total; bonferroni compares
the z statistic with a threshold corrected for 2^(number of pixels) hypotheses;
figsize 4x4, font 12, etc.).

Results of piecewise-linear approximation (PWL) models are selected with --act_fn /
--pwl_knots (reads the result tree ../results/pwl/{act_fn}{K}/). If omitted, the
existing ReLU results are used.

Usage:
    cd src
    uv run python plot_fpr.py --act_fn silu --pwl_knots 8   # PWL SiLU (K=8)
    uv run python plot_fpr.py                               # ReLU (original)
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")  # also works without a GUI (e.g., on a server)
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

from model_paths import model_result_tag

# ===== Settings (edit here to match the sweep setup) ============
RESULTS_BASE = "../results"                 # root directory of the FPR results
OUTPUT_DIR = "../plots/figures/fpr"         # output directory for figures
CATEGORIES = ["iid", "corr"]                # categories to plot (one figure each)
SIZES = [8, 16, 32, 64]                     # image sizes (x-axis labels: pixels, size^2)
SEEDS = list(range(10))                     # seeds to aggregate (missing ones skipped)
ALPHA = 0.05                                # significance level
PER_SEED_LIMIT = 100                        # iterations used per seed (same as the ipynb)
TOTAL_LIMIT = 1000                          # total iterations used (same as the ipynb)
# Off by default: the current experiment.py does not compute permutation (fixed at -1).
# Set to True only to plot older results with valid permutation_p_values.
INCLUDE_PERMUTATION = False
# ================================================================


def compute_threshold(alpha: float, base: float, power: float) -> float:
    """Return the Bonferroni rejection threshold (same computation as plot_fpr.ipynb).

    Solves for the two-sided threshold th corrected for base^power (= 2^(number of pixels))
    hypotheses, using the log form of 2 * Phi(-th) = alpha / base^power.
    """
    def target_func(th):
        log_bonf_alpha = np.log(alpha) - np.log(base) * power
        return np.log(2.0) + norm.logcdf(-th) - log_bonf_alpha

    return brentq(target_func, 0.0, 1000.0)


def load_results(results_base: str, category: str, size: int) -> dict[str, np.ndarray]:
    """Return (category, size) results merged over all seeds, truncated as in the ipynb."""
    keys = ["naive_p_values", "selective_p_values", "oc_p_values", "z", "permutation_p_values"]
    values: dict[str, list] = {key: [] for key in keys}
    for seed in SEEDS:
        path = (
            f"{results_base}/{category}/fpr/"
            f"{category}_size{size}_signal0_seed{seed}.pickle"
        )
        if not os.path.exists(path):
            print(f"  pickle not found: {path}")
            continue
        with open(path, "rb") as f:
            result = pickle.load(f)
        for key in keys:
            values[key].extend(result[key][:PER_SEED_LIMIT])
    return {key: np.asarray(v[:TOTAL_LIMIT], dtype=float) for key, v in values.items()}


def plot_category(results_base: str, category: str, tag: str) -> None:
    """Plot the FPR of one category in a single figure, one line per method, and save it."""
    fpr = lambda p_values: np.sum(p_values < ALPHA) / len(p_values)

    # Collect the FPR of each method across sizes
    fpr_lists: dict[str, list] = {
        "proposed": [], "w/o-pp": [], "bonferroni": [], "permutation": [], "naive": []
    }
    for size in SIZES:
        results = load_results(results_base, category, size)
        if results["selective_p_values"].size == 0:
            print(f"[{category}] size={size}: no data")
            continue
        print(f"[{category}] size={size}: iter={results['selective_p_values'].size}")
        fpr_lists["proposed"].append(fpr(results["selective_p_values"]))
        fpr_lists["w/o-pp"].append(fpr(results["oc_p_values"]))
        fpr_lists["bonferroni"].append(
            np.mean(results["z"] > compute_threshold(ALPHA, 2, size**2))
        )
        fpr_lists["permutation"].append(fpr(results["permutation_p_values"]))
        fpr_lists["naive"].append(fpr(results["naive_p_values"]))

    if not fpr_lists["proposed"]:
        print(f"[{category}] No data found (check {results_base})")
        return

    if not INCLUDE_PERMUTATION:
        fpr_lists.pop("permutation")

    # Same styling as plot_fpr.ipynb (evenly spaced positions + pixel-count labels)
    # Fix each method's color so it matches the original figure even without permutation
    colors = {
        "proposed": "C0", "w/o-pp": "C1", "bonferroni": "C2",
        "permutation": "C3", "naive": "C4",
    }
    positions = list(range(1, len(SIZES) + 1))
    plt.rcParams["font.size"] = 12
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(positions, [ALPHA] * len(positions), linestyle="dashed", color="black")
    for zorder, (label, fprs) in enumerate(fpr_lists.items(), start=1):
        ax.plot(positions, fprs, "x-", label=label, color=colors[label], zorder=10 - zorder)
    ax.grid()
    ax.set_yticks(np.arange(0.0, 0.51, 0.1))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(size**2) for size in SIZES])
    ax.set_xlabel("Image size")
    ax.set_ylabel("Type I Error Rate")
    plt.legend(loc="upper left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_tag = f"_{tag.replace('/', '_')}" if tag else ""
    out_path = os.path.join(OUTPUT_DIR, f"{category}_fpr{file_tag}.pdf")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"[{category}] Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-act_fn", "--act_fn", type=str, default="relu")
    parser.add_argument("-pwl_knots", "--pwl_knots", type=int, default=0)
    args = parser.parse_args()

    # PWL model results are kept separately under ../results/pwl/{act_fn}{K}/
    tag = model_result_tag(args.act_fn, args.pwl_knots)
    results_base = f"{RESULTS_BASE}/{tag}" if tag else RESULTS_BASE

    for category in CATEGORIES:
        plot_category(results_base, category, tag)
