#!/bin/bash
# -------------------------------------------------------------
#  Submit the FPR experiments on synthetic data (iid / corr)
#  NOTE: run from the scripts/ directory (results go to ../results)
#
#  Usage:
#      ./submit_synthetic_fpr.sh          # both iid and corr
#      ./submit_synthetic_fpr.sh iid      # iid only
#      ./submit_synthetic_fpr.sh corr     # corr only
# -------------------------------------------------------------
source "$(dirname "$0")/common.sh"

# --- Experiment parameters -----------------------------------
ALLOWED=("iid" "corr")     # selectable categories
resolve_categories "$@"    # filter by the arguments (result in CATEGORIES)
SIZES=(8 16 32 64)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
SIGNAL=0.0
DISTANCE=0
THR=0.8       # threshold for ReferenceMeanDiff (required; None causes a crash)

for category in "${CATEGORIES[@]}"; do
    for size in "${SIZES[@]}"; do
        # more iterations for size=8 only (kept from the original scripts)
        if [ "$size" -eq 8 ]; then iter=300; else iter=120; fi
        for seed in "${SEEDS[@]}"; do
            submit_experiment "fpr_${category}_${size}_${seed}" \
                "--category $category --size $size --thr $THR --signal $SIGNAL --workers $WORKERS --iter $iter --seed $seed --distance $DISTANCE"
        done
    done
done
