#!/bin/bash
# -------------------------------------------------------------
#  Submit the power experiments on synthetic data (iid / corr)
#  NOTE: run from the scripts/ directory (results go to ../results)
#
#  Usage:
#      ./submit_synthetic_power.sh        # both iid and corr
#      ./submit_synthetic_power.sh iid    # iid only
#      ./submit_synthetic_power.sh corr   # corr only
# -------------------------------------------------------------
source "$(dirname "$0")/common.sh"

# --- Experiment parameters -----------------------------------
ALLOWED=("iid" "corr")     # selectable categories
resolve_categories "$@"    # filter by the arguments (result in CATEGORIES)
SIZE=64
SEEDS=(0 1 2 3 4 5 6 7 8 9)
SIGNALS=(1.0 2.0 3.0 4.0)
ITER=110
DISTANCE=0
THR=0.8       # threshold for ReferenceMeanDiff (required; None causes a crash)

for category in "${CATEGORIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for signal in "${SIGNALS[@]}"; do
            submit_experiment "pow_${category}_${signal%.*}_${seed}" \
                "--category $category --size $SIZE --thr $THR --signal $signal --workers $WORKERS --iter $ITER --seed $seed --distance $DISTANCE"
        done
    done
done
