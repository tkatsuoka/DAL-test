#!/bin/bash
# -------------------------------------------------------------
#  Submit the robustness experiments (non-Gaussian noise)
#  NOTE: run from the scripts/ directory (results go to ../results)
#
#  Usage:
#      ./submit_robust.sh                 # all distributions
#      ./submit_robust.sh t skewnorm      # only the given distributions
# -------------------------------------------------------------
source "$(dirname "$0")/common.sh"

# --- Experiment parameters -----------------------------------
ALLOWED=("skewnorm" "exponnorm" "gennormsteep" "gennormflat" "t")   # selectable categories
resolve_categories "$@"    # filter by the arguments (result in CATEGORIES)
DISTANCES=(0.01 0.02 0.03 0.04)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
SIZE=16
SIGNAL=0
ITER=120
THR=0.8       # threshold for ReferenceMeanDiff (required; None causes a crash)

for category in "${CATEGORIES[@]}"; do
    for distance in "${DISTANCES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            submit_experiment "rob_${category:0:6}_${seed}" \
                "--category $category --size $SIZE --thr $THR --signal $SIGNAL --workers $WORKERS --iter $ITER --seed $seed --distance $distance"
        done
    done
done
