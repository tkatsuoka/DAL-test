#!/bin/bash
# -------------------------------------------------------------
#  Submit the interval-count experiment (Appendix D)
#  Each iteration trains its own model (one p-value = one model) and
#  counts the intervals over the whole parametric line (--exhaustive).
#  NOTE: run from the scripts/ directory
#
#  Usage:
#      ./submit_intervals.sh             # both iid and corr
#      ./submit_intervals.sh iid         # iid only
# -------------------------------------------------------------
source "$(dirname "$0")/common.sh"

# --- Experiment parameters -----------------------------------
ALLOWED=("iid" "corr")     # selectable categories
resolve_categories "$@"     # filter by the arguments (result in CATEGORIES)
SIZES=(8 16 32 64)
THR=0.8                     # threshold for ReferenceMeanDiff
SIGNAL=0.0                  # under the null

# One job per chunk of at most WORKERS iterations
# (data seed = chunk index, model seed = base + running index)
for category in "${CATEGORIES[@]}"; do
    # different model seeds for iid and corr
    case "$category" in
        iid)  base=0 ;;
        corr) base=1000 ;;
    esac
    for size in "${SIZES[@]}"; do
        count=$(model_seed_count "$size")
        chunks=$(( (count + WORKERS - 1) / WORKERS ))
        chunk_size=$(( (count + chunks - 1) / chunks ))
        offset=0
        chunk_idx=0
        while [ "$offset" -lt "$count" ]; do
            n=$(( count - offset ))
            [ "$n" -gt "$chunk_size" ] && n=$chunk_size
            submit_experiment "iv_${category:0:4}_${size}_c${chunk_idx}" \
                "--category $category --size $size --thr $THR --signal $SIGNAL --workers $n --n_jobs 1 --iter $n --seed $chunk_idx --model_seed_base $(( base + offset )) --exhaustive"
            offset=$(( offset + n ))
            chunk_idx=$(( chunk_idx + 1 ))
        done
    done
done
