#!/bin/bash
# -------------------------------------------------------------
#  Submit the interval-count experiment (exhaustive mode)
#
#  Design: each p-value (= one interval computation) gets its own
#  independently trained model and its own data. Each iteration
#  (= model) runs in a separate worker process, with 1 core = 1 model,
#  and the workers run "train a throwaway model (CPU, in memory) ->
#  compute intervals" in parallel (via experiment.py --model_seed_base;
#  the models are never saved).
#
#  The target is 100 successful runs per (category, size). The total
#  number of iterations, padded for failures, is set in
#  model_seed_count() in common.sh:
#      size 8 -> 300 (3x), size 16/32/64 -> 120 each (1.2x)
#  The aggregation step (plot_intervals.py) keeps only the first 100
#  successful runs.
#
#  A node has 96 cores, so the iterations are split evenly into chunks
#  of at most WORKERS (96), and each job finishes in a single round:
#      size 8 -> 4 jobs x 75 iterations, size 16/32/64 -> 2 jobs x 60 iterations each
#      -> 10 jobs per category, 20 jobs in total for iid+corr
#
#  Seed assignment (unique across all iterations):
#      model seed = base + running index (iid: base=0, corr: base=1000,
#                   so that no model is reproduced across categories)
#      data seed  = chunk index (each chunk has its own random stream,
#                   and data are drawn independently per iteration)
#
#  Results are kept in a tree separate from the main experiments
#  (one pickle per chunk):
#      ../results/intervals/{iid|corr}/fpr/{cat}_size{S}_signal0_seed{chunk}.pickle
#
#  NOTE: run from the scripts/ directory
#
#  Usage:
#      ./submit_intervals.sh             # both iid and corr
#      ./submit_intervals.sh iid         # iid only (all sizes)
# -------------------------------------------------------------
source "$(dirname "$0")/common.sh"

# --- Experiment parameters -----------------------------------
ALLOWED=("iid" "corr")     # selectable categories (synthetic data only)
resolve_categories "$@"     # filter by the arguments (result in CATEGORIES)
SIZES=(8 16 32 64)
THR=0.8                     # threshold for ReferenceMeanDiff (same as the main synthetic experiments)
SIGNAL=0.0                  # collected under the null (FPR setting), as in the main FPR experiments

# one chunk = one job (--workers = --iter = number of iterations in the chunk)
for category in "${CATEGORIES[@]}"; do
    # offset the model seed base so that no model is reproduced across categories
    case "$category" in
        iid)  base=0 ;;
        corr) base=1000 ;;
    esac
    for size in "${SIZES[@]}"; do
        count=$(model_seed_count "$size")
        # split evenly so that each job has <= WORKERS iterations (one round)
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
