#!/bin/bash
# -------------------------------------------------------------
#  Submit the FPR experiments on synthetic data with SiLU diffusion
#  models whose activations are replaced by a piecewise-linear (PWL)
#  approximation with K=8 knots (PiecewiseLinearApprox; Appendix C)
#  NOTE: run from the scripts/ directory (results go to ../results)
#
#  Requires: trained models ../model/syn/sim_diffusion_silu_size{S}_...
#            (if missing, first run `python model.py --category syn --act_fn silu`).
#            Missing pwl8_ models are generated on the login node
#            before submission (takes a few seconds).
#
#  Usage:
#      ./submit_pwl_synthetic.sh          # both iid and corr
#      ./submit_pwl_synthetic.sh iid      # iid only
#
#  Output: under ../results/pwl/silu8/ (kept apart from the main ReLU results)
# -------------------------------------------------------------
source "$(dirname "$0")/common.sh"

# --- Experiment parameters -----------------------------------
ACT_FN="silu"
KNOTS=8                    # number of knots K (K+1 linear pieces)
ALLOWED=("iid" "corr")     # selectable categories
resolve_categories "$@"    # filter by the arguments (result in CATEGORIES)
SIZES=(8 16 32 64)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
SIGNAL=0.0
DISTANCE=0
THR=0.8                    # threshold for ReferenceMeanDiff (required)
# model settings for the syn category (must match the naming scheme in src/model_paths.py)
TIMESTEPS=460
STEP=115

# --- Check the PWL models and generate missing ones ----------
#  Exit with an error if a trained sim_ model is missing; generate a
#  missing pwl{K}_ model with rewrite_pwl.py (it only rewrites the
#  graph, so it is light enough to run on the login node)
ensure_pwl_model() {
    local size="$1"
    local sim="${PROJECT_DIR}/model/syn/sim_diffusion_${ACT_FN}_size${size}_timesteps${TIMESTEPS}_step${STEP}.onnx"
    local pwl="${PROJECT_DIR}/model/syn/pwl${KNOTS}_diffusion_${ACT_FN}_size${size}_timesteps${TIMESTEPS}_step${STEP}.onnx"
    if [ ! -f "$pwl" ]; then
        if [ ! -f "$sim" ]; then
            echo "Error: trained model not found: $sim" >&2
            echo "       run (cd src && python model.py --category syn --act_fn ${ACT_FN}) first" >&2
            exit 1
        fi
        echo "Generating: $(basename "$pwl")"
        (cd "${PROJECT_DIR}/src" && "$PYTHON" rewrite_pwl.py \
            --input "$sim" --output "$pwl" --act_fn "$ACT_FN" \
            --num_knots "$KNOTS") || exit 1
    fi
}

# --- Submit --------------------------------------------------
for size in "${SIZES[@]}"; do
    ensure_pwl_model "$size"
done
for category in "${CATEGORIES[@]}"; do
    for size in "${SIZES[@]}"; do
        # more iterations for size 8, where no anomalous region is detected more often
        if [ "$size" -eq 8 ]; then iter=300; else iter=120; fi
        for seed in "${SEEDS[@]}"; do
            submit_experiment "pwlfpr_${category}_${size}_k${KNOTS}_${seed}" \
                "--category $category --size $size --thr $THR --signal $SIGNAL --workers $WORKERS --iter $iter --seed $seed --distance $DISTANCE --act_fn $ACT_FN --pwl_knots $KNOTS"
        done
    done
done
