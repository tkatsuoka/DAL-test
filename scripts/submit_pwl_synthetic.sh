#!/bin/bash
# -------------------------------------------------------------
#  Submit the synthetic-data experiments with piecewise-linear (PWL)
#  approximated models
#  Measures FPR / power with SiLU/GELU diffusion models whose
#  activations are replaced by PiecewiseLinearApprox, and sweeps
#  the number of knots K
#  NOTE: run from the scripts/ directory (results go to ../results)
#
#  Requires: trained models ../model/syn/sim_diffusion_{ACT_FN}_size{S}_...
#            (if missing, first run `python model.py --category syn --act_fn silu`).
#            Missing pwl{K}_ models are generated on the login node
#            before submission (takes a few seconds).
#
#  Usage:
#      ./submit_pwl_synthetic.sh fpr          # FPR (both iid and corr)
#      ./submit_pwl_synthetic.sh fpr iid      # FPR (iid only)
#      ./submit_pwl_synthetic.sh power iid    # power (iid only)
#
#  Output: under ../results/pwl/{ACT_FN}{K}/ (kept apart from the main ReLU results)
# -------------------------------------------------------------
source "$(dirname "$0")/common.sh"

# --- Experiment parameters -----------------------------------
ACT_FN="silu"              # silu / gelu
KNOTS=(8)                  # number of knots K (K+1 linear pieces; fixed at K=8)
                           # to compare several K, use e.g. (8 16 32)
KNOT_METHOD="curvature"    # knot placement (curvature: equidistribute curvature, uniform: equal spacing)
                           # Note: the pwl{K}_ file name does not record the placement method,
                           #       so after changing it, delete the existing pwl models and regenerate them
ALLOWED=("iid" "corr")     # selectable categories
FPR_SIZES=(8 16 32 64)     # image sizes swept in the FPR experiment
POWER_SIZE=64              # image size for power (same as the main power experiment)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
SIGNALS=(1.0 2.0 3.0 4.0)  # signal strengths for power
DISTANCE=0
THR=0.8                    # threshold for ReferenceMeanDiff (required; None causes a crash)
# model settings for the syn category (must match the naming scheme in src/model_paths.py)
TIMESTEPS=460
STEP=115

# --- Mode (fpr / power) --------------------------------------
MODE="$1"
if [ "$MODE" != "fpr" ] && [ "$MODE" != "power" ]; then
    echo "Usage: $0 <fpr|power> [category...]" >&2
    exit 1
fi
shift
resolve_categories "$@"    # filter by the arguments (result in CATEGORIES)

# --- Check the PWL models and generate missing ones ----------
#  Exit with an error if a trained sim_ model is missing; generate a
#  missing pwl{K}_ model with rewrite_pwl.py (it only rewrites the
#  graph, so it is light enough to run on the login node)
ensure_pwl_model() {
    local size="$1" knots="$2"
    local sim="${PROJECT_DIR}/model/syn/sim_diffusion_${ACT_FN}_size${size}_timesteps${TIMESTEPS}_step${STEP}.onnx"
    local pwl="${PROJECT_DIR}/model/syn/pwl${knots}_diffusion_${ACT_FN}_size${size}_timesteps${TIMESTEPS}_step${STEP}.onnx"
    if [ ! -f "$pwl" ]; then
        if [ ! -f "$sim" ]; then
            echo "Error: trained model not found: $sim" >&2
            echo "       run (cd src && python model.py --category syn --act_fn ${ACT_FN}) first" >&2
            exit 1
        fi
        echo "Generating: $(basename "$pwl")"
        (cd "${PROJECT_DIR}/src" && "$PYTHON" rewrite_pwl.py \
            --input "$sim" --output "$pwl" --act_fn "$ACT_FN" \
            --num_knots "$knots" --knot_method "$KNOT_METHOD") || exit 1
    fi
}

# --- Submit --------------------------------------------------
if [ "$MODE" = "fpr" ]; then
    for size in "${FPR_SIZES[@]}"; do
        for knots in "${KNOTS[@]}"; do
            ensure_pwl_model "$size" "$knots"
        done
    done
    for category in "${CATEGORIES[@]}"; do
        for size in "${FPR_SIZES[@]}"; do
            # more iterations for size=8 only (same as the main FPR experiment)
            if [ "$size" -eq 8 ]; then iter=300; else iter=120; fi
            for knots in "${KNOTS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    submit_experiment "pwlfpr_${category}_${size}_k${knots}_${seed}" \
                        "--category $category --size $size --thr $THR --signal 0.0 --workers $WORKERS --iter $iter --seed $seed --distance $DISTANCE --act_fn $ACT_FN --pwl_knots $knots"
                done
            done
        done
    done
else
    for knots in "${KNOTS[@]}"; do
        ensure_pwl_model "$POWER_SIZE" "$knots"
    done
    for category in "${CATEGORIES[@]}"; do
        for knots in "${KNOTS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for signal in "${SIGNALS[@]}"; do
                    submit_experiment "pwlpow_${category}_${signal%.*}_k${knots}_${seed}" \
                        "--category $category --size $POWER_SIZE --thr $THR --signal $signal --workers $WORKERS --iter 110 --seed $seed --distance $DISTANCE --act_fn $ACT_FN --pwl_knots $knots"
                done
            done
        done
    done
fi
