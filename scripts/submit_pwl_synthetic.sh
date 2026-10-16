#!/bin/bash
# -------------------------------------------------------------
#  Submit the FPR experiments with SiLU models whose activations are
#  replaced by the piecewise-linear approximation (K=8; Appendix C)
#  Requires the trained SiLU models (python model.py --category syn --act_fn silu)
#  NOTE: run from the scripts/ directory
#
#  Usage:
#      ./submit_pwl_synthetic.sh          # both iid and corr
#      ./submit_pwl_synthetic.sh iid      # iid only
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
THR=0.8                    # threshold for ReferenceMeanDiff
TIMESTEPS=460              # used in the model file names (src/model_paths.py)
STEP=115

# --- Generate the pwl8_ models from the trained SiLU models if missing
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
