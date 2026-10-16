#!/bin/bash
# Shared settings and helpers, sourced by the drivers (submit_*.sh)

# --- Environment ---------------------------------------------
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # <project>/scripts
PROJECT_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"                # <project>
PYTHON="${PROJECT_DIR}/.venv/bin/python3"   # uv-managed environment (`uv sync`)
EO_DIR="${PROJECT_DIR}/eo"          # PBS stdout/stderr logs
mkdir -p "$EO_DIR"

# --- PBS submission settings ---------------------------------
QUEUE="cpu-large"                   # queue name (also set in jobs/experiment.pbs)
WORKERS=96                          # --workers for experiment.py (= ncpus in jobs/experiment.pbs)
EXPERIMENT_JOB="${SCRIPTS_DIR}/jobs/experiment.pbs"

# --- Interval-count experiment -------------------------------
#  Number of iterations (= independently trained models) per image size.
#  Iterations with no detected region yield no p-value, so more than 100 are run.
model_seed_count() {
    case "$1" in
        8) echo 300 ;;
        *) echo 120 ;;
    esac
}

# --- Helpers -------------------------------------------------
#  Submit one experiment.py job: submit_experiment <job name> "<MYARGS string>"
submit_experiment() {
    local job_name="$1"
    local myargs="$2"
    qsub -N "$job_name" -o "$EO_DIR" -e "$EO_DIR" -v MYARGS="$myargs" "$EXPERIMENT_JOB"
}

#  Set CATEGORIES from the arguments (all of ALLOWED if no argument is given)
resolve_categories() {
    if [ "$#" -eq 0 ]; then
        CATEGORIES=("${ALLOWED[@]}")
    else
        CATEGORIES=()
        local arg allowed ok
        for arg in "$@"; do
            ok=0
            for allowed in "${ALLOWED[@]}"; do
                [ "$arg" = "$allowed" ] && ok=1 && break
            done
            if [ "$ok" -eq 1 ]; then
                CATEGORIES+=("$arg")
            else
                echo "Error: unknown category '$arg' (allowed: ${ALLOWED[*]})" >&2
                exit 1
            fi
        done
    fi
    echo "Categories to submit: ${CATEGORIES[*]}"
}
