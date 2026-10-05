#!/bin/bash
# =============================================================
#  Shared settings and helpers for the job-submission scripts
#  - Parameters that may need changing are collected here
#  - Meant to be `source`d by the drivers (submit_*.sh)
#    (the PBS job scripts jobs/*.pbs do not source it, because
#     PBS runs them from a spooled copy)
# =============================================================

# --- Environment ---------------------------------------------
#  The project location is derived from the location of this file,
#  so renaming the project directory does not break anything
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # <project>/scripts
PROJECT_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"                # <project>
#  Python runs from the uv-managed virtual environment (create it with
#  `uv sync` in the project root; dependencies are pinned in pyproject.toml / uv.lock.
#  jobs/*.pbs uses the same .venv through a relative path)
PYTHON="${PROJECT_DIR}/.venv/bin/python3"
EO_DIR="${PROJECT_DIR}/eo"          # where PBS writes stdout/stderr logs
mkdir -p "$EO_DIR"                  # create it if missing (must exist before submission)

# --- PBS submission settings ---------------------------------
QUEUE="cpu-large"                   # one queue for all experiments (also set in jobs/*.pbs)
WORKERS=96                          # --workers for experiment.py (keep equal to ncpus)

# --- Worker job scripts (the files actually passed to qsub) --
EXPERIMENT_JOB="${SCRIPTS_DIR}/jobs/experiment.pbs"   # for src/experiment.py

# --- Independent-model experiment (interval counts) ----------
#  In this experiment one p-value = one independently trained model.
#  The function below gives, per image size, the total number of
#  throwaway models to train (1 core = 1 model). The target is 100
#  successful runs; small sizes fail more often at inference, so more
#  runs are launched, and the aggregation step (plot_intervals.py)
#  keeps only the first 100 successful ones.
#  submit_intervals.sh splits the runs evenly so that each job has
#  at most WORKERS (96) iterations.
model_seed_count() {
    case "$1" in
        8) echo 300 ;;   # size 8: 3x to allow for failures
        *) echo 120 ;;   # size 16/32/64: 20% margin
    esac
}

# --- Submission helpers --------------------------------------
#  The log directory (-o/-e) is always passed explicitly as EO_DIR
#  (jobs/*.pbs contains no hard-coded paths)
#  Submit one experiment.py job
#    Usage: submit_experiment <job name> "<MYARGS string>"
submit_experiment() {
    local job_name="$1"
    local myargs="$2"
    qsub -N "$job_name" -o "$EO_DIR" -e "$EO_DIR" -v MYARGS="$myargs" "$EXPERIMENT_JOB"
}

# --- Category selection --------------------------------------
#  Shared logic for choosing which categories to submit from the
#  command-line arguments
#    - no arguments  : all default categories (array ALLOWED)
#    - with arguments: only the given categories (a name not in
#                      ALLOWED is an error)
#  Usage: define ALLOWED, then call resolve_categories "$@"
#         The result is stored in the array CATEGORIES
#      e.g. ./submit_synthetic_fpr.sh          # both iid and corr
#           ./submit_synthetic_fpr.sh iid      # iid only
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
