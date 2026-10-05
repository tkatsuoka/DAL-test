# Job-submission scripts

Shell scripts that submit the synthetic-data experiments to a PBS cluster with `qsub`.
Every job runs `src/experiment.py` through the job script `jobs/experiment.pbs`.

> **Note.** These scripts were written for a specific PBS cluster. Before using them, adapt the
> following to your environment:
> the queue name (`#PBS -q` in `jobs/experiment.pbs`, which is what `qsub` actually uses, and
> `QUEUE` in `common.sh`), the CPU count and memory (`#PBS -l select=...` in `jobs/experiment.pbs`
> and `WORKERS` in `common.sh`, which should match `ncpus`), and the Python path (`PYTHON` in
> `common.sh` and the `../.venv/bin/python3` line in `jobs/experiment.pbs`).

## Usage

Create the Python environment first by running `uv sync` in the repository root (this creates `.venv/`).

**Run the drivers (`submit_*.sh`) from the `scripts/` directory.**
`experiment.py` saves results to the relative path `../results/...`, so the working directory is
assumed to be `scripts/` (`../results` is `results/` in the repository root).
PBS logs (stdout/stderr) go to `eo/` in the repository root.

```bash
cd scripts
./submit_synthetic_fpr.sh       # synthetic data, FPR (iid + corr)
./submit_synthetic_power.sh     # synthetic data, power (iid + corr)
./submit_robust.sh              # robustness (non-Gaussian noise)
./submit_intervals.sh           # interval counts (exhaustive): iid/corr x size 8/16/32/64
./submit_pwl_synthetic.sh fpr   # piecewise-linear (PWL) approximation of SiLU models
```

The default sweeps are listed below. All drivers use the threshold `--thr 0.8` of
`ReferenceMeanDiff`, and all except `submit_intervals.sh` run `--workers 96` (`WORKERS` in `common.sh`).

| Driver | Categories | Sweep | `--iter` per job | Jobs | Output |
| --- | --- | --- | --- | --- | --- |
| `submit_synthetic_fpr.sh` | `iid`, `corr` | size 8/16/32/64, signal 0, seed 0-9 | 300 (size 8), 120 (others) | 80 | `../results/{cat}/fpr/` |
| `submit_synthetic_power.sh` | `iid`, `corr` | size 64, signal 1/2/3/4, seed 0-9 | 110 | 80 | `../results/{cat}/power/` |
| `submit_robust.sh` | `skewnorm`, `exponnorm`, `gennormsteep`, `gennormflat`, `t` | size 16, signal 0, distance 0.01/0.02/0.03/0.04, seed 0-9 | 120 | 200 | `../results/robust/fpr/` |
| `submit_intervals.sh` | `iid`, `corr` | size 8/16/32/64 | see below | 20 | `../results/intervals/` |
| `submit_pwl_synthetic.sh` | `iid`, `corr` | see below | same as FPR / power | 80 per mode (K=8) | `../results/pwl/` |

To change a sweep, edit the variables at the top of each script (`SIZES`, `SEEDS`, `SIGNALS`, etc.).

### Selecting categories

All drivers accept category names as arguments (for `submit_pwl_synthetic.sh`, after the mode).
**With no arguments, every category is submitted**; with arguments, only the given categories are submitted.

```bash
./submit_synthetic_fpr.sh iid        # iid only
./submit_synthetic_power.sh corr     # corr only
./submit_robust.sh t skewnorm        # only the listed distributions (several allowed)
./submit_intervals.sh iid            # iid only
```

An unknown category name stops the script and prints the list of allowed names.

### Interval-count experiment

**Design: one p-value (= one interval computation) uses one independently trained model and its own data.**
If one model were reused for several p-values, the p-values would be dependent through the model,
so a fresh throwaway model is trained for every iteration. Each iteration runs in its own worker
process, with **1 core = 1 model**, and the workers run "train, then compute intervals" in parallel
(`experiment.py --model_seed_base` calls `train_synthetic_model_ephemeral` in `model.py`).
The trained models live only in memory and are **never written to disk**; only the result pickle is saved.
The goal is **100 independent interval counts** per (category, size). More iterations are run to allow
for failures, and only the first 100 successful models are used when aggregating.

#### Collecting interval counts (`submit_intervals.sh`)

The jobs run `experiment.py --exhaustive --model_seed_base {B}`, which searches the whole parametric
line and **adds** the following entries to `result_dict` in the result pickle:

- `searched_intervals` / `truncated_intervals`: the searched intervals / the truncated intervals
- `search_count` / `detect_count`: the number of searched / detected intervals
- `model_seeds`: the model seed of each successful iteration (to keep track of failures)

```bash
./submit_intervals.sh                 # iid and corr (size 8/16/32/64, 20 jobs in total)
./submit_intervals.sh iid             # iid only (10 jobs)
```

- The total number of iterations (= models) is set in `model_seed_count()` in `common.sh`:
  **size 8: 300 (3x the target of 100), size 16/32/64: 120 each (1.2x)**.
- A node has 96 cores. To finish in a single round, the iterations are **split evenly into chunks**
  of at most 96, and each chunk is one job (size 8: 4 jobs x 75 iterations; other sizes: 2 jobs x
  60 iterations; `--workers` = `--iter` = chunk size).
- Model seed = `base + running index` (iid: base=0, corr: base=1000), so that deterministic training
  does not reproduce the same model in both categories. The training-data seed is offset to
  `10000 + model seed`, so it never overlaps with the evaluation data.
- Each chunk draws its evaluation data from its own random stream (`--seed <chunk index>`),
  independently for every iteration.
- Training and inference use one core per worker (`--n_jobs 1`). Only synthetic data and ReLU models are supported.
- **Results are kept in a separate tree**, `../results/intervals/` (one pickle per chunk).
  Example for iid: `../results/intervals/iid/fpr/iid_size{8..64}_signal0_seed{0..3}.pickle`.
- `--exhaustive` and `--model_seed_base` are off by default, so the other drivers are not affected
  (their output paths and results are unchanged).
- If some iterations fail, the rest continue and only the successful ones are stored in the pickle.
  100 models over all chunks are enough for aggregation.
- One iteration takes "training on a single core (long for size 64) + exhaustive search".

#### Plotting (`src/plot_intervals.py`)

[`src/plot_intervals.py`](../src/plot_intervals.py) plots the results with **image size on the x-axis**.

```bash
cd src
python plot_intervals.py     # writes PDFs to ../plots/figures/intervals/
```

- For each (category, size), the chunk pickles are concatenated in chunk order and **the first 100
  successful models** are used (with fewer than 100, it prints a warning and plots what is available).
- One figure is written per metric, with the categories overlaid as lines. By default the metrics are
  `search_count` (number of intervals, `num_of_intervals.pdf`) and `time` (computation time of the
  parametric SI inference, `intervals_time.pdf`). `detect_count` is also stored in the pickles and can be
  added to `METRICS`. The number of models used (`N_ADOPT`) and other options are set at the top of the script.

### Piecewise-linear (PWL) experiment

`submit_pwl_synthetic.sh` measures FPR / power on synthetic data with SiLU diffusion models whose
activations are replaced by a piecewise-linear approximation (`PiecewiseLinearApprox`).
The number of knots K is set by `KNOTS` at the top of the script (default: K=8 only; list several
values to sweep over K). Knots are placed by equidistributing the curvature (`KNOT_METHOD=curvature`),
which gives high accuracy with few linear pieces.

```bash
./submit_pwl_synthetic.sh fpr          # FPR: iid, corr x size 8/16/32/64 x K x seed
./submit_pwl_synthetic.sh fpr iid      # FPR: iid only
./submit_pwl_synthetic.sh power iid    # power: iid only (size 64, signal 1-4)
```

- **Requirement**: trained models `model/syn/sim_diffusion_silu_size{S}_...onnx`
  (run `python model.py --category syn --act_fn silu` in `src/` first).
- Missing `pwl{K}_` models are **generated automatically** by `rewrite_pwl.py` on the login node before
  submission (graph rewriting only; takes a few seconds). If the trained `sim_` model is missing,
  the script prints an error and stops.
- The `pwl{K}_` file names do not record the knot placement method. After changing `KNOT_METHOD`,
  delete the existing `pwl` models so that they are regenerated.
- Number of jobs: categories x sizes x K values x seeds for FPR, i.e. 2 x 4 x 1 x 10 = **80 jobs** with
  the default K=8 (320 jobs with four K values). Narrow it down with category arguments or by editing
  `KNOTS` / `FPR_SIZES`.
- **Results are kept in a separate tree**, `../results/pwl/{act_fn}{K}/`
  (e.g. `../results/pwl/silu8/iid/fpr/iid_size16_signal0_seed0.pickle`), apart from the ReLU results.
- The activation function is set by `ACT_FN` at the top of the script (`silu` by default; `gelu` is also accepted).

## Shared settings (`common.sh`)

`common.sh` is sourced by every driver and collects the settings that may need changing.

| Name | Default | Role |
| --- | --- | --- |
| `PROJECT_DIR`, `SCRIPTS_DIR` | derived from the location of `common.sh` | repository root and `scripts/` |
| `PYTHON` | `${PROJECT_DIR}/.venv/bin/python3` | Python of the uv-managed environment (used on the login node to run `rewrite_pwl.py`) |
| `EO_DIR` | `${PROJECT_DIR}/eo` | PBS stdout/stderr directory (created automatically) |
| `QUEUE` | `cpu-large` | queue name for reference; the queue actually used is `#PBS -q` in `jobs/experiment.pbs` |
| `WORKERS` | `96` | `--workers` of `experiment.py`; keep it equal to `ncpus` in `jobs/experiment.pbs` |
| `model_seed_count()` | 300 (size 8), 120 (others) | number of models per size in the interval-count experiment |
| `submit_experiment` | | submits one `experiment.py` job: `submit_experiment <job name> "<MYARGS string>"` |
| `resolve_categories` | | turns the category arguments into the array `CATEGORIES` |

The experiment arguments (`--category`, `--size`, `--signal`, ...) are passed to the job in the
environment variable `MYARGS` (`qsub -v MYARGS=...`), and `jobs/experiment.pbs` runs
`../.venv/bin/python3 ../src/experiment.py $MYARGS` from the submission directory.

## Files

| File | Role |
| --- | --- |
| `common.sh` | Shared settings (Python path, queue, worker count, job sizes of the interval-count experiment) and submission helpers |
| `jobs/experiment.pbs` | PBS job script that runs `src/experiment.py` |
| `submit_synthetic_fpr.sh` | Synthetic data, FPR (iid / corr) |
| `submit_synthetic_power.sh` | Synthetic data, power (iid / corr) |
| `submit_robust.sh` | Robustness to non-Gaussian noise |
| `submit_intervals.sh` | Interval counts with one independently trained model per p-value |
| `submit_pwl_synthetic.sh` | FPR / power of PWL-approximated SiLU models |
