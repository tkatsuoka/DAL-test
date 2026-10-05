# DAL-Test: Statistical Test for Diffusion-Based Anomaly Localization

This repository contains the code for the **numerical experiments** in

> Teruyuki Katsuoka, Tomohiro Shiraishi, Daiki Miwa, Vo Nguyen Le Duy, and Ichiro Takeuchi.
> **Statistical Test for Diffusion-Based Anomaly Localization via Selective Inference.**
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=S1df8b69fh)

DAL-Test computes valid *p*-values for the anomalous regions detected by a diffusion model.
A test image is reconstructed by a trained denoising diffusion model, the region whose
(filtered) reconstruction error exceeds a threshold is selected, and the mean pixel values
of the test image and a reference image in that region are compared. Because the region is
selected from the same data, the test is conducted conditionally on the selection event
(selective inference), which keeps the type I error rate at the significance level.

The trained diffusion model (a U-Net with ReLU activations) is exported to ONNX, and the bundled
library [`src/si4onnx/`](src/si4onnx/) executes the ONNX graph to compute the truncation intervals
of the test statistic exactly via parametric programming.

The code covers the experiments on synthetic data:

| Paper | Experiment |
| --- | --- |
| Section 5.1, Figures 3 and 4 | Type I error rate and power (independence and correlation settings) |
| Appendix C | U-Nets with SiLU activations handled via a piecewise-linear approximation |
| Appendix D | Number of searched intervals and computation time |
| Appendix I | Robustness to non-Gaussian noise |

The real-world experiments (BraTS and MVTec AD) are not included.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` / `uv.lock`).
Use Python 3.10 or 3.11 (required by the `torchvision==0.15.2` wheels).

```bash
uv sync --python 3.11
```

All commands below are run from `src/` through `uv run`, because models and results are
written to the relative paths `../model/` and `../results/`.

## Usage

### 1. Train the diffusion models (GPU required)

```bash
cd src
uv run python model.py --category syn                # ReLU U-Nets for image sizes 8, 16, 32, 64
uv run python model.py --category syn --act_fn silu  # SiLU U-Nets (Appendix C)
```

This writes `../model/syn/sim_diffusion_size{8,16,32,64}_timesteps460_step115.onnx`
(the `sim_` files are the onnxsim-simplified models used for inference). Trained models are
not included in this repository.

### 2. Run the experiments

One call of `experiment.py` generates `--iter` test/reference image pairs, computes the
selective, over-conditioned (`w/o-pp`), naive, and permutation *p*-values in parallel
(`--workers` processes), and saves them to `../results/`. Results with `--signal 0` go to
`fpr/`, and results with a positive signal go to `power/`.

```bash
# Type I error rate (Figure 3): --category iid or corr, --size 8/16/32/64
uv run python experiment.py --category iid --size 16 --thr 0.8 --signal 0 \
    --workers 8 --iter 100 --seed 0

# Power (Figure 4): image size 64 with signal 1, 2, 3, or 4
uv run python experiment.py --category corr --size 64 --thr 0.8 --signal 2 \
    --workers 8 --iter 100 --seed 0

# Robustness to non-Gaussian noise (Appendix I):
#   --category skewnorm / exponnorm / gennormsteep / gennormflat / t
#   --distance is the 1-Wasserstein distance from N(0, 1)
uv run python experiment.py --category t --size 16 --thr 0.8 --signal 0 \
    --distance 0.04 --workers 8 --iter 100 --seed 0

# Number of searched intervals (Appendix D): one independently trained model per p-value
uv run python experiment.py --category iid --size 16 --thr 0.8 --signal 0 \
    --workers 8 --iter 100 --seed 0 --exhaustive --model_seed_base 0
```

Image size `--size d` means a `d x d` image, i.e., `n = d^2` pixels
(`--size 8/16/32/64` corresponds to `n = 64/256/1024/4096` in the paper).

### 3. SiLU U-Nets via the piecewise-linear approximation (Appendix C)

Replace each SiLU in the trained model by a piecewise-linear function with 8 knots
(placed by curvature equidistribution), then run the same experiments on the rewritten model.
The approximated network is used both to detect the region and to compute the truncation
intervals, so the selective *p*-value is exactly valid for the detector that is actually used.

```bash
uv run python rewrite_pwl.py \
    --input ../model/syn/sim_diffusion_silu_size16_timesteps460_step115.onnx \
    --act_fn silu --num_knots 8
uv run python experiment.py --category iid --size 16 --thr 0.8 --signal 0 \
    --workers 8 --iter 100 --seed 0 --act_fn silu --pwl_knots 8
```

The rewritten model uses a custom ONNX operator that only si4onnx can execute (not onnxruntime).
Its results are stored separately under `../results/pwl/silu8/`.
`uv run python validate_pwl.py` checks the implementation of the approximation (CPU only,
no trained model needed).

### 4. Batch submission

The experiments in the paper were run as sweeps over image sizes, signals, and seeds on a PBS
cluster. The submission scripts and the exact sweeps are in [`scripts/`](scripts/)
(see [`scripts/README.md`](scripts/README.md)). The queue name, CPU counts, and Python path in
`scripts/common.sh` and `scripts/jobs/experiment.pbs` need to be adapted to your environment.

### 5. Figures

```bash
uv run python plot_fpr.py                          # type I error rate (Figure 3)
uv run python plot_fpr.py --act_fn silu --pwl_knots 8   # type I error rate of the SiLU U-Nets (Appendix C)
uv run python plot_intervals.py                    # number of intervals and computation time (Appendix D)
uv run python plot_pwl_activation.py               # SiLU and its piecewise-linear approximation (Appendix C)
```

Figures are written to `../plots/figures/`.

## Settings used in the paper

| Hyperparameter | Value |
| --- | --- |
| Threshold `--thr` (lambda) | 0.8 |
| Kernel size of the averaging filter | 3 |
| Total diffusion steps T | 1000 |
| Initial step of the reverse process T' | 460 |
| Number of sampling steps | 5 |
| Stochasticity eta | 1 |
| Significance level alpha | 0.05 |

## Repository structure

| Path | Contents |
| --- | --- |
| `src/model.py` | Diffusion model (U-Net), training, and ONNX export |
| `src/experiment.py` | Selective inference experiments (type I error rate, power, robustness, interval counts) |
| `src/dataset.py` | Synthetic data (independence / correlation / non-Gaussian noise) |
| `src/model_paths.py` | Naming convention and path resolution of the ONNX models |
| `src/rewrite_pwl.py` | Rewrites SiLU / GELU in an ONNX graph into a piecewise-linear operator |
| `src/validate_pwl.py` | Tests of the piecewise-linear approximation |
| `src/plot_*.py` | Figures |
| `src/si4onnx/` | Selective inference engine for ONNX models (bundled) |
| `scripts/` | PBS job-submission scripts |

## Citation

```bibtex
@article{katsuoka2026statistical,
  title   = {Statistical Test for Diffusion-Based Anomaly Localization via Selective Inference},
  author  = {Katsuoka, Teruyuki and Shiraishi, Tomohiro and Miwa, Daiki and Duy, Vo Nguyen Le and Takeuchi, Ichiro},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=S1df8b69fh}
}
```

## License

This project is licensed under the MIT License (see [LICENSE](LICENSE)).
