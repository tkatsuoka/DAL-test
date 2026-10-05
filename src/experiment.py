import os
import argparse
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import onnx
import si4onnx
from si4onnx.operators import AverageFilter, InputDiff, L1Norm
from environment import set_env
from model_paths import resolve_onnx_path, model_result_tag
from tqdm import tqdm
from dataset import (
    generate_images_non_iid,
    generate_images_corr,
    generate_images_iid,
)

# Synthetic data categories: Gaussian noise (iid / corr) and the non-Gaussian
# noise families used in the robustness experiment.
NON_GAUSSIAN_CATEGORIES = {"skewnorm", "exponnorm", "gennormsteep", "gennormflat", "t"}
SYNTHETIC_CATEGORIES = {"iid", "corr"} | NON_GAUSSIAN_CATEGORIES


def resolve_results_base(exhaustive: bool, act_fn: str, pwl_knots: int) -> str:
    """Resolve the base directory for saving results (the separation rules live here).

    - exhaustive (interval counting) results go to ../results/intervals/ so that they
      do not mix with the main results
    - results of PWL-approximated models go one level deeper, to pwl/{act}{K}/
    With the default settings (non-exhaustive, ReLU) this returns ../results.
    """
    results_base = "../results/intervals" if exhaustive else "../results"
    model_tag = model_result_tag(act_fn, pwl_knots)
    if model_tag:
        results_base = f"{results_base}/{model_tag}"
    return results_base


def fn(args):
    (
        input_x,
        ref_x,
        mask,
        category,
        thr,
        image_size,
        var,
        seed,
        idx,
        n_jobs,
        exhaustive,
        act_fn,
        pwl_knots,
        model_seed,
    ) = args

    if n_jobs == 1:
        set_env(parallel=False)
        torch.set_num_threads(1)
    else:
        set_env(parallel=True)

    input_x = input_x.clone().detach().to(dtype=torch.float64)
    ref_x = ref_x.clone().detach().to(dtype=torch.float64)

    # Choose the model:
    #   model_seed given (independent-model experiment, one p-value = one model): train a
    #       throwaway model for this iteration inside the worker. The ONNX model lives only in
    #       memory and is not saved to disk. One worker = one core = one model (training uses
    #       one thread), and many workers train and run inference in parallel.
    #   otherwise: load the shared trained model resolved from the settings.
    if model_seed is not None:
        if act_fn != "relu" or pwl_knots != 0:
            raise ValueError("model_seed (in-job training) supports only ReLU models")
        # Lazy import so that the training-only dependencies (onnxsim etc.) are not needed
        # for the regular experiments
        from model import train_synthetic_model_ephemeral

        onnx_model = train_synthetic_model_ephemeral(image_size, model_seed, act_fn)
    else:
        # Resolve the model path including the activation function and the PWL setting
        # (SiLU / GELU models can be used for inference only after the approximation, pwl_knots > 0)
        onnx_path = resolve_onnx_path(category, image_size, act_fn, pwl_knots)
        onnx_model = onnx.load(onnx_path)

    si_model = si4onnx.load(
        model=onnx_model,
        hypothesis=si4onnx.ReferenceMeanDiff(
            threshold=thr,
            post_process=[InputDiff(), AverageFilter(), L1Norm()],
        ),
        seed=seed,
    )

    try:
        oc_result = si_model.inference(
            (input_x, ref_x),
            var=var,
            mask=mask,
            inference_mode="over_conditioning",
        )
        start_time = time.time()
        pp_result = si_model.inference(
            (input_x, ref_x),
            var=var,
            mask=mask,
            inference_mode="parametric",
            n_jobs=n_jobs,
            max_iter=2e7,
        )
        calc_time = time.time() - start_time
        selective_p = pp_result.p_value
        oc_p = oc_result.p_value
        naive_p = oc_result.naive_p_value()
        z = oc_result.stat

        output = si_model.output
        salient_region = si_model.roi
        score_map = si_model.score_map

        # Exhaustive mode (only with --exhaustive):
        # search the whole parametric line to collect the interval information for counting.
        # The outputs above (selective_p etc.) are already fixed and are not affected by this re-run.
        searched_intervals = None
        truncated_intervals = None
        search_count = None
        detect_count = None
        if exhaustive:
            ex_result = si_model.inference(
                (input_x, ref_x),
                var=var,
                mask=mask,
                inference_mode="exhaustive",
                n_jobs=n_jobs,
                max_iter=2e7,
            )
            searched_intervals = ex_result.searched_intervals
            truncated_intervals = ex_result.truncated_intervals
            search_count = ex_result.search_count
            detect_count = ex_result.detect_count

        # Compute permutation p-value
        # Pool and permute the input and reference images to build the distribution of |z|
        # under the null. construct_hypothesis takes the (input, reference) tuple and var
        # (reference_data is set inside it, so it does not need to be assigned beforehand).
        perm_rng = np.random.default_rng(seed)
        corr_z_list = []
        B = 1000
        # About half of the permuted samples give an empty ROI, so no hypothesis can be formed
        # (NoHypothesisError); the error limit allows for this (with a limit equal to B the
        # loop would always stop early)
        max_permutation_error = 10 * B
        cnt = 0
        permutation_error = 0
        while cnt < B:
            if permutation_error > max_permutation_error:
                print(
                    f"permutation failed: idx={idx}, "
                    f"success={cnt}/{B}, error={permutation_error}"
                )
                return None
            try:
                x_permutated = torch.cat([input_x, ref_x], dim=0)
                x_permutated = (
                    x_permutated.view(-1)[
                        perm_rng.permutation(x_permutated.numel())
                    ]
                    .reshape(x_permutated.shape)
                    .double()
                )

                input_x_permutated = x_permutated[:1]
                ref_x_permutated = x_permutated[1:]

                si_model.construct_hypothesis(
                    (input_x_permutated, ref_x_permutated), var
                )
                permutation_z = si_model.si_calculator.stat
                corr_z_list.append(np.abs(permutation_z))
                cnt += 1
            except Exception:
                permutation_error += 1
                continue
        permutation_p_value = 1 / B * np.sum(np.array(corr_z_list) > np.abs(z))
    except:
        print(None)
        traceback.print_exc()
        return None

    return (
        selective_p,
        oc_p,
        naive_p,
        z,
        output,
        salient_region,
        ref_x,
        permutation_p_value,
        calc_time,
        searched_intervals,
        truncated_intervals,
        search_count,
        detect_count,
    )


def experiment(
    category: str,
    image_size: int,
    thr: float,
    signal: float,
    seed: int,
    number_of_workers: int,
    num_iter: int,
    n_jobs: int = 1,
    distance: float = None,
    exhaustive: bool = False,
    act_fn: str = "relu",
    pwl_knots: int = 0,
    model_seed_base: int | None = None,
    **kwargs,
):
    print(
        f"category: {category}",
        f"image_size: {image_size}",
        f"signal: {signal}",
        f"seed: {seed}",
    )
    if category not in SYNTHETIC_CATEGORIES:
        raise ValueError(
            f"Unknown category: {category} (choose from {sorted(SYNTHETIC_CATEGORIES)})"
        )

    match category:
        case "iid":
            input_image_list, _, _ = generate_images_iid(
                num=num_iter,
                img_size=image_size,
                scale=1,
                signal=signal,
                seed=seed,
            )
            reference_image_list, _, _ = generate_images_iid(
                num=num_iter,
                img_size=image_size,
                scale=1,
                signal=0,
                seed=int(seed + 2023),
            )
            input_image_list = torch.from_numpy(input_image_list).to(torch.float64)
            reference_image_list = torch.from_numpy(reference_image_list).to(
                torch.float64
            )
            mask_list = [None] * num_iter
            var = 1.0

        case "corr":
            input_image_list, cov = generate_images_corr(
                num=num_iter, img_size=image_size, signal=signal, seed=seed
            )
            reference_image_list, cov = generate_images_corr(
                num=num_iter, img_size=image_size, signal=0, seed=int(seed + 2023)
            )
            input_image_list = torch.from_numpy(input_image_list).to(torch.float64)
            reference_image_list = torch.from_numpy(reference_image_list).to(
                torch.float64
            )
            ZERO = np.zeros((image_size**2, image_size**2))
            top = np.concatenate([cov, ZERO], axis=1)
            bottom = np.concatenate([ZERO, cov], axis=1)
            var = np.concatenate([top, bottom], axis=0)
            mask_list = [None] * num_iter

        case "skewnorm" | "exponnorm" | "gennormsteep" | "gennormflat" | "t":
            image_list = generate_images_non_iid(
                image_size, category, distance, num_samples=num_iter * 2
            )
            input_image_list = image_list[:num_iter]
            reference_image_list = image_list[num_iter:]
            input_image_list = torch.from_numpy(input_image_list).to(torch.float64)
            reference_image_list = torch.from_numpy(reference_image_list).to(
                torch.float64
            )
            var = 1.0
            mask_list = [None] * num_iter

    seeds = np.arange(num_iter)

    with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        args = (
            (
                input_image_list[idx : idx + 1, :, :, :],
                reference_image_list[idx : idx + 1, :, :, :],
                mask_list[idx],
                category,
                thr,
                image_size,
                var,
                seeds[idx],
                idx,
                n_jobs,
                exhaustive,
                act_fn,
                pwl_knots,
                # In the independent-model experiment, each iteration trains a throwaway model with its own seed
                model_seed_base + idx if model_seed_base is not None else None,
            )
            for idx in range(num_iter)
        )
        outputs = list(tqdm(executor.map(fn, args), total=num_iter))

        p_values = []
        input_images = []
        output_images = []
        permutation_p_values = []
        salient_regions = []
        reference_images = []
        times = []
        # Interval information from the exhaustive mode (one entry per iteration)
        searched_intervals = []
        truncated_intervals = []
        search_count = []
        detect_count = []
        # Model seeds of the successful iterations in the independent-model experiment (to track failures)
        model_seeds = []
        for i in range(num_iter):
            if (
                outputs[i] is None
                or None in outputs[i][0:4]  # selective, oc, naive, z
                or outputs[i][5] is None  # salient_region
            ):
                continue
            if model_seed_base is not None:
                model_seeds.append(model_seed_base + i)
            p_values.append(outputs[i][0:4])
            input_images.append(input_image_list[i : i + 1, :, :, :])
            output_images.append(outputs[i][4])
            salient_regions.append(outputs[i][5])
            reference_images.append(outputs[i][6])
            permutation_p_values.append(outputs[i][7])
            times.append(outputs[i][8])
            searched_intervals.append(outputs[i][9])
            truncated_intervals.append(outputs[i][10])
            search_count.append(outputs[i][11])
            detect_count.append(outputs[i][12])

        result = np.array([p_value for p_value in p_values if p_value is not None])
        if result.size == 0:
            raise RuntimeError(
                f"All {num_iter} iterations failed (category={category}, "
                f"size={image_size}, seed={seed}). Check the worker tracebacks."
            )
        selective_p_values = result[:, 0]
        oc_p_values = result[:, 1]
        naive_p_values = result[:, 2]
        z = result[:, 3]

    print("naive:", len(naive_p_values))

    result_dict = {
        "category": category,
        "image_size": image_size,
        "signal": signal,
        "num_iter": num_iter,
        "seed": seed,
        "selective_p_values": selective_p_values,
        "oc_p_values": oc_p_values,
        "naive_p_values": naive_p_values,
        "permutation_p_values": permutation_p_values,
        "z": z,
        "time": times,
    }

    # Save the interval-counting results only in the exhaustive mode
    if exhaustive:
        result_dict["searched_intervals"] = searched_intervals
        result_dict["truncated_intervals"] = truncated_intervals
        result_dict["search_count"] = search_count
        result_dict["detect_count"] = detect_count

    # In the independent-model experiment, also record the model seed of each successful iteration
    if model_seed_base is not None:
        result_dict["model_seeds"] = model_seeds

    if signal == 0:
        error = "fpr"
    else:
        error = "power"

    # Keep the exhaustive (interval counting) and PWL results separate from the main
    # FPR/power results (the rules are in resolve_results_base).
    # With the default settings, results_base=../results.
    results_base = resolve_results_base(exhaustive, act_fn, pwl_knots)

    match category:
        case "iid":
            save_dir = f"{results_base}/iid/{error}/"
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"iid_size{image_size}_signal{int(signal)}_seed{seed}.pickle"
            with open(os.path.join(save_dir, file_name), "wb") as f:
                pickle.dump(result_dict, f)
        case "corr":
            save_dir = f"{results_base}/corr/{error}/"
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"corr_size{image_size}_signal{int(signal)}_seed{seed}.pickle"
            with open(os.path.join(save_dir, file_name), "wb") as f:
                pickle.dump(result_dict, f)
        case "skewnorm" | "exponnorm" | "gennormsteep" | "gennormflat" | "t":
            save_dir = f"{results_base}/robust/{error}/"
            os.makedirs(save_dir, exist_ok=True)
            file_name = (
                f"{category}_size{image_size}_distance{distance}_seed{seed}.pickle"
            )
            with open(os.path.join(save_dir, file_name), "wb") as f:
                pickle.dump(result_dict, f)


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("-category", "--category", type=str)
    cmdline_parser.add_argument("-size", "--size", type=int)
    cmdline_parser.add_argument("-thr", "--thr", type=float)
    cmdline_parser.add_argument("-signal", "--signal", type=float)
    cmdline_parser.add_argument("-workers", "--workers", type=int)
    cmdline_parser.add_argument("-n_jobs", "--n_jobs", type=int, default=1)
    cmdline_parser.add_argument("-iter", "--iter", type=int)
    cmdline_parser.add_argument("-seed", "--seed", type=int)
    cmdline_parser.add_argument("-distance", "--distance", type=float, default=None)
    # Exhaustive mode that collects the interval counts (searched/truncated intervals, search/detect count)
    cmdline_parser.add_argument("--exhaustive", action="store_true")
    # Base value of the model seed for the independent-model experiment (one p-value = one model).
    # If given, each iteration idx trains a throwaway model with seed = base + idx inside the worker
    # (one core = one model; models are not saved to disk). Shifting the base per category prevents
    # the same model from being reproduced across categories.
    # If omitted, the shared trained models are used.
    cmdline_parser.add_argument(
        "-model_seed_base", "--model_seed_base", type=int, default=None
    )
    # Activation function and number of knots of the piecewise-linear approximation
    # (silu/gelu require pwl_knots > 0)
    cmdline_parser.add_argument("-act_fn", "--act_fn", type=str, default="relu")
    cmdline_parser.add_argument("-pwl_knots", "--pwl_knots", type=int, default=0)

    args, unknowns = cmdline_parser.parse_known_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    experiment(
        category=args.category,
        image_size=args.size,
        thr=args.thr,
        signal=args.signal,
        seed=args.seed,
        number_of_workers=args.workers,
        n_jobs=args.n_jobs,
        num_iter=args.iter,
        distance=args.distance,
        exhaustive=args.exhaustive,
        act_fn=args.act_fn,
        pwl_knots=args.pwl_knots,
        model_seed_base=args.model_seed_base,
    )
