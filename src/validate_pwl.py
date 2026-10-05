"""Validation script for the PiecewiseLinearApprox (piecewise-linear approximation) implementation.

Requires no training and runs locally on CPU only. Checks:
  1. Continuity of the approximation parameters, and monotone decrease of the
     approximation error as the number of knots increases
  2. The custom node's forward exactly matches a NumPy reference implementation
  3. The forward / forward_si outputs and the truncation interval [l, u] match those
     of a ReLU decomposition (a mathematically equivalent graph built only from
     already-supported ops)
     (the intervals should be identical, since the segment-selection event is the
     sign pattern over all knots)
  4. Export -> simplify -> rewrite SiLU / GELU CNNs, and check that (a, b, l, u)
     from forward_si is consistent with forward for any z in the interval
  5. End-to-end check on a small untrained diffusion model
     (a real graph containing RandomNormalLike, Resize, Concat, etc.)

Usage:
    cd src && python validate_pwl.py
"""

import os
import tempfile

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify

from rewrite_pwl import make_pwl_params, rewrite_model
from si4onnx.nn import NN

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED = 1234
INPUT_SHAPE = (1, 1, 8, 8)  # input shape of the single-activation graphs in Tests 2 and 3
NUM_KNOTS_SWEEP = [8, 32, 128]  # knot counts for checking the monotone decrease of the error
NUM_Z_SAMPLES = 7  # number of z values sampled from the interval in the forward_si check
ATOL_EXACT = 1e-10  # tolerance for comparisons that should be mathematically exact
ATOL_CONSISTENCY = 1e-7  # tolerance for forward_si consistency on deep graphs

NEG_INF = torch.tensor(-torch.inf).double()
POS_INF = torch.tensor(torch.inf).double()


# ---------------------------------------------------------------------------
# Reference implementation and test graph construction
# ---------------------------------------------------------------------------
def pwl_reference(x: np.ndarray, knots, slopes, intercepts) -> np.ndarray:
    """NumPy reference implementation of the piecewise-linear function (left-closed,
    right-open interval assignment, as in bucketize)."""
    index = np.searchsorted(knots, x, side="right")
    return slopes[index] * x + intercepts[index]


def build_pwl_graph(shape, knots, slopes, intercepts) -> onnx.ModelProto:
    """Build a minimal float64 graph: input -> PiecewiseLinearApprox -> output."""
    from rewrite_pwl import make_pwl_node

    node = make_pwl_node("x", "y", "pwl_test", knots, slopes, intercepts)
    graph = helper.make_graph(
        [node],
        "pwl_single_op",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, shape)],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, shape)],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("si4onnx", 1)],
    )


def build_relu_sum_graph(shape, knots, slopes, intercepts) -> onnx.ModelProto:
    """Graph of the same piecewise-linear function built only from existing ops (Relu/Mul/Add/Sub).

    y = intercepts[0] + slopes[0] * x + sum_k (slopes[k+1] - slopes[k]) * ReLU(x - knots[k])
    This is the standard form of a continuous piecewise-linear function, so the graph is
    mathematically equivalent to PiecewiseLinearApprox.
    """
    coefs = np.diff(slopes)
    nodes = []
    inits = [
        numpy_helper.from_array(np.float64(slopes[0]), name="s0"),
        numpy_helper.from_array(np.float64(intercepts[0]), name="c0"),
    ]
    nodes.append(helper.make_node("Mul", ["x", "s0"], ["lin"]))
    nodes.append(helper.make_node("Add", ["lin", "c0"], ["acc_0"]))
    acc = "acc_0"
    for k in range(len(knots)):
        inits.append(numpy_helper.from_array(np.float64(knots[k]), name=f"t_{k}"))
        inits.append(numpy_helper.from_array(np.float64(coefs[k]), name=f"coef_{k}"))
        nodes.append(helper.make_node("Sub", ["x", f"t_{k}"], [f"d_{k}"]))
        nodes.append(helper.make_node("Relu", [f"d_{k}"], [f"r_{k}"]))
        nodes.append(helper.make_node("Mul", [f"r_{k}", f"coef_{k}"], [f"m_{k}"]))
        nodes.append(helper.make_node("Add", [acc, f"m_{k}"], [f"acc_{k + 1}"]))
        acc = f"acc_{k + 1}"
    graph = helper.make_graph(
        nodes,
        "relu_sum",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, shape)],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, shape)],
        initializer=inits,
    )
    # Rename the output of the last node to y
    graph.node[-1].output[0] = "y"
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TinyCNN(torch.nn.Module):
    """Small CNN (with BatchNorm) for testing the rewrite."""

    def __init__(self, act_fn: torch.nn.Module):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            act_fn,
            torch.nn.Conv2d(4, 4, 3, padding=1),
            act_fn,
            torch.nn.Conv2d(4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def export_and_rewrite(model, dummy_input, act_fn, num_knots, work_dir, tag):
    """Apply export -> onnxsim simplify -> PWL rewrite to a torch model and return the result."""
    path = os.path.join(work_dir, f"{tag}.onnx")
    model.eval()
    torch.onnx.export(model, dummy_input, path)
    simplified, ok = simplify(onnx.load(path))
    assert ok, "onnxsim simplification failed"
    rewritten, num_replaced = rewrite_model(simplified, act_fn, num_knots)
    assert num_replaced > 0, f"no {act_fn} pattern found in {tag}"
    print(f"  [{tag}] replaced {num_replaced} {act_fn} subgraph(s)")
    return rewritten


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
def check_forward_si_consistency(model_proto, x, seed=None, label=""):
    """Check that (a, b, l, u) from forward_si is consistent with forward for any z in the interval.

    Runs forward_si at z = 0 with the observed x as a and a random direction b, and then
    checks forward(x + b z') == a + b z' at several points z' in the resulting interval (l, u).
    """
    net = NN(model_proto, seed=seed)
    x = x.double()
    y_obs = net.forward(x)  # required to freeze the RandomNormalLike noise

    gen = torch.Generator().manual_seed(SEED)
    b = torch.randn(x.shape, generator=gen, dtype=torch.float64)
    _, a_out, b_out, l, u = net.forward_si(x, x, b, NEG_INF, POS_INF, 0.0)
    assert l < 0.0 < u, f"[{label}] observed z=0 is not inside (l, u) = ({l}, {u})"

    # The reconstruction at z = 0 must match the observed forward output
    err_obs = (y_obs - a_out).abs().max().item()
    assert err_obs < ATOL_CONSISTENCY, f"[{label}] mismatch at z=0: {err_obs}"

    # Linearity must hold at several points inside the interval
    lo = max(float(l), -3.0)
    hi = min(float(u), 3.0)
    margin = (hi - lo) * 0.01
    max_err = 0.0
    for z in np.linspace(lo + margin, hi - margin, NUM_Z_SAMPLES):
        y = net.forward(x + b * z)
        pred = a_out + b_out * z
        max_err = max(max_err, (y - pred).abs().max().item())
    assert max_err < ATOL_CONSISTENCY, f"[{label}] forward_si inconsistency: {max_err}"
    print(f"  [{label}] interval=({float(l):.6f}, {float(u):.6f}), max_err={max_err:.2e}")


def test_1_pwl_params():
    print("Test 1: PWL parameter continuity and error decay")
    # Outside the approximation range we extrapolate along the asymptotes (0 on the left, x on the right),
    # so |silu(+-8)| ~ 2.7e-3 is
    # a lower bound on the overall error. We therefore check the monotone decrease using the
    # error inside the approximation range, which is not affected by this floor
    grid = np.linspace(-7.9, 7.9, 10001)
    silu_true = grid / (1.0 + np.exp(-grid))
    for method in ["curvature", "uniform"]:
        prev_err = np.inf
        for num_knots in NUM_KNOTS_SWEEP:
            knots, slopes, intercepts = make_pwl_params("silu", num_knots, method=method)
            # Continuity: the left and right segments take the same value at each knot
            left = slopes[:-1] * knots + intercepts[:-1]
            right = slopes[1:] * knots + intercepts[1:]
            assert np.max(np.abs(left - right)) < 1e-12, "discontinuity at knots"
            # Monotone decrease of the approximation error
            err = np.max(
                np.abs(pwl_reference(grid, knots, slopes, intercepts) - silu_true)
            )
            print(f"  method={method}, num_knots={num_knots}: max_error={err:.2e}")
            assert err < prev_err, "approximation error did not decrease"
            prev_err = err
        assert prev_err < 1e-3, "error with the largest num_knots is too large"
    print("  PASS")


def test_2_forward_exactness():
    print("Test 2: custom node forward matches numpy reference")
    knots, slopes, intercepts = make_pwl_params("silu", 32)
    net = NN(build_pwl_graph(INPUT_SHAPE, knots, slopes, intercepts))
    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(INPUT_SHAPE, generator=gen, dtype=torch.float64) * 4
    y = net.forward(x).numpy()
    y_ref = pwl_reference(x.numpy(), knots, slopes, intercepts)
    err = np.max(np.abs(y - y_ref))
    assert err < ATOL_EXACT, f"forward mismatch: {err}"
    print(f"  max_err={err:.2e}  PASS")


def test_3_relu_sum_equivalence():
    print("Test 3: equivalence with ReLU-sum decomposition (forward / forward_si)")
    knots, slopes, intercepts = make_pwl_params("silu", 16)
    net_pwl = NN(build_pwl_graph(INPUT_SHAPE, knots, slopes, intercepts))
    net_relu = NN(build_relu_sum_graph(INPUT_SHAPE, knots, slopes, intercepts))

    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(INPUT_SHAPE, generator=gen, dtype=torch.float64) * 4
    b = torch.randn(INPUT_SHAPE, generator=gen, dtype=torch.float64)

    err_fwd = (net_pwl.forward(x) - net_relu.forward(x)).abs().max().item()
    assert err_fwd < ATOL_EXACT, f"forward mismatch: {err_fwd}"

    _, a1, b1, l1, u1 = net_pwl.forward_si(x, x, b, NEG_INF, POS_INF, 0.0)
    _, a2, b2, l2, u2 = net_relu.forward_si(x, x, b, NEG_INF, POS_INF, 0.0)
    err_a = (a1 - a2).abs().max().item()
    err_b = (b1 - b2).abs().max().item()
    err_l = abs(float(l1) - float(l2))
    err_u = abs(float(u1) - float(u2))
    print(
        f"  err_a={err_a:.2e}, err_b={err_b:.2e}, "
        f"interval=({float(l1):.6f}, {float(u1):.6f}), err_l={err_l:.2e}, err_u={err_u:.2e}"
    )
    assert max(err_a, err_b) < ATOL_EXACT, "a/b mismatch with ReLU decomposition"
    assert max(err_l, err_u) < 1e-9, "truncated interval mismatch with ReLU decomposition"
    print("  PASS")


def test_4_cnn_consistency(work_dir):
    print("Test 4: exported CNN (SiLU / GELU) forward_si consistency")
    gen = torch.Generator().manual_seed(SEED)
    dummy = torch.randn(1, 1, 16, 16)
    x = torch.randn((1, 1, 16, 16), generator=gen, dtype=torch.float64)
    for act_fn, module in [("silu", torch.nn.SiLU()), ("gelu", torch.nn.GELU())]:
        torch.manual_seed(SEED)
        model = export_and_rewrite(TinyCNN(module), dummy, act_fn, 32, work_dir, f"cnn_{act_fn}")
        check_forward_si_consistency(model, x, label=f"cnn_{act_fn}")
    print("  PASS")


def test_5_diffusion_end_to_end(work_dir):
    print("Test 5: small diffusion model end-to-end")
    from model import DiffusionModel

    torch.manual_seed(SEED)
    diffusion = DiffusionModel(
        test_num_timesteps=4, sampling_step=2, act_fn="silu", in_ch=1
    )
    dummy = torch.randn(1, 1, 16, 16)
    model = export_and_rewrite(diffusion, dummy, "silu", 32, work_dir, "diffusion_silu")

    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn((1, 1, 16, 16), generator=gen, dtype=torch.float64)
    check_forward_si_consistency(model, x, seed=SEED, label="diffusion_silu")
    print("  PASS")


def main():
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as work_dir:
        test_1_pwl_params()
        test_2_forward_exactness()
        test_3_relu_sum_equivalence()
        test_4_cnn_consistency(work_dir)
        test_5_diffusion_end_to_end(work_dir)
    print("All validation tests passed.")


if __name__ == "__main__":
    main()
