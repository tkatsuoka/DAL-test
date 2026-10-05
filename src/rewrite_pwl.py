"""Replace smooth activation functions in an ONNX model with piecewise-linear approximation nodes.

Detects SiLU (Swish) / GELU subgraphs in a trained model and replaces them with
the custom node "PiecewiseLinearApprox" in the si4onnx domain.
si4onnx interprets and executes the ONNX graph itself, so it can handle custom
nodes outside the standard opset (the rewritten model cannot be run with
onnxruntime; to measure the prediction gap from the original function, keep
the original model separately).

Target patterns (as exported by torch.onnx):
    SiLU : X -> Sigmoid -> Mul(X, .)
    GELU : X -> Div(sqrt(2)) or Mul(1/sqrt(2)) -> Erf -> Add(1) -> Mul -> Mul(0.5)

Usage:
    python rewrite_pwl.py --input ../model/syn/sim_diffusion_silu_size16_....onnx \
        --act_fn silu --num_knots 32
    By default, the output path is the input path with the "sim_" prefix replaced by "pwl{num_knots}_".
"""

import argparse
import math
import os
from collections import defaultdict

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

# ---------------------------------------------------------------------------
# Parameters (all settings of the piecewise-linear approximation are managed here)
# ---------------------------------------------------------------------------
CUSTOM_DOMAIN = "si4onnx"
CUSTOM_OP_TYPE = "PiecewiseLinearApprox"
DEFAULT_NUM_KNOTS = 32
# Knot placement method. "curvature" equidistributes |f''|^(1/2) (the asymptotically optimal
# placement for piecewise-linear interpolation); it concentrates knots near the origin, where
# the curvature is high, and is therefore accurate even with few knots.
# "uniform" places knots at equal spacing (for comparison).
DEFAULT_KNOT_METHOD = "curvature"
KNOT_METHODS = ("curvature", "uniform")
# Range where the SiLU / GELU nonlinearity is concentrated. Outside this range, we
# extrapolate along the asymptotes (TAIL_SLOPES below)
DEFAULT_KNOT_RANGE = (-8.0, 8.0)
# Asymptotic slopes (left tail, right tail) of each activation. SiLU and GELU both approach 0 and x
TAIL_SLOPES = {"silu": (0.0, 1.0), "gelu": (0.0, 1.0)}
# Operators that must not remain on the input-dependent path after the rewrite.
# si4onnx cannot treat them as piecewise linear (Sigmoid/Exp are passed through with b=None)
SMOOTH_OP_TYPES = {"Sigmoid", "Erf", "Tanh", "Exp", "Softmax"}
# Operators whose output has no linear dependence on the input (they break the dependency)
DEPENDENCE_BREAKING_OP_TYPES = {"RandomNormalLike", "Shape", "ConstantOfShape"}


def activation_values(act_fn: str, x: np.ndarray) -> np.ndarray:
    """Compute the exact values of the target activation function in float64."""
    t = torch.from_numpy(np.asarray(x, dtype=np.float64))
    if act_fn == "silu":
        return torch.nn.functional.silu(t).numpy()
    if act_fn == "gelu":
        return torch.nn.functional.gelu(t).numpy()
    raise ValueError(f"Unsupported activation function: {act_fn}")


def _curvature_knots(
    act_fn: str,
    num_knots: int,
    knot_range: tuple[float, float],
    grid_size: int = 200001,
) -> np.ndarray:
    """Return knots that equidistribute |f''(x)|^(1/2).

    This is the asymptotically optimal knot placement for piecewise-linear interpolation
    (the interpolation error is equalized across segments). On a fine grid, we build the
    cumulative distribution of the density |f''|^(1/2) from a numerical second derivative
    and place the knots at its equally spaced quantiles. Both endpoints (x_min, x_max)
    are always included.
    """
    x = np.linspace(*knot_range, grid_size)
    values = activation_values(act_fn, x)
    step = x[1] - x[0]
    curvature = np.abs(np.gradient(np.gradient(values, step), step))
    density = np.sqrt(curvature)
    cdf = np.cumsum(density)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    knots = np.interp(np.linspace(0.0, 1.0, num_knots), cdf, x)
    if not np.all(np.diff(knots) > 0):
        raise ValueError(
            "curvature-based knots are not strictly increasing; "
            "use method='uniform' or reduce num_knots"
        )
    return knots


def make_pwl_params(
    act_fn: str,
    num_knots: int = DEFAULT_NUM_KNOTS,
    knot_range: tuple[float, float] = DEFAULT_KNOT_RANGE,
    method: str = DEFAULT_KNOT_METHOD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the piecewise-linear approximation parameters (knots, slopes, intercepts).

    Builds a polyline (linear interpolant) that matches the true function values at the
    knots, and extrapolates continuously outside the approximation range with the same
    slopes as the asymptotes. Segment j covers [knots[j-1], knots[j]) (j=0 is the left
    tail and j=M the right tail; the number of segments is the number of knots + 1).
    The knot placement is selected by method (default: curvature).
    """
    if num_knots < 2:
        raise ValueError("num_knots must be >= 2")
    if method not in KNOT_METHODS:
        raise ValueError(f"Unknown knot method: {method} (available: {KNOT_METHODS})")
    x_min, x_max = knot_range
    if method == "curvature":
        knots = _curvature_knots(act_fn, num_knots, knot_range)
    else:
        knots = np.linspace(x_min, x_max, num_knots, dtype=np.float64)
    values = activation_values(act_fn, knots)
    left_slope, right_slope = TAIL_SLOPES[act_fn]

    inner_slopes = np.diff(values) / np.diff(knots)
    slopes = np.concatenate([[left_slope], inner_slopes, [right_slope]])
    # Continuity: set each intercept so that the segment passes through the true value at a knot
    # (its left-end knot; the left tail uses knots[0], its right end)
    intercepts = np.empty(num_knots + 1, dtype=np.float64)
    intercepts[0] = values[0] - left_slope * knots[0]
    intercepts[1:-1] = values[:-1] - inner_slopes * knots[:-1]
    intercepts[-1] = values[-1] - right_slope * knots[-1]
    return knots, slopes, intercepts


def make_pwl_node(
    input_name: str,
    output_name: str,
    node_name: str,
    knots: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
) -> onnx.NodeProto:
    """Create a PiecewiseLinearApprox custom node (attributes are float64 tensors)."""
    return helper.make_node(
        CUSTOM_OP_TYPE,
        inputs=[input_name],
        outputs=[output_name],
        name=node_name,
        domain=CUSTOM_DOMAIN,
        knots=numpy_helper.from_array(knots),
        slopes=numpy_helper.from_array(slopes),
        intercepts=numpy_helper.from_array(intercepts),
    )


# ---------------------------------------------------------------------------
# Graph analysis utilities
# ---------------------------------------------------------------------------
def _build_consumer_map(graph) -> dict[str, list]:
    consumers = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)
    return consumers


def _build_constant_map(graph) -> dict[str, np.ndarray]:
    """Build a name -> ndarray lookup of the values of initializers and Constant nodes."""
    values = {}
    for init in graph.initializer:
        values[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    values[node.output[0]] = numpy_helper.to_array(attr.t)
    return values


def _is_scalar_close(constants: dict, name: str, target: float) -> bool:
    if name not in constants:
        return False
    arr = constants[name]
    return arr.size == 1 and np.isclose(float(arr.reshape(-1)[0]), target, rtol=1e-4)


def _sole_consumer(consumers: dict, tensor_name: str):
    nodes = consumers.get(tensor_name, [])
    return nodes[0] if len(nodes) == 1 else None


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------
def _find_silu_patterns(graph) -> list[dict]:
    """Detect SiLU patterns (X -> Sigmoid -> Mul(X, .)).

    Returns
    -------
    list[dict]
        {"input": name of X, "output": output name,
         "removed_outputs": output names of the nodes to remove,
         "anchor_output": output name of the node that marks the replacement position
         (the last node of the pattern)}
    """
    consumers = _build_consumer_map(graph)
    patterns = []
    for node in graph.node:
        if node.op_type != "Sigmoid":
            continue
        x_name = node.input[0]
        mul = _sole_consumer(consumers, node.output[0])
        if mul is None or mul.op_type != "Mul":
            continue
        if set(mul.input) != {x_name, node.output[0]}:
            continue
        # Identify nodes by their output tensor names (id() of protobuf wrappers is not stable)
        patterns.append(
            {
                "input": x_name,
                "output": mul.output[0],
                "removed_outputs": {node.output[0], mul.output[0]},
                "anchor_output": mul.output[0],
            }
        )
    return patterns


def _find_gelu_patterns(graph) -> list[dict]:
    """Detect GELU (erf variant) patterns.

    Assumes the subgraph for 0.5 * x * (1 + erf(x / sqrt(2))) exported by torch.onnx:
        X -> Div(sqrt(2)) or Mul(1/sqrt(2)) -> Erf -> Add(1) -> Mul -> Mul(0.5) -> Y
    The two Mul nodes (multiplying by X / by 0.5) may appear in either order.
    """
    consumers = _build_consumer_map(graph)
    constants = _build_constant_map(graph)
    patterns = []
    for erf in graph.node:
        if erf.op_type != "Erf":
            continue
        # The input of Erf is produced by the node that scales X by 1/sqrt(2)
        scale = next(
            (n for n in graph.node if erf.input[0] in n.output),
            None,
        )
        if scale is None:
            continue
        if scale.op_type == "Div" and _is_scalar_close(
            constants, scale.input[1], math.sqrt(2.0)
        ):
            x_name = scale.input[0]
        elif scale.op_type == "Mul":
            const_inputs = [i for i in scale.input if _is_scalar_close(constants, i, 1.0 / math.sqrt(2.0))]
            if len(const_inputs) != 1:
                continue
            x_name = next(i for i in scale.input if i != const_inputs[0])
        else:
            continue

        # Erf -> Add(1)
        add = _sole_consumer(consumers, erf.output[0])
        if add is None or add.op_type != "Add":
            continue
        if not any(
            _is_scalar_close(constants, i, 1.0) for i in add.input if i != erf.output[0]
        ):
            continue

        # Add -> Mul -> Mul (multiply by X and 0.5 in either order)
        mul1 = _sole_consumer(consumers, add.output[0])
        if mul1 is None or mul1.op_type != "Mul":
            continue
        mul2 = _sole_consumer(consumers, mul1.output[0])
        if mul2 is None or mul2.op_type != "Mul":
            continue
        others1 = [i for i in mul1.input if i != add.output[0]]
        others2 = [i for i in mul2.input if i != mul1.output[0]]
        if len(others1) != 1 or len(others2) != 1:
            continue
        pair = (others1[0], others2[0])
        is_x_then_half = pair[0] == x_name and _is_scalar_close(constants, pair[1], 0.5)
        is_half_then_x = _is_scalar_close(constants, pair[0], 0.5) and pair[1] == x_name
        if not (is_x_then_half or is_half_then_x):
            continue

        patterns.append(
            {
                "input": x_name,
                "output": mul2.output[0],
                "removed_outputs": {
                    scale.output[0],
                    erf.output[0],
                    add.output[0],
                    mul1.output[0],
                    mul2.output[0],
                },
                "anchor_output": mul2.output[0],
            }
        )
    return patterns


PATTERN_FINDERS = {"silu": _find_silu_patterns, "gelu": _find_gelu_patterns}


# ---------------------------------------------------------------------------
# Graph rewriting
# ---------------------------------------------------------------------------
def _prune_dead_nodes(graph):
    """Remove nodes (e.g., constants) that are no longer referenced anywhere."""
    output_names = {o.name for o in graph.output}
    while True:
        used = set()
        for node in graph.node:
            used.update(node.input)
        dead = [
            node
            for node in graph.node
            if not (set(node.output) & (used | output_names))
        ]
        if not dead:
            break
        for node in dead:
            graph.node.remove(node)


def _check_no_smooth_ops_on_input_path(graph):
    """Verify that no smooth activation consuming an input-dependent tensor remains.

    si4onnx passes Sigmoid / Exp etc. through with b=None, so if any of them remain on
    the input-dependent path, it silently returns incorrect (invalid) inference results.
    Assuming the nodes are in topological order, the dependent set is propagated in a
    single pass.
    """
    dependent = {inp.name for inp in graph.input}
    offending = []
    for node in graph.node:
        depends = any(name in dependent for name in node.input)
        if depends and node.op_type in SMOOTH_OP_TYPES:
            offending.append(f"{node.op_type}({node.name})")
        if depends and node.op_type not in DEPENDENCE_BREAKING_OP_TYPES:
            dependent.update(node.output)
    if offending:
        raise RuntimeError(
            "Smooth activation nodes remain on the input-dependent path: "
            + ", ".join(offending)
            + ". Selective inference on this model would be invalid."
        )


def rewrite_model(
    model: onnx.ModelProto,
    act_fn: str,
    num_knots: int = DEFAULT_NUM_KNOTS,
    knot_range: tuple[float, float] = DEFAULT_KNOT_RANGE,
    method: str = DEFAULT_KNOT_METHOD,
) -> tuple[onnx.ModelProto, int]:
    """Replace the act_fn subgraphs in the model with PWL custom nodes.

    Returns
    -------
    (rewritten_model, num_replaced)
    """
    knots, slopes, intercepts = make_pwl_params(act_fn, num_knots, knot_range, method)
    graph = model.graph
    patterns = PATTERN_FINDERS[act_fn](graph)

    removed_outputs = set()
    anchor_to_pattern = {}
    for i, pat in enumerate(patterns):
        removed_outputs.update(pat["removed_outputs"])
        anchor_to_pattern[pat["anchor_output"]] = (i, pat)

    # Insert the custom node at the anchor (the last node of the pattern) to keep topological order
    new_nodes = []
    for node in graph.node:
        if node.output[0] in anchor_to_pattern:
            i, pat = anchor_to_pattern[node.output[0]]
            new_nodes.append(
                make_pwl_node(
                    input_name=pat["input"],
                    output_name=pat["output"],
                    node_name=f"pwl_{act_fn}_{i}",
                    knots=knots,
                    slopes=slopes,
                    intercepts=intercepts,
                )
            )
        elif node.output[0] not in removed_outputs:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    _prune_dead_nodes(graph)

    # Register the custom domain in opset_import
    if not any(imp.domain == CUSTOM_DOMAIN for imp in model.opset_import):
        imp = model.opset_import.add()
        imp.domain = CUSTOM_DOMAIN
        imp.version = 1

    _check_no_smooth_ops_on_input_path(graph)
    return model, len(patterns)


def default_output_path(input_path: str, num_knots: int) -> str:
    """Return the output path with the "sim_" prefix replaced by "pwl{K}_"."""
    directory, filename = os.path.split(input_path)
    if filename.startswith("sim_"):
        filename = f"pwl{num_knots}_" + filename[len("sim_"):]
    else:
        filename = f"pwl{num_knots}_" + filename
    return os.path.join(directory, filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Path to the input ONNX model")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: pwl{K}_ prefix)")
    parser.add_argument("--act_fn", type=str, default="silu", choices=["silu", "gelu"])
    parser.add_argument("--num_knots", type=int, default=DEFAULT_NUM_KNOTS,
                        help="Number of knots K (the number of segments is K + 1)")
    parser.add_argument("--knot_method", type=str, default=DEFAULT_KNOT_METHOD,
                        choices=list(KNOT_METHODS),
                        help="Knot placement method (curvature: curvature-equidistributed, uniform: equally spaced)")
    parser.add_argument("--x_min", type=float, default=DEFAULT_KNOT_RANGE[0])
    parser.add_argument("--x_max", type=float, default=DEFAULT_KNOT_RANGE[1])
    args = parser.parse_args()

    model = onnx.load(args.input)
    model, num_replaced = rewrite_model(
        model, args.act_fn, args.num_knots, (args.x_min, args.x_max), args.knot_method
    )
    if num_replaced == 0:
        raise RuntimeError(
            f"No {args.act_fn} pattern was found in {args.input}. "
            "Check that the model was exported with the expected activation."
        )

    output_path = args.output or default_output_path(args.input, args.num_knots)
    onnx.save(model, output_path)
    print(f"Replaced {num_replaced} {args.act_fn} subgraph(s) with {CUSTOM_OP_TYPE}.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
