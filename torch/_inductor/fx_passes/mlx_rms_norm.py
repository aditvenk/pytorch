from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

CONVERT_TO_FLOAT32 = {
    torch.ops.aten._to_copy.default,
    torch.ops.prims.convert_element_type.default,
}

CONVERT_BACK = {
    torch.ops.aten._to_copy.default,
    torch.ops.prims.convert_element_type.default,
}


def _mlx_rms_norm_stub(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, x.shape[-1:], eps=eps, weight=weight)


def _is_convert_node(node: Node, targets: Sequence) -> bool:
    return node.op == "call_function" and node.target in targets


def fuse_mlx_rms_norm(gm: GraphModule) -> bool:
    changed = False
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.mul.Tensor:
            continue

        weight_node = None
        convert_back_node = None
        for arg in node.args:
            if isinstance(arg, Node) and _is_convert_node(arg, CONVERT_BACK):
                convert_back_node = arg
            else:
                weight_node = arg

        if convert_back_node is None or weight_node is None:
            continue

        mul_float_node = convert_back_node.args[0]
        if (
            not isinstance(mul_float_node, Node)
            or mul_float_node.op != "call_function"
            or mul_float_node.target != torch.ops.aten.mul.Tensor
        ):
            continue

        convert32_node = None
        rsqrt_node = None
        for arg in mul_float_node.args:
            if isinstance(arg, Node) and arg.target == torch.ops.aten.rsqrt.default:
                rsqrt_node = arg
            elif isinstance(arg, Node) and _is_convert_node(arg, CONVERT_TO_FLOAT32):
                convert32_node = arg

        if convert32_node is None or rsqrt_node is None:
            continue

        input_node = convert32_node.args[0]
        add_node = rsqrt_node.args[0] if len(rsqrt_node.args) > 0 else None
        if (
            not isinstance(add_node, Node)
            or add_node.target != torch.ops.aten.add.Tensor
            or len(add_node.args) < 2
        ):
            continue

        mean_node = add_node.args[0]
        eps_value = add_node.args[1]
        if not isinstance(eps_value, (int, float)):
            continue

        if not isinstance(mean_node, Node) or not str(mean_node.target).startswith(
            "aten.mean"
        ):
            continue

        pow_node = mean_node.args[0]
        if (
            not isinstance(pow_node, Node)
            or pow_node.target != torch.ops.aten.pow.Tensor_Scalar
            or len(pow_node.args) != 2
        ):
            continue
        if pow_node.args[1] != 2:
            continue
        if pow_node.args[0] is not convert32_node:
            continue

        with graph.inserting_before(node):
            fused = graph.call_function(
                _mlx_rms_norm_stub,
                args=(input_node, weight_node, float(eps_value)),
            )
        node.replace_all_uses_with(fused)
        fused.meta = dict(node.meta)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        gm.recompile()
    return changed
