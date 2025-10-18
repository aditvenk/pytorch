from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import sympy
import torch
from torch.fx import GraphModule, Node

from ..utils import IndentedBuffer, ValueWithLineMap, normalize_name


@dataclass
class _TensorMeta:
    dtype: Optional[torch.dtype]
    device: Optional[torch.device]
    shape: Optional[Iterable[Any]]


@dataclass
class _MLXValue:
    name: str
    meta: Optional[_TensorMeta]


@dataclass
class _OutputSpec:
    expr: str
    meta: Optional[_TensorMeta]
    convert: bool


def _extract_tensor_meta(node: Node) -> Optional[_TensorMeta]:
    meta_val = node.meta.get("val") if hasattr(node, "meta") else None
    if meta_val is None:
        return None
    if isinstance(meta_val, torch.Tensor):
        dtype = meta_val.dtype
        device = meta_val.device
        shape = tuple(meta_val.shape)
    else:
        dtype = getattr(meta_val, "dtype", None)
        device = getattr(meta_val, "device", None)
        shape = getattr(meta_val, "shape", None)
        if shape is not None:
            shape = tuple(shape)
    return _TensorMeta(dtype=dtype, device=device, shape=shape)


def _format_dtype(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return "None"
    return repr(dtype)


def _format_device(device: Optional[torch.device]) -> str:
    if device is None:
        return "torch.device('cpu')"
    return f"torch.device({str(device)!r})"


class MLXCodegenError(Exception):
    pass


class MLXGraphCodegen:
    """
    Translate an FX GraphModule into a Python source string that executes the model using
    MLX operations. The generated module mirrors the structure emitted by Inductor for
    other Python backends, exposing a `call` entry point that accepts the original
    positional arguments tuple.
    """

    def __init__(self, graph_lowering):
        self._graph = graph_lowering
        self._gm: GraphModule = graph_lowering.orig_gm
        self._values: Dict[Node, _MLXValue] = {}
        self._name_counter = itertools.count()
        self._body = IndentedBuffer()
        self._converted_constants: Dict[str, str] = {}
        self._output_specs: list[_OutputSpec] = []
        self._input_metas: list[Optional[_TensorMeta]] = []

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def generate(self) -> ValueWithLineMap:
        """
        Produce a ValueWithLineMap containing Python source that evaluates the FX graph with MLX.
        """
        self._emit_function_body()

        module = IndentedBuffer()
        module.writelines(
            [
                "import numpy as _np",
                "import torch",
                "from torch.utils import dlpack as _dlpack",
                "",
                "try:",
                "    import mlx.core as mx",
                "except ImportError as exc:",
                "    raise RuntimeError(",
                "        \"TORCH_INDUCTOR_MLX requires the 'mlx' package to be installed.\"",
                "    ) from exc",
                "",
                "",
                "def _to_mx(value):",
            ]
        )
        with module.indent():
            module.writelines(
                [
                    "if isinstance(value, mx.array):",
                    "    return value",
                    "if not isinstance(value, torch.Tensor):",
                    "    return value",
                    "tensor = value",
                    "if not tensor.is_contiguous():",
                    "    tensor = tensor.contiguous()",
                    "if hasattr(tensor, 'device') and tensor.device.type != 'cpu':",
                    "    try:",
                    "        return mx.array(_dlpack.to_dlpack(tensor))",
                    "    except Exception:",
                    "        pass",
                    "cpu_tensor = tensor.detach().to(\"cpu\")",
                    "return mx.array(_np.asarray(cpu_tensor))",
                ]
            )

        module.writelines(
            [
                "",
                "def _from_mx(value, *, dtype, device):",
            ]
        )
        with module.indent():
            module.writelines(
                [
                    "tensor = None",
                    "try:",
                    "    tensor = _dlpack.from_dlpack(value.__dlpack__())",
                    "except Exception:",
                    "    tensor = torch.from_numpy(_np.asarray(value))",
                    "if dtype is not None:",
                    "    tensor = tensor.to(dtype=dtype)",
                    "if device is not None:",
                    "    tensor = tensor.to(device=device)",
                    "# Copy to decouple the PyTorch view from the MLX buffer so autograd",
                    "# does not see mutations that happen on the MLX side during backward.",
                    "return tensor.clone()",
                ]
            )

        module.writelines(["", "def _mlx_impl(*args):"])
        with module.indent():
            module.splice(self._body)

        module.writelines(["", "_MLX_COMPILED = mx.compile(_mlx_impl)"])

        module.writelines(["", "_OUTPUT_SPECS = ["])
        with module.indent():
            for spec in self._output_specs:
                if spec.convert:
                    meta = spec.meta
                    dtype_expr = _format_dtype(meta.dtype if meta else None)
                    device_expr = _format_device(meta.device if meta else None)
                    module.writeline(f"({dtype_expr}, {device_expr}),")
                else:
                    module.writeline("None,")
        module.writelines(["]"])

        module.writelines(["", "_INPUT_METAS = ["])
        with module.indent():
            for meta in self._input_metas:
                if meta is None:
                    module.writeline("None,")
                    continue
                dtype_expr = _format_dtype(meta.dtype)
                device_expr = _format_device(meta.device)
                if meta.shape is None:
                    shape_literal = "()"
                else:
                    shape_values = tuple(
                        self._to_int(dim) for dim in meta.shape
                    )
                    shape_literal = repr(shape_values)
                module.writeline(
                    f"({dtype_expr}, {device_expr}, {shape_literal}),"
                )
        module.writelines(["]", "", "def call(args):"])
        with module.indent():
            module.writelines(
                [
                    "mx_args = [_to_mx(value) for value in args]",
                    "mx_results = _MLX_COMPILED(*mx_args)",
                    "if not _OUTPUT_SPECS:",
                ]
            )
            with module.indent():
                module.writeline("return tuple()")
            module.writelines(
                [
                    "if not isinstance(mx_results, (tuple, list)):",
                ]
            )
            with module.indent():
                module.writeline("mx_results = (mx_results,)")
            module.writelines(["else:"])
            with module.indent():
                module.writeline("mx_results = tuple(mx_results)")
            module.writelines(["assert len(mx_results) == len(_OUTPUT_SPECS)"])
            module.writelines(
                [
                    "results = []",
                    "for _value, _spec in zip(mx_results, _OUTPUT_SPECS):",
                ]
            )
            with module.indent():
                module.writelines(["if _spec is None:"])
                with module.indent():
                    module.writeline("results.append(_value)")
                module.writelines(["else:"])
                with module.indent():
                    module.writelines(
                        [
                            "dtype, device = _spec",
                            "results.append(_from_mx(_value, dtype=dtype, device=device))",
                        ]
                    )
            module.writeline("return tuple(results)")

        module.writelines(
            [
                "",
                "def benchmark_compiled_module(*_args, **_kwargs):",
            ]
        )
        with module.indent(1):
            module.writeline(
                'raise NotImplementedError("MLX backend benchmark harness not implemented")'
            )

        module.writelines(
            [
                "",
                "",
                'if __name__ == "__main__":',
            ]
        )
        with module.indent():
            module.writelines(
                [
                    "import torch",
                    "",
                    "def _random_tensor(meta):",
                ]
            )
            with module.indent():
                module.writelines(
                    [
                        "if meta is None:",
                        "    shape = ()",
                        "    dtype = torch.float32",
                        '    device = torch.device("cpu")',
                        "else:",
                        "    dtype, device, shape = meta",
                        "    dtype = dtype or torch.float32",
                        '    device = device or torch.device("cpu")',
                        "return torch.randn(shape, dtype=dtype, device=device)",
                    ]
                )
            module.writelines(
                [
                    "",
                    "args = []",
                    "for meta in _INPUT_METAS:",
                    "    args.append(_random_tensor(meta))",
                    "",
                    "result = call(tuple(args))",
                    'print("MLX compiled module output shapes:")',
                    "if isinstance(result, tuple):",
                ]
            )

            with module.indent():
                module.writelines(
                    [
                        "for idx, item in enumerate(result):",
                        "    if isinstance(item, torch.Tensor):",
                        '        print(f"  output[{idx}]: {tuple(item.shape)} {item.dtype}")',
                        "    else:",
                        '        print(f"  output[{idx}]: {type(item).__name__}")',
                    ]
                )
            module.writelines(
                [
                    "else:",
                ]
            )
            with module.indent():
                module.writelines(
                    [
                        "if isinstance(result, torch.Tensor):",
                        '    print(f"  output: {tuple(result.shape)} {result.dtype}")',
                        "else:",
                        '    print(f"  output: {type(result).__name__}")',
                    ]
                )

        return module.getvaluewithlinemap()

    # ------------------------------------------------------------------ #
    # Core translation helpers
    # ------------------------------------------------------------------ #

    def _emit_function_body(self) -> None:
        self._output_specs = []
        self._input_metas = []
        placeholder_nodes = [
            node for node in self._gm.graph.nodes if node.op == "placeholder"
        ]
        if placeholder_nodes:
            self._body.writeline("_args = tuple(args)")
        for index, node in enumerate(placeholder_nodes):
            mlx_name = self._new_var(
                normalize_name(node.name or f"arg{index}")
            )
            meta = _extract_tensor_meta(node)
            self._body.writeline(f"{mlx_name} = _args[{index}]")
            self._values[node] = _MLXValue(mlx_name, meta)
            self._input_metas.append(meta)
        if placeholder_nodes:
            self._body.writeline("")

        output_node: Optional[Node] = None
        for node in self._gm.graph.nodes:
            if node.op in ("placeholder", "output"):
                if node.op == "output":
                    output_node = node
                continue
            if node.op == "get_attr":
                self._emit_get_attr(node)
            elif node.op == "call_function":
                self._emit_call_function(node)
            elif node.op == "call_method":
                self._emit_call_method(node)
            elif node.op == "call_module":
                raise MLXCodegenError(
                    f"MLX codegen does not yet support call_module nodes; encountered {node.target!r}"
                )
            else:
                raise MLXCodegenError(
                    f"Unsupported FX node kind for MLX codegen: {node.op}"
                )

        if output_node is None:
            raise MLXCodegenError("FX graph lacks an output node")
        self._emit_output(output_node)

    def _emit_get_attr(self, node: Node) -> None:
        target = node.target
        assert isinstance(target, str)
        constant_name = self._ensure_constant(target)
        torch_name = self._new_var(f"{normalize_name(target)}_torch")
        mlx_name = self._new_var(normalize_name(node.name or target))
        self._body.writeline(f"{torch_name} = {constant_name}")
        self._body.writeline(f"{mlx_name} = _to_mx({torch_name})")
        self._values[node] = _MLXValue(mlx_name, _extract_tensor_meta(node))

    def _emit_call_function(self, node: Node) -> None:
        target = node.target
        if target == torch.ops.aten.relu.default:
            value = self._value(node.args[0])
            self._values[node] = self._assign(node, f"mx.maximum({value}, 0)")
        elif target == torch.ops.aten.neg.default:
            value = self._value(node.args[0])
            self._values[node] = self._assign(node, f"-({value})")
        elif target in _UNARY_OPS:
            self._emit_unary(node, _UNARY_OPS[target])
        elif target in _BINARY_OPS:
            self._emit_binary(node, target)
        elif target == torch.ops.aten.matmul.default:
            lhs, rhs = (self._value(arg) for arg in node.args[:2])
            result = self._assign(node, f"mx.matmul({lhs}, {rhs})")
            self._values[node] = result
        elif target in _BMM_TARGETS:
            self._emit_bmm(node)
        elif target in _ADDMM_TARGETS:
            input_tensor = self._value(node.args[0])
            mat1 = self._value(node.args[1])
            mat2 = self._value(node.args[2])
            beta = node.kwargs.get("beta")
            alpha = node.kwargs.get("alpha")
            if beta is None and len(node.args) > 3:
                beta = node.args[3]
            if alpha is None and len(node.args) > 4:
                alpha = node.args[4]
            beta = 1 if beta is None else beta
            alpha = 1 if alpha is None else alpha
            beta_expr = self._format_scalar(beta)
            alpha_expr = self._format_scalar(alpha)
            beta_is_one = isinstance(beta, (int, float)) and beta == 1
            alpha_is_one = isinstance(alpha, (int, float)) and alpha == 1

            matmul_expr = f"mx.matmul({mat1}, {mat2})"
            if not alpha_is_one:
                matmul_expr = f"({alpha_expr}) * ({matmul_expr})"

            input_expr = input_tensor
            if not beta_is_one:
                input_expr = f"({beta_expr}) * ({input_expr})"

            expr = f"{input_expr} + {matmul_expr}"
            self._values[node] = self._assign(node, expr)
        elif target in _SOFTMAX_TARGETS:
            self._emit_softmax(node)
        elif target in (
            torch.ops.aten.reshape.default,
            torch.ops.aten.view.default,
        ):
            self._emit_reshape(node)
        elif target == torch.ops.aten.permute.default:
            self._emit_permute(node, tuple(node.args[1]))
        elif target == torch.ops.aten.transpose.int:
            dim0 = int(self._to_int(node.args[1]))
            dim1 = int(self._to_int(node.args[2]))
            axes = list(range(len(node.meta["val"].shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            self._emit_permute(node, tuple(axes))
        elif target == torch.ops.aten.unsqueeze.default:
            self._emit_unsqueeze(node)
        elif target == torch.ops.aten.mm.default:
            self._emit_mm(node)
        elif target in _FMA_TARGETS:
            self._emit_fma(node)
        elif target in _SQUEEZE_TARGETS:
            self._emit_squeeze(node)
        elif target == torch.ops.aten.clone.default:
            val = self._value(node.args[0])
            self._values[node] = self._assign(node, val)
        elif target in _EXPAND_TARGETS:
            self._emit_expand(node)
        elif target in _AMAX_TARGETS:
            self._emit_reduce(node, "max")
        elif target in _SUM_TARGETS:
            self._emit_reduce(node, "sum")
        elif target == torch.ops.aten.le.Scalar:
            self._emit_comparison_scalar(node, "mx.less_equal")
        elif target == torch.ops.aten.full.default:
            self._emit_full(node)
        elif target == torch.ops.aten.where.self:
            cond = self._value(node.args[0])
            a = self._value(node.args[1])
            b = self._value(node.args[2])
            self._values[node] = self._assign(
                node, f"mx.where({cond}, {a}, {b})"
            )
        elif target == torch.ops.aten.to.dtype:
            tensor = self._value(node.args[0])
            dtype = _format_dtype(node.args[1])
            self._values[node] = self._assign(
                node, f"mx.astype({tensor}, {dtype})"
            )
        else:
            raise MLXCodegenError(
                f"MLX codegen does not yet support aten op {getattr(target, '__name__', target)}"
            )

    def _emit_call_method(self, node: Node) -> None:
        method = node.target
        if method in ("view", "reshape"):
            self._emit_reshape(node)
        elif method == "permute":
            self._emit_permute(node, tuple(node.args[1]))
        else:
            raise MLXCodegenError(
                f"Unsupported Tensor method for MLX codegen: {method}"
            )

    def _emit_output(self, node: Node) -> None:
        output = node.args[0]
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        self._output_specs = []
        for out in outputs:
            if isinstance(out, Node):
                mlx_value = self._values[out]
                meta = mlx_value.meta or _extract_tensor_meta(out)
                self._output_specs.append(
                    _OutputSpec(expr=mlx_value.name, meta=meta, convert=True)
                )
            else:
                self._output_specs.append(
                    _OutputSpec(expr=repr(out), meta=None, convert=False)
                )
        exprs = [spec.expr for spec in self._output_specs]
        tuple_expr = ", ".join(exprs)
        if len(exprs) == 1:
            tuple_expr = f"{tuple_expr},"
        self._body.writeline(f"return ({tuple_expr})")

    # ------------------------------------------------------------------ #
    # Individual op emitters
    # ------------------------------------------------------------------ #

    def _emit_unary(self, node: Node, func_name: str) -> None:
        value = self._value(node.args[0])
        expr = f"{func_name}({value})"
        self._values[node] = self._assign(node, expr)

    def _emit_binary(self, node: Node, target) -> None:
        lhs = self._value(node.args[0])
        rhs = self._value(node.args[1])
        if target == torch.ops.aten.add.Tensor:
            alpha = node.kwargs.get("alpha", 1)
            if alpha != 1:
                rhs = f"({rhs}) * {alpha}"
            expr = f"{lhs} + {rhs}"
        elif target == torch.ops.aten.mul.Tensor:
            expr = f"{lhs} * {rhs}"
        elif target == torch.ops.aten.sub.Tensor:
            expr = f"{lhs} - {rhs}"
        elif target == torch.ops.aten.div.Tensor:
            expr = f"{lhs} / {rhs}"
        elif target == torch.ops.aten.maximum.default:
            expr = f"mx.maximum({lhs}, {rhs})"
        elif target == torch.ops.aten.minimum.default:
            expr = f"mx.minimum({lhs}, {rhs})"
        else:
            raise MLXCodegenError(f"Unsupported binary aten op {target}")
        self._values[node] = self._assign(node, expr)

    def _emit_softmax(self, node: Node) -> None:
        tensor = self._value(node.args[0])
        dim = int(node.args[1])
        expr = f"mx.softmax({tensor}, axis={dim})"
        self._values[node] = self._assign(node, expr)

    def _emit_reshape(self, node: Node) -> None:
        tensor = self._value(node.args[0])
        meta = _extract_tensor_meta(node)
        if meta is None or meta.shape is None:
            raise MLXCodegenError(
                "Missing shape metadata for reshape/view operation"
            )
        shape_literal = self._format_shape(meta.shape)
        expr = f"mx.reshape({tensor}, {shape_literal})"
        self._values[node] = self._assign(node, expr, meta=meta)

    def _emit_permute(self, node: Node, axes: Iterable[Any]) -> None:
        tensor = self._value(node.args[0])
        axes_literal = self._format_shape(tuple(axes))
        expr = f"mx.transpose({tensor}, axes={axes_literal})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_expand(self, node: Node) -> None:
        tensor = self._value(node.args[0])
        size = node.args[1]
        if not isinstance(size, (tuple, list)):
            raise MLXCodegenError(
                "Expected expand sizes to be a tuple or list"
            )
        size_literal = self._format_shape(size)
        expr = f"mx.broadcast_to({tensor}, {size_literal})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_unsqueeze(self, node: Node) -> None:
        tensor = self._value(node.args[0])
        if len(node.args) <= 1:
            raise MLXCodegenError("Unsqueeze operation missing axis argument")
        axis = self._to_int(node.args[1])
        axis_literal = str(axis)
        expr = f"mx.expand_dims({tensor}, axis={axis_literal})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_squeeze(self, node: Node) -> None:
        tensor = self._value(node.args[0])
        if len(node.args) > 1:
            dim = self._to_int(node.args[1])
            expr = f"mx.squeeze({tensor}, axis={dim})"
        else:
            expr = f"mx.squeeze({tensor})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_mm(self, node: Node) -> None:
        lhs = self._value(node.args[0])
        rhs = self._value(node.args[1])
        expr = f"mx.matmul({lhs}, {rhs})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_comparison_scalar(self, node: Node, func: str) -> None:
        tensor = self._value(node.args[0])
        scalar = self._format_scalar(node.args[1])
        expr = f"{func}({tensor}, {scalar})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_full(self, node: Node) -> None:
        shape = node.args[0]
        value_expr = self._format_scalar(node.args[1])
        if isinstance(shape, (list, tuple)) and len(shape) == 0:
            expr = f"mx.array({value_expr})"
        else:
            shape_literal = self._format_shape(shape)
            expr = f"mx.full({shape_literal}, {value_expr})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_fma(self, node: Node) -> None:
        x = self._value(node.args[0])
        y = self._value(node.args[1])
        z = self._value(node.args[2])
        expr = f"({x} * {y}) + {z}"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_bmm(self, node: Node) -> None:
        lhs = self._value(node.args[0])
        rhs = self._value(node.args[1])
        expr = f"mx.matmul({lhs}, {rhs})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_reduce(self, node: Node, op: str) -> None:
        input_arg = node.args[0]
        tensor = self._value(input_arg)
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            bool(node.args[2])
            if len(node.args) > 2
            else bool(node.kwargs.get("keepdim", False))
        )

        if dims is None:
            axes_literal = "None"
        else:
            if isinstance(dims, int):
                dims = [dims]
            elif not isinstance(dims, (list, tuple)):
                raise MLXCodegenError(
                    "Expected reduction dims to be int, list, or tuple"
                )

            meta = (
                _extract_tensor_meta(input_arg)
                if isinstance(input_arg, Node)
                else None
            )
            rank = len(meta.shape) if meta and meta.shape is not None else None
            normalized = []
            for dim in dims:
                dim_int = self._to_int(dim)
                if rank is not None and dim_int < 0:
                    dim_int += rank
                normalized.append(dim_int)
            axes_literal = (
                "None"
                if not normalized
                else self._format_shape(tuple(normalized))
            )

        keepdim_literal = "True" if keepdim else "False"
        expr = f"mx.{op}({tensor}, axis={axes_literal}, keepdims={keepdim_literal})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #

    def _assign(
        self, node: Node, expr: str, *, meta: Optional[_TensorMeta] = None
    ) -> _MLXValue:
        name = self._new_var(normalize_name(node.name or "tmp"))
        self._body.writeline(f"{name} = {expr}")
        return _MLXValue(name, meta or _extract_tensor_meta(node))

    def _value(self, arg: Any) -> str:
        if isinstance(arg, Node):
            if arg not in self._values:
                raise MLXCodegenError(
                    f"Value for FX node {arg.name} not computed"
                )
            return self._values[arg].name
        if isinstance(arg, (int, float)):
            return repr(arg)
        if isinstance(arg, bool):
            return "True" if arg else "False"
        raise MLXCodegenError(
            f"Unsupported argument type for MLX expression: {type(arg)}"
        )

    def _format_scalar(self, value: Any) -> str:
        if isinstance(value, Node):
            return self._value(value)
        if isinstance(value, (int, float)):
            return repr(value)
        raise MLXCodegenError(
            f"Unsupported scalar value for MLX expression: {type(value)}"
        )

    def _format_shape(self, shape: Iterable[Any]) -> str:
        formatted = []
        for dim in shape:
            formatted.append(str(self._to_int(dim)))
        return (
            "("
            + ", ".join(formatted)
            + ("," if len(formatted) == 1 else "")
            + ")"
        )

    def _to_int(self, value: Any) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, torch.SymInt):
            return value.node.hint  # type: ignore[return-value]
        if isinstance(value, sympy.Expr):
            return int(self._graph.sizevars.size_hint(value))
        if hasattr(value, "node") and hasattr(value.node, "hint"):
            return int(value.node.hint)
        raise MLXCodegenError(
            f"Unable to resolve symbolic dimension {value!r}"
        )

    def _new_var(self, prefix: str) -> str:
        base = normalize_name(prefix) or "tmp"
        return f"{base}_{next(self._name_counter)}"

    def _ensure_constant(self, target: str) -> str:
        if target in self._converted_constants:
            return self._converted_constants[target]
        value = getattr(self._gm, target)
        if not isinstance(value, torch.Tensor):
            raise MLXCodegenError(
                f"Expected tensor constant for attribute {target!r}, found {type(value)}"
            )
        name = self._graph.add_tensor_constant(value, name=target)
        self._converted_constants[target] = name
        return name


# ---------------------------------------------------------------------- #
# Operator registries
# ---------------------------------------------------------------------- #


def _maybe_get_op(namespace: str, name: str, overload: str = "default"):
    try:
        ns = getattr(torch.ops, namespace)
    except AttributeError:
        return None
    op = getattr(ns, name, None)
    if op is None:
        return None
    try:
        return getattr(op, overload)
    except AttributeError:
        return None


def _maybe_get_aten_op(name: str, overload: str = "default"):
    return _maybe_get_op("aten", name, overload)
    try:
        op = getattr(torch.ops.aten, name)
    except AttributeError:
        return None
    try:
        return getattr(op, overload)
    except AttributeError:
        return None


_UNARY_OPS = {
    _maybe_get_aten_op("abs"): "mx.abs",
    _maybe_get_aten_op("sin"): "mx.sin",
    _maybe_get_aten_op("cos"): "mx.cos",
    _maybe_get_aten_op("exp"): "mx.exp",
    _maybe_get_aten_op("sqrt"): "mx.sqrt",
    _maybe_get_aten_op("tanh"): "mx.tanh",
}
# prune missing ops
_UNARY_OPS = {k: v for k, v in _UNARY_OPS.items() if k is not None}

_BINARY_OPS = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.maximum.default,
    torch.ops.aten.minimum.default,
}

_SOFTMAX_TARGETS = tuple(
    op
    for op in (
        _maybe_get_aten_op("softmax"),
        _maybe_get_aten_op("_softmax"),
    )
    if op is not None
)

_ADDMM_TARGETS = tuple(
    op for op in (_maybe_get_aten_op("addmm"),) if op is not None
)

_EXPAND_TARGETS = tuple(
    op for op in (_maybe_get_aten_op("expand"),) if op is not None
)

_BMM_TARGETS = tuple(
    op for op in (_maybe_get_aten_op("bmm"),) if op is not None
)

_AMAX_TARGETS = tuple(
    op for op in (_maybe_get_aten_op("amax"),) if op is not None
)

_SUM_TARGETS = tuple(
    op
    for op in [
        _maybe_get_aten_op("sum"),
        _maybe_get_aten_op("sum", "default"),
        _maybe_get_aten_op("sum", "dim_IntList"),
        _maybe_get_aten_op("sum", "dim"),
    ]
    if op is not None
)

_SQUEEZE_TARGETS = tuple(
    op
    for op in (
        _maybe_get_aten_op("squeeze"),
        _maybe_get_aten_op("squeeze", "dim"),
    )
    if op is not None
)

_FMA_TARGETS = tuple(
    op
    for op in (
        _maybe_get_op("prims", "fma") if hasattr(torch.ops, "prims") else None,
    )
    if op is not None
)


__all__ = ["MLXGraphCodegen", "MLXCodegenError"]
