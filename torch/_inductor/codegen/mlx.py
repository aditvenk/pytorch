from __future__ import annotations

import itertools
from dataclasses import dataclass
import operator
from typing import Any, Dict, Iterable, Optional, Sequence

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
    alias_input_idx: Optional[int] = None


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
        return "None"
    return f"torch.device({str(device)!r})"


class MLXCodegenError(Exception):
    pass


_TORCH_DTYPE_TO_MX = {
    torch.bool: "mx.bool_",
    torch.uint8: "mx.uint8",
    torch.int8: "mx.int8",
    torch.int16: "mx.int16",
    torch.int32: "mx.int32",
    torch.int64: "mx.int64",
    torch.float16: "mx.float16",
    torch.bfloat16: "mx.bfloat16",
    torch.float32: "mx.float32",
    torch.float64: "mx.float64",
}


def _torch_dtype_to_mx(dtype: torch.dtype) -> str:
    mx_dtype = _TORCH_DTYPE_TO_MX.get(dtype)
    if mx_dtype is None:
        raise MLXCodegenError(
            f"MLX codegen does not yet support dtype {dtype!r}"
        )
    return mx_dtype


class MLXGraphCodegen:
    """
    Translate an FX GraphModule into a Python source string that executes the model using
    MLX operations. The generated module mirrors the structure emitted by Inductor for
    other Python backends, exposing a `call` entry point that accepts the original
    positional arguments tuple.
    """

    def __init__(
        self,
        graph_lowering,
        mutated_input_idxs: Optional[Sequence[int]] = None,
    ):
        self._graph = graph_lowering
        self._gm: GraphModule = graph_lowering.orig_gm
        self._values: Dict[Node, _MLXValue] = {}
        self._name_counter = itertools.count()
        self._body = IndentedBuffer()
        self._converted_constants: Dict[str, str] = {}
        self._output_specs: list[_OutputSpec] = []
        self._input_metas: list[Optional[_TensorMeta]] = []
        self._placeholder_indices: dict[Node, int] = {}
        self._output_aliases: list[Optional[int]] = []
        self._mutated_input_idxs = (
            set(mutated_input_idxs) if mutated_input_idxs is not None else set()
        )
        self._mutated_input_specs: list[tuple[int, Optional[_TensorMeta]]] = []
        self._additional_mutated_idxs: set[int] = set()
        self._placeholder_nodes: list[Node] = []
        self._mutated_exprs: list[str] = []

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
                "    from mlx.core import fast as _mx_fast",
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

        module.writelines(["", "_OUTPUT_ALIASES = ["])
        with module.indent():
            for alias in self._output_aliases:
                if alias is None:
                    module.writeline("None,")
                else:
                    module.writeline(f"{alias},")
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
        module.writelines(["]"])

        module.writelines(["", "_MUTATED_INPUT_SPECS = ["])
        with module.indent():
            for index, meta in self._mutated_input_specs:
                if meta is None:
                    module.writeline(f"({index}, None, None),")
                    continue
                dtype_expr = _format_dtype(meta.dtype)
                device_expr = _format_device(meta.device)
                module.writeline(f"({index}, {dtype_expr}, {device_expr}),")
        module.writelines(["]", "", "def call(args):"])
        with module.indent():
            module.writeline("mx_args = []")
            module.writeline("for _value in args:")
            with module.indent():
                module.writeline("mx_args.append(_to_mx(_value))")
            module.writeline("mx_results = _MLX_COMPILED(*mx_args)")
            module.writeline("if not isinstance(mx_results, (tuple, list)):")
            with module.indent():
                module.writeline("mx_results = (mx_results,)")
            module.writeline("else:")
            with module.indent():
                module.writeline("mx_results = tuple(mx_results)")
            module.writeline("mutated_count = len(_MUTATED_INPUT_SPECS)")
            module.writeline("if mutated_count:")
            with module.indent():
                module.writeline("if mutated_count > len(mx_results):")
                with module.indent():
                    module.writeline(
                        "raise RuntimeError('MLX kernel did not return enough mutated values')"
                    )
                module.writeline("mutated_values = mx_results[-mutated_count:]")
                module.writeline("mx_outputs = mx_results[:-mutated_count]")
            module.writeline("else:")
            with module.indent():
                module.writeline("mutated_values = tuple()")
                module.writeline("mx_outputs = mx_results")
            module.writeline("if mutated_count:")
            with module.indent():
                module.writeline("if len(mutated_values) != mutated_count:")
                with module.indent():
                    module.writeline(
                        "raise RuntimeError('Unexpected number of mutated values returned by MLX kernel')"
                    )
                module.writeline(
                    "for (_idx, _dtype, _device), _value in zip(_MUTATED_INPUT_SPECS, mutated_values):"
                )
                with module.indent():
                    module.writeline(
                        "if _idx < len(args) and isinstance(args[_idx], torch.Tensor):"
                    )
                    with module.indent():
                        module.writelines(
                            [
                                "_tensor = args[_idx]",
                                "_updated = _from_mx(_value, dtype=_dtype, device=_device)",
                                "_tensor.copy_(",
                                "    _updated.to(device=_tensor.device, dtype=_tensor.dtype)",
                                ")",
                            ]
                        )
            module.writeline("results = []")
            module.writeline("if _OUTPUT_SPECS:")
            with module.indent():
                module.writeline("if len(mx_outputs) != len(_OUTPUT_SPECS):")
                with module.indent():
                    module.writeline(
                        "raise RuntimeError('Unexpected number of MLX outputs')"
                    )
                module.writeline("for _value, _spec in zip(mx_outputs, _OUTPUT_SPECS):")
                with module.indent():
                    module.writeline("if _spec is None:")
                    with module.indent():
                        module.writeline("results.append(_value)")
                    module.writeline("else:")
                    with module.indent():
                        module.writelines(
                            [
                                "dtype, device = _spec",
                                "results.append(_from_mx(_value, dtype=dtype, device=device))",
                            ]
                        )
            module.writeline("for _idx, _alias in enumerate(_OUTPUT_ALIASES):")
            with module.indent():
                module.writeline("if _alias is not None:")
                with module.indent():
                    module.writeline("results[_idx] = args[_alias]")
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
        self._output_aliases = []
        self._placeholder_indices = {}
        self._mutated_input_specs = []
        placeholder_nodes = [
            node for node in self._gm.graph.nodes if node.op == "placeholder"
        ]
        if placeholder_nodes:
            self._body.writeline("_args = tuple(args)")
        for index, node in enumerate(placeholder_nodes):
            self._placeholder_nodes.append(node)
            self._placeholder_indices[node] = index
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

        mutated_indices = self._mutated_input_idxs | self._additional_mutated_idxs
        self._mutated_input_specs = []
        self._mutated_exprs = []
        for index in sorted(mutated_indices):
            meta = self._input_metas[index] if index < len(self._input_metas) else None
            self._mutated_input_specs.append((index, meta))
            placeholder_node = (
                self._placeholder_nodes[index]
                if index < len(self._placeholder_nodes)
                else None
            )
            if placeholder_node is None or placeholder_node not in self._values:
                raise MLXCodegenError(
                    f"Missing MLX value for mutated input at index {index}"
                )
            self._mutated_exprs.append(self._values[placeholder_node].name)

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
        if target == operator.getitem:
            self._emit_getitem(node)
        elif target == torch.ops.aten.relu.default:
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
        elif target == torch.ops.aten.bitwise_and.Tensor:
            lhs = self._value(node.args[0])
            rhs = self._value(node.args[1])
            expr = f"mx.bitwise_and({lhs}, {rhs})"
            self._values[node] = self._assign(
                node, expr, meta=_extract_tensor_meta(node)
            )
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
            self._values[node] = self._assign(
                node, f"mx.array({val})", meta=_extract_tensor_meta(node)
            )
        elif target == torch.ops.aten.cat.default:
            self._emit_cat(node)
        elif target == torch.ops.aten.embedding.default:
            self._emit_embedding(node)
        elif target == torch.ops.aten.index.Tensor:
            self._emit_index(node)
        elif target == torch.ops.aten.pow.Tensor_Scalar:
            base = self._value(node.args[0])
            exponent = self._format_scalar(node.args[1])
            expr = f"mx.power({base}, {exponent})"
            self._values[node] = self._assign(
                node, expr, meta=_extract_tensor_meta(node)
            )
        elif target == torch.ops.prims.convert_element_type.default:
            value = self._value(node.args[0])
            dtype_arg = node.args[1]
            if not isinstance(dtype_arg, torch.dtype):
                raise MLXCodegenError(
                    "prims.convert_element_type expects a concrete torch.dtype"
                )
            mx_dtype = _torch_dtype_to_mx(dtype_arg)
            expr = f"({value}).astype({mx_dtype})"
            self._values[node] = self._assign(
                node, expr, meta=_extract_tensor_meta(node)
            )
        elif target == torch.ops.prims.iota.default:
            self._emit_prims_iota(node)
        elif target == torch.ops.aten.rsqrt.default:
            value = self._value(node.args[0])
            self._values[node] = self._assign(
                node, f"mx.rsqrt({value})", meta=_extract_tensor_meta(node)
            )
        elif target == torch.ops.aten.sigmoid.default:
            value = self._value(node.args[0])
            self._values[node] = self._assign(
                node, f"mx.sigmoid({value})", meta=_extract_tensor_meta(node)
            )
        elif target == torch.ops.aten.sin.default:
            value = self._value(node.args[0])
            self._values[node] = self._assign(
                node, f"mx.sin({value})", meta=_extract_tensor_meta(node)
            )
        elif target == torch.ops.aten.slice.Tensor:
            self._emit_slice(node)
        elif target in _EXPAND_TARGETS:
            self._emit_expand(node)
        elif target in _AMAX_TARGETS:
            self._emit_reduce(node, "max")
        elif target in _SUM_TARGETS:
            self._emit_reduce(node, "sum")
        elif target in _MEAN_TARGETS:
            self._emit_reduce(node, "mean")
        elif target == torch.ops.aten.le.Tensor:
            self._emit_binary_compare(node, "mx.less_equal")
        elif target == torch.ops.aten.le.Scalar:
            self._emit_comparison_scalar(node, "mx.less_equal")
        elif target == torch.ops.aten.full.default:
            self._emit_full(node)
        elif target == torch.ops.aten.copy_.default:
            dest = node.args[0]
            src = self._value(node.args[1])
            result = self._assign(
                node, f"mx.array({src})", meta=_extract_tensor_meta(node)
            )
            self._values[node] = result
            if isinstance(dest, Node):
                idx = self._placeholder_indices.get(dest)
                if idx is not None:
                    self._additional_mutated_idxs.add(idx)
                    self._values[dest] = result
        elif (
            target
            == torch.ops.aten._scaled_dot_product_attention_math_for_mps.default
        ):
            self._emit_sdpa_mps(node)
        elif target == torch.ops.aten.where.self:
            cond = self._value(node.args[0])
            a = self._value(node.args[1])
            b = self._value(node.args[2])
            self._values[node] = self._assign(
                node, f"mx.where({cond}, {a}, {b})"
            )
        elif target == torch.ops.aten.to.dtype:
            tensor = self._value(node.args[0])
            desired_dtype = node.args[1]
            if not isinstance(desired_dtype, torch.dtype):
                raise MLXCodegenError(
                    "MLX codegen expects dtype arguments to be concrete values"
                )
            mx_dtype = _torch_dtype_to_mx(desired_dtype)
            expr = f"({tensor}).astype({mx_dtype})"
            self._values[node] = self._assign(
                node, expr, meta=_extract_tensor_meta(node)
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

    def _emit_getitem(self, node: Node) -> None:
        base = node.args[0]
        if not isinstance(node.args[1], int):
            index = self._to_int(node.args[1])
        else:
            index = node.args[1]
        expr = f"{self._value(base)}[{index}]"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_output(self, node: Node) -> None:
        output = node.args[0]
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        self._output_specs = []
        for out in outputs:
            alias_idx: Optional[int] = None
            if isinstance(out, Node):
                mlx_value = self._values[out]
                meta = mlx_value.meta or _extract_tensor_meta(out)
                alias_idx = self._placeholder_indices.get(out)
                convert = alias_idx is None
                self._output_specs.append(
                    _OutputSpec(
                        expr=mlx_value.name,
                        meta=meta,
                        convert=convert,
                        alias_input_idx=alias_idx,
                    )
                )
            else:
                self._output_specs.append(
                    _OutputSpec(expr=repr(out), meta=None, convert=False)
                )
            self._output_aliases.append(alias_idx)
        exprs = [spec.expr for spec in self._output_specs]
        all_exprs = exprs + self._mutated_exprs
        if not all_exprs:
            self._body.writeline("return tuple()")
            return
        tuple_expr = ", ".join(all_exprs)
        if len(all_exprs) == 1:
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
        input_arg = node.args[0]
        tensor = self._value(input_arg)
        size = node.args[1]
        if not isinstance(size, (tuple, list)):
            raise MLXCodegenError(
                "Expected expand sizes to be a tuple or list"
            )
        resolved = self._normalize_expand_sizes(
            tuple(size),
            _extract_tensor_meta(input_arg)
            if isinstance(input_arg, Node)
            else None,
        )
        size_literal = self._format_shape(resolved)
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

    def _emit_cat(self, node: Node) -> None:
        tensors = node.args[0]
        if not isinstance(tensors, (list, tuple)):
            raise MLXCodegenError("torch.cat expects a list or tuple of tensors")
        values = [self._value(arg) for arg in tensors]
        dim_arg = self._get_kwarg_or_pos(node, 1, "dim")
        dim = 0 if dim_arg is None else self._to_int(dim_arg)
        list_var = self._new_var("cat_inputs")
        self._body.writeline(f"{list_var} = [{', '.join(values)}]")
        expr = f"mx.concatenate({list_var}, axis={dim})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_embedding(self, node: Node) -> None:
        weight_expr = self._value(node.args[0])
        indices_node = node.args[1]
        indices_expr = self._value(indices_node)

        padding_idx = self._get_kwarg_or_pos(node, 2, "padding_idx")
        padding_val = -1 if padding_idx is None else self._to_int(padding_idx)
        if padding_val >= 0:
            raise MLXCodegenError(
                "MLX embedding codegen does not yet support non-negative padding_idx"
            )

        scale_grad_arg = self._get_kwarg_or_pos(node, 3, "scale_grad_by_freq") or False
        sparse_arg = self._get_kwarg_or_pos(node, 4, "sparse") or False
        if scale_grad_arg:
            raise MLXCodegenError(
                "MLX embedding codegen does not support scale_grad_by_freq=True"
            )
        if sparse_arg:
            raise MLXCodegenError("MLX embedding codegen does not support sparse=True")

        gather_var = self._new_var("embedding_gather")
        self._body.writeline(
            f"{gather_var} = mx.take({weight_expr}, {indices_expr}, axis=0)"
        )

        self._values[node] = _MLXValue(
            gather_var,
            _extract_tensor_meta(node),
        )

    def _emit_index(self, node: Node) -> None:
        base_expr = self._value(node.args[0])
        indices = node.args[1]
        if not isinstance(indices, (list, tuple)):
            raise MLXCodegenError("MLX index lowering expects tuple/list of indices")
        if len(indices) != 1:
            raise MLXCodegenError(
                "MLX index lowering currently supports a single tensor index"
            )
        index_node = indices[0]
        if not isinstance(index_node, Node):
            raise MLXCodegenError(
                "MLX index lowering requires tensor indices (slices/ellipsis unsupported)"
            )
        index_meta = self._require_tensor_meta(index_node, "index")
        if index_meta.dtype not in (
            torch.int32,
            torch.int64,
            torch.int16,
            torch.int8,
            torch.uint8,
        ):
            raise MLXCodegenError(
                "MLX index lowering expects integer tensor indices"
            )
        index_expr = self._value(index_node)
        result = self._assign(
            node,
            f"mx.take({base_expr}, {index_expr}, axis=0)",
            meta=_extract_tensor_meta(node),
        )
        self._values[node] = result

    def _emit_slice(self, node: Node) -> None:
        tensor = self._value(node.args[0])
        dim = self._to_int(node.args[1])
        start = self._to_int(node.args[2])
        end = self._to_int(node.args[3])
        step = self._to_int(node.args[4]) if len(node.args) > 4 else 1

        if step != 1:
            raise MLXCodegenError("MLX slice lowering does not yet support step != 1")

        slice_meta = self._require_tensor_meta(node, "slice")
        input_meta = self._require_tensor_meta(node.args[0], "slice_input")

        start_indices = [0] * len(input_meta.shape)
        start_indices[dim] = start
        start_array = self._new_var("slice_start")
        self._body.writeline(
            f"{start_array} = mx.array({start_indices}, dtype=mx.int32)"
        )

        axes = list(range(len(input_meta.shape)))
        axes_tuple = "(" + ", ".join(str(axis) for axis in axes) + ")"
        slice_sizes = "(" + ", ".join(str(self._to_int(s)) for s in slice_meta.shape) + ")"

        expr = (
            f"mx.slice({tensor}, start_indices={start_array}, axes={axes_tuple}, slice_size={slice_sizes})"
        )
        self._values[node] = self._assign(node, expr, meta=slice_meta)

    def _emit_prims_iota(self, node: Node) -> None:
        length = self._to_int(node.args[0])
        if length < 0:
            raise MLXCodegenError("prims.iota length must be non-negative")

        start = self._to_int(node.kwargs.get("start", 0))
        step = self._to_int(node.kwargs.get("step", 1))
        dtype_arg = node.kwargs.get("dtype")
        if not isinstance(dtype_arg, torch.dtype):
            raise MLXCodegenError("prims.iota expects a concrete torch.dtype")
        mx_dtype = _torch_dtype_to_mx(dtype_arg)

        device_arg = node.kwargs.get("device")
        if device_arg is not None and device_arg.type != "cpu":
            raise MLXCodegenError(
                f"MLX iota backend expects CPU device, got {device_arg}"
            )

        end = start + length * step
        expr = f"mx.arange({start}, {end}, {step}, dtype={mx_dtype})"
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

    def _emit_binary_compare(self, node: Node, func: str) -> None:
        lhs = self._value(node.args[0])
        rhs = self._value(node.args[1])
        expr = f"{func}({lhs}, {rhs})"
        self._values[node] = self._assign(
            node, expr, meta=_extract_tensor_meta(node)
        )

    def _emit_full(self, node: Node) -> None:
        shape = node.args[0]
        value_expr = self._format_scalar(node.args[1])
        dtype_arg = self._get_kwarg_or_pos(node, 2, "dtype")
        mx_dtype_expr = None
        if dtype_arg is not None:
            if not isinstance(dtype_arg, torch.dtype):
                raise MLXCodegenError(
                    "MLX codegen expects dtype arguments to be concrete torch.dtype values"
                )
            mx_dtype_expr = _torch_dtype_to_mx(dtype_arg)
        device_arg = self._get_kwarg_or_pos(node, 4, "device")
        normalized_device = self._normalize_device_arg(device_arg)
        if normalized_device is not None and normalized_device.type != "cpu":
            raise MLXCodegenError(
                f"MLX codegen only supports CPU tensors, requested device {normalized_device}"
            )
        if isinstance(shape, (list, tuple)) and len(shape) == 0:
            expr = f"mx.array({value_expr})"
            if mx_dtype_expr is not None:
                expr = f"({expr}).astype({mx_dtype_expr})"
        else:
            shape_literal = self._format_shape(shape)
            expr = f"mx.full({shape_literal}, {value_expr}"
            if mx_dtype_expr is not None:
                expr = f"{expr}, dtype={mx_dtype_expr}"
            expr = f"{expr})"
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

    def _emit_sdpa_mps(self, node: Node) -> None:
        query_node = node.args[0]
        key_node = node.args[1]
        value_node = node.args[2]

        query_expr = self._value(query_node)
        key_expr = self._value(key_node)
        value_expr = self._value(value_node)

        query_meta = self._require_tensor_meta(query_node, "query")
        key_meta = self._require_tensor_meta(key_node, "key")

        if len(query_meta.shape) < 2 or len(key_meta.shape) < 2:
            raise MLXCodegenError(
                "Scaled dot-product attention requires rank >= 2 for query/key tensors"
            )

        q_len = self._to_int(query_meta.shape[-2])
        head_dim = self._to_int(query_meta.shape[-1])
        k_len = self._to_int(key_meta.shape[-2])

        scale_arg = node.kwargs.get("scale", None)
        if scale_arg is None:
            scale_expr = f"(1.0 / ({head_dim} ** 0.5))"
        else:
            scale_expr = self._format_scalar(scale_arg)

        scores_dtype = (
            _torch_dtype_to_mx(query_meta.dtype)
            if query_meta.dtype is not None
            else "mx.float32"
        )

        attn_mask_arg = self._get_kwarg_or_pos(node, 3, "attn_mask")
        mask_add_expr: Optional[str] = None
        if attn_mask_arg is not None:
            if not isinstance(attn_mask_arg, Node):
                raise MLXCodegenError(
                    "MLX codegen expects attention masks to be tensor inputs"
                )
            mask_add_expr = self._maybe_cast_mask(
                self._value(attn_mask_arg), scores_dtype
            )

        dropout_p_arg = self._get_kwarg_or_pos(node, 4, "dropout_p")
        if dropout_p_arg is None:
            dropout_p = 0.0
        elif isinstance(dropout_p_arg, (int, float)):
            dropout_p = float(dropout_p_arg)
        else:
            raise MLXCodegenError(
                "MLX codegen requires dropout probability to be a numeric literal"
            )

        is_causal_arg = self._get_kwarg_or_pos(node, 5, "is_causal")
        if is_causal_arg is None:
            is_causal = False
        elif isinstance(is_causal_arg, bool):
            is_causal = is_causal_arg
        else:
            raise MLXCodegenError(
                "MLX codegen requires is_causal to be a boolean literal"
            )

        dropout_mask_arg = self._get_kwarg_or_pos(node, 6, "dropout_mask")
        if dropout_mask_arg is not None:
            raise MLXCodegenError(
                "MLX codegen does not yet support custom dropout masks in SDPA"
            )

        fast_mask_arg = "None"
        if mask_add_expr is not None:
            fast_mask_arg = mask_add_expr

        if is_causal:
            causal_mask = self._build_causal_additive_mask(
                q_len, k_len, scores_dtype
            )
            if mask_add_expr is None:
                fast_mask_arg = "'causal'"
                mask_add_expr = causal_mask
            else:
                mask_add_expr = self._combine_masks(mask_add_expr, causal_mask)
                fast_mask_arg = mask_add_expr

        output_var = self._new_var("sdpa_output")
        self._body.writeline(
            f"{output_var} = _mx_fast.scaled_dot_product_attention({query_expr}, {key_expr}, {value_expr}, scale={scale_expr}, mask={fast_mask_arg})"
        )

        probs_var = self._emit_sdpa_probs(
            query_expr,
            key_expr,
            scale_expr,
            mask_add_expr,
        )

        tuple_var = self._new_var("sdpa_result")
        self._body.writeline(f"{tuple_var} = ({output_var}, {probs_var})")
        self._values[node] = _MLXValue(tuple_var, None)

    def _emit_sdpa_probs(
        self,
        query_expr: str,
        key_expr: str,
        scale_expr: str,
        mask_expr: Optional[str],
    ) -> str:
        scores_var = self._new_var("sdpa_scores")
        self._body.writeline(
            f"{scores_var} = mx.matmul({query_expr}, mx.swapaxes({key_expr}, -2, -1))"
        )
        self._body.writeline(f"{scores_var} = ({scores_var}) * ({scale_expr})")
        if mask_expr is not None:
            self._body.writeline(f"{scores_var} = ({scores_var}) + ({mask_expr})")
        probs_var = self._new_var("sdpa_probs")
        self._body.writeline(f"{probs_var} = mx.softmax({scores_var}, axis=-1)")
        return probs_var

    def _normalize_expand_sizes(
        self,
        target_sizes: Iterable[Any],
        input_meta: Optional[_TensorMeta],
    ) -> tuple[int, ...]:
        sizes = list(target_sizes)
        input_shape: Optional[tuple[int, ...]] = None
        if input_meta is not None and input_meta.shape is not None:
            input_shape = tuple(self._to_int(dim) for dim in input_meta.shape)
        input_rank = len(input_shape) if input_shape is not None else 0
        resolved = [0] * len(sizes)
        for offset in range(1, len(sizes) + 1):
            target_dim = sizes[-offset]
            input_dim = None
            has_input_dim = False
            if input_shape is not None and offset <= input_rank:
                input_dim = input_shape[-offset]
                has_input_dim = True
            elif input_shape is not None:
                input_dim = 1
            resolved[-offset] = self._resolve_expand_dim_value(
                target_dim, input_dim, has_input_dim
            )
        return tuple(resolved)

    def _resolve_expand_dim_value(
        self, target_dim: Any, input_dim: Optional[int], has_input_dim: bool
    ) -> int:
        dim_value = self._to_int(target_dim)
        if dim_value == -1:
            if not has_input_dim or input_dim is None:
                raise MLXCodegenError(
                    "expand(-1, ...) requires shape metadata for the corresponding dimension"
                )
            return input_dim
        return dim_value

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
        if isinstance(value, bool):
            return "True" if value else "False"
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

    def _require_tensor_meta(self, node: Node, context: str) -> _TensorMeta:
        meta = _extract_tensor_meta(node)
        if meta is None or meta.shape is None:
            raise MLXCodegenError(
                f"Missing tensor metadata for {context} in MLX codegen"
            )
        return meta

    def _expand_mask_for_attention(self, mask_var: str) -> str:
        expanded = self._new_var("sdpa_mask")
        self._body.writeline(
            f"{expanded} = mx.expand_dims(mx.expand_dims({mask_var}, 0), 0)"
        )
        return expanded

    def _maybe_cast_mask(self, mask_expr: str, dtype_expr: str) -> str:
        casted = self._new_var("sdpa_mask_cast")
        self._body.writeline(f"{casted} = ({mask_expr}).astype({dtype_expr})")
        return casted

    def _combine_masks(self, a_expr: str, b_expr: str) -> str:
        combined = self._new_var("sdpa_mask_combined")
        self._body.writeline(f"{combined} = ({a_expr}) + ({b_expr})")
        return combined

    def _build_causal_additive_mask(
        self, q_len: int, k_len: int, dtype_expr: str
    ) -> str:
        mask_bool = self._new_var("sdpa_causal_mask_bool")
        base_mask_shape = self._format_shape((q_len, k_len))
        self._body.writeline(
            f"{mask_bool} = mx.triu(mx.ones({base_mask_shape}, dtype=mx.bool_), 1)"
        )
        expanded_bool = self._expand_mask_for_attention(mask_bool)
        mask_shape = self._format_shape((1, 1, q_len, k_len))
        neg_inf = self._new_var("sdpa_neg_inf")
        self._body.writeline(
            f"{neg_inf} = mx.full({mask_shape}, float('-inf'), dtype={dtype_expr})"
        )
        zeros = self._new_var("sdpa_zero_mask")
        self._body.writeline(f"{zeros} = mx.zeros({mask_shape}, dtype={dtype_expr})")
        mask_add = self._new_var("sdpa_causal_mask_add")
        self._body.writeline(
            f"{mask_add} = mx.where({expanded_bool}, {neg_inf}, {zeros})"
        )
        return mask_add

    def _get_kwarg_or_pos(
        self, node: Node, position: int, name: str
    ) -> Any:
        if len(node.args) > position:
            return node.args[position]
        return node.kwargs.get(name)

    def _normalize_device_arg(
        self, value: Any
    ) -> Optional[torch.device]:
        if value is None:
            return None
        if isinstance(value, torch.device):
            return value
        if isinstance(value, str):
            return torch.device(value)
        raise MLXCodegenError(
            f"Unsupported device specification for MLX codegen: {value!r}"
        )


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

_MEAN_TARGETS = tuple(
    op
    for op in [
        _maybe_get_aten_op("mean"),
        _maybe_get_aten_op("mean", "dim"),
        _maybe_get_aten_op("mean", "dim_IntList"),
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
