import enum
import graphviz
import io
import logging
import torch
import torch.jit

from collections import defaultdict
from dataclasses import dataclass, field
from functorch.compile import (
    aot_module,
    ts_compile,
    make_boxed_compiler
)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Set,
    Sequence,
    Optional,
    Tuple,
)

import torch_graph_visualizer.attribute_generator as attr
from torch_graph_visualizer.attribute_generator import AttributeGenerator
from torch_graph_visualizer.profile import ProfiledKernel, StatKind
from torch_graph_visualizer.utils import get_milliseconds

_GLOBAL_GRAPH_ATTR = {
    "newrank": "true"
}

_KERNEL_DENYLIST = {
    "CallFunction",
    "CudaFusionGuard",
    "Float",
    "ListConstruct",
    "_size_if_not_equal",
    "slice",
    "TupleUnpack",
}

_IGNORE_NODE_OPS = {
    "aten::size",
    "aten::__getitem__",
    "aten::_set_item"
}

_IGNORE_INPUT_TYPES = [
    torch.BoolType,
    torch.FloatType,
    torch.IntType,
]

logger = logging.getLogger(__name__)


def _should_ignore_type(ty: Any) -> bool:
    if any(isinstance(ty, ignore_ty) for ignore_ty in _IGNORE_INPUT_TYPES):
        return True
    if isinstance(ty, torch._C.ListType):
        return _should_ignore_type(ty.getElementType())
    return False


def _should_ignore_node(node: torch._C.Node) -> bool:
    return (
        node.kind() in _IGNORE_NODE_OPS
        or all(_should_ignore_type(inp.type()) for inp in node.inputs())
    )


def _value_is_tensor_like(val: torch.Value) -> bool:
    ty = val.type()
    return (
        isinstance(ty, torch.TensorType)
        or (
            isinstance(ty, torch.ListType)
            and isinstance(ty.getElementType(), torch.TensorType)
        )
    )


class _DigraphNode:
    def __init__(self, name, node, shape=None):
        self._name = name
        self._node = node
        self._shape = shape

    def name(self):
        return self._name

    def node(self):
        return self._node

    def is_value(self):
        return self._name.startswith("val_")

    def tpl(self):
        return (self._name, self._node)

    def __eq__(self, rhs):
        return self.tpl() == rhs.tpl()

    def __hash__(self):
        return hash(self.tpl())

    @staticmethod
    def from_node(node, shape=None):
        outputs = list(node.outputs())
        outputs_str = f"o{len(outputs)}"
        for o in outputs:
            outputs_str += "." + o.debugName()

        inputs = list(node.inputs())
        inputs_str = f"i{len(inputs)}"
        for i in node.inputs():
            inputs_str += "." + i.debugName()

        kind = str(node.kind()).replace("::", ".")
        name = f"node-{kind}--{outputs_str}-{inputs_str}--{id(node)}"
        return _DigraphNode(name=name, node=node, shape=shape)

    @staticmethod
    def from_value(value, shape=None):
        return _DigraphNode(
            name=f"val_{id(value)}_{value.debugName()}",
            node=value.node(),
            shape=shape,
        )


class _NodeKind(enum.Enum):
    GET_ATTR = "get_attr"
    FUSION_GROUP = "fusion_group"
    DIFFERENTIABLE_GRAPH = "differentiable_graph"
    STR = "str"
    IF = "if"
    PASSTHROUGH = "passthrough"
    GROUPCHECK = "gradcheck"

    @staticmethod
    def from_node(node):
        k = str(node.kind()).replace("::", ".")

        if "GetAttr" in k:
            return _NodeKind.GET_ATTR

        elif "CudaFusionGroup" in k:
            return _NodeKind.FUSION_GROUP

        elif "DifferentiableGraph" in k:
            return _NodeKind.DIFFERENTIABLE_GRAPH

        elif any(word in k for word in ["profile", "Return", "Constant", "aten.len"]):
            return _NodeKind.PASSTHROUGH

        elif "If" in k and "IfThenElse" not in k:
            return _NodeKind.IF

        elif any(word in k for word in ["RequiresGradCheck", "AutogradAll"]):
            return _NodeKind.GROUPCHECK

        else:
            return _NodeKind.STR

    def needs_wiring_after_creation(self):
        return self in (_NodeKind.GET_ATTR, _NodeKind.STR)


class _GraphDrawer:
    def __init__(
            self,
            graph: torch._C.Graph,
            digraph: graphviz.Digraph,
            every_node: Optional[Dict[str, str]] = None,
            parent: Optional[graphviz.Digraph] = None,
            color_fused: str = "mistyrose2",
            kernels: Optional[Dict[str, List[ProfiledKernel]]] = None,
            kernels_count: Optional[Dict[str, int]] = None,
            attr_gen: Optional[AttributeGenerator] = None,
            values_to_pyvalues_map: Optional[Dict[torch.Value, Any]] = None
    ) -> None:
        self._graph = graph
        self._digraph = digraph
        self._every_node = every_node
        self._parent = parent
        self._color_fused = color_fused
        self._kernels = kernels if kernels is not None else {}
        self._kernels_count = {k: 0 for k in self._kernels} \
            if kernels_count is None else kernels_count
        self._nodes: Set[torch._C.Node] = set()
        self._kernels_not_found: Dict[str, List[str]] = defaultdict(list)
        self._kernels_found: Dict[str, List[str]] = defaultdict(list)
        self._attr_gen = attr_gen
        self._values_to_pyvalues_map = values_to_pyvalues_map \
            if values_to_pyvalues_map is not None else {}

    def _inputs(self):
        if isinstance(self._graph, torch._C.Graph):
            return self._graph.inputs()

        elif isinstance(self._graph, torch._C.Block):
            inputs = []
            outputs = set()

            for node in self._graph.nodes():
                for inp in node.inputs():
                    if inp not in outputs:
                        inputs.append(inp)
                for out in node.outputs():
                    outputs.add(out)

            return inputs

        raise ValueError(f"invalid graph instance: {type(self._graph)}")

    def _update_kernel_stats(self, drawer: "_GraphDrawer") -> None:
        for k, v in drawer._kernels_not_found.items():
            self._kernels_not_found[k].extend(v)
        for k, v in drawer._kernels_found.items():
            self._kernels_found[k].extend(v)

    def _node(self, digraph_node, _digraph=None, **attrs):
        fused_attrs = {}
        if self._every_node is not None:
            fused_attrs.update(self._every_node)
        fused_attrs.update(attrs)
        self._nodes.add(digraph_node)

        digraph = _digraph if _digraph is not None else self._digraph
        digraph.node(digraph_node.name(), **fused_attrs)

    def _edge(self, src, tgt, **attrs):
        src_name = src.name()
        if "src_loc" in attrs:
            src_name += f""":{attrs.pop("src_loc")}"""

        tgt_name = tgt.name()
        if "tgt_loc" in attrs:
            tgt_name += f""":{attrs.pop("tgt_loc")}"""

        use_parent = src not in self._nodes and self._parent is not None
        g = self._parent if use_parent else self._digraph
        g.edge(src_name, tgt_name, **attrs)

    def _process_node(self, node, nodeof):
        def get_next_kernel(name: str) -> ProfiledKernel:
            position = self._kernels_count[name]
            self._kernels_count[name] += 1
            return self._kernels[name][position]

        def update_nodeof_outputs(digraph_node):
            for out in node.outputs():
                nodeof[out] = digraph_node

        def invisible_input_edges(node):
            kind = node.kind()
            return (
                "GradCheck" in kind
                or "FusionGuard" in kind
                or "CallFunction" in kind
                or "AutogradAll" in kind
            )

        def wire(inputs, node, **attrs):
            if invisible_input_edges(node):
                attrs["style"] = "invis"

            inp_digraph_nodes = set()
            for inp in node.inputs():
                if inp in nodeof:
                    if nodeof[inp] not in inp_digraph_nodes:
                        inp_digraph_nodes.add(nodeof[inp])
                        self._edge(nodeof[inp], _DigraphNode.from_node(node), **attrs)
                else:
                    if _NodeKind.from_node(inp.node()) not in (
                        _NodeKind.PASSTHROUGH,
                        _NodeKind.GROUPCHECK
                    ):
                        logger.debug("input node not processed:")
                        logger.debug(f"    {inp.node()}")

        kind = _NodeKind.from_node(node)

        node_attr = {}
        if str(node.kind()).startswith("prim::"):
            node_attr["fillcolor"] = "gray"
            node_attr["style"] = "filled"

        if kind is _NodeKind.GET_ATTR:
            digraph_node = _DigraphNode.from_node(node)
            self._node(digraph_node, label=f"""GetAttr[{node.s("name")}]""", **node_attr)
            update_nodeof_outputs(digraph_node)
            wire(node.inputs(), node)

        elif kind is _NodeKind.STR:
            outputs = list(node.outputs())
            inputs = list(node.inputs())

            if "aten::size" in node.kind():
                assert len(outputs) == 1 and len(inputs) <= 2, f"bad node: {node}"
                if inputs[0] in nodeof:
                    nodeof[outputs[0]] = nodeof[inputs[0]]

            elif "aten::__getitem__" in node.kind():
                assert len(outputs) == 1 and len(inputs) == 2, f"bad node: {node}"
                if inputs[0] in nodeof:
                    nodeof[outputs[0]] = nodeof[inputs[0]]

            elif "aten::_set_item" in node.kind():
                assert len(outputs) == 1 and len(inputs) == 3, f"bad node: {node}"
                if inputs[0] in nodeof:
                    nodeof[outputs[0]] = nodeof[inputs[0]]

            elif _should_ignore_node(node):
                logger.debug("node of scalar inputs ignored:")
                logger.debug(f"    {node}")
                pass

            else:
                namespace, fnname = node.kind().split("::")
                label = f"{namespace}.{fnname}"
                output_shape = None

                IGNORE = ["CudaFusionGuard", "CallFunction"]
                has_metadata = (
                    all(s not in fnname for s in IGNORE)
                    and all(inp in self._values_to_pyvalues_map for inp in node.inputs())
                )
                if has_metadata:
                    all_args = [self._values_to_pyvalues_map[inp] for inp in node.inputs()]

                    try:
                        if fnname == "ListConstruct":
                            output = all_args
                        else:
                            schema = torch._C.parse_schema(node.schema())

                            args = []
                            kwargs = {}
                            for schema_arg, arg in zip(schema.arguments, all_args):
                                if schema_arg.kwarg_only or len(kwargs) > 0:
                                    kwargs[schema_arg.name] = arg
                                else:
                                    args.append(arg)

                            output = getattr(torch.ops.aten, fnname)(*args, **kwargs)
                    except Exception:
                        logger.error(f"Failed evaluating node: {node}", end="")
                        logger.error([self._value_to_label(inp) for inp in node.inputs()])
                        raise

                    if isinstance(output, torch.Tensor) or isinstance(output, list):
                        assert node.outputsSize() == 1, f"invalid output for: {node}"
                        self._values_to_pyvalues_map[next(iter(node.outputs()))] = output
                    elif isinstance(output, tuple):
                        assert node.outputsSize() == len(outputs), f"invalid output for: {node}"
                        for node_output, out in zip(node.outputs(), output):
                            self._values_to_pyvalues_map[node_output] = out
                    else:
                        assert False, f"invalid output type: {type(output)} ({node.kind()})"

                    input_str = " ".join(
                        self._value_to_label(inp)
                        for inp in node.inputs() if _value_is_tensor_like(inp)
                    )
                    output_str = " ".join(
                        self._value_to_label(out)
                        for out in node.outputs() if _value_is_tensor_like(out)
                    )
                    label = f"{{ {label} | In: {input_str} | Out: {output_str} }}"
                    node_attr["shape"] = "record"
                else:
                    output = None

                kernel_found = False
                graph_attr = _GLOBAL_GRAPH_ATTR.copy()
                if (
                        fnname in self._kernels
                        and self._kernels_count[fnname] < len(self._kernels[fnname])
                ):
                    kernel = get_next_kernel(fnname)

                    while "matmul" in fnname and "elementwise" in kernel.name():
                        kernel = get_next_kernel(fnname)

                    if self._attr_gen is not None:
                        if has_metadata:
                            graph_attr.update(self._attr_gen.cluster(node, kernel))
                        else:
                            node_attr.update(self._attr_gen.node(node, kernel))

                    self._kernels_found[fnname].append(node.kind())
                    kernel_found = True
                elif fnname not in _KERNEL_DENYLIST:
                    self._kernels_not_found[fnname].append(node.kind())

                digraph_node = _DigraphNode.from_node(node, shape=output_shape)

                if has_metadata and kernel_found:
                    sub_digraph = graphviz.Digraph(
                        name=f"cluster_{fnname}_{id(node)}",
                        graph_attr=graph_attr,
                        node_attr=node_attr,
                    )
                    d = sub_digraph
                else:
                    d = self._digraph

                self._node(digraph_node, label=label, _digraph=d, **node_attr)

                if has_metadata and kernel_found:
                    self._digraph.subgraph(d)

                update_nodeof_outputs(digraph_node)
                wire(node.inputs(), node)

        elif kind is _NodeKind.GROUPCHECK:
            digraph_node = _DigraphNode.from_node(node)
            inputs = list(node.inputs())
            outputs = list(node.outputs())

            assert len(inputs) + 1 == len(outputs) or len(outputs) == 1, (
                "Assumption: check nodes return an extra boolean "
                "value (except for passing through the inputs) representing the "
                "result of the check."
            )

            self._node(digraph_node, label=node.kind().replace("::", "."), **node_attr)
            nodeof[outputs[-1]] = digraph_node
            wire(inputs, node)

            for i, o in zip(inputs, outputs[:-1]):
                nodeof[o] = nodeof[i]

        elif kind in (_NodeKind.FUSION_GROUP, _NodeKind.DIFFERENTIABLE_GRAPH):
            digraph_node = _DigraphNode.from_node(node)
            subgraph = node.g("Subgraph")

            label = kind.value
            if kind is _NodeKind.FUSION_GROUP:
                cache_id = node.i("cache_id") + 1
                label = f"{label}_{cache_id}"

            self._node(digraph_node, label=label, style="filled", fillcolor="gray")
            wire(node.inputs(), node, style="invis")

            for inp, diff_inp in zip(node.inputs(), subgraph.inputs()):
                if inp in nodeof:
                    nodeof[diff_inp] = nodeof[inp]
                if inp in self._values_to_pyvalues_map:
                    self._values_to_pyvalues_map[diff_inp] = self._values_to_pyvalues_map[inp]
                else:
                    logger.debug(f"Input not in: {inp.debugName()} -> {diff_inp.debugName()}")
                    logger.debug(inp.node())

            node_attr = {"group": f"graph_{id(node)}"}

            if kind is _NodeKind.FUSION_GROUP:
                if self._color_fused is not None:
                    node_attr["style"] = "filled"
                    node_attr["fillcolor"] = self._color_fused

                graph_attr = _GLOBAL_GRAPH_ATTR.copy()
                fnname = f"kernel{cache_id}"

                if fnname in self._kernels:
                    kernel = get_next_kernel(fnname)
                    if self._attr_gen is not None:
                        graph_attr.update(self._attr_gen.cluster(node, kernel))

                sub_digraph = graphviz.Digraph(
                    name=f"cluster_{id(node)}",
                    graph_attr=graph_attr,
                    node_attr=node_attr,
                )

                drawer = _GraphDrawer(
                    graph=subgraph,
                    digraph=sub_digraph,
                    every_node=node_attr,
                    parent=self._digraph,
                    values_to_pyvalues_map=self._values_to_pyvalues_map
                )

                drawer._draw_graph(nodeof)
                self._digraph.subgraph(sub_digraph)
            else:
                drawer = _GraphDrawer(
                    graph=subgraph,
                    digraph=self._digraph,
                    every_node=node_attr,
                    parent=self._digraph,
                    kernels=self._kernels,
                    kernels_count=self._kernels_count,
                    attr_gen=self._attr_gen,
                    values_to_pyvalues_map=self._values_to_pyvalues_map
                )

                drawer._draw_graph(nodeof)
                self._update_kernel_stats(drawer)

            for sub_node in {n for inp in drawer._inputs() for n in drawer.get_head_nodes(inp)}:
                self._edge(digraph_node, _DigraphNode.from_node(sub_node))

            node_outputs = list(node.outputs())
            subgraph_outputs = list(subgraph.outputs())

            for out, diff_out in zip(node_outputs, subgraph_outputs):
                if diff_out in self._values_to_pyvalues_map:
                    self._values_to_pyvalues_map[out] = self._values_to_pyvalues_map[diff_out]
                nodeof[out] = nodeof[diff_out]

            for i in range(len(node_outputs), len(subgraph_outputs)):
                out = subgraph_outputs[i]
                if out in nodeof and nodeof[out] != _DigraphNode.from_value(out):
                    back_node = _DigraphNode.from_value(out)
                    self._node(back_node, label=f"%{out.debugName()}")
                    self._edge(nodeof[out], back_node)

        elif kind is _NodeKind.IF:
            digraph_node = _DigraphNode.from_node(node)
            self._node(digraph_node, label="{ If | { <yes> true | <no> false } }", shape="record")
            wire(node.inputs(), node)

            for i, (label, block) in enumerate(zip(["yes", "no"], node.blocks())):
                drawer = _GraphDrawer(
                    graph=block,
                    digraph=self._digraph,
                    parent=self._digraph,
                    every_node={"group": f"if_{label}_{id(node)}"},
                    kernels=self._kernels,
                    kernels_count=self._kernels_count,
                    attr_gen=self._attr_gen,
                    values_to_pyvalues_map=self._values_to_pyvalues_map
                )
                drawer._draw_graph(nodeof)
                self._update_kernel_stats(drawer)

                block_head_nodes = {
                    n
                    for inp in drawer._inputs()
                    for n in drawer.get_head_nodes(inp)
                }

                for block_node in block_head_nodes:
                    self._edge(digraph_node, _DigraphNode.from_node(block_node), src_loc=label)

                for if_out, block_out in zip(node.outputs(), block.outputs()):
                    if block_out in self._values_to_pyvalues_map:
                        self._values_to_pyvalues_map[if_out] = \
                            self._values_to_pyvalues_map[block_out]

                    if block_out in nodeof:
                        if nodeof[block_out] == _DigraphNode.from_value(block_out):
                            nodeof[if_out] = nodeof[block_out]
                        else:
                            if_out_digraph_node = _DigraphNode.from_value(if_out)
                            nodeof[if_out] = if_out_digraph_node

                            if if_out in self._values_to_pyvalues_map:
                                value_label = self._value_to_label(if_out)
                                label = f"{{ %{if_out.debugName()} | {value_label} }}"
                                self._node(if_out_digraph_node, shape="record", label=label)
                            else:
                                self._node(if_out_digraph_node, label=f"%{if_out.debugName()}")

                            self._edge(nodeof[block_out], if_out_digraph_node, style="dashed")

        elif kind is _NodeKind.PASSTHROUGH:
            # The number of outputs should be the same as the number
            # of inputs (passthrough).
            inputs = list(node.inputs())
            outputs = list(node.outputs())

            assert len(inputs) == 0 or len(inputs) == len(outputs), (
                "Passthrough nodes must either have same number of "
                f"inputs ({len(inputs)}) and outputs ({len(outputs)}) "
                f"or constants: {node}"
            )

            if "Constant" in node.kind():
                out = next(iter(node.outputs()))
                try:
                    out_type = out.type()

                    if isinstance(out_type, torch.NoneType):
                        pyval = None
                    else:
                        pyval = getattr(node, node.kindOf("value"))("value")

                        if out_type == torch.BoolType.get():
                            pyval = bool(pyval)

                    self._values_to_pyvalues_map[out] = pyval
                except Exception:
                    logger.debug(f"Failed retrieving value: {node}")

            if len(inputs) > 0:
                # The 'nodeof' each output is the same as its corresponding
                # input. In this case, we assume that the order of the inputs
                # is the same as the order of the outputs.
                for inp, out in zip(inputs, outputs):
                    nodeof[out] = nodeof[inp]

    def _value_to_label(self, inp):
        value = self._values_to_pyvalues_map[inp]
        if isinstance(value, torch.Tensor):
            return str(list(value.shape))
        if isinstance(value, (int, str, bool, float, complex)):
            return str(value)
        return f"{type(value).__name__}"

    def _draw_inputs(self, nodeof, args):
        for i, inp in enumerate(self._inputs()):
            if args is not None:
                self._values_to_pyvalues_map[inp] = args[i]
                label = f"{{ %{inp.debugName()} | {self._value_to_label(inp)} }}"
                shape = "record"
            else:
                label = f"%{inp.debugName()}"
                shape = "ellipse"
            nodeof[inp] = _DigraphNode.from_value(inp)
            self._node(nodeof[inp], label=label, shape=shape)

    def _draw_graph(self, nodeof):
        for node in self._graph.nodes():
            # Skip if all output values were processed.
            # This is the case when we are actually trying to draw the
            # input nodes of a subgraph (e.g. FusionGroup or
            # DifferentiableGraph).
            if all(out in nodeof for out in node.outputs()):
                continue

            # Otherwise, all outputs should not be processed.
            assert all(out not in nodeof for out in node.outputs()), (
                "Some outputs are unexpectedly processed."
            )

            self._process_node(node, nodeof)

    def draw(self, nodeof=None, args=None):
        nodeof = nodeof if nodeof is not None else {}
        self._draw_inputs(nodeof, args)
        self._draw_graph(nodeof)

    def get_head_nodes(self, inp):
        value = inp

        for node in self._graph.nodes():
            if value not in set(node.inputs()):
                continue

            kind = _NodeKind.from_node(node)
            if kind == _NodeKind.PASSTHROUGH or _should_ignore_node(node):
                return [n for out in node.outputs() for n in self.get_head_nodes(out)]
            else:
                return [node]

        return []


def _reload_jit_model(model):
    tmp = io.BytesIO()
    torch.jit.save(model, tmp)
    tmp.seek(0)
    return torch.jit.load(tmp)

################################################################################
# Code from: torch/jit/_fuser.py ###############################################
################################################################################


def _get_differentiable_graph_node(node, diff_node):
    if node.kind() == 'prim::DifferentiableGraph':
        diff_node.append(node)
    else:
        for block in node.blocks():
            for n in block.nodes():
                _get_differentiable_graph_node(n, diff_node)


def _graph_for(debug_state):
    eps = list(debug_state.execution_plans.values())
    assert len(eps) == 1
    graph = eps[0].graph.copy()

    # graph_executor_states for differentiable node
    fw_states = eps[0].code.differentiable_op_executor_states()
    diff_nodes: List[torch._C.Node] = []
    for n in graph.nodes():
        _get_differentiable_graph_node(n, diff_nodes)

    assert len(fw_states) == len(diff_nodes)
    # swap each differentiable graph with optimized graph in their execution plan
    for n, state in zip(diff_nodes, fw_states):
        fw_execution_plans = list(state.execution_plans.values())
        # we can only update the subgraph when there's a unique execution
        # plan. Avoid assert here so we would skip the ones that can't be
        # updated while try the best effort to update other nodes.
        if len(fw_execution_plans) == 1:
            n.g_('Subgraph', fw_execution_plans[0].graph)

    return graph

################################################################################
################################################################################
################################################################################


def draw_graph(
        graph: torch.Graph,
        output_name: str = "graph",
        profiled_kernels: Sequence[ProfiledKernel] = [],
        attribute_generator: Optional[AttributeGenerator] = None,
        input_data: Optional[Tuple] = None,
        is_nn_module: bool = True,
) -> None:
    op_to_profiled_kernel = defaultdict(list)
    for i, kernel in enumerate(profiled_kernels):
        if kernel.jit:
            op_to_profiled_kernel[kernel.op_name].append(kernel)

    digraph = graphviz.Digraph(
        name=output_name,
        graph_attr=_GLOBAL_GRAPH_ATTR
    )
    drawer = _GraphDrawer(
        graph=graph,
        digraph=digraph,
        kernels=op_to_profiled_kernel,
        attr_gen=attribute_generator
    )

    if is_nn_module and input_data is not None:
        args = [None] + list(input_data)
    elif input_data is not None:
        args = list(input_data)
    else:
        args = None

    drawer.draw(args=args)
    digraph.render()

    logger.debug("Drawn:", digraph.name)
    logger.debug("found:", {k: len(v) for k, v in drawer._kernels_found.items()})
    logger.debug("not found:", {k: len(v) for k, v in drawer._kernels_not_found.items()})
    logger.debug("processed:", {k: v for k, v in drawer._kernels_count.items()})
    logger.debug("all:", {k: len(v) for k, v in op_to_profiled_kernel.items()})


@dataclass
class _AOTHelper:
    graphs: List[torch.Graph] = field(default_factory=list)
    call_args_list: List[Tuple] = field(default_factory=list)
    capture: bool = False

    def get_compiler_fn(self) -> Callable:
        @make_boxed_compiler
        def draw_compiler(gm: torch.fx.GraphModule, inputs):
            class _Drawer:
                def __init__(self, f: Callable, helper: "_AOTHelper"):
                    self.f = f
                    self.helper = helper

                def __call__(self, *call_args):
                    r = self.f(call_args)
                    if self.helper.capture:
                        self.helper.graphs.append(torch.jit.last_executed_optimized_graph())
                        self.helper.call_args_list.append(call_args)
                    return r

            f = ts_compile(gm, inputs)
            return _Drawer(f, self)

        return draw_compiler


def draw_model(
        model: torch.nn.Module,
        input_data: Tuple = (),
        output_name_template: str = "{attr}-{mode}{index}-{jit}-{fuser}",
        profiled_kernels_groups: List[Sequence[ProfiledKernel]] = [],
        attribute_generator: AttributeGenerator = attr.Nop(),
        allow_backward: bool = True,
        fuser: str = "none",
        jit: str = "none"
) -> None:
    model.train()

    base_kwargs = {
        "attr": attribute_generator.attribute_name(),
        "jit": jit,
        "fuser": fuser,
    }

    def run(model, repeat: int = 1, ninputs: int = -1):
        for i in range(repeat):
            n = min(ninputs, len(input_data)) if ninputs > 0 else len(input_data)
            for args in input_data[:n]:
                r = model(*args)

                if allow_backward:
                    r.mean().backward()

    if jit == "none" and fuser != "none":
        logger.warning(f"ignoring fuser ({fuser}) value, since 'jit' is 'none.")
        fuser = "none"

    aot_helper = _AOTHelper()
    with torch.jit.fuser(fuser):
        if jit == "script" or jit == "none":
            ts_model = torch.jit.script(model)

        elif jit == "trace":
            ts_model = torch.jit.trace(model, input_data[0])

        elif jit == "aot":
            ts_model = aot_module(model, fw_compiler=aot_helper.get_compiler_fn())

        else:
            raise ValueError(f"unexpected value for 'jit': {jit}")

        with torch.jit.optimized_execution(True):
            run(ts_model, repeat=3)

    if jit == "aot":
        aot_helper.capture = True

        with torch.jit.fuser(fuser):
            with torch.jit.optimized_execution(True):
                run(ts_model, ninputs=1)

        for i, (g, args) in enumerate(zip(aot_helper.graphs, aot_helper.call_args_list)):
            draw_graph(
                graph=g,
                output_name=output_name_template.format(mode="g", index=i, **base_kwargs),
                profiled_kernels=profiled_kernels_groups[i],
                attribute_generator=attribute_generator,
                input_data=args,
                is_nn_module=True,
            )
        return

    draw_graph(
        graph=_graph_for(ts_model.get_debug_state()),
        output_name=output_name_template.format(mode="forward", index=0, **base_kwargs),
        profiled_kernels=profiled_kernels_groups[0],
        attribute_generator=attribute_generator,
        input_data=input_data,
        is_nn_module=True,
    )

    if allow_backward:
        fw_execution_plan = list(ts_model.get_debug_state().execution_plans.values())[0]
        for i, bw_state in enumerate(fw_execution_plan.code.grad_executor_states()):
            output_name = output_name_template.format(mode="backward", index=i, **base_kwargs)
            group_index = min(len(profiled_kernels_groups) - 1, i + 1)
            draw_graph(
                graph=_graph_for(bw_state),
                output_name=output_name,
                profiled_kernels=profiled_kernels_groups[group_index],
                attribute_generator=attribute_generator,
                is_nn_module=True,
            )
