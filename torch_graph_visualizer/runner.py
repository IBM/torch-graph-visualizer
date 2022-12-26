import contextlib
import logging
import torch
import torch.jit

from dataclasses import dataclass
from functorch.compile import aot_module, ts_compile, make_boxed_compiler
from typing import (
    Callable,
    Optional,
    Tuple,
)

logger = logging.getLogger(__name__)


@dataclass
class _AOTHelper:
    graph_count: int = 0

    def get_compiler_fn(self) -> Callable:
        @make_boxed_compiler
        def draw_compiler(gm: torch.fx.GraphModule, inputs):
            class _Drawer:
                def __init__(self, f: Callable, helper: "_AOTHelper"):
                    self.f = f
                    self.helper = helper
                    self.id = helper.graph_count
                    helper.graph_count += 1

                def __call__(self, *call_args):
                    with nvtx_sync(f"aot-graph{self.helper.graph_count}"):
                        r = self.f(call_args)
                    return r

            f = ts_compile(gm, inputs)
            return _Drawer(f, self)

        return draw_compiler


@contextlib.contextmanager
def nvtx_sync(name: str):
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(name)

    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


def run_model(
        model: torch.nn.Module,
        input_data: Optional[Tuple] = None,
        fuser: str = "none",
        jit: str = "none",
        warmup_steps: int = 1,
        number: int = 1,
) -> None:
    model.train()

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

    with torch.jit.fuser(fuser), torch.jit.optimized_execution(True):
        with nvtx_sync("warm-up"):
            for i in range(max(len(input_data), warmup_steps)):
                pred = ts_model(*input_data[i % len(input_data)])
                loss = pred.mean()
                loss.backward()

        for i in range(max(len(input_data), number)):
            input = input_data[i % len(input_data)]

            with nvtx_sync(f"fw-graph{i}"):
                pred = ts_model(*input)

            with nvtx_sync(f"loss-{i}"):
                loss = pred.mean()

            with nvtx_sync(f"bw-graph{i}"):
                loss.backward()
