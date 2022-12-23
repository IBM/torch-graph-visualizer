import re

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Union,
)

_RE_TORCH_OP = re.compile(
    r"(?P<prefix>at::_ops|at::native)::(?P<name>[^:]*)(::(?P<method>.*)|)\(.*"
)

_RE_FUSION = re.compile(
    r"^CudaCodeGen::(?P<name>kernel\d*)\("
)

Stat = Union[str, "StatKind"]


class StatKind(Enum):
    ComputeThroughput = auto()
    MemoryThroughput = auto()
    Duration = auto()


@dataclass(frozen=True)
class TorchOp:
    prefix: str
    name: str
    method: Optional[str]

    def __str__(self) -> str:
        method = f"::{self.method}" if self.method is not None else ""
        return f"{self.prefix}::{self.name}{method}"


class CallStackEntry(ABC):
    @abstractmethod
    def name(self) -> str: ...


class ProfiledKernel(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def callstack(self) -> Sequence[CallStackEntry]: ...

    @abstractmethod
    def get(self, stat: Stat) -> Any: ...

    def _prune_name(self, name: str) -> str:
        for suffix in ["_Tensor", "_ScalarOpt_dim", "_symint", "_dim_IntList"]:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name

    @property
    def torch_op_callstack(self) -> List[TorchOp]:
        ops = []
        for c in reversed(self.callstack()):
            m = _RE_TORCH_OP.match(c.name())
            if m is not None:
                ops.append(TorchOp(**m.groupdict()))
        return ops

    @property
    def op_name(self) -> str:
        if len(self.torch_op_callstack) == 0:
            assert self.fusion, f"kernel not a fusion nor an operation: {self.name()}"
            m = _RE_FUSION.match(self.name())

            assert m is not None, f"bad fused kernel name: {self.name()}"
            return m.groupdict()["name"]

        first_op = self.torch_op_callstack[0]
        return self._prune_name(first_op.name)

    @property
    def jit(self) -> bool:
        for c in reversed(self.callstack()):
            if "jit::InterpreterState::run" in c.name():
                return True
        return False

    @property
    def fusion(self) -> bool:
        for c in reversed(self.callstack()):
            if "cuda::runCudaFusionGroup" in c.name():
                return True
        return False
