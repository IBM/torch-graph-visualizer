import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

from torch_graph_visualizer.profile import ProfiledKernel, StatKind
from torch_graph_visualizer.utils import get_milliseconds


class AttributeGenerator(ABC):
    @abstractmethod
    def node(self, node: torch.Node, kernel: ProfiledKernel) -> Dict[str, str]: ...

    @abstractmethod
    def cluster(self, node: torch.Node, kernel: ProfiledKernel) -> Dict[str, str]: ...

    @staticmethod
    def attribute_name() -> str:
        raise NotImplementedError()


class Nop(AttributeGenerator):
    def node(self, node: torch.Node, kernel: ProfiledKernel) -> Dict[str, str]:
        return {}

    def cluster(self, node: torch.Node, kernel: ProfiledKernel) -> Dict[str, str]:
        return {}

    @staticmethod
    def attribute_name() -> str:
        return "none"


class MemoryAndCompute(AttributeGenerator):
    def get_memory_latency_compute(self, kernel: ProfiledKernel) -> Tuple[float, float, float]:
        compute_str, _ = kernel.get(StatKind.ComputeThroughput)
        compute = float(compute_str) / 100 / 2
        memory_str, _ = kernel.get(StatKind.MemoryThroughput)
        memory = float(memory_str) / 100 / 2
        latency = 1 - memory - compute
        return (memory, latency, compute)

    def get_common(self, kernel: ProfiledKernel) -> Dict[str, str]:
        memory, latency, compute = self.get_memory_latency_compute(kernel)
        return {
            "style":      "striped",
            "fillcolor": f"green;{memory}:white;{latency}:yellow;{compute}"
        }

    def node(self, node, kernel):
        common = self.get_common(kernel)
        common["shape"] = "rect"
        return common

    def cluster(self, node, kernel):
        return self.get_common(kernel)

    @staticmethod
    def attribute_name() -> str:
        return "bottleneck"


@dataclass
class Duration(AttributeGenerator):
    max_duration: float

    def get_duration_and_slack(self, kernel: ProfiledKernel) -> Tuple[float, float]:
        duration_str, unit = kernel.get(StatKind.Duration)
        duration = float(duration_str)
        relative_duration = get_milliseconds(duration, unit) / self.max_duration
        return (relative_duration, 1 - relative_duration)

    def get_common(self, kernel: ProfiledKernel) -> Dict[str, str]:
        duration, slack = self.get_duration_and_slack(kernel)
        return {
            "style":      "striped",
            "fillcolor": f"lightblue;{duration}:white;{slack}",
        }

    def node(self, node, kernel):
        common = self.get_common(kernel)
        common["shape"] = "rect"
        return common

    def cluster(self, node, kernel):
        return self.get_common(kernel)

    @staticmethod
    def attribute_name() -> str:
        return "duration"
