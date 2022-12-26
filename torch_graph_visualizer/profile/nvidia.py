import datetime
import logging
import re

from abc import ABC, abstractclassmethod
from dataclasses import dataclass, field, fields
from enum import Enum, auto

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import torch_graph_visualizer.profile as prof

logger = logging.getLogger(__name__)

_RE_PROCESS = re.compile(
    r"^\[(?P<pid>\d*)\] (?P<name>[^@]*)@(?P<hostip>[0-9\.]*)"
)

_RE_KERNEL = re.compile(
    r"^  (?P<_name>\S.*)"
    r"(, (?P<_date>\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2})|)"
    r"(, Context (?P<context>\d*)|)"
    r"(, Stream (?P<stream>\d*)|)"
    r"(, Device (?P<device>\d*)|)"
    r"(, CC (?P<cc>\d*\.\d*)|)"
)

_RE_CALLSTACK_ENTRY = re.compile(
    r"      #(?P<depth>\d*) (?P<address>0x[0-9a-f]*) (?P<mode>in|of) (?P<_name>.*)"
)

_RE_NVTX = re.compile(
    r"        <(?P<depth>[0-9]*),(?P<name>[^>]*)>"
)

_RE_STATS_SECTION = re.compile(
    r"^    Section: (?P<name>.*)"
)

_RE_STATS_SEPARATOR = re.compile(
    r"^    -[- ]*"
)

_RE_STATS_ROW = re.compile(
    r"^    (?P<description>([^ ]+ )*) *(?P<unit>[A-Za-z/%]*) *(?P<value>[0-9\.]+)"
)

_RE_TORCH_OP = re.compile(
    r"(?P<prefix>at::_ops|at::native)::(?P<name>[^:]*)(::(?P<method>.*)|)\(.*"
)

_RE_FUSION = re.compile(
    r"^CudaCodeGen::(?P<name>kernel\d*)\("
)


class LineKind(Enum):
    Process = auto()
    Kernel = auto()
    CallStackEntry = auto()
    NVTX = auto()
    StatsSection = auto()
    StatsSeparator = auto()
    StatsRow = auto()


class _BaseNewWithExtraKwargs(ABC):
    @abstractclassmethod
    def new(cls, **kwargs):
        pass


class _NewWithExtraKwargs(_BaseNewWithExtraKwargs):
    @classmethod
    def new(cls, **kwargs):
        kw = {f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs}
        return cls(**kw)


@dataclass
class NVIDIACallStackEntry(_NewWithExtraKwargs, prof.CallStackEntry):
    _name: str

    def name(self) -> str:
        return self._name


@dataclass
class NVIDIAProfiledKernel(_BaseNewWithExtraKwargs, prof.ProfiledKernel):
    CALLSTACK: ClassVar[str] = "_callstack"
    NVTX: ClassVar[str] = "_nvtx"
    STATS: ClassVar[str] = "_stats"

    _name: str
    _date: Optional[datetime.datetime]
    _callstack: List[NVIDIACallStackEntry]
    _nvtx: List[str]
    _stats: Dict[str, Any]

    @classmethod
    def new(cls, **kwargs):
        kw = {f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs}
        kw["_date"] = datetime.datetime.strptime(kw["_date"], "%Y-%b-%d %H:%M:%S") \
            if kw["_date"] is not None else None
        return cls(**kw)

    def name(self) -> str:
        return self._name

    def callstack(self) -> Sequence[prof.CallStackEntry]:
        return self._callstack

    def get(self, stat: prof.Stat) -> Any:
        print(self._stats)
        if stat == prof.StatKind.ComputeThroughput:
            return self._stats["GPUSpeedOfLightThroughput"]["Compute"]
        elif stat == prof.StatKind.MemoryThroughput:
            return self._stats["GPUSpeedOfLightThroughput"]["Memory"]
        elif stat == prof.StatKind.Duration:
            return self._stats["GPUSpeedOfLightThroughput"]["Duration"]
        raise ValueError(f"unexpected stat: {stat}")


@dataclass
class NVIDIAProfiledProcess(_NewWithExtraKwargs):
    KERNELS_NAME: ClassVar[str] = "kernels"

    name: str
    kernels: List[NVIDIAProfiledKernel]


@dataclass(frozen=True)
class PrettyPrinter:
    indent: int = 2

    def get_indent(self, level: int) -> str:
        return " " * (level * self.indent)

    def _print_each(self, lst: List, level: int) -> List[str]:
        return [self.print(el, level) for el in lst]

    def _print_callstack_entry(self, entry: NVIDIACallStackEntry, level: int) -> str:
        return f"{self.get_indent(level)}Entry({entry.name})"

    def _print_kernel(self, kernel: NVIDIAProfiledKernel, level: int) -> str:
        self_indent = self.get_indent(level)
        fields_indent = self.get_indent(level + 1)
        fields_item_indent = self.get_indent(level + 2)

        nvtx = ", ".join(kernel._nvtx)
        nvtx = f"[{nvtx}]"

        stats = [f"{fields_item_indent}{k}: {v}" for k, v in kernel._stats.items()]

        lines = [
            f"{self_indent}Kernel(",
            f"{fields_indent}name:      {kernel._name}",
            f"{fields_indent}op:        {kernel.torch_op_callstack[0]}",
            f"{fields_indent}jit:       {kernel.jit}",
            f"{fields_indent}fusion:    {kernel.fusion}",
            f"{fields_indent}nvtx:      {nvtx}",
            f"{fields_indent}stats:",
            *stats,
            f"{fields_indent}callstack:",
            *self._print_each(kernel._callstack, level + 2),
            f"{self_indent})"
        ]

        return "\n".join(lines)

    def _print_process(self, process: NVIDIAProfiledProcess, level: int) -> str:
        self_indent = self.get_indent(level)
        fields_indent = self.get_indent(level + 1)

        lines = [
            f"{self_indent}Process(",
            f"{fields_indent}name:    {process.name}",
            f"{fields_indent}kernels:",
            *self._print_each(process.kernels, level + 2),
            f"{self_indent})"
        ]

        return "\n".join(lines)

    def print(self, thing: Any, level: int = 0) -> str:
        if isinstance(thing, NVIDIACallStackEntry):
            return self._print_callstack_entry(thing, level + 1)
        if isinstance(thing, NVIDIAProfiledKernel):
            return self._print_kernel(thing, level + 1)
        if isinstance(thing, NVIDIAProfiledProcess):
            return self._print_process(thing, level + 1)
        raise ValueError(f"can't print type: {type(thing)}")


@dataclass(frozen=True)
class PredicatedPrinter(PrettyPrinter):
    predicate: Callable[[Any], bool] = lambda x: True

    def print(self, thing: Any, level: int = 0) -> str:
        if self.predicate(thing):
            return super().print(thing, level)
        return ""


class SectionState(Enum):
    Title = auto()
    Header = auto()
    Body = auto()
    Outside = auto()


@dataclass
class _Parser:
    processes: List[NVIDIAProfiledProcess] = field(default_factory=list)

    _current_stack_entry_data: Optional[Dict[str, Any]] = None
    _current_kernel_data: Optional[Dict[str, Any]] = None
    _current_process_data: Optional[Dict[str, Any]] = None
    _current_section_data: Optional[Tuple[str, bool]] = None
    _current_section_state: SectionState = SectionState.Outside

    def _append_process(self) -> None:
        if self._current_process_data is not None:
            self.processes.append(
                NVIDIAProfiledProcess.new(**self._current_process_data)
            )

    def _append_kernel(self) -> None:
        if self._current_kernel_data is not None:
            assert self._current_process_data is not None
            self._current_process_data[NVIDIAProfiledProcess.KERNELS_NAME].append(
                NVIDIAProfiledKernel.new(**self._current_kernel_data)
            )

    def handle_process(self, groupdict: Dict[str, Any]) -> None:
        self._append_process()
        self._current_process_data = groupdict
        self._current_process_data[NVIDIAProfiledProcess.KERNELS_NAME] = []

    def handle_kernel(self, groupdict: Dict[str, Any]) -> None:
        self._append_kernel()
        self._current_kernel_data = groupdict
        self._current_kernel_data[NVIDIAProfiledKernel.CALLSTACK] = []
        self._current_kernel_data[NVIDIAProfiledKernel.NVTX] = []

    def handle_callstack_entry(self, groupdict: Dict[str, str]) -> None:
        assert self._current_kernel_data is not None

        # Hack for better printing shared library addresses.
        if groupdict["mode"] == "of":
            groupdict["_name"] = groupdict["address"] + " of " + groupdict["_name"]

        self._current_kernel_data[NVIDIAProfiledKernel.CALLSTACK].append(
            NVIDIACallStackEntry.new(**groupdict)
        )

    def handle_nvtx(self, groupdict: Dict[str, str]) -> None:
        assert self._current_kernel_data is not None
        self._current_kernel_data[NVIDIAProfiledKernel.NVTX].append(
            groupdict["name"]
        )

    def handle_stats_section(self, groupdict: Dict[str, str]) -> None:
        name = groupdict["name"].replace(" ", "")
        self._current_section_data = (name, False)
        self._current_section_state = SectionState.Title

    def handle_stats_separator(self, groupdict: Dict[str, str]) -> None:
        assert self._current_kernel_data is not None

        if self._current_section_data is None:
            return

        if self._current_section_state == SectionState.Title:
            self._current_section_state = SectionState.Header
            name, active = self._current_section_data

            if active:
                self._current_section_data = None
            else:
                stats = self._current_kernel_data.setdefault(NVIDIAProfiledKernel.STATS, dict())
                stats[name] = dict()
                self._current_section_data = (name, True)

        elif self._current_section_state == SectionState.Header:
            self._current_section_state = SectionState.Body
        elif self._current_section_state == SectionState.Body:
            self._current_section_state = SectionState.Outside
        else:
            raise RuntimeError("unexpected section separator")

    def handle_stats_row(self, groupdict: Dict[str, str]) -> None:
        assert self._current_kernel_data is not None

        if self._current_section_state not in (SectionState.Header, SectionState.Body):
            logger.warning(
                "ignoring row due to unexpected section state: "
                f"{self._current_section_state}"
            )
            logger.warning(f"    {groupdict}")
            return

        if self._current_section_data is None:
            return

        name, active = self._current_section_data

        assert active, f"stats row before separator for section: {name}"

        key = groupdict["description"].replace(" ", "").replace("/", "Per")
        key = re.sub(r"(\[.*\]|\(.*\))", "", key)
        self._current_kernel_data[NVIDIAProfiledKernel.STATS][name][key] = (
            groupdict["value"],
            groupdict["unit"],
        )

    def finalize(self) -> None:
        self._append_kernel()
        self._append_process()


def parse_ncu_file(prof: str) -> List[NVIDIAProfiledProcess]:
    parser = _Parser()

    kind_and_regex = [
        (LineKind.Process,        _RE_PROCESS),
        (LineKind.Kernel,         _RE_KERNEL),
        (LineKind.CallStackEntry, _RE_CALLSTACK_ENTRY),
        (LineKind.NVTX,           _RE_NVTX),
        (LineKind.StatsSection,   _RE_STATS_SECTION),
        (LineKind.StatsSeparator, _RE_STATS_SEPARATOR),
        (LineKind.StatsRow,       _RE_STATS_ROW),
    ]

    dispatch = {
        LineKind.Process:        parser.handle_process,
        LineKind.Kernel:         parser.handle_kernel,
        LineKind.CallStackEntry: parser.handle_callstack_entry,
        LineKind.NVTX:           parser.handle_nvtx,
        LineKind.StatsSection:   parser.handle_stats_section,
        LineKind.StatsSeparator: parser.handle_stats_separator,
        LineKind.StatsRow:       parser.handle_stats_row,
    }

    with open(prof, "r", encoding="utf-8") as f:
        for rawline in f.readlines():
            line = rawline[:-1] if rawline[-1] == "\n" else rawline

            if len(line) == 0:
                continue

            for kind, regex in kind_and_regex:
                m = regex.match(line)
                if m:
                    # print(f"Matched {kind} for line: {line}")
                    dispatch[kind](m.groupdict())
                    break
                else:
                    # print(f"Failed matching {kind} for line: {line}")
                    ...

    parser.finalize()
    return parser.processes
