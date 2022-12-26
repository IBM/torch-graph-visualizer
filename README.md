# torch-graph-visualizer

A visualization tool to investigate bottlenecks of the computational graph in
PyTorch.

This tool embeds NVIDIA profiling data into the execution graph of the
model. Profiling data, such as:

- Kernel latency (a.k.a. duration)
- Kernel memory and computation throughput usage
- Tensor shapes

## Feature

The main feature present in this tool is to correlate NVIDIA NSight Compute
profiled low-level kernels (e.g. `volta_sgemm_XXXX`) with PyTorch high-level
operations (e.g. `torch.bmm`).

## Description

`torch-graph-visualizer` works in 2 steps:

1. profiling with NVTX annotations
2. drawing

The tool assumes that both steps have access to the same model (same
optimizations applied). So, when drawing, you should provide the same
(optimized) model. In order to make things easier, this tool provides basic
boilerplate for using PyTorch JIT options:

- `torch_graph_visualizer.run_model`: uses NVTX to annotate and run the model
- `torch_graph_visualizer.default_draw_model`: parses the profiling data,
  assuming a set of NVTX annotations


## Example

See [this file](example/vgg.py) for an example using VGG.
