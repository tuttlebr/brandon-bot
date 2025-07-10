---
title: What is a Load/Store Unit?
abbreviation: LSU
---

The Load/Store Units (LSUs) dispatch requests to load or store data to the
memory subsystems of the GPU.

![The internal architecture of an H100 SM. Load/Store Units are shown in pink, along with the [Special Function Units](/device-hardware/special-function-unit). Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](https://modal-cdn.com/gpu-glossary/terminal-gh100-sm.svg)

Most importantly for
[CUDA programmers](/host-software/cuda-software-platform) they
interact with the
[Streaming Multiprocessor](/device-hardware/streaming-multiprocessor)'s
on-chip SRAM [L1 data cache](/device-hardware/l1-data-cache) and
the off-chip, on-device [global RAM](/device-hardware/gpu-ram) that
respectively implement the lowest and highest levels of the
[memory hierarchy](/device-software/memory-hierarchy) in the
[CUDA programming model](/device-software/cuda-programming-model).
