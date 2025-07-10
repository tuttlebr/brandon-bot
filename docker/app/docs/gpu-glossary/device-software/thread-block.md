---
title: What is a Thread Block?
---

![Thread blocks are an intermediate level of the thread group hierarchy of the [CUDA programming model](/device-software/cuda-programming-model) (left). A thread block executes on a single [Streaming Multiprocessor](/device-hardware/streaming-multiprocessor) (right, middle). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

A thread block is a level of the CUDA programming model's thread hierarchy below
a grid but above a [warp](/device-software/warp). It is the CUDA
programming model's abstract equivalent of the concrete
[cooperative thread arrays](/device-software/cooperative-thread-array)
in
[PTX](/device-software/parallel-thread-execution)/[SASS](/device-software/streaming-assembler).

Blocks are the smallest unit of thread coordination exposed to programmers.
Blocks must execute independently, so that any execution order for blocks is
valid, from fully serial in any order to all interleavings.

A single CUDA [kernel](/device-software/kernel) launch produces one
or more thread blocks (in the form of a
[block grid](/device-software/thread-block-grid)), each of which
contains one or more [warps](/device-software/warp). Blocks can be
arbitrarily sized, but they are typically multiples of the
[warp](/device-software/warp) size (32 on all current CUDA GPUs).
