---
title: What is a Thread Block Grid?
---

![Thread block grids are the highest level of the thread group hierarchy of the [CUDA programming model](/device-software/cuda-programming-model) (left). They map onto multiple [Streaming Multiprocessors](/device-hardware/streaming-multiprocessor) (right, bottom). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

When a CUDA [kernel](/device-software/kernel) is launched, it
creates a collection of [threads](/device-software/thread) known as
a thread block grid. Grids can be one, two, or three dimensional. They are made
up of [thread blocks](/device-software/thread-block).

The matching level of the
[memory hierarchy](/device-software/memory-hierarchy) is the
[global memory](/device-software/global-memory).

[Thread blocks](/device-software/thread-block) are effectively
independent units of computation. They execute concurrently, that is, with
indeterminate order, ranging from fully sequentially in the case of a GPU with a
single
[Streaming Multiprocessor](/device-hardware/streaming-multiprocessor)
to fully in parallel when run on a GPU with sufficient resources to run them all
simultaneously.
