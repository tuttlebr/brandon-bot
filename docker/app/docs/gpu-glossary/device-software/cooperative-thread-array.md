---
title: What is a Cooperative Thread Array?
---

![Cooperative thread arrays correspond to the [thread block](/device-software/thread-block) level of the thread block hierarchy in the [CUDA programming model](/device-software/cuda-programming-model). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

A cooperative thread array (CTA) is a collection of threads scheduled onto the
same
[Streaming Multiprocessor (SM)](/device-hardware/streaming-multiprocessor).
CTAs are the
[PTX](/device-software/parallel-thread-execution)/[SASS](/device-software/streaming-assembler)
implementation of the
[CUDA programming model](/device-software/cuda-programming-model)'s
[thread blocks](/device-software/thread-block). CTAs are composed
of one or more [warps](/device-software/warp).

Programmers can direct [threads](/device-software/thread) within a
CTA to coordinate with each other. The programmer-managed
[shared memory](/device-software/shared-memory), in the
[L1 data cache](/device-hardware/l1-data-cache) of the
[SMs](/device-hardware/streaming-multiprocessor), makes this
coordination fast. Threads in different CTAs cannot coordinate with each other
via barriers, unlike threads within a CTA, and instead must coordinate via
[global memory](/device-software/global-memory), e.g. via atomic
update instructions. Due to driver control over the scheduling of CTAs at
runtime, CTA execution order is indeterminate and blocking a CTA on another CTA
can easily lead to deadlock.

The number of CTAs that can be scheduled onto a single
[SM](/device-hardware/streaming-multiprocessor) depends on a number
of factors. Fundamentally, the
[SM](/device-hardware/streaming-multiprocessor) has a limited set
of resources — lines in the
[register file](/device-hardware/register-file), "slots" for
[warps](/device-software/warp), bytes of
[shared memory](/device-software/shared-memory) in the
[L1 data cache](/device-hardware/l1-data-cache) — and each CTA uses
a certain amount of those resources (as calculated at
[compile](/host-software/nvcc) time) when scheduled onto an
[SM](/device-hardware/streaming-multiprocessor).
