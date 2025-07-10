---
title: What is the Memory Hierarchy?
---

![[Shared memory](/device-software/shared-memory) and [global memory](/device-software/global-memory) are two levels of the memory hierarchy in the [CUDA programming model](/device-software/cuda-programming-model) (left), mapping onto the [L1 data cache](/device-hardware/l1-data-cache) and [GPU RAM](/device-hardware/gpu-ram), respectively. Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

As part of the
[CUDA programming model](/device-software/cuda-programming-model),
each level of the thread group hierarchy has access to a distinct block of
memory shared by all threads in a group at that level: a "memory hierarchy" to
match the thread group hierarchy. This memory can be used for coordination and
communication and is managed by the programmer (not the hardware or a runtime).

For a [thread block grid](/device-software/thread-block-grid), that
shared memory is in the [GPU's RAM](/device-hardware/gpu-ram) and
is known as the [global memory](/device-software/global-memory).
Access to this memory can be coordinated with atomic operations and barriers,
but execution order across
[thread blocks](/device-software/thread-block) is indeterminate.

For a single [thread](/device-software/thread), the memory is a
chunk of the
[Streaming Multiprocessor's (SM's)](/device-hardware/streaming-multiprocessor)
[register file](/device-hardware/register-file). In keeping with
the memory semantics of the
[CUDA programming model](/device-software/cuda-programming-model),
this memory is private.

In between, the [shared memory](/device-software/shared-memory) for
the [thread block](/device-software/thread-block) level of the
thread hierarchy is stored in the
[L1 data cache](/device-hardware/l1-data-cache) of each
[SM](/device-hardware/streaming-multiprocessor). Careful management
of this cache — e.g. loading data into it to support the maximum number of
arithmetic operations before new data is loaded — is key to the art of designing
high-performance CUDA [kernels](/device-software/kernel).
