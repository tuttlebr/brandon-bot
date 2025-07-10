---
title: What is Global Memory?
---

![Global memory is the highest level of the [memory hierarchy](/device-software/memory-hierarchy) in the [CUDA programming model](/device-software/cuda-programming-model). It is stored in the [GPU RAM](/device-hardware/gpu-ram). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

As part of the
[CUDA programming model](/device-software/cuda-programming-model),
each level of the thread group hierarchy has access to matching memory from the
[memory hierarchy](/device-software/memory-hierarchy). This memory
can be used for coordination and communication and is managed by the programmer
(not the hardware or a runtime).

The highest level of that memory hierarchy is the global memory. Global memory
is global in its scope and its lifetime. That is, it is accessible by every
[thread](/device-software/thread) in a
[thread block grid](/device-software/thread-block-grid) and its
lifetime is as long as the execution of the program.

Access to data structures in the global memory can be synchronized across all
accessors using atomic instructions, as with CPU memory. Within a
[cooperative thread array](/device-software/cooperative-thread-array),
access can be more tightly synchronized, e.g. with barriers.

This level of the
[memory hierarchy](/device-software/memory-hierarchy) is typically
implemented in the [GPU's RAM](/device-hardware/gpu-ram) and
allocated from the host using a memory allocator provided by the
[CUDA Driver API](/host-software/cuda-driver-api) or the
[CUDA Runtime API](/host-software/cuda-runtime-api).
