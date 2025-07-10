---
title: What is a Thread?
---

![Threads are the lowest level of the thread group hierarchy (top, left) and are mapped onto the [cores](/device-hardware/core) of a [Streaming Multiprocessor](/device-hardware/streaming-multiprocessor). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

A _thread of execution_ (or "thread" for short) is the lowest unit of
programming for GPUs, the atom of the
[CUDA programming model](/device-software/cuda-programming-model)'s
thread group hierarchy. A thread has its own
[registers](/device-software/registers), but little else.

Both [SASS](/device-software/streaming-assembler) and
[PTX](/device-software/parallel-thread-execution) programs target
threads. Compare this to a typical C program in a POSIX environment, which
targets a process, itself a collection of one or more threads.

Like a thread on a CPU, a GPU thread can have a private instruction
pointer/program counter. However, for performance reasons, GPU programs are
generally written so that all the threads in a
[warp](/device-software/warp) share the same instruction pointer,
executing instructions in lock-step (see also
[Warp Scheduler](/device-hardware/warp-scheduler)).

Also like threads on CPUs, GPU threads have stacks in
[global memory](/device-hardware/gpu-ram) for storing spilled
registers and a function call stack, but high-performance
[kernels](/device-software/kernel) generally avoid using either.

A single [CUDA Core](/device-hardware/cuda-core) executes
instructions from a single thread.
