---
title: What is the CUDA Programming Model?
---

CUDA stands for _Compute Unified Device Architecture_. Depending on the context,
"CUDA" can refer to multiple distinct things: a
[high-level device architecture](/device-hardware/cuda-device-architecture),
a parallel programming model for architectures with that design, or a
[software platform](/host-software/cuda-software-platform) that
extends high-level languages like C to add that programming model.

The vision for CUDA is laid out in the
[Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
white paper. We highly recommend this paper, which is the original source for
many claims, diagrams, and even specific turns of phrase in NVIDIA's
documentation.

Here, we focus on the CUDA _programming model_.

The Compute Unified Device Architecture (CUDA) programming model is a
programming model for programming massively parallel processors.

Per the
[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#a-scalable-programming-model),
there are three key abstractions in the CUDA programming model:

- **Hierarchy of thread groups**. Programs are executed in threads but can make
  reference to groups of threads in a nested hierarchy, from
  [blocks](/device-software/thread-block) to
  [grids](/device-software/thread-block-grid).
- **Hierarchy of memories**. Thread groups have access to a
  [memory resource](/device-software/memory-hierarchy) for
  communication between [threads](/device-software/thread) in the
  group. Accessing the
  [lowest layer](/device-software/shared-memory) of the memory
  hierarchy should be
  [nearly as fast as executing an instruction](/device-hardware/l1-data-cache).
- **Barrier synchronization.** Thread groups can coordinate execution by means
  of barriers.

The hierarchies of execution and memory and their mapping onto
[device hardware](/device-hardware) are summarized in the following
diagram.

![Left: the abstract thread group and memory hierarchies of the CUDA programming model. Right: the matching hardware implementing those abstractions. Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

Together, these three abstractions encourage the expression of programs in a way
that scales transparently as GPU devices scale in their parallel execution
resources.

Put provocatively: this programming model prevents programmers from writing
programs for NVIDIA's
[CUDA-architected](/device-hardware/cuda-device-architecture) GPUs
that fail to get faster when the program's user buys a new NVIDIA GPU.

For example, each [thread block](/device-software/thread-block) in
a CUDA program can coordinate tightly, but coordination between blocks is
limited. This ensures blocks capture parallelizable components of the program
and can be scheduled in any order — in the terminology of NVIDIA documentation,
the programmer "exposes" this parallelism to the compiler and hardware. When the
program is executed on a new GPU that has more scheduling units (specifically,
more
[Streaming Multiprocessors](/device-hardware/streaming-multiprocessor)),
more of these blocks can be executed in parallel.

![A CUDA program with eight [blocks](/device-software/thread-block) runs in four sequential steps (waves) on a GPU with two [SMs](/device-hardware/streaming-multiprocessor) but in half as many steps on one with twice as many [SMs](/device-hardware/streaming-multiprocessor). Modified from the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).](https://modal-cdn.com/gpu-glossary/terminal-wave-scheduling.svg)

The CUDA programming model abstractions are made available to programmers as
extensions to high-level CPU programming languages, like the
[CUDA C++ extension of C++](/host-software/cuda-c). The programming
model is implemented in software by an instruction set architecture
[(Parallel Thread eXecution, or PTX)](/device-software/parallel-thread-execution)
and low-level assembly language
[(Streaming Assembler, or SASS)](/device-software/streaming-assembler).
For example, the [thread block](/device-software/thread-block)
level of the thread hierarchy is implemented via
[cooperative thread arrays](/device-software/cooperative-thread-array)
in these languages.
