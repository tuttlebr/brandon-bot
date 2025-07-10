---
title: What are Registers?
---

![Registers are the memory of the [memory hierarchy](/device-software/memory-hierarchy) associated with individual [threads](/device-software/thread) (left). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

At the lowest level of the
[memory hierarchy](/device-software/memory-hierarchy) are the
registers, which store information manipulated by a single
[thread](/device-software/thread).

The values in registers are generally stored in the
[register file](/device-hardware/register-file) of the
[Streaming Multiprocessor (SM)](/device-hardware/streaming-multiprocessor),
but they can also spill to the
[global memory](/device-software/global-memory) in the
[GPU RAM](/device-hardware/gpu-ram) at a substantial performance
penalty.

As when programming CPUs, these registers are not directly manipulated by
high-level languages like [CUDA C](/host-software/cuda-c). They are
only visible to lower-level languages like
[Parallel Thread Execution (PTX)](/device-software/parallel-thread-execution)
or
[Streaming Assembler (SASS)](/device-software/streaming-assembler)
and so are typically managed by a compiler like
[nvcc](/host-software/nvcc). Among the compiler's goals is to limit
the register space used by each [thread](/device-software/thread)
so that more [thread blocks](/device-software/thread-block) can be
simultaneously scheduled into a single
[SM](/device-hardware/streaming-multiprocessor).

The registers used in the
[PTX](/device-software/parallel-thread-execution) instruction set
architecture are documented
[here](https://docs.nvidia.com/cuda/parallel-thread-execution/#register-state-space).
The registers used in [SASS](/device-software/streaming-assembler)
are not, to our knowledge, documented.
