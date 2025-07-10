---
title: What is a Warp Scheduler?
---

The Warp Scheduler of the
[Streaming Multiprocessor (SM)](/device-hardware/streaming-multiprocessor)
decides which group of [threads](/device-software/thread) to
execute.

![The internal architecture of an H100 SM. The Warp Scheduler and Dispatch Unit are shown in orange. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](https://modal-cdn.com/gpu-glossary/terminal-gh100-sm.svg)

These groups of threads, known as [warps](/device-software/warp),
are switched out on a per clock cycle basis — roughly one nanosecond.

CPU thread context switches, on the other hand, take few hundred to a few
thousand clock cycles (more like a microsecond than a nanosecond) due to the
need to save the context of one thread and restore the context of another.
Additionally, context switches on CPUs lead to reduced locality, further
reducing performance by increasing cache miss rates (see
[Mogul and Borg, 1991](https://www.researchgate.net/publication/220938995_The_Effect_of_Context_Switches_on_Cache_Performance)).

Because each [thread](/device-software/thread) has its own private
[registers](/device-software/registers) allocated from the
[register file](/device-hardware/register-file) of the
[SM](/device-hardware/streaming-multiprocessor), context switches
on the GPU do not require any data movement to save or restore contexts.

Because the [L1 caches](/device-hardware/l1-data-cache) on GPUs can
be entirely programmer-managed and are
[shared](/device-software/shared-memory) between the
[warps](/device-software/warp) scheduled together onto an
[SM](/device-hardware/streaming-multiprocessor) (see
[cooperative thread array](/device-software/cooperative-thread-array)),
context switches on the GPU have much less impact on cache hit rates. For
details on the interaction between programmer-managed caches and
hardware-managed caches in GPUs, see
[the "Maximize Memory Throughput" section of the CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput)
