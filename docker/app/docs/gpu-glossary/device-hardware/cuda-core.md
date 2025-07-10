---
title: What is a CUDA Core?
---

The CUDA Cores are GPU [cores](/device-hardware/core) that execute
scalar arithmetic instructions.

![The internal architecture of an H100 SM. The CUDA Cores and Tensor Cores are depicted in green. Note the larger size and lower number of Tensor Cores. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](https://modal-cdn.com/gpu-glossary/terminal-gh100-sm.svg)

They are to be contrasted with the
[Tensor Cores](/device-hardware/tensor-core), which execute matrix
operations.

Unlike CPU cores, instructions issued to CUDA Cores are not generally
independently scheduled. Instead, groups of cores are issued the same
instruction simultaneously by the
[Warp Scheduler](/device-hardware/warp-scheduler) but apply it to
different [registers](/device-software/registers). Commonly, these
groups are of size 32, the size of a [warp](/device-software/warp),
but for contemporary GPUs groups can contain as little as one thread, at a cost
to performance.

The term "CUDA Core" is slightly slippery: in different
[Streaming Multiprocessor architectures](/device-hardware/streaming-multiprocessor-architecture)
CUDA Cores can consist of different units -- a different mixture of 32 bit
integer and 32 bit and 64 bit floating point units.

So, for example, the
[H100 whitepaper](https://resources.nvidia.com/en-us-tensor-core) indicates that
an H100 GPU's
[Streaming Multiprocessors (SMs)](/device-hardware/streaming-multiprocessor)
each have 128 "FP32 CUDA Cores", which accurately counts the number of 32 bit
floating point units but is double the number of 32 bit integer or 64 bit
floating point units (as evidenced by the diagram above). For estimating
performance, it's best to look directly at the number of hardware units for a
given operation.
