---
title: What is a Graphics/GPU Processing Cluster?
abbreviation: GPC
---

A GPC is a collection of
[Texture Processing Clusters (TPCs)](/device-hardware/texture-processing-cluster)
(themselves groups of
[Streaming Multiprocessors](/device-hardware/streaming-multiprocessor)
or SMs) plus a raster engine. Apparently, some people use NVIDIA GPUs for
graphics, for which the raster engine is important. Relatedly, the name used to
stand for Graphics Processing Cluster, but is now, e.g. in the
[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html),
expanded as "GPU Processing Cluster".

For the latest
[compute capability](/device-software/compute-capability) 9.0 GPUs
like H100s, there is an additional layer of the
[CUDA programming model](/device-software/cuda-programming-model)'s
thread hierarchy, a "cluster" of
[thread blocks](/device-software/thread-block), that are scheduled
onto the same GPC, just as the threads of a
[thread block](/device-software/thread-block) are scheduled onto
the same [SM](/device-hardware/streaming-multiprocessor), and have
their own level of the
[memory hierarchy](/device-software/memory-hierarchy). Elsewhere,
we elide discussion of this feature.
