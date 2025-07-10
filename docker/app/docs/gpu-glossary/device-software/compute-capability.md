---
title: What is Compute Capability?
---

Instructions in the
[Parallel Thread Execution](/device-software/parallel-thread-execution)
instruction set are compatible with only certain physical GPUs. The versioning
system used to abstract away details of physical GPUs from the instruction set
and [compiler](/host-software/nvcc) is called "Compute Capability".

Most compute capability version numbers have two components: a major version and
a minor version. NVIDIA promises forward compatibility (old
[PTX](/device-software/parallel-thread-execution) code runs on new
GPUs) across both major and minor versions following the
[onion layer](https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-module-directives-target)
model.

With Hopper, NVIDIA has introduced an additional version suffix, the `a` in
`9.0a`, which includes features that deviate from the onion model: their future
support is not guaranteed.

Target compute capabilities for
[PTX](/device-software/parallel-thread-execution) compilation can
be specified when invoking `nvcc`, the
[NVIDIA CUDA Compiler Driver](/host-software/nvcc). By default, the
compiler will also generate optimized
[SASS](/device-software/streaming-assembler) for the matching
[Streaming Multiprocessor (SM) architecture](/device-hardware/streaming-multiprocessor-architecture).
The
[documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architectures)
for [nvcc](/host-software/nvcc) refers to compute capability as a
"virtual GPU architecture", in contrast to the "physical GPU architecture"
expressed by the [SM](/device-hardware/streaming-multiprocessor)
version.

The technical specifications for each compute capability version can be found in
the
[Compute Capability section of the NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=compute%2520capability#compute-capabilities).
