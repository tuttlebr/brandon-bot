---
title: What is the CUDA C++ programming language?
---

CUDA C++ is an implementation of the
[CUDA programming model](/device-software/cuda-programming-model)
as an extension of the C++ programming language.

CUDA C++ adds several features to C++ to implement the
[CUDA programming model](/device-software/cuda-programming-model),
including:

- **[Kernel](/device-software/kernel) definition** with
  **`global`**. CUDA [kernels](/device-software/kernel) are
  implemented as C functions that take in pointers and have return type `void`,
  annotated with this keyword.
- **[Kernel](/device-software/kernel) launches** with **`<<<>>>`**.
  [Kernels](/device-software/kernel) are executed from the CPU host
  using a triple bracket syntax that sets the
  [thread block grid](/device-software/thread-block-grid)
  dimensions.
- **[Shared memory](/device-software/shared-memory) allocation**
  with the `shared` keyword, **barrier synchronization** with the
  `__syncthreads()` intrinsic function, and
  **[thread block](/device-software/thread-block)** and
  **[thread](/device-software/thread) indexing** with the
  `blockDim` and `threadIdx` built-in variables.

CUDA C++ programs are compiled by a combination of host C/C++ compiler drivers
like `gcc` and the
[NVIDIA CUDA Compiler Driver](/host-software/nvcc), `nvcc`.

For information on how to use CUDA C++ on Modal, see
[this guide](https://modal.com/docs/guide/cuda).
