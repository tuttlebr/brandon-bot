---
title: What are the NVIDIA GPU Drivers?
---

The NVIDIA GPU drivers mediate the interaction between host programs or the host
operating system and the GPU device. The primary interfaces to the GPU drivers
for applications are, in order, the
[CUDA Runtime API](/host-software/cuda-runtime-api) and the
[CUDA Driver API](/host-software/cuda-driver-api).

![The CUDA Toolkit. The NVIDIA GPU Driver, is the only component that communicates directly with the GPU. Adapted from the *Professional CUDA C Programming Guide*.](https://modal-cdn.com/gpu-glossary/terminal-cuda-toolkit.svg)

NVIDIA has released the
[source](https://github.com/NVIDIA/open-gpu-kernel-modules) for their Linux Open
GPU [Kernel Module](/host-software/nvidia-ko).
