---
title: What is a Special Function Unit?
abbreviation: SFU
---

The Special Function Units (SFUs) in
[Streaming Multiprocessors (SMs)](/device-hardware/streaming-multiprocessor)
accelerate certain arithmetic operations.

![The internal architecture of an H100 SM. Special Function Units are shown in maroon, along with the [Load/Store Units](/device-hardware/load-store-unit). Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](https://modal-cdn.com/gpu-glossary/terminal-gh100-sm.svg)

Notable for neural network workloads are transcendental mathematical operations,
like `exp`, `sin`, and `cos`.
