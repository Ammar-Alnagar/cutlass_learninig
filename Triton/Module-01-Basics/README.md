# Module 1: Introduction to Triton and Basic Tensor Operations

## Overview
Welcome to Triton! This module introduces you to the basics of Triton programming for GPU acceleration. Triton is a domain-specific language (DSL) that enables you to write custom PyTorch operations that run on GPUs, with performance close to hand-written CUDA while being much easier to develop and maintain.

## Key Concepts
- **GPU Programming**: Understanding why we need specialized GPU programming
- **Triton vs CUDA**: Why Triton can be easier than raw CUDA
- **Tensor Operations**: Basic operations on tensors using Triton
- **Kernel Functions**: Writing your first Triton kernel

## What is Triton?
Triton is a Python-based language that compiles to GPU code. It provides:
- High-level abstractions for GPU programming
- Automatic optimization for GPU architectures
- Easy integration with PyTorch
- Performance comparable to hand-tuned CUDA

## Prerequisites
Before starting this module, ensure you have:
- Basic Python knowledge
- Understanding of tensors (like NumPy arrays)
- Familiarity with PyTorch (optional but helpful)

## Learning Objectives
By the end of this module, you will:
1. Understand the basic structure of a Triton program
2. Know how to define and launch a simple kernel
3. Perform basic tensor operations using Triton
4. Understand the relationship between Triton and PyTorch

## Basic Triton Program Structure

```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_function(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Kernel implementation goes here
    pass

# Launch the kernel
grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
kernel_function[grid](input_ptr, output_ptr, n_elements, BLOCK_SIZE=1024)
```

## Key Components Explained:
1. **`@triton.jit`**: Decorator that marks a function as a Triton kernel
2. **`tl.constexpr`**: A compile-time constant that gets optimized during compilation
3. **Grid**: Defines how many times to launch the kernel (parallel execution)
4. **Block Size**: Number of elements processed per kernel instance

## Important Notes for Beginners:
- Triton kernels run on the GPU, not the CPU
- The `@triton.jit` decorator compiles the function to GPU code
- Always ensure tensors are on the correct device (GPU)
- Memory access patterns significantly affect performance

## Next Steps
After mastering this module, proceed to Module 2 to learn about memory operations and data movement between CPU and GPU.