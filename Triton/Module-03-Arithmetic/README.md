# Module 3: Basic Arithmetic and Element-wise Operations

## Overview
This module covers basic arithmetic operations and element-wise computations in Triton. You'll learn how to perform mathematical operations on tensors element by element, which is fundamental for many GPU algorithms.

## Key Concepts
- **Element-wise Operations**: Performing operations on corresponding elements of tensors
- **Arithmetic Operations**: Addition, subtraction, multiplication, division, etc.
- **Mathematical Functions**: Trigonometric, exponential, logarithmic functions
- **Broadcasting**: How Triton handles operations between tensors of different shapes

## Learning Objectives
By the end of this module, you will:
1. Perform basic arithmetic operations in Triton kernels
2. Use mathematical functions available in Triton
3. Understand how to implement element-wise operations efficiently
4. Handle operations between tensors of different shapes

## Triton Mathematical Operations
Triton provides a rich set of mathematical operations through `triton.language`:
- Basic arithmetic: `+`, `-`, `*`, `/`, `//`, `%`
- Comparison: `<`, `<=`, `>`, `>=`, `==`, `!=`
- Mathematical functions: `tl.sqrt`, `tl.exp`, `tl.log`, `tl.sin`, `tl.cos`, etc.
- Logical operations: `&`, `|`, `^`, `~`

## Important Notes:
- All operations happen element-wise
- Operations are performed in parallel across tensor elements
- Always consider numerical precision when performing floating-point operations

## Next Steps
After mastering arithmetic operations, proceed to Module 4 to learn about block operations and tiling concepts.