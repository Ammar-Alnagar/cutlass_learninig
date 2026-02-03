# Module 02: CuTe Tensors

## Overview
This module focuses on CuTe tensors - the fundamental abstraction for wrapping raw pointers with layouts to create multidimensional views of memory.

## Key Concepts
- **Tensor Creation**: Wrapping raw pointers with layouts to create indexed views
- **Slicing Operations**: Extracting sub-tensors using layout transformations
- **Memory Access Patterns**: How tensors enable safe and efficient memory access
- **Layout Composition**: Combining multiple layouts to create complex access patterns

## Learning Objectives
By the end of this module, you will understand:
1. How to create tensors from raw pointers and layouts
2. Techniques for slicing tensors to access sub-regions
3. How tensor operations map to memory access patterns
4. Best practices for tensor composition and manipulation

## Building on Module 01
This module extends the layout concepts from Module 01 by showing how layouts become the foundation for safe, high-performance memory access through tensors.

## Files
- `tensor_basics.cu` - Basic tensor creation and access
- `tensor_slicing.cu` - Advanced slicing techniques
- `tensor_composition.cu` - Combining tensors and layouts
- `BUILD.md` - Build instructions for sm_89