# Module 03: Tiled Copy

## Overview
This module explores CuTe's TiledCopy mechanism for efficient data movement, focusing on vectorized loads and cp.async operations optimized for sm_89.

## Key Concepts
- **Tiled Copy Atoms**: Hardware-aware copy operations for optimal bandwidth
- **Vectorized Access**: 128-bit loads and stores for maximum throughput
- **Async Memory Operations**: Using cp.async for overlapping computation and memory transfer
- **Coalescing Strategies**: Ensuring memory accesses align with warp execution

## Learning Objectives
By the end of this module, you will understand:
1. How to configure TiledCopy for different data types and access patterns
2. Techniques for maximizing memory bandwidth with vectorized operations
3. Proper use of cp.async for asynchronous memory transfers
4. How tiled copies integrate with thread block arrangements

## Building on Previous Modules
This module combines layout algebra (Module 01) and tensor concepts (Module 02) to implement efficient data movement patterns that feed computational units.

## Files
- `tiled_copy_basics.cu` - Basic TiledCopy usage
- `vectorized_loads.cu` - 128-bit vectorized operations
- `async_copy_sm89.cu` - cp.async implementation for sm_89
- `coalescing_patterns.cu` - Memory access optimization
- `BUILD.md` - Build instructions for sm_89