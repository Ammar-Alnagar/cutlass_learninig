# Module 04: MMA Atoms

## Overview
This module delves into CuTe's MMA (Matrix Multiply-Accumulate) atoms, providing direct access to Tensor Core operations on sm_89 hardware.

## Key Concepts
- **MMA Atoms**: Hardware-specific matrix multiplication units
- **Thread Block Tiling**: Partitioning work among threads for Tensor Core operations
- **Accumulator Registers**: Managing intermediate and final results
- **Data Movement Coordination**: Aligning inputs with MMA operation requirements

## Learning Objectives
By the end of this module, you will understand:
1. How to instantiate and configure MMA atoms for different precisions
2. Thread-to-Tensor-Core mapping strategies
3. How to orchestrate data movement to feed MMA operations
4. Accumulation patterns and register management

## Building on Previous Modules
This module integrates layout algebra, tensor operations, and efficient data movement to implement high-performance matrix multiplication using Tensor Cores.

## Files
- `mma_atom_basics.cu` - Basic MMA atom usage
- `thread_tensor_mapping.cu` - Thread-to-Tensor-Core assignment
- `accumulator_management.cu` - Working with accumulator registers
- `mixed_precision_mma.cu` - Different precision configurations
- `BUILD.md` - Build instructions for sm_89