# Module 05: Shared Memory & Swizzling

## Overview
This module addresses shared memory optimization and bank conflict resolution using layout algebra, specifically targeting sm_89 architecture characteristics.

## Key Concepts
- **Shared Memory Layouts**: Optimizing data placement in limited shared memory
- **Bank Conflict Resolution**: Using swizzling patterns to eliminate conflicts
- **Memory Padding**: Strategic padding to avoid bank conflicts
- **Access Pattern Analysis**: Identifying and resolving conflict patterns

## Learning Objectives
By the end of this module, you will understand:
1. How to design shared memory layouts that minimize bank conflicts
2. Swizzling techniques to redistribute memory accesses
3. Trade-offs between memory utilization and conflict avoidance
4. How layout algebra enables systematic conflict resolution

## Building on Previous Modules
This module applies layout algebra principles to solve shared memory bank conflicts while maintaining efficient data access patterns established in previous modules.

## Files
- `shared_memory_layouts.cu` - Basic shared memory layout design
- `bank_conflict_analysis.cu` - Identifying and diagnosing conflicts
- `swizzling_techniques.cu` - Implementing swizzling patterns
- `conflict_resolution_examples.cu` - Practical conflict resolution
- `BUILD.md` - Build instructions for sm_89