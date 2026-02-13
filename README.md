# AI Kernel Engineer Learning Repository

## Mastering High-Performance GPU Computing & Deep Learning Kernels

This comprehensive learning repository is designed to transform software engineers into expert AI kernel developers, focusing on the cutting-edge technologies required for developing high-performance GPU kernels using NVIDIA's latest architectures and libraries.

---

## Table of Contents

- [Repository Overview](#repository-overview)
- [Complete Directory Structure](#complete-directory-structure)
- [Detailed Module Descriptions](#detailed-module-descriptions)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Recommended Learning Path](#recommended-learning-path)
- [Quick Start Guide](#quick-start-guide)
- [Assessment & Certification](#assessment--certification)
- [Contributing](#contributing)
- [Career Impact](#career-impact)

---

## Repository Overview

This repository represents a complete curriculum for mastering GPU kernel development, from foundational concepts to production-level optimization. It covers multiple programming paradigms, optimization techniques, and tools essential for modern AI infrastructure development.

### Target Hardware

- **Primary**: NVIDIA RTX 4060 (Compute Capability 8.9 / Ada Lovelace)
- **Tensor Core Support**: FP16, BF16, INT8, and FP8 operations
- **Memory Hierarchy**: Global, shared, and register memory optimization
- **Warp-level Primitives**: Cooperative thread operations
- **Asynchronous Operations**: cp.async for overlapping computation and memory transfer

### Learning Philosophy

This repository emphasizes **composable abstractions** over manual indexing:

- **Mathematical Representation**: Thinking in terms of linear algebra and tensor operations
- **Functional Composition**: Building complex operations from simple, reusable components
- **Hardware Mapping**: Understanding how abstractions map to physical GPU resources
- **Performance Analysis**: Measuring and optimizing each component individually

---

## Complete Directory Structure

```
AI-Kernel-learning/
├── Cuda/                           # CUDA Programming Mastery (Fundamentals to Advanced)
│   ├── fundamentals/               # Core CUDA concepts
│   │   ├── thread_heir/            # Thread hierarchy organization
│   │   ├── mem_hier/               # Memory hierarchy concepts
│   │   ├── vec_add/                # Vector addition examples
│   │   ├── matmul/                 # Matrix multiplication
│   │   ├── reduction/              # Reduction operations
│   │   ├── shared_mem/             # Shared memory usage
│   │   ├── memory_coal/            # Memory coalescing
│   │   ├── atomics/                # Atomic operations
│   │   ├── cuda_streams/           # Asynchronous execution
│   │   └── warp/                   # Warp-level operations
│   ├── memory_optimization/        # Memory access optimization
│   │   ├── coalescing.md           # Memory coalescing patterns
│   │   ├── banking.md              # Shared memory banking
│   │   └── swizzling.md            # Swizzling techniques
│   ├── execution_optimization/     # Execution optimization
│   │   ├── occupancy.md            # Occupancy optimization
│   │   ├── warp_primitives.md      # Warp-level primitives
│   │   ├── register_pressure.md    # Register usage optimization
│   │   └── synchronization.md      # Synchronization primitives
│   ├── advanced_memory/            # Advanced memory operations
│   │   ├── async_copy.md           # Asynchronous copy operations
│   │   ├── tensor_cores.md         # Tensor Core usage
│   │   └── pipelining.md           # Software pipelining
│   ├── performance_analysis/       # Performance profiling
│   │   ├── roofline.md             # Roofline model analysis
│   │   └── profiling.md            # Nsight Compute profiling
│   ├── cute_specific/              # CuTe-specific concepts
│   │   ├── layout_algebra.md       # Layout algebra fundamentals
│   │   ├── tiled_layouts.md        # Hierarchical tiling
│   │   ├── copy_atoms.md           # Copy atom abstractions
│   │   └── mma_atoms.md            # MMA atom abstractions
│   └── advanced_concepts/          # Advanced CUDA topics
│       ├── memory_optimization/    # Advanced memory techniques
│       ├── execution_optimization/  # Advanced execution patterns
│       ├── mathematical_kernels/   # Optimized math kernels
│       ├── tensor_cores/           # Tensor Core programming
│       └── profiling_tools/        # Profiling methodologies
│
├── CuTE/                           # CuTe (CUTLASS 3.x) Progressive Learning
│   ├── Module_01_Layout_Algebra/   # Shapes, strides, hierarchical layouts
│   │   ├── README.md               # Layout algebra concepts
│   │   ├── layout_study.cu         # Practical demonstrations
│   │   └── BUILD.md                # Build instructions
│   ├── Module_02_CuTe_Tensors/     # Tensor creation and manipulation
│   │   ├── README.md               # Tensor concepts
│   │   ├── tensor_basics.cu        # Tensor operations
│   │   └── BUILD.md                # Build instructions
│   ├── Module_03_Tiled_Copy/       # Vectorized loads and cp.async
│   │   ├── README.md               # Tiled copy concepts
│   │   ├── tiled_copy_basics.cu    # Copy implementations
│   │   └── BUILD.md                # Build instructions
│   ├── Module_04_MMA_Atoms/        # Tensor Core operations
│   │   ├── README.md               # MMA atom concepts
│   │   ├── mma_atom_basics.cu      # MMA implementations
│   │   └── BUILD.md                # Build instructions
│   ├── Module_05_Shared_Memory_Swizzling/  # Bank conflict resolution
│   │   ├── README.md               # Swizzling concepts
│   │   ├── shared_memory_layouts.cu # Swizzling implementations
│   │   └── BUILD.md                # Build instructions
│   ├── Module_06_Collective_Mainloops/  # Producer-consumer pipelines
│   │   ├── README.md               # Collective concepts
│   │   ├── producer_consumer_pipeline.cu  # Pipeline implementations
│   │   └── BUILD.md                # Build instructions
│   ├── cutlass/                    # CUTLASS library (submodule)
│   ├── SETUP.md                    # Environment setup guide
│   ├── SUMMARY.md                  # Module summaries
│   └── COMPLETION_CERTIFICATE.md   # Certification information
├── Cutlass3.x/ # CUTLASS 3.x Advanced Linear Algebra
│ ├── module1-Layouts and Tensors/ # Foundation abstractions
│ │ ├── README.md # Layout and tensor concepts
│ │ ├── main.cu # Implementation examples
│ │ ├── CMakeLists.txt # Build configuration
│ │ └── build.sh # Build script
│ ├── module2-Tiled Copy/ # Vectorized memory operations
│ │ ├── README.md # Tiled copy concepts
│ │ ├── main.cu # Implementation examples
│ │ └── build.sh # Build script
│ ├── module3-Tiled MMA/ # Tensor Core operations
│ │ ├── README.md # Tiled MMA concepts
│ │ ├── main.cu # Implementation examples
│ │ └── build.sh # Build script
│ ├── module4-Fused Bias-Add/ # Fused operations
│ │ ├── README.md # Fusion concepts
│ │ ├── main.cu # Implementation examples
│ │ └── build.sh # Build script
│ ├── module5-Mainloop Pipelining/ # Temporal overlap optimization
│ │ ├── README.md # Pipelining concepts
│ │ ├── main.cu # Implementation examples
│ │ └── build.sh # Build script
│ ├── module6-Fused Epilogues/ # Eliminating VRAM roundtrips
│ │ ├── README.md # Epilogue concepts
│ │ ├── main.cu # Implementation examples
│ │ └── build.sh # Build script
│ ├── CMakeLists.txt # Top-level build configuration
│ ├── build_all.sh # Build all modules script
│ └── README.md # CUTLASS overview
│
├── DSA/ # Data Structures & Algorithms Fundamentals
│ ├── 00-Introduction-BigO/ # Complexity analysis
│ │ ├── README.md # Big O notation concepts
│ │ └── main.cpp # Example implementations
│ ├── 01-Arrays-Strings/ # Array and string operations
│ │ ├── README.md # Array/string concepts
│ │ └── main.cpp # Example implementations
│ ├── 02-Stacks-Queues/ # Stack and queue data structures
│ │ ├── README.md # Stack/queue concepts
│ │ └── main.cpp # Example implementations
│ ├── 03-Linked-Lists/ # Linked list implementations
│ │ ├── README.md # Linked list concepts
│ │ └── main.cpp # Example implementations
│ ├── 04-Searching-Sorting/ # Search and sort algorithms
│ │ ├── README.md # Algorithm concepts
│ │ └── main.cpp # Example implementations
│ ├── 05-Trees/ # Tree data structures
│ │ ├── README.md # Tree concepts
│ │ └── main.cpp # Example implementations
│ └── 06-Graphs/ # Graph algorithms
│ ├── README.md # Graph concepts
│ └── main.cpp # Example implementations
│
├── GPU-DSA/ # GPU-Optimized Data Structures & Algorithms
│ ├── Parallel_Reduction/ # Tree-based and warp shuffle reduction
│ ├── Parallel_Prefix_Sum/ # Blelloch and Kogge-Stone algorithms
│ ├── Parallel_Histogramming/ # Privatization and atomic aggregation
│ ├── Radix_Sort/ # LSB/MSB radix sort implementations
│ ├── Bitonic_Sort/ # Parallel bitonic sort
│ ├── Tiled_Matrix_Multiplication/ # GEMM optimizations
│ ├── Double_Buffering_Async_Copy/ # Overlapping computation and transfer
│ ├── Shared_Memory_Swizzling/ # XOR layouts for bank conflicts
│ ├── Z_Curve_Morton_Order/ # Space-filling curves
│ ├── Online_Softmax/ # Safe softmax algorithm
│ ├── FlashAttention/ # Fused attention with recomputation
│ ├── PagedAttention/ # Virtual memory for KV-Cache
│ ├── Fused_Layer_Normalization/ # Optimized layer normalization
│ └── Compressed_Sparse_Formats/ # CSR and Blocked-Ellpack

├── Triton/ # Triton High-Level GPU Programming
│ ├── Module-01-Basics/ # Introduction and basic operations
│ │ └── README.md # Basic Triton concepts
│ ├── Module-02-Memory/ # Memory operations and data movement
│ │ └── README.md # Memory management concepts
│ ├── Module-03-Arithmetic/ # Element-wise operations
│ │ └── README.md # Arithmetic operations
│ ├── Module-04-Blocks/ # Block operations and tiling
│ │ └── README.md # Tiling concepts
│ ├── Module-05-Matrix-Multiplication/ # Matrix multiplication fundamentals
│ │ └── README.md # GEMM concepts
│ ├── Module-06-Advanced-Memory/ # Advanced memory layouts
│ │ └── README.md # Memory optimization
│ ├── Module-07-Reductions/ # Reduction operations
│ │ └── README.md # Reduction algorithms
│ ├── Module-08-Advanced-Techniques/ # Best practices
│ │ └── README.md # Advanced techniques
│ └── README.md # Triton overview
│
├── NCCL/ # Multi-GPU Collective Communications
│ ├── Module-01/ # Introduction to NCCL
│ │ └── README.md # NCCL concepts and setup
│ ├── Module-02/ # Basic collective operations
│ │ └── README.md # AllReduce, Broadcast
│ ├── Module-03/ # Advanced collective operations
│ │ └── README.md # Reduce, AllGather, Scatter
│ ├── Module-04/ # Multi-GPU programming
│ │ └── README.md # Multi-GPU patterns
│ ├── Module-05/ # Multi-node communication
│ │ └── README.md # Distributed computing
│ ├── Module-06/ # Performance optimization
│ │ └── README.md # NCCL optimization
│ └── Module-07/ # Framework integration
│ └── README.md # Deep learning integration
│
├── Profiling/ # GPU Kernel Profiling and Optimization
│ ├── Module_1_Introduction_to_GPU_Computing/ # GPU computing concepts
│ ├── Module_2_Setting_Up_Profiling_Tools/ # Tool setup
│ ├── Module_3_Basic_Profiling_Techniques/ # Basic profiling
│ ├── Module_4_Identifying_Common_Bottlenecks/ # Bottleneck identification
│ ├── Module_5_Memory_Optimization_Techniques/ # Memory optimization
│ ├── Module_6_Computational_Optimization_Strategies/ # Compute optimization
│ ├── Module_7_Advanced_Profiling_Analysis/ # Advanced analysis
│ ├── Module_8_Real_world_Case_Studies/ # Real-world examples
│ └── README.md # Profiling overview
│
├── PTX/ # PTX Assembly Programming
│ ├── PTX-Basics/ # PTX syntax and structure
│ ├── PTX-Memory-Management/ # Memory spaces and optimization
│ ├── PTX-Debugging-Profiling/ # Debugging and profiling
│ ├── PTX-Advanced-Optimizations/ # Advanced optimization techniques
│ ├── PTX-Custom-Kernels/ # Custom kernel development
│ ├── getting_started.sh # Setup script
│ └── README.md # PTX overview
│
├── Temp-Meta/ # C++ Template Metaprogramming
│ ├── module_1_foundations/ # Modern C++ foundations
│ ├── module_2_fundamentals/ # Template fundamentals
│ ├── module_3_basics/ # Metaprogramming basics
│ ├── module_4_advanced/ # Advanced techniques
│ ├── module_5_cuda_fundamentals/ # CUDA fundamentals
│ ├── module_6_cutlass_architecture/ # CUTLASS architecture
│ ├── module_7_cutlass_patterns/ # CUTLASS patterns
│ ├── module_8_advanced_customization/ # Advanced customization
│ ├── module_9_real_world_applications/ # Real-world applications
│ ├── module_10_performance_optimization/ # Performance optimization
│ ├── quick_reference.md # Quick reference guide
│ ├── SUMMARY.md # Module summaries
│ └── README.md # Template metaprogramming overview

├── cmakeguide/ # CMake and Build Systems Mastery
│ ├── training_projects/ # Hands-on training projects
│ │ ├── modules/ # Modular project examples
│ │ ├── intermediate/ # Intermediate examples
│ │ └── README.md # Training project guide
│ ├── cmake_guide.md # Comprehensive CMake guide
│ ├── make_guide.md # GNU Make guide
│ ├── hands_on_tutorial.md # Step-by-step tutorials
│ ├── exercises.md # Practice exercises
│ ├── cheat_sheet.md # Quick command reference
│ ├── reference_sheet.md # Detailed reference
│ ├── quick_reference.md # Quick lookup guide
│ ├── troubleshooting.md # Common issues and solutions
│ ├── learning_path.md # Structured learning path
│ ├── example_project.md # Example project walkthrough
│ ├── example_projects.md # Multiple project examples
│ ├── setup_check.sh # Environment verification script
│ ├── COMPLETE_TRAINING_SUMMARY.md # Training summary
│ ├── SUMMARY.md # Module summaries
│ └── README.md # CMake guide overview
│
├── concurrency/ # Concurrency & Parallel Programming
│ ├── intro/ # Introduction to concurrency
│ ├── threads/ # Thread fundamentals
│ ├── synchronization/ # Synchronization primitives
│ ├── async/ # Asynchronous programming
│ ├── advanced/ # Advanced concurrency topics
│ ├── concurrency-tutorial/ # Comprehensive tutorial
│ └── README.md # Concurrency overview
│
├── third_party/ # External dependencies
│ └── cutlass/ # CUTLASS library (submodule)
│
├── .git/ # Git repository metadata
├── .gitignore # Git ignore patterns
├── .gitmodules # Git submodule configuration
├── LEARNING_PATH.md # Comprehensive learning roadmap
├── LICENSE # Repository license
└── README.md # This file

```

---

## Detailed Module Descriptions

### Cuda - CUDA Programming Mastery

**Location**: `Cuda/`

**Purpose**: Complete CUDA programming curriculum from fundamentals to advanced optimization techniques.

#### Fundamentals (`Cuda/fundamentals/`)

Core CUDA concepts with hands-on implementations:

- **Thread Hierarchy** (`thread_heir/`): Grid-block-thread organization, indexing, warp behavior
- **Memory Hierarchy** (`mem_hier/`): Global, shared, constant, and register memory
- **Vector Addition** (`vec_add/`): Basic parallel operations
- **Matrix Multiplication** (`matmul/`): Tiled GEMM implementations
- **Reduction** (`reduction/`): Parallel reduction algorithms
- **Shared Memory** (`shared_mem/`): Shared memory usage patterns
- **Memory Coalescing** (`memory_coal/`): Optimized memory access
- **Atomics** (`atomics/`): Atomic operations for race conditions
- **CUDA Streams** (`cuda_streams/`): Asynchronous execution
- **Warp Operations** (`warp/`): Warp-level primitives

#### Memory Optimization (`Cuda/memory_optimization/`)

Advanced memory access patterns:

- **Coalescing** (`coalescing.md`, `coalescing_tutorial.cu`): Maximizing memory bandwidth
- **Banking** (`banking.md`, `banking_tutorial.cu`): Avoiding bank conflicts
- **Swizzling** (`swizzling.md`, `swizzling_tutorial.cu`): Systematic conflict elimination

#### Execution Optimization (`Cuda/execution_optimization/`)

Thread execution optimization:

- **Occupancy** (`occupancy.md`, `occupancy_tutorial.cu`): Resource balancing
- **Warp Primitives** (`warp_primitives.md`, `warp_primitives_tutorial.cu`): Intra-warp communication
- **Register Pressure** (`register_pressure.md`, `register_pressure_tutorial.cu`): Register optimization
- **Synchronization** (`synchronization.md`, `synchronization_tutorial.cu`): Synchronization primitives

#### Advanced Memory (`Cuda/advanced_memory/`)

Cutting-edge memory techniques:

- **Async Copy** (`async_copy.md`, `async_copy_tutorial.cu`): cp.async operations
- **Tensor Cores** (`tensor_cores.md`, `tensor_cores_tutorial.cu`): Tensor Core programming
- **Pipelining** (`pipelining.md`, `pipelining_tutorial.cu`): Software pipelining

#### Performance Analysis (`Cuda/performance_analysis/`)

Performance measurement and optimization:

- **Roofline Model** (`roofline.md`, `roofline_tutorial.cu`): Bottleneck identification
- **Profiling** (`profiling.md`, `profiling_tutorial.cu`): Nsight Compute usage

#### CuTe-Specific (`Cuda/cute_specific/`)

CuTe abstraction concepts:

- **Layout Algebra** (`layout_algebra.md`, `layout_algebra_tutorial.cu`): Algebraic memory layouts
- **Tiled Layouts** (`tiled_layouts.md`, `tiled_layouts_tutorial.cu`): Hierarchical tiling
- **Copy Atoms** (`copy_atoms.md`, `copy_atoms_tutorial.cu`): Hardware-agnostic data movement
- **MMA Atoms** (`mma_atoms.md`, `mma_atoms_tutorial.cu`): Tensor Core abstractions

#### Advanced Concepts (`Cuda/advanced_concepts/`)

Production-level optimization techniques organized by category:

- **Memory Optimization**: Global coalescing, shared tiling, bank conflict resolution
- **Execution Optimization**: Warp divergence, warp shuffle, thread coarsening, reductions, kernel fusion
- **Mathematical Kernels**: GEMM, softmax, attention, layer normalization
- **Tensor Cores**: WMMA API, Tensor Core GEMM, fused operations
- **Profiling Tools**: Occupancy calculation, Nsight profiling, PTX analysis

**Key Files**:

- `README.md`: Overview and learning path
- `COMPREHENSIVE_CURRICULUM_README.md`: Detailed curriculum structure
- `ADVANCED_CUDA_CURRICULUM.md`: Advanced topics guide
- `QUICK_REFERENCE_GUIDE.md`: Quick command reference

---

```

### CuTE - Modern GPU Programming with CUTLASS 3.x

**Location**: `CuTE/`

**Purpose**: Progressive 6-module curriculum mastering CuTe (CUDA Templates for Element-wise operations) for RTX 4060 (sm_89) architecture.

#### Module 01: Layout Algebra (`CuTE/Module_01_Layout_Algebra/`)

Foundation of CuTe programming:

- **Concepts**: Shapes, strides, hierarchical layouts
- **Key Learning**: Logical-to-physical memory mapping
- **Files**:
  - `README.md`: Layout algebra theory
  - `layout_study.cu`: Practical demonstrations
  - `BUILD.md`: Compilation instructions
  - `mock_cute.hpp`: Reference header
- **Skills**: Understanding `cute::Layout`, shape/stride composition, debugging with `cute::print()`

#### Module 02: CuTe Tensors (`CuTE/Module_02_CuTe_Tensors/`)

Tensor creation and manipulation:

- **Concepts**: Wrapping pointers, slicing, sub-tensors
- **Key Learning**: Tensor creation, layout composition, memory access patterns
- **Files**:
  - `README.md`: Tensor concepts
  - `tensor_basics.cu`: Tensor operations
  - `BUILD.md`: Build instructions
- **Skills**: Creating tensors, slicing operations, multidimensional views

#### Module 03: Tiled Copy (`CuTE/Module_03_Tiled_Copy/`)

Efficient memory movement:

- **Concepts**: Vectorized 128-bit loads, cp.async operations
- **Key Learning**: Coalesced memory access, async copy for sm_89
- **Files**:
  - `README.md`: Tiled copy theory
  - `tiled_copy_basics.cu`: Copy implementations
  - `BUILD.md`: Build instructions
- **Skills**: Vectorized loads, asynchronous memory operations, bandwidth optimization

#### Module 04: MMA Atoms (`CuTE/Module_04_MMA_Atoms/`)

Tensor Core programming:

- **Concepts**: Direct Tensor Core access using hardware atoms
- **Key Learning**: Matrix multiply-accumulate operations, WMMA instructions
- **Files**:
  - `README.md`: MMA atom concepts
  - `mma_atom_basics.cu`: MMA implementations
  - `BUILD.md`: Build instructions
- **Skills**: Tensor Core operations, MMA atom usage, hardware-specific optimizations

#### Module 05: Shared Memory & Swizzling (`CuTE/Module_05_Shared_Memory_Swizzling/`)

Bank conflict resolution:

- **Concepts**: Solving bank conflicts with algebra
- **Key Learning**: Shared memory optimization, swizzling patterns
- **Files**:
  - `README.md`: Swizzling theory
  - `shared_memory_layouts.cu`: Swizzling implementations
  - `BUILD.md`: Build instructions
- **Skills**: Bank conflict identification, XOR swizzling, layout transformations

#### Module 06: Collective Mainloops (`CuTE/Module_06_Collective_Mainloops/`)

Complete kernel orchestration:

- **Concepts**: Full producer-consumer pipeline
- **Key Learning**: Thread cooperation, collective operations
- **Files**:
  - `README.md`: Collective concepts
  - `producer_consumer_pipeline.cu`: Pipeline implementations
  - `BUILD.md`: Build instructions
- **Skills**: Producer-consumer patterns, collective operations, complete kernel design

**Additional Resources**:

- `SETUP.md`: Environment setup and CUTLASS installation
- `SUMMARY.md`: Module summaries and learning outcomes
- `COMPLETION_CERTIFICATE.md`: Certification information
- `cutlass/`: CUTLASS library submodule

**Target Architecture**: NVIDIA RTX 4060 (sm_89)
**Compilation**: `nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr`

---

### Cutlass3.x - Advanced Linear Algebra

**Location**: `Cutlass3.x/`

**Purpose**: Deep dive into NVIDIA's premier CUDA C++ linear algebra library with 6 comprehensive modules.

#### Module 1: Layouts and Tensors (`module1-Layouts and Tensors/`)

Foundation of composable abstractions:

- **Concepts**: CUTLASS layout system, tensor representations
- **Files**: `README.md`, `main.cu`, `CMakeLists.txt`, `build.sh`
- **Skills**: Layout creation, tensor manipulation, memory mapping

#### Module 2: Tiled Copy (`module2-Tiled Copy/`)

Vectorized global-to-shared memory movement:

- **Concepts**: Tiled memory operations, vectorization
- **Files**: `README.md`, `main.cu`, `build.sh`
- **Skills**: Efficient data movement, memory bandwidth optimization

#### Module 3: Tiled MMA (`module3-Tiled MMA/`)

Tensor Core operations via CuTe atoms:

- **Concepts**: Matrix multiply-accumulate, tiled operations
- **Files**: `README.md`, `main.cu`, `build.sh`
- **Skills**: Tensor Core programming, tiled GEMM

#### Module 4: Fused Bias-Add (`module4-Fused Bias-Add/`)

Bias-add and activation function implementations:

- **Concepts**: Operation fusion, epilogue operations
- **Files**: `README.md`, `main.cu`, `build.sh`
- **Skills**: Fused operations, activation functions

#### Module 5: Mainloop Pipelining (`module5-Mainloop Pipelining/`)

Temporal overlap and throughput optimization:

- **Concepts**: Software pipelining, latency hiding
- **Files**: `README.md`, `main.cu`, `build.sh`
- **Skills**: Pipeline design, throughput optimization

#### Module 6: Fused Epilogues (`module6-Fused Epilogues/`)

Eliminating VRAM roundtrips:

- **Concepts**: Functional fusion, memory traffic reduction
- **Files**: `README.md`, `main.cu`, `build.sh`
- **Skills**: Epilogue fusion, memory optimization

**Build System**:

- `CMakeLists.txt`: Top-level build configuration
- `build_all.sh`: Build all modules at once
- Individual `build.sh` scripts per module

---

### DSA - Data Structures & Algorithms

**Location**: `DSA/`

**Purpose**: 6-module foundational course essential for kernel development.

#### Module 00: Introduction & Big O (`00-Introduction-BigO/`)

- **Concepts**: Complexity analysis, Big O notation
- **Files**: `README.md`, `main.cpp`
- **Skills**: Algorithm analysis, time/space complexity

#### Module 01: Arrays & Strings (`01-Arrays-Strings/`)

- **Concepts**: Memory layout, access patterns
- **Files**: `README.md`, `main.cpp`
- **Skills**: Array manipulation, string operations

#### Module 02: Stacks & Queues (`02-Stacks-Queues/`)

- **Concepts**: Abstract data types, LIFO/FIFO
- **Files**: `README.md`, `main.cpp`
- **Skills**: Stack/queue implementations, applications

#### Module 03: Linked Lists (`03-Linked-Lists/`)

- **Concepts**: Dynamic memory management, pointer manipulation
- **Files**: `README.md`, `main.cpp`
- **Skills**: Linked list operations, memory management

#### Module 04: Searching & Sorting (`04-Searching-Sorting/`)

- **Concepts**: Search algorithms, sorting algorithms
- **Files**: `README.md`, `main.cpp`
- **Skills**: Binary search, merge sort, quicksort

#### Module 05: Trees (`05-Trees/`)

- **Concepts**: Hierarchical data structures, tree traversal
- **Files**: `README.md`, `main.cpp`
- **Skills**: Binary trees, BST operations, traversals

#### Module 06: Graphs (`06-Graphs/`)

- **Concepts**: Graph representation, graph algorithms
- **Files**: `README.md`, `main.cpp`
- **Skills**: BFS, DFS, shortest path algorithms

---

### GPU-DSA - GPU-Optimized Algorithms

**Location**: `GPU-DSA/`

**Purpose**: Specialized algorithms optimized for GPU architectures.

#### Core Algorithms

- **Parallel_Reduction/**: Tree-based and warp shuffle optimization
- **Parallel_Prefix_Sum/**: Blelloch and Kogge-Stone algorithms
- **Parallel_Histogramming/**: Privatization and atomic aggregation
- **Radix_Sort/**: LSB/MSB implementations
- **Bitonic_Sort/**: Parallel sorting algorithm

#### Matrix Operations

- **Tiled_Matrix_Multiplication/**: GEMM optimizations with tiling
- **Double_Buffering_Async_Copy/**: Overlapping computation and memory transfer

#### Memory Optimization

- **Shared_Memory_Swizzling/**: XOR layouts to avoid bank conflicts
- **Z_Curve_Morton_Order/**: Space-filling curves for locality

#### Deep Learning Kernels

- **Online_Softmax/**: Safe softmax algorithm with numerical stability
- **FlashAttention/**: Fused attention with recomputation
- **PagedAttention/**: Virtual memory mapping for KV-Cache
- **Fused_Layer_Normalization/**: Optimized GPU implementation

#### Sparse Operations

- **Compressed_Sparse_Formats/**: CSR and Blocked-Ellpack implementations

---
```

### Triton - High-Level GPU Programming

**Location**: `Triton/`

**Purpose**: Comprehensive 8-module curriculum for Triton programming, a domain-specific language for GPU programming with performance close to hand-written CUDA.

#### Module 1: Basics (`Module-01-Basics/`)

- **Concepts**: Introduction to Triton, basic tensor operations
- **Skills**: Kernel structure, decorators, simple operations
- **Files**: `README.md` with examples and exercises

#### Module 2: Memory (`Module-02-Memory/`)

- **Concepts**: Memory operations and data movement
- **Skills**: GPU memory hierarchy, coalesced access, boundary conditions
- **Files**: `README.md` with memory optimization examples

#### Module 3: Arithmetic (`Module-03-Arithmetic/`)

- **Concepts**: Basic arithmetic and element-wise operations
- **Skills**: Mathematical functions, broadcasting
- **Files**: `README.md` with arithmetic examples

#### Module 4: Blocks (`Module-04-Blocks/`)

- **Concepts**: Block operations and tiling concepts
- **Skills**: 2D blocks, tiling strategies, thread coordination
- **Files**: `README.md` with tiling examples

#### Module 5: Matrix Multiplication (`Module-05-Matrix-Multiplication/`)

- **Concepts**: Matrix multiplication fundamentals
- **Skills**: Tiled GEMM, optimization strategies
- **Files**: `README.md` with GEMM implementations

#### Module 6: Advanced Memory (`Module-06-Advanced-Memory/`)

- **Concepts**: Advanced memory layouts and optimizations
- **Skills**: Memory coalescing, bank conflict avoidance, prefetching
- **Files**: `README.md` with advanced memory techniques

#### Module 7: Reductions (`Module-07-Reductions/`)

- **Concepts**: Reduction operations
- **Skills**: Sum, max, min reductions, numerical stability
- **Files**: `README.md` with reduction algorithms

#### Module 8: Advanced Techniques (`Module-08-Advanced-Techniques/`)

- **Concepts**: Advanced techniques and best practices
- **Skills**: Profiling, optimization, numerical precision
- **Files**: `README.md` with advanced patterns

**Prerequisites**: Basic Python, understanding of tensors and linear algebra, PyTorch familiarity helpful

---

### NCCL - Multi-GPU Communication

**Location**: `NCCL/`

**Purpose**: 7-module learning path covering collective communications for multi-GPU and distributed computing.

#### Module 1: Introduction (`Module-01/`)

- **Concepts**: NCCL fundamentals, concepts, and setup
- **Skills**: Installation, basic architecture, communicators
- **Files**: `README.md` with setup instructions

#### Module 2: Basic Collective Operations (`Module-02/`)

- **Concepts**: AllReduce, Broadcast
- **Skills**: Basic collective patterns, synchronization
- **Files**: `README.md` with basic examples

#### Module 3: Advanced Collective Operations (`Module-03/`)

- **Concepts**: Reduce, AllGather, Scatter
- **Skills**: Advanced communication patterns
- **Files**: `README.md` with advanced examples

#### Module 4: Multi-GPU Programming (`Module-04/`)

- **Concepts**: Multi-GPU programming with NCCL
- **Skills**: GPU coordination, data distribution
- **Files**: `README.md` with multi-GPU patterns

#### Module 5: Multi-Node Communication (`Module-05/`)

- **Concepts**: Distributed computing across nodes
- **Skills**: Network communication, topology awareness
- **Files**: `README.md` with distributed examples

#### Module 6: Performance Optimization (`Module-06/`)

- **Concepts**: NCCL performance tuning
- **Skills**: Bandwidth optimization, latency reduction
- **Files**: `README.md` with optimization techniques

#### Module 7: Framework Integration (`Module-07/`)

- **Concepts**: Integration with deep learning frameworks
- **Skills**: PyTorch/TensorFlow integration, distributed training
- **Files**: `README.md` with integration examples

---

### Profiling - Performance Analysis

**Location**: `Profiling/`

**Purpose**: 8-module mastery course for identifying and optimizing GPU kernel bottlenecks.

#### Module 1: Introduction (`Module_1_Introduction_to_GPU_Computing/`)

- **Concepts**: GPU computing and profiling concepts
- **Skills**: GPU architecture basics, performance metrics

#### Module 2: Setting Up Tools (`Module_2_Setting_Up_Profiling_Tools/`)

- **Concepts**: Profiling tools and environment setup
- **Skills**: Nsight Compute, Nsight Systems, nvprof

#### Module 3: Basic Profiling (`Module_3_Basic_Profiling_Techniques/`)

- **Concepts**: Basic profiling techniques
- **Skills**: Kernel timing, basic metrics analysis

#### Module 4: Identifying Bottlenecks (`Module_4_Identifying_Common_Bottlenecks/`)

- **Concepts**: Common performance bottlenecks
- **Skills**: Memory-bound vs compute-bound identification

#### Module 5: Memory Optimization (`Module_5_Memory_Optimization_Techniques/`)

- **Concepts**: Memory optimization techniques
- **Skills**: Bandwidth optimization, coalescing, caching

#### Module 6: Computational Optimization (`Module_6_Computational_Optimization_Strategies/`)

- **Concepts**: Computational optimization strategies
- **Skills**: Occupancy, instruction throughput, warp efficiency

#### Module 7: Advanced Profiling (`Module_7_Advanced_Profiling_Analysis/`)

- **Concepts**: Advanced profiling and analysis
- **Skills**: Roofline analysis, detailed metrics interpretation

#### Module 8: Real-world Case Studies (`Module_8_Real_world_Case_Studies/`)

- **Concepts**: Real-world optimization scenarios
- **Skills**: End-to-end optimization, production considerations

---

### PTX - Assembly-Level Programming

**Location**: `PTX/`

**Purpose**: Low-level GPU programming with NVIDIA's Parallel Thread Execution (PTX) assembly.

#### PTX-Basics

- **Concepts**: PTX syntax, structure, basic concepts
- **Skills**: PTX instructions, data types, basic programs

#### PTX-Memory-Management

- **Concepts**: Memory spaces, access patterns, optimization
- **Skills**: Different memory spaces, coalescing, shared memory

#### PTX-Debugging-Profiling

- **Concepts**: Debugging tools, profiling methods
- **Skills**: cuobjdump, cuda-gdb, performance analysis

#### PTX-Advanced-Optimizations

- **Concepts**: Advanced optimization techniques
- **Skills**: Instruction-level parallelism, warp-level primitives

#### PTX-Custom-Kernels

- **Concepts**: Complete custom kernel development
- **Skills**: End-to-end kernel design, integration

**Tools**: nvcc, cuobjdump, nvdisasm, Nsight Compute, cuda-gdb
**Setup**: `getting_started.sh` script for environment configuration

---

### Temp-Meta - Template Metaprogramming

**Location**: `Temp-Meta/`

**Purpose**: 10-module intensive program from C++ fundamentals to advanced CUTLASS optimization.

#### Module 1: Foundations (`module_1_foundations/`)

- **Concepts**: Modern C++ features
- **Skills**: RAII, smart pointers, move semantics, lambdas

#### Module 2: Fundamentals (`module_2_fundamentals/`)

- **Concepts**: Template fundamentals
- **Skills**: Class/function templates, specialization, variadic templates

#### Module 3: Basics (`module_3_basics/`)

- **Concepts**: Template metaprogramming basics
- **Skills**: Compile-time computation, type traits, SFINAE

#### Module 4: Advanced (`module_4_advanced/`)

- **Concepts**: Advanced template techniques
- **Skills**: Expression templates, type lists, policy-based design

#### Module 5: CUDA Fundamentals (`module_5_cuda_fundamentals/`)

- **Concepts**: CUDA and GPU programming fundamentals
- **Skills**: GPU architecture, memory hierarchies, thread organization

#### Module 6: CUTLASS Architecture (`module_6_cutlass_architecture/`)

- **Concepts**: CUTLASS 3.x architecture overview
- **Skills**: GEMM fundamentals, tile-based approach, components

#### Module 7: CUTLASS Patterns (`module_7_cutlass_patterns/`)

- **Concepts**: CUTLASS template patterns and idioms
- **Skills**: Template conventions, dispatch patterns, specialization

#### Module 8: Advanced Customization (`module_8_advanced_customization/`)

- **Concepts**: Advanced CUTLASS customization
- **Skills**: Custom epilogues, non-standard types, performance tuning

#### Module 9: Real-world Applications (`module_9_real_world_applications/`)

- **Concepts**: Real-world applications and case studies
- **Skills**: Framework integration, quantized operations, mixed precision

#### Module 10: Performance Optimization (`module_10_performance_optimization/`)

- **Concepts**: Performance optimization and profiling
- **Skills**: Profiling tools, bandwidth optimization, occupancy tuning

**Duration**: 8-10 months (20-25 hours/week)
**Resources**: `quick_reference.md`, `SUMMARY.md`, `README.md`

---

### cmakeguide - Build Systems

**Location**: `cmakeguide/`

**Purpose**: Complete build system expertise for GPU development.

#### Core Documentation

- **cmake_guide.md**: Comprehensive CMake guide from basics to advanced
- **make_guide.md**: GNU Make for efficient project orchestration
- **hands_on_tutorial.md**: Step-by-step practical tutorials
- **exercises.md**: Progressive hands-on challenges
- **cheat_sheet.md**: Quick command reference
- **reference_sheet.md**: Detailed syntax reference
- **quick_reference.md**: Quick lookup guide
- **troubleshooting.md**: Common issues and solutions
- **learning_path.md**: Structured learning progression
- **example_project.md**: Complete project walkthrough
- **example_projects.md**: Multiple project examples

#### Training Projects (`training_projects/`)

- **modules/**: Modular project examples
- **intermediate/**: Intermediate-level projects
- **README.md**: Training project guide

#### Utilities

- **setup_check.sh**: Environment verification script
- **COMPLETE_TRAINING_SUMMARY.md**: Comprehensive training summary
- **SUMMARY.md**: Module summaries

**Skills Covered**: CMake basics to advanced, Make fundamentals, cross-platform builds, dependency management, testing integration

---

### concurrency - Parallel Programming

**Location**: `concurrency/`

**Purpose**: Advanced concepts in concurrent and parallel programming for GPU computing.

#### Module 1: Introduction (`intro/`)

- **Concepts**: Concurrency vs parallelism, race conditions, critical sections
- **Skills**: Basic terminology, performance measurement

#### Module 2: Threads (`threads/`)

- **Concepts**: Thread lifecycle, creation, joining, detaching
- **Skills**: Thread pools, producer-consumer patterns

#### Module 3: Synchronization (`synchronization/`)

- **Concepts**: Mutexes, semaphores, condition variables, atomics
- **Skills**: Deadlock prevention, reader-writer problem

#### Module 4: Async (`async/`)

- **Concepts**: Futures/promises, async/await, event loops
- **Skills**: Asynchronous pipelines, async patterns

#### Module 5: Advanced (`advanced/`)

- **Concepts**: Lock-free programming, memory models, transactional memory
- **Skills**: Lock-free data structures, hazard pointers

#### Tutorial (`concurrency-tutorial/`)

- Comprehensive hands-on tutorial with examples in C++ and Python

**Time Estimate**: 16-24 hours total
**Languages**: C++, Python, conceptual explanations

---

## Prerequisites

Before starting this learning journey, ensure you have:

### Required Knowledge

- **Solid understanding of CUDA programming fundamentals**
- **Proficiency in C++** (especially templates, metaprogramming, and modern C++ features)
- **Basic knowledge of GPU memory hierarchy** (global, shared, registers)
- **Understanding of Tensor Core concepts** and matrix multiplication algorithms (GEMM)
- **Experience with performance profiling tools** (Nsight Compute, nvprof)

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with Compute Capability 6.0+ (RTX 4060 or equivalent recommended)
- **CUDA Toolkit**: Version 12.x or later
- **Compiler**: GCC 7+ or Clang 5+ with C++17 support
- **CMake**: Version 3.12 or higher
- **Memory**: 16GB+ system RAM, 8GB+ GPU memory
- **Storage**: 50GB+ free disk space

### Software Dependencies

```bash
# Verify NVIDIA GPU and driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify compiler
gcc --version
g++ --version

# Check CMake
cmake --version

# Python (for Triton modules)
python3 --version
pip3 --version
```

---

## Setup Instructions

### 1. Clone Repository with Submodules

```bash
# Clone the repository with all submodules
git clone --recursive https://github.com/[your-username]/AI-Kernel-learning.git
cd AI-Kernel-learning

# If already cloned without --recursive, initialize submodules
git submodule update --init --recursive
```

### 2. Install CUDA Toolkit

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-0

# Verify installation
nvcc --version
nvidia-smi
```

### 3. Install Build Tools

```bash
# Essential build tools
sudo apt-get install build-essential cmake git

# Additional tools
sudo apt-get install python3-dev python3-pip
pip3 install numpy torch triton
```

### 4. Build CuTE Modules

```bash
cd CuTE
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"  # For RTX 4060
make -j$(nproc)
```

### 5. Build CUTLASS 3.x Modules

```bash
cd Cutlass3.x
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"
make -j$(nproc)

# Or build all modules at once
cd Cutlass3.x
./build_all.sh
```

### 6. Verify Setup

```bash
# Test CUDA compilation
cd Cuda/fundamentals/vec_add
nvcc vector_add.cu -o vector_add
./vector_add

# Test CMake
cd cmakeguide
./setup_check.sh
```

### 7. Individual Module Compilation

Each module contains standalone `.cu` files that can be compiled directly:

```bash
# Example: Compile a CuTE module
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I/path/to/cutlass/include \
     module_file.cu -o module_output

# Example: Compile a CUDA tutorial
nvcc -arch=sm_89 tutorial.cu -o tutorial
./tutorial
```

---

## Recommended Learning Path

This comprehensive learning path is designed to take you from beginner to expert level in approximately 46 weeks (11-12 months) with 20-25 hours per week of dedicated study.

### Phase 1: Foundational Skills (Weeks 1-8)

#### Week 1-2: Data Structures & Algorithms

- **Focus**: DSA fundamentals essential for kernel development
- **Modules**: `DSA/00-Introduction-BigO/` through `DSA/03-Linked-Lists/`
- **Activities**:
  - Complete Big O notation exercises
  - Implement array/string algorithms
  - Practice stack/queue and linked list operations
- **Deliverable**: Complete all exercises in main.cpp files

#### Week 3-4: Advanced DSA & Build Systems Introduction

- **Focus**: Complete DSA and introduce CMake
- **Modules**: `DSA/04-Searching-Sorting/` through `DSA/06-Graphs/`, `cmakeguide/`
- **Activities**:
  - Complete searching/sorting algorithms
  - Implement tree and graph algorithms
  - Read CMake fundamentals
- **Deliverable**: Implement merge sort, binary tree, and graph traversal exercises

#### Week 5-6: CMake Mastery

- **Focus**: Master build systems for GPU development
- **Modules**: `cmakeguide/hands_on_tutorial.md`, `cmakeguide/exercises.md`
- **Activities**:
  - Complete hands-on CMake tutorial
  - Practice with CMake exercises
  - Build example projects
- **Deliverable**: Successfully build multi-module CMake projects

#### Week 7-8: C++ Template Fundamentals

- **Focus**: Begin template metaprogramming
- **Modules**: `Temp-Meta/module_1_foundations/`, `Temp-Meta/module_2_fundamentals/`
- **Activities**:
  - Study modern C++ features
  - Learn template basics
  - Complete foundation exercises
- **Deliverable**: Implement basic template examples

---

### Phase 2: GPU Programming Fundamentals (Weeks 9-16)

#### Week 9-10: CUDA Fundamentals

- **Focus**: Core CUDA concepts
- **Modules**: `Cuda/fundamentals/thread_heir/`, `Cuda/fundamentals/mem_hier/`
- **Activities**:
  - Study thread hierarchy
  - Understand memory hierarchy
  - Implement vector addition and matrix multiplication
- **Deliverable**: Working CUDA kernels with proper indexing

#### Week 11-12: CUDA Memory Optimization

- **Focus**: Optimize memory access patterns
- **Modules**: `Cuda/memory_optimization/`, `Cuda/fundamentals/shared_mem/`
- **Activities**:
  - Study memory coalescing
  - Learn shared memory banking
  - Implement optimized kernels
- **Deliverable**: Optimized memory access implementations

#### Week 13-14: CuTE Layout Algebra & Tensors

- **Focus**: Master CuTe fundamentals
- **Modules**: `CuTE/Module_01_Layout_Algebra/`, `CuTE/Module_02_CuTe_Tensors/`
- **Activities**:
  - Study layout algebra
  - Implement tensor operations
  - Debug with cute::print()
- **Deliverable**: Working CuTe layout and tensor examples

#### Week 15-16: CuTE Tiled Copy & CUTLASS Introduction

- **Focus**: Efficient memory movement
- **Modules**: `CuTE/Module_03_Tiled_Copy/`, `Cutlass3.x/module1-Layouts and Tensors/`
- **Activities**:
  - Study tiled copy mechanisms
  - Implement vectorized loads
  - Compare CuTe and CUTLASS approaches
- **Deliverable**: Optimized tiled copy implementations

---

### Phase 3: Advanced GPU Programming (Weeks 17-24)

#### Week 17-18: Tensor Cores & MMA Atoms

- **Focus**: Master Tensor Core programming
- **Modules**: `CuTE/Module_04_MMA_Atoms/`, `Cutlass3.x/module3-Tiled MMA/`
- **Activities**:
  - Study MMA atom operations
  - Implement Tensor Core kernels
  - Compare different MMA approaches
- **Deliverable**: Working Tensor Core implementations

#### Week 19-20: Shared Memory Optimization

- **Focus**: Bank conflict resolution and swizzling
- **Modules**: `CuTE/Module_05_Shared_Memory_Swizzling/`, `Cuda/memory_optimization/swizzling.md`
- **Activities**:
  - Study swizzling patterns
  - Implement bank conflict solutions
  - Measure performance improvements
- **Deliverable**: Optimized shared memory layouts

#### Week 21-22: Collective Operations & Pipelining

- **Focus**: Complete kernel orchestration
- **Modules**: `CuTE/Module_06_Collective_Mainloops/`, `Cutlass3.x/module5-Mainloop Pipelining/`
- **Activities**:
  - Study producer-consumer patterns
  - Implement pipelined kernels
  - Measure throughput improvements
- **Deliverable**: Complete pipelined kernel implementations

#### Week 23-24: Advanced CUTLASS & Fusion

- **Focus**: Fused operations and epilogues
- **Modules**: `Cutlass3.x/module4-Fused Bias-Add/`, `Cutlass3.x/module6-Fused Epilogues/`
- **Activities**:
  - Study operation fusion
  - Implement fused epilogues
  - Eliminate VRAM roundtrips
- **Deliverable**: Fused kernel implementations

---

### Phase 4: Production-Level Skills (Weeks 25-32)

#### Week 25-26: Advanced Template Metaprogramming

- **Focus**: Master advanced C++ templates
- **Modules**: `Temp-Meta/module_3_basics/`, `Temp-Meta/module_4_advanced/`
- **Activities**:
  - Study compile-time computation
  - Learn expression templates
  - Implement advanced template patterns
- **Deliverable**: Complex template metaprogramming examples

#### Week 27-28: CUTLASS Architecture Deep Dive

- **Focus**: Understand CUTLASS internals
- **Modules**: `Temp-Meta/module_6_cutlass_architecture/`, `Temp-Meta/module_7_cutlass_patterns/`
- **Activities**:
  - Study CUTLASS architecture
  - Learn template patterns
  - Implement custom operations
- **Deliverable**: Custom CUTLASS components

#### Week 29-30: GPU-DSA Algorithms

- **Focus**: GPU-optimized algorithms
- **Modules**: `GPU-DSA/Parallel_Reduction/`, `GPU-DSA/Tiled_Matrix_Multiplication/`, `GPU-DSA/FlashAttention/`
- **Activities**:
  - Implement parallel reduction
  - Study FlashAttention
  - Optimize matrix multiplication
- **Deliverable**: Optimized GPU algorithm implementations

#### Week 31-32: Performance Profiling

- **Focus**: Master profiling and optimization
- **Modules**: `Profiling/Module_1_Introduction_to_GPU_Computing/` through `Profiling/Module_4_Identifying_Common_Bottlenecks/`
- **Activities**:
  - Learn Nsight Compute
  - Profile existing kernels
  - Identify bottlenecks
- **Deliverable**: Profiling reports with optimization recommendations

---

### Phase 5: Triton Programming (Weeks 33-38)

#### Week 33-34: Triton Fundamentals

- **Focus**: High-level GPU programming with Triton
- **Modules**: `Triton/Module-01-Basics/`, `Triton/Module-02-Memory/`
- **Activities**:
  - Learn Triton syntax
  - Implement basic operations
  - Study memory patterns
- **Deliverable**: Working Triton kernels

#### Week 35-36: Triton Intermediate

- **Focus**: Advanced Triton operations
- **Modules**: `Triton/Module-03-Arithmetic/`, `Triton/Module-04-Blocks/`, `Triton/Module-05-Matrix-Multiplication/`
- **Activities**:
  - Implement arithmetic operations
  - Study tiling in Triton
  - Implement matrix multiplication
- **Deliverable**: Optimized Triton GEMM

#### Week 37-38: Triton Advanced

- **Focus**: Advanced techniques and optimization
- **Modules**: `Triton/Module-06-Advanced-Memory/`, `Triton/Module-07-Reductions/`, `Triton/Module-08-Advanced-Techniques/`
- **Activities**:
  - Study advanced memory layouts
  - Implement reductions
  - Apply best practices
- **Deliverable**: Production-ready Triton kernels

---

### Phase 6: Multi-GPU & Distributed Computing (Weeks 39-42)

#### Week 39-40: NCCL Fundamentals

- **Focus**: Multi-GPU communication
- **Modules**: `NCCL/Module-01/`, `NCCL/Module-02/`, `NCCL/Module-03/`
- **Activities**:
  - Setup NCCL environment
  - Implement basic collectives
  - Study advanced collectives
- **Deliverable**: Working multi-GPU applications

#### Week 41-42: NCCL Advanced & Optimization

- **Focus**: Distributed computing and optimization
- **Modules**: `NCCL/Module-04/`, `NCCL/Module-05/`, `NCCL/Module-06/`, `NCCL/Module-07/`
- **Activities**:
  - Implement multi-node communication
  - Optimize NCCL performance
  - Integrate with frameworks
- **Deliverable**: Distributed training implementation

---

### Phase 7: Advanced Profiling & Optimization (Weeks 43-46)

#### Week 43-44: Advanced Profiling

- **Focus**: Deep performance analysis
- **Modules**: `Profiling/Module_5_Memory_Optimization_Techniques/`, `Profiling/Module_6_Computational_Optimization_Strategies/`
- **Activities**:
  - Study memory optimization
  - Learn computational optimization
  - Apply roofline analysis
- **Deliverable**: Comprehensive optimization reports

#### Week 45-46: Real-world Case Studies & Capstone

- **Focus**: Apply all knowledge to production scenarios
- **Modules**: `Profiling/Module_7_Advanced_Profiling_Analysis/`, `Profiling/Module_8_Real_world_Case_Studies/`
- **Activities**:
  - Study real-world optimizations
  - Complete capstone project
  - Integrate multiple concepts
- **Deliverable**: Complete production-ready kernel with documentation

---

### Optional Advanced Tracks

#### PTX Assembly Programming (4-6 weeks)

- **Modules**: `PTX/PTX-Basics/` through `PTX/PTX-Custom-Kernels/`
- **Focus**: Low-level GPU programming and manual optimization
- **When**: After completing Phase 3

#### Concurrency & Parallel Programming (3-4 weeks)

- **Modules**: `concurrency/intro/` through `concurrency/advanced/`
- **Focus**: Advanced concurrent programming patterns
- **When**: Parallel to Phase 2 or 3

#### Advanced Template Metaprogramming (6-8 weeks)

- **Modules**: `Temp-Meta/module_8_advanced_customization/` through `Temp-Meta/module_10_performance_optimization/`
- **Focus**: Expert-level CUTLASS customization
- **When**: After completing Phase 4

---

## Quick Start Guide

### For Complete Beginners

1. **Start Here**: `DSA/00-Introduction-BigO/`
2. **Then**: `cmakeguide/hands_on_tutorial.md`
3. **Next**: `Cuda/fundamentals/README.md`
4. **Follow**: The complete learning path above

### For CUDA Programmers

1. **Start Here**: `CuTE/Module_01_Layout_Algebra/`
2. **Then**: `Cutlass3.x/module1-Layouts and Tensors/`
3. **Next**: `Temp-Meta/module_5_cuda_fundamentals/`
4. **Follow**: Phase 2-7 of the learning path

### For Template Metaprogramming Experts

1. **Start Here**: `Temp-Meta/module_6_cutlass_architecture/`
2. **Then**: `CuTE/Module_01_Layout_Algebra/`
3. **Next**: `Cutlass3.x/module1-Layouts and Tensors/`
4. **Follow**: Phase 3-7 of the learning path

### For Quick Experimentation

```bash
# Try a simple CUDA example
cd Cuda/fundamentals/vec_add
nvcc vector_add.cu -o vector_add -arch=sm_89
./vector_add

# Try a CuTE example
cd CuTE/Module_01_Layout_Algebra
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"
make
./layout_study

# Try a Triton example
cd Triton/Module-01-Basics
python3 basic_example.py
```

---

## Assessment & Certification

### Weekly Assessments

Each week includes practical exercises that reinforce the concepts covered. Complete all exercises before moving to the next week.

### Monthly Reviews

At the end of each month, conduct a self-assessment:

- Can you explain the key concepts from the past month?
- Have you completed all practical exercises?
- Are you comfortable with the code examples?
- Can you modify examples to solve new problems?

### Mid-Program Checkpoint (Week 24)

- Successfully implement a complete CuTe-based kernel
- Demonstrate understanding of layout algebra, tensors, and MMA operations
- Show proficiency with profiling and optimization tools
- Complete a mid-program assessment project

### Final Assessment (Week 46)

- Complete the capstone project implementing a high-performance kernel
- Demonstrate mastery of all concepts covered
- Prepare a presentation explaining implementation choices and optimizations
- Achieve measurable performance improvements over baseline implementations

### Completion Certificates

Certificates are available in:

- `CuTE/COMPLETION_CERTIFICATE.md` - CuTe module completion
- Individual module directories for CUTLASS 3.x
- Individual module directories for Triton, NCCL, and Profiling tracks

### Capstone Project Requirements

Your final project should demonstrate:

1. **Correctness**: Numerically accurate results
2. **Performance**: Significant speedup over baseline
3. **Code Quality**: Clean, maintainable, well-documented code
4. **Optimization**: Application of multiple optimization techniques
5. **Analysis**: Comprehensive performance analysis and profiling

---

## Contributing

This repository is designed to be a collaborative learning resource. Contributions are welcome in the form of:

### How to Contribute

- **Bug Fixes**: Correct errors in code or documentation
- **Additional Examples**: Provide new examples and exercises
- **Performance Improvements**: Optimize existing implementations
- **Documentation Enhancements**: Improve clarity and completeness
- **New Modules**: Add new learning modules or topics

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Test your changes thoroughly
5. Submit a pull request with detailed description

### Areas for Contribution

- Additional GPU-DSA algorithm implementations
- More Triton examples and tutorials
- Extended profiling case studies
- Cross-platform build improvements
- Translation of documentation to other languages

---

## Career Impact

Upon completing this learning path, you will be prepared for roles such as:

### Job Titles

- **Senior GPU Kernel Engineer** - Design and implement high-performance GPU kernels
- **High-Performance Computing Specialist** - Optimize scientific computing applications
- **AI Infrastructure Engineer** - Build and optimize AI training infrastructure
- **Deep Learning Compiler Developer** - Develop compiler optimizations for ML frameworks
- **GPU Performance Optimization Expert** - Analyze and optimize GPU workloads
- **CUDA Library Developer** - Contribute to GPU-accelerated libraries
- **ML Systems Engineer** - Design efficient ML training and inference systems

### Skills Acquired

- Expert-level CUDA programming and optimization
- Mastery of CUTLASS 3.x and CuTe abstractions
- Advanced C++ template metaprogramming
- Multi-GPU and distributed computing with NCCL
- High-level GPU programming with Triton
- Performance profiling and bottleneck analysis
- PTX assembly-level optimization
- Build system expertise for GPU projects

### Industry Applications

- **AI/ML**: Training and inference optimization for large language models
- **Scientific Computing**: HPC simulations and numerical methods
- **Computer Graphics**: Real-time rendering and ray tracing
- **Data Analytics**: Large-scale data processing and analytics
- **Autonomous Systems**: Real-time perception and decision making
- **Financial Computing**: High-frequency trading and risk analysis

### Salary Expectations

Professionals with these skills typically command:

- **Entry Level** (0-2 years): $120,000 - $160,000
- **Mid Level** (3-5 years): $160,000 - $220,000
- **Senior Level** (6+ years): $220,000 - $350,000+
- **Principal/Staff** (10+ years): $350,000 - $500,000+

_Salaries vary by location, company, and specific role_

---

## Key Resources

### Internal Documentation

- `LEARNING_PATH.md` - Comprehensive roadmap for the entire curriculum
- `cmakeguide/README.md` - CMake and build systems guide
- `CuTE/README.md` - CuTe programming guide
- `CuTE/SUMMARY.md` - CuTe module summaries
- `Cutlass3.x/README.md` - CUTLASS 3.x guide
- `Triton/README.md` - Triton programming learning path
- `NCCL/Module-01/README.md` - NCCL collective communications guide
- `Profiling/README.md` - GPU profiling and optimization resources
- `PTX/README.md` - PTX assembly programming guide
- `Temp-Meta/README.md` - Template metaprogramming curriculum

### External References

- [NVIDIA CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Nsight Compute Profiler](https://developer.nvidia.com/nsight-compute)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Triton Documentation](https://triton-lang.org/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/)

### Community Resources

- NVIDIA Developer Forums
- CUDA Programming Discord/Slack channels
- GPU Computing Stack Overflow
- Research papers on GPU optimization

---

## Support

For questions, clarifications, or discussions about the material:

- Open an issue in the repository
- Refer to the documentation in each module directory
- Consult the external resources listed above
- Join GPU computing communities for peer support

---

## License

See the `LICENSE` file for licensing information.

---

## Acknowledgments

This curriculum builds upon the foundational work of:

- NVIDIA's CUDA and CUTLASS documentation
- Open-source GPU computing community
- Academic research in GPU optimization
- Industry best practices in kernel development

---
