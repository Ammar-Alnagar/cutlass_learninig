# AI Kernel Engineer Learning Repository

## Mastering High-Performance GPU Computing & Deep Learning Kernels

This comprehensive learning repository is designed to transform software engineers into expert AI kernel developers, focusing on the cutting-edge technologies required for developing high-performance GPU kernels using NVIDIA's latest architectures and libraries.

## Learning Tracks

### 1. CuTE (CUTLASS 3.x) - Modern GPU Programming
Progressive 6-module curriculum mastering CuTe (CUDA Templates for Element-wise operations) for RTX 4060 (sm_89) architecture:

- **Module 01**: Layout Algebra - Shapes, Strides, and Hierarchical Layouts
- **Module 02**: CuTe Tensors - Wrapping raw pointers with layouts for multidimensional views
- **Module 03**: Tiled Copy - Vectorized loads and cp.async operations for optimal bandwidth
- **Module 04**: MMA Atoms - Direct access to Tensor Core operations
- **Module 05**: Shared Memory & Swizzling - Bank conflict resolution and optimization
- **Module 06**: Collective Mainloops - Producer-consumer pipelines with collective operations

### 2. CUTLASS 3.x - Advanced Linear Algebra Libraries
Deep dive into NVIDIA's premier CUDA C++ linear algebra library with 6 comprehensive modules:

- **Module 1**: Layouts and Tensors - Foundation of composable abstractions
- **Module 2**: Tiled Copy - Vectorized global-to-shared memory movement
- **Module 3**: Tiled MMA - Tensor Core operations via CuTe atoms
- **Module 4**: Fused Epilogues - Bias-add and activation function implementations
- **Module 5**: Mainloop Pipelining - Temporal overlap and throughput optimization
- **Module 6**: Advanced Fused Operations - Eliminating VRAM roundtrips

### 3. CMake & Build Systems Mastery
Complete build system expertise for GPU development:

- **CMake Guide**: Comprehensive coverage from basics to advanced topics
- **Make Guide**: GNU Make for efficient project orchestration
- **Hands-on Tutorials**: Practical exercises and step-by-step tutorials
- **Reference Sheets**: Quick reference for commands, variables, and best practices
- **Real-world Examples**: Complete project demonstrating CMake + Make integration

### 4. C++ Template Metaprogramming for GPU Computing
10-module intensive program from fundamentals to advanced optimization:

- **Modules 1-4**: C++ foundations, template fundamentals, metaprogramming basics, and advanced techniques
- **Module 5**: CUDA and GPU programming fundamentals
- **Module 6**: CUTLASS architecture and design principles
- **Module 7**: CUTLASS template patterns and idioms
- **Module 8**: Advanced customization techniques
- **Module 9**: Real-world applications and case studies
- **Module 10**: Performance optimization and profiling

### 5. Data Structures & Algorithms for Systems Programming
6-module foundational course essential for kernel development:

- **Module 00**: Introduction to DSA & Big O notation
- **Module 01**: Arrays & Strings - Memory layout and access patterns
- **Module 02**: Stacks & Queues - Abstract data types
- **Module 03**: Linked Lists - Dynamic memory management
- **Module 04**: Searching & Sorting algorithms
- **Module 05**: Trees - Hierarchical data structures
- **Module 06**: Graphs - Complex relationship modeling

## Target Hardware & Architecture

- **Primary Target**: NVIDIA RTX 4060 (Compute Capability 8.9 / Ada Lovelace)
- **Tensor Core Support**: FP16, BF16, INT8, and FP8 operations
- **Memory Hierarchy**: Global, shared, and register memory optimization
- **Warp-level Primitives**: Cooperative thread operations
- **Asynchronous Operations**: cp.async for overlapping computation and memory transfer

## Learning Philosophy

This repository emphasizes **composable abstractions** over manual indexing. Instead of traditional nested loops, we focus on:

- **Mathematical Representation**: Thinking in terms of linear algebra and tensor operations
- **Functional Composition**: Building complex operations from simple, reusable components
- **Hardware Mapping**: Understanding how abstractions map to physical GPU resources
- **Performance Analysis**: Measuring and optimizing each component individually

### Key Principles

1. **Composability**: Each component should be reusable and combinable with others
2. **Abstraction**: Hide complexity behind clean, mathematically-grounded interfaces
3. **Performance**: Every abstraction should maintain or improve performance
4. **Correctness**: Mathematical precision and numerical accuracy are paramount

## Prerequisites

Before starting this learning journey, ensure you have:

- **Solid understanding of CUDA programming fundamentals**
- **Proficiency in C++ (especially templates, metaprogramming, and modern C++ features)**
- **Basic knowledge of GPU memory hierarchy (global, shared, registers)**
- **Understanding of Tensor Core concepts and matrix multiplication algorithms (GEMM)**
- **Experience with performance profiling tools (Nsight Compute, nvprof)**
- **Linux development environment with NVIDIA GPU**

## Setup Instructions

### System Requirements
```bash
# Verify NVIDIA GPU and driver
nvidia-smi

# Install CUDA Toolkit 12.x or later
# Download from: https://developer.nvidia.com/cuda-downloads
```

### Repository Setup
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/[your-username]/cutlass_learning.git
cd cutlass_learning

# Initialize submodules (contains CUTLASS library)
git submodule update --init --recursive
```

### Build Configuration

#### CuTE Learning Modules
```bash
cd CuTE
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"  # For RTX 4060
make -j$(nproc)
```

#### CUTLASS 3.x Modules
```bash
cd Cutlass3.x
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"
make -j$(nproc)
```

#### CMake Learning Examples
```bash
cd Cmake_tr/complete_example
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Individual Module Compilation
Each module contains standalone `.cu` files that can be compiled directly:
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I/path/to/cutlass/include \
     module_file.cu -o module_output
```

## Recommended Learning Path

### Track 1: Foundational Skills (Weeks 1-8)
1. **DSA Fundamentals** (2 weeks) - Complete all 6 modules
2. **C++ Template Metaprogramming** (4 weeks) - Modules 1-4
3. **Build Systems** (2 weeks) - Complete CMake and Make guides

### Track 2: GPU Programming (Weeks 9-16)
1. **Template Metaprogramming** (2 weeks) - Modules 5-6 (CUDA fundamentals)
2. **CuTE Basics** (4 weeks) - Complete all 6 CuTE modules
3. **CUTLASS 3.x Introduction** (2 weeks) - Modules 1-2

### Track 3: Advanced Optimization (Weeks 17-24)
1. **CUTLASS 3.x Advanced** (4 weeks) - Modules 3-6
2. **Template Metaprogramming** (2 weeks) - Modules 7-8
3. **Performance Optimization** (2 weeks) - Modules 9-10

## Key Resources

### Internal Documentation
- `COMPLETE_LEARNING_GUIDE.md` - Comprehensive roadmap for the entire curriculum
- `HANDS_ON_IMPLEMENTATION_GUIDE.md` - Practical implementation strategies
- `LEARNING_CHECKLIST.md` - Progress tracking and milestones
- `LEARNING_PATH_SUMMARY.md` - Executive summary of the learning journey

### External References
- [NVIDIA CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Nsight Compute Profiler](https://developer.nvidia.com/nsight-compute)
- [GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/ampere-architecture-whitepaper/)

## Assessment & Certification

Each learning track includes:
- **Knowledge Checks**: Conceptual understanding assessments
- **Implementation Projects**: Hands-on coding challenges
- **Performance Benchmarks**: Optimization and measurement exercises
- **Capstone Projects**: Integration of multiple concepts in real-world scenarios

Completion certificates are available in:
- `CuTE/COMPLETION_CERTIFICATE.md`
- Individual module directories for CUTLASS 3.x

## Contributing

This repository is designed to be a collaborative learning resource. Contributions are welcome in the form of:
- Bug fixes and improvements to existing modules
- Additional examples and exercises
- Performance optimization techniques
- Documentation enhancements

## Career Impact

Upon completing this learning path, you will be prepared for roles such as:
- **Senior GPU Kernel Engineer**
- **High-Performance Computing Specialist**
- **AI Infrastructure Engineer**
- **Deep Learning Compiler Developer**
- **GPU Performance Optimization Expert**

## Support

For questions, clarifications, or discussions about the material:
- Open an issue in the repository
- Refer to the documentation in each module directory
- Consult the external resources listed above

---

*This repository represents a comprehensive investment in your GPU computing education. Dedication to completing all modules will provide you with industry-leading expertise in AI kernel development.*