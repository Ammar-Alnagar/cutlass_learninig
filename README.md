# AI Kernel Learning

A comprehensive repository dedicated to mastering GPU kernel programming, high-performance computing (HPC) primitives, and the mathematical foundations of modern AI architectures. This project serves as a structured learning path from fundamental CUDA concepts to advanced optimizations using industry-standard libraries like CUTLASS and Triton.

## 🚀 Core Learning Modules

### 1. GPU Kernel Programming
*   **CUDA Fundamentals:** Core execution model, memory hierarchy (Global, Shared, Constant), and thread indexing.
*   **Advanced CUDA:** Warp-level primitives (`__shfl_sync`), kernel fusion, thread coarsening, and asynchronous copies.
*   **PTX (Parallel Thread Execution):** Low-level virtual machine and instruction set architecture (ISA) for NVIDIA GPUs.
*   **Triton:** Python-based domain-specific language for writing highly efficient GPU kernels with automated tiling and memory management.

### 2. Specialized Libraries
*   **CuTE (C++ Unrolled Template Engine):** Layout abstractions, hierarchical tensors, and tiled operations for complex GEMM schedules.
*   **CUTLASS 3.x:** High-performance template library for GEMM and convolution, focusing on the latest Hopper (SM90) and Blackwell architectures.
*   **NCCL (NVIDIA Collective Communications Library):** Multi-GPU and multi-node communication primitives (AllReduce, Broadcast, etc.).

### 3. AI & Transformer Math
*   **Attention Mechanisms:** Scalable Dot-Product, Multi-Head Attention (MHA), Grouped Query Attention (GQA), and FlashAttention (v1/v2).
*   **Inference Optimizations:** KV Cache management, Paged Attention (vLLM style), Speculative Decoding, and Quantization (INT8/FP8).
*   **Architectural Primitives:** RoPE (Rotary Positional Embeddings), Mixture of Experts (MoE) routing, and Layer Normalization.

### 4. GPU Data Structures & Algorithms (DSA)
*   **Parallel Primitives:** Reductions, Prefix Sum (Scan), Radix Sort, and Bitonic Sort.
*   **Optimized Kernels:** Tiled Matrix Multiplication, Online Softmax, and Fused LayerNorm.

---

## 📂 Repository Structure

```text
.
├── Cuda/                     # CUDA learning path (Fundamentals -> Advanced)
│   ├── fundamentals/         # Basic execution model and memory hierarchy
│   ├── memory_optimization/  # Global coalescing, bank conflicts, and tiling
│   ├── execution_optimization/# Warp shuffle, reductions, and kernel fusion
│   ├── mathematical_kernels/ # GEMM, Softmax, Attention, and Layernorm
│   └── cudalings/            # Interactive exercises for CUDA mastery
├── Triton/                   # Triton kernel development (Basics -> Mastery)
│   ├── Module-01-Basics/     # Vector addition and basic kernels
│   ├── Module-05-Matrix-Multiplication/ # Tiled matrix multiplication in Triton
│   ├── optimization/         # Pipelining, shared memory, and warp specialization
│   └── mastery/              # Custom attention and production-ready kernels
├── CuTE/                     # Layouts and Tensors abstraction (Modules 01-06)
├── Cutlass3.x/               # CUTLASS 3.x template usage and fusion
├── transformer_math/         # Theoretical foundations and Python simulations
│   ├── 01_attention/         # SDPA and MHA analysis
│   ├── 05_flash_attention/   # Tiling insights and online softmax
│   └── 07_paged_attention/   # Memory fragmentation and block tables
├── GPU-DSA/                  # High-performance GPU algorithms
│   ├── Parallel_Reduction/   # Tree-based and warp-level reductions
│   ├── FlashAttention/       # Implementation of memory-efficient attention
│   └── Radix_Sort/           # GPU-accelerated sorting algorithms
├── NCCL/                     # Multi-GPU collective communication modules
├── PTX/                      # Low-level PTX programming and debugging
├── Profiling/                # Nsight Compute (ncu) and Nsight Systems (nsys)
├── Concurrency/              # C++ host-side multithreading and async patterns
├── Template_Metaprogramming/ # Modern C++ patterns used in CUTLASS/CuTE
├── DSA/                      # Standard Data Structures and Algorithms (Host-side)
├── Cmake-guide/              # Comprehensive build system training
├── learning_material.md      # Curated list of external resources
├── LEARNING_PATH.md          # Suggested order of study
└── plan.md                   # Roadmap for project evolution
```

---

## 🛠️ Tooling & Environment
*   **Compiler:** `nvcc` (CUDA Toolkit) and `clang++`.
*   **Build System:** `CMake` (3.20+) and `Make`.
*   **Profiling:** NVIDIA Nsight Compute (NCU) for kernel-level analysis and Nsight Systems (NSYS) for application-level timelines.
*   **Environment:** Python 3.x with `torch` and `triton` for DSL development.

## 📖 How to Use
1.  Consult `LEARNING_PATH.md` to identify your current skill level.
2.  Review `learning_material.md` for pre-requisite reading.
3.  Navigate to a specific module (e.g., `Cuda/fundamentals`) and follow the local README instructions.
4.  Run provided exercises and verify performance using the tools in the `Profiling/` directory.
