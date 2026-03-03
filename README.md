# AI Kernel Learning

A comprehensive repository dedicated to mastering GPU kernel programming, high-performance computing (HPC) primitives, and the mathematical foundations of modern AI architectures. This project serves as a structured learning path from fundamental CUDA concepts to advanced optimizations using industry-standard libraries like CUTLASS and Triton.

---

## 📂 Repository Structure & Module Deep Dive

### 1. [CUDA Learning Path](./Cuda/) (Fundamentals to Mastery)
*Comprehensive curriculum covering the entire CUDA ecosystem, from thread hierarchy to hardware-specific optimizations.*
- **[Fundamentals](./Cuda/fundamentals/):** Core concepts including Grids, Blocks, Threads, and the Memory Hierarchy (Global vs. Shared vs. Registers).
- **[Memory Optimization](./Cuda/memory_optimization/):** Techniques for maximizing bandwidth via Coalescing, avoiding Shared Memory Bank Conflicts, and implementing Swizzling.
- **[Execution Optimization](./Cuda/execution_optimization/):** Balancing resources with Occupancy analysis and utilizing Warp-Level Primitives (`__shfl_sync`) for intra-warp communication.
- **[Advanced Concepts](./Cuda/advanced_concepts/):** Asynchronous copies (`cp.async`), Tensor Core programming, and Software Pipelining to hide memory latency.
- **[Cudalings](./Cuda/cudalings/):** A series of interactive, small-scale exercises to reinforce CUDA syntax and logic.

### 2. [Triton Learning Path](./Triton/) (Python-First Kernel Development)
*Structured modules for writing high-performance GPU kernels using the Triton DSL.*
- **[Module 01-04: Basics & Tiling](./Triton/Module-01-Basics/):** Vector addition, memory operations, boundary handling, and the core philosophy of block-based programming.
- **[Module 05: Matrix Multiplication](./Triton/Module-05-Matrix-Multiplication/):** Implementation of tiled GEMM and optimization strategies for different shapes.
- **[Module 06-08: Advanced Techniques](./Triton/Module-06-Advanced-Memory/):** Cache hierarchy optimization, parallel reductions (sum/max/min), and numerical stability.
- **[Optimization](./Triton/optimization/):** Deep dives into Pipelining, Shared Memory management, and Warp Specialization.
- **[Mastery](./Triton/mastery/):** Custom attention and production-ready kernels.

### 3. [CuTE & LLM Inference Engineering](./CuTE/) (The Modern Stack)
*Hands-on curriculum focused on building production-grade LLM kernels using CuTE and CUTLASS 3.x.*
- **[Module 01: Layouts](./CuTE/Module_01_Layouts/):** Layout algebra (`make_layout`), hierarchical tiling, and GQA (Grouped Query Attention) stride-0 patterns.
- **[Module 02: Tensors](./CuTE/Module_02_Tensors/):** Tensor views, underscore slicing for per-head access, and `local_tile` for block iteration.
- **[Module 03-04: Tiled Operations](./CuTE/Module_03_TiledCopy/):** Vectorized 128-bit loads, `cp.async` pipelines, and MMA Atoms for Tensor Core acceleration.
- **[Module 05-06: Pipelining](./CuTE/Module_05_Swizzle/):** Bank-conflict-free swizzling and multi-stage asynchronous pipelines (prologue/mainloop/epilogue).
- **[Projects](./CuTE/Projects/):** Capstone implementations of **Tiled GEMM** and **FlashAttention-2 Prefill**.

### 4. [Transformer Math](./transformer_math/) (Theoretical Foundations)
*The mathematical "why" behind kernel engineering, specifically for LLM inference.*
- **[Attention & KV Cache](./transformer_math/01_attention/):** Scaled dot-product derivations, causal masking, and the "Memory Wall" of KV caching.
- **[FlashAttention](./transformer_math/05_flash_attention/):** Tiling insights, the IO problem, and the complex math of Online Softmax.
- **[Optimization Variants](./transformer_math/03_attention_variants/):** MHA, MQA, GQA, and MLA architectures and their hardware implications.
- **[Systems Thinking](./transformer_math/10_arithmetic_intensity/):** Roofline analysis, decode vs. prefill characteristics, and batch size effects on arithmetic intensity.

### 5. [GPU Data Structures & Algorithms (DSA)](./GPU-DSA/)
*Implementations of classic and modern parallel algorithms.*
- **Parallel Primitives:** Reductions, Prefix Sum (Scan), and Z-Curve/Morton Order indexing.
- **Sorting:** High-performance Radix Sort and Bitonic Sort kernels.
- **Advanced Kernels:** Fused LayerNorm, Online Softmax, Paged Attention (vLLM style), and Double Buffering patterns.

### 6. [Specialized Tooling & Systems](./)
- **[NCCL](./NCCL/):** 7 modules covering multi-GPU collectives (AllReduce, AllGather) and ring/tree algorithms.
- **[PTX](./PTX/):** Low-level virtual ISA programming, debugging, and PTX-specific optimizations.
- **[Profiling](./Profiling/):** 8 modules on using Nsight Compute (NCU) and Nsight Systems (NSYS) to identify bottlenecks.
- **[Cmake-guide](./Cmake-guide/):** Comprehensive training for building complex CUDA/C++ projects with modern CMake.
- **[Concurrency](./Concurrency/):** Host-side C++ multithreading, synchronization, and asynchronous workload driving.
- **[Template Metaprogramming](./Template_Metaprogramming/):** The C++ patterns required to understand the CUTLASS and CuTE source code.

---

## 🛠️ Tooling & Environment
*   **Compilers:** `nvcc` (CUDA Toolkit 12.x+), `clang++` (C++17/20).
*   **Libraries:** CUTLASS 3.x, CuTE, NCCL, Triton.
*   **Profiling:** NVIDIA Nsight Compute & Nsight Systems.
*   **Hardware Target:** Optimized for NVIDIA Ampere (SM80), Ada Lovelace (SM89), and Hopper (SM90).

## 📖 Getting Started
1.  Read **[LEARNING_PATH.md](./LEARNING_PATH.md)** to choose your entry point.
2.  Follow the **[Transformer Math](./transformer_math/README.md)** sequence to understand the implementation depth.
3.  Transition to **[CuTE](./CuTE/README.md)** or **[Triton](./Triton/README.md)** for hands-on kernel development.
4.  Use **[Profiling](./Profiling/README.md)** tools to verify your performance against the hardware roofline.
