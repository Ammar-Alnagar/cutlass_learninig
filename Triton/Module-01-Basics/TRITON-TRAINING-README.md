# Triton Kernel Training Program

A comprehensive hands-on training program for learning Triton GPU kernels from basics to mastery.

## 📚 Training Structure

This training program consists of **4 learning tracks** with **19 modules** total:

```
Triton Training Program
├── 📝 Missing Elements (5 modules) - Fix incomplete kernels
├── ⚡ Optimization (5 modules) - Learn performance techniques
├── 🐛 Debugging (4 modules) - Master debugging skills
└── 🎓 Mastery (5 modules) - Advanced kernel development
```

## 🗂️ Directory Structure

```
Module-01-Basics/
├── basic_vector_add.py          # Your starting point
├── README.md
│
├── missing-elements/            # Track 1: Fix broken kernels
│   ├── module-01-vector-add/
│   ├── module-02-matrix-mul/
│   ├── module-03-softmax/
│   ├── module-04-layer-norm/
│   └── module-05-conv2d/
│
├── optimization/                # Track 2: Performance optimization
│   ├── module-01-tiling/
│   ├── module-02-pipelining/
│   ├── module-03-shared-memory/
│   ├── module-04-warp-specialization/
│   └── module-05-advanced-optimizations/
│
├── debugging/                   # Track 3: Debugging skills
│   ├── module-01-basic-debugging/
│   ├── module-02-error-handling/
│   ├── module-03-performance-profiling/
│   └── module-04-memory-debugging/
│
└── mastery/                     # Track 4: Advanced mastery
    ├── module-01-custom-kernels/
    ├── module-02-fused-operations/
    ├── module-03-attention-kernels/
    ├── module-04-training-kernels/
    └── module-05-production-ready/
```

## 📖 Learning Tracks

### Track 1: Missing Elements (5 modules)
**Goal:** Learn Triton basics by fixing incomplete kernels

| Module | Topic | Challenge |
|--------|-------|-----------|
| 01 | Vector Add | Add missing imports, decorators, loads/stores |
| 02 | Matrix Mul | Implement 2D indexing, tl.dot(), accumulation |
| 03 | Softmax | Implement numerically stable softmax |
| 04 | Layer Norm | Add mean/variance computation, normalization |
| 05 | Conv2D | Implement im2col-style convolution |

**How to use:**
1. Open `kernel.py` in each module
2. Find the `# TODO` comments
3. Fill in the missing code using the hints
4. Run `python kernel.py` to test

---

### Track 2: Optimization (5 modules)
**Goal:** Learn to optimize Triton kernels for maximum performance

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| 01 | Tiling | Block sizes, memory coalescing, cache utilization |
| 02 | Pipelining | NUM_STAGES, latency hiding, prefetching |
| 03 | Shared Memory | SRAM usage, data reuse, tree reduction |
| 04 | Warp Specialization | Program ID swizzling, fused operations |
| 05 | Advanced | Autotuning, persistent kernels, combined optimizations |

**How to use:**
1. Read the README.md for concepts
2. Run the benchmark scripts
3. Experiment with different parameters
4. Compare with PyTorch baseline

---

### Track 3: Debugging (4 modules)
**Goal:** Master debugging techniques for Triton kernels

| Module | Topic | Skills Learned |
|--------|-------|----------------|
| 01 | Basic Debugging | Print debugging, output validation, boundary testing |
| 02 | Error Handling | Input validation, fallback mechanisms, recovery |
| 03 | Performance Profiling | Timing, bandwidth analysis, bottleneck identification |
| 04 | Memory Debugging | Bounds checking, alignment, corruption detection |

**How to use:**
1. Learn debugging patterns
2. Apply to your own kernels
3. Use the provided utilities
4. Practice with intentional bugs

---

### Track 4: Mastery (5 modules)
**Goal:** Become a Triton expert with production-ready skills

| Module | Topic | Projects |
|--------|-------|----------|
| 01 | Custom Kernels | Polynomial activation, GLU, complex multiplication |
| 02 | Fused Operations | Linear+Bias+Activation, LayerNorm+Linear |
| 03 | Attention Kernels | FlashAttention, causal masking, multi-head |
| 04 | Training Kernels | Forward/backward pass, Adam optimizer |
| 05 | Production-Ready | Testing, benchmarking, documentation |

**How to use:**
1. Study the reference implementations
2. Build your own custom kernels
3. Create a portfolio project
4. Follow production checklist

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support
- Triton (`pip install triton`)
- NVIDIA GPU (Compute Capability 7.0+)

### Installation
```bash
# Install Triton
pip install triton

# Verify installation
python -c "import triton; print(triton.__version__)"

# Run the basic example
python basic_vector_add.py
```

### Recommended Learning Path

1. **Start with `basic_vector_add.py`** - Understand the basic structure
2. **Track 1: Missing Elements** - Learn by fixing (1-2 weeks)
3. **Track 2: Optimization** - Learn performance techniques (1-2 weeks)
4. **Track 3: Debugging** - Learn debugging skills (1 week)
5. **Track 4: Mastery** - Build advanced kernels (2-3 weeks)

**Total estimated time: 5-8 weeks**

---

## 📝 Module Format

Each module contains:

```
module-XX-topic/
├── README.md          # Concepts, explanations, exercises
├── topic_module.py    # Code examples and benchmarks
└── kernel.py          # (Missing Elements only) Fix this file
```

### README.md Structure
- Learning objectives
- Key concepts explained
- Code examples
- Exercises
- Best practices
- Next steps

### Code Files Structure
- Working implementations
- Benchmark utilities
- Test cases
- Performance comparisons

---

## 🎯 Learning Outcomes

After completing this training, you will be able to:

### Basics
- ✅ Write basic Triton kernels
- ✅ Understand program ID and indexing
- ✅ Handle boundary conditions with masks
- ✅ Load/store data correctly

### Optimization
- ✅ Choose optimal block sizes
- ✅ Enable pipelining for latency hiding
- ✅ Maximize shared memory usage
- ✅ Fuse operations for performance

### Debugging
- ✅ Debug kernel errors systematically
- ✅ Profile and identify bottlenecks
- ✅ Handle edge cases gracefully
- ✅ Validate kernel outputs

### Mastery
- ✅ Design custom kernels from scratch
- ✅ Implement attention mechanisms
- ✅ Create training kernels with backward pass
- ✅ Write production-ready code

---

## 🏆 Challenges & Exercises

### Missing Elements Challenges
- Fix all 14 missing elements in Vector Add
- Complete the 21+ missing elements in Matrix Mul
- Implement stable softmax with 16+ fixes
- Complete LayerNorm with 17+ additions
- Master Conv2D with 25+ missing pieces

### Optimization Challenges
- Find optimal block size for your GPU
- Achieve >80% memory bandwidth
- Match or beat PyTorch performance
- Tune NUM_STAGES for best performance

### Debugging Challenges
- Fix intentionally broken kernels
- Add comprehensive input validation
- Profile and optimize a slow kernel
- Detect and fix memory corruption

### Mastery Projects
- Implement Swish activation kernel
- Create fused LayerNorm+GELU kernel
- Build FlashAttention from scratch
- Implement Adam optimizer kernel

---

## 📚 Resources

### Official Documentation
- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Recommended Reading
- [Triton Paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [GPU Architecture Guides](https://docs.nvidia.com/cuda/)

### Tools
- **Nsight Compute** - Kernel profiling
- **cuda-memcheck** - Memory debugging
- **PyTorch Profiler** - End-to-end profiling

---

## 💡 Tips for Success

1. **Start simple** - Master basics before optimization
2. **Test frequently** - Run tests after each change
3. **Read error messages** - They're usually helpful
4. **Compare with PyTorch** - Verify correctness
5. **Profile before optimizing** - Find real bottlenecks
6. **Join the community** - Ask questions on Discord/GitHub

---

## 🤝 Contributing

Found a bug? Have a suggestion? Want to add a module?

1. Fork the repository
2. Create a branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This training program is provided for educational purposes. Feel free to use, modify, and share.

---

## 🎓 Certificate of Completion

Track your progress by completing all modules:

- [ ] Track 1: Missing Elements (5/5 modules)
- [ ] Track 2: Optimization (5/5 modules)
- [ ] Track 3: Debugging (4/4 modules)
- [ ] Track 4: Mastery (5/5 modules)

**Total: 19/19 modules completed** 🎉

---

## 📞 Support

- **Issues:** Open an issue on GitHub
- **Discussions:** Join the Triton Discord
- **Questions:** Check existing issues first

---

**Happy Kernel Writing! 🚀**
