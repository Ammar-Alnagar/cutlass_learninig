# AI Kernel Engineer Learning Path

## Complete Curriculum for Mastering High-Performance GPU Computing

This learning path is designed to take you from beginner to expert level in AI kernel engineering, focusing on NVIDIA's CUTLASS 3.x, CuTe, and advanced GPU optimization techniques. The curriculum is structured to build knowledge progressively with hands-on practice at each stage.

## Prerequisites

Before starting this learning path, ensure you have:

- **Programming Background**: Proficiency in C++ (at least intermediate level)
- **Mathematics**: Understanding of linear algebra concepts (matrices, vectors, transformations)
- **System Setup**: Linux environment with NVIDIA GPU (RTX 4060 or equivalent recommended)
- **Development Tools**: CUDA Toolkit 12.x installed, basic familiarity with command line
- **Version Control**: Git proficiency for managing code and tracking progress

## Learning Path Overview

| Phase | Duration | Focus Area | Key Components |
|-------|----------|------------|----------------|
| Foundation | Weeks 1-4 | Core Skills | DSA, CMake, Basic C++ Templates |
| GPU Programming Fundamentals | Weeks 5-10 | CUDA & CuTe Basics | CuTE Modules 1-3 |
| Advanced GPU Programming | Weeks 11-16 | CUTLASS & Optimization | CuTE Modules 4-6, CUTLASS Modules 1-3 |
| Production-Level Skills | Weeks 17-24 | Advanced Techniques | CUTLASS Modules 4-6, Template Metaprogramming |

## Week-by-Week Breakdown

### Phase 1: Foundation (Weeks 1-4)

#### Week 1: Data Structures & Algorithms
- **Objective**: Strengthen foundational programming skills essential for kernel development
- **Materials**: 
  - DSA/00-Introduction-BigO/README.md
  - DSA/00-Introduction-BigO/main.cpp
  - DSA/01-Arrays-Strings/README.md
  - DSA/01-Arrays-Strings/main.cpp
- **Activities**:
  - Complete Introduction to DSA & Big O Notation
  - Implement array/string algorithms
  - Practice complexity analysis
- **Deliverable**: Complete exercises in main.cpp files

#### Week 2: More DSA & Build Systems
- **Objective**: Continue DSA mastery and introduce build systems
- **Materials**:
  - DSA/02-Stacks-Queues/README.md
  - DSA/02-Stacks-Queues/main.cpp
  - DSA/03-Linked-Lists/README.md
  - DSA/03-Linked-Lists/main.cpp
  - Cmake_tr/cmake_guide.md
- **Activities**:
  - Complete Stacks & Queues module
  - Complete Linked Lists module
  - Read through CMake fundamentals
- **Deliverable**: Implement stack/queue and linked list exercises

#### Week 3: Advanced DSA & CMake Basics
- **Objective**: Complete DSA and get comfortable with CMake
- **Materials**:
  - DSA/04-Searching-Sorting/README.md
  - DSA/04-Searching-Sorting/main.cpp
  - DSA/05-Trees/README.md
  - DSA/05-Trees/main.cpp
  - Cmake_tr/hands_on_tutorial.md
- **Activities**:
  - Complete Searching & Sorting algorithms
  - Complete Trees module
  - Start hands-on CMake tutorial
- **Deliverable**: Implement merge sort and binary tree exercises

#### Week 4: CMake Mastery & Graphs
- **Objective**: Master build systems and complete DSA
- **Materials**:
  - DSA/06-Graphs/README.md
  - DSA/06-Graphs/main.cpp
  - Cmake_tr/hands_on_tutorial.md
  - Cmake_tr/cheat_sheet.md
  - Cmake_tr/reference_sheet.md
- **Activities**:
  - Complete Graphs module
  - Finish CMake hands-on tutorial
  - Practice with CMake cheat sheet
- **Deliverable**: Implement graph algorithms and complete CMake exercises

### Phase 2: GPU Programming Fundamentals (Weeks 5-10)

#### Week 5: CUDA & Template Metaprogramming Introduction
- **Objective**: Begin GPU programming fundamentals
- **Materials**:
  - Temp-Meta/module_1_foundations.md
  - Temp-Meta/module_5_cuda_fundamentals.md
  - CuTE/README.md
  - CuTE/SETUP.md
- **Activities**:
  - Complete Module 1: Foundations of Modern C++
  - Begin Module 5: CUDA and GPU Programming Fundamentals
  - Set up CuTE environment
- **Deliverable**: Complete exercises in Module 1 and 5

#### Week 6: CuTE Layout Algebra
- **Objective**: Master CuTE's layout algebra concepts
- **Materials**:
  - CuTE/Module_01_Layout_Algebra/README.md
  - CuTE/Module_01_Layout_Algebra/layout_study.cu
  - CuTE/Module_01_Layout_Algebra/BUILD.md
- **Activities**:
  - Study layout algebra fundamentals
  - Implement and run layout_study.cu
  - Experiment with different layout configurations
- **Deliverable**: Successfully compile and run layout study examples

#### Week 7: CuTE Tensors
- **Objective**: Understand CuTE tensors and memory access patterns
- **Materials**:
  - CuTE/Module_02_CuTe_Tensors/README.md
  - CuTE/Module_02_CuTe_Tensors/tensor_basics.cu
  - CuTE/Module_02_CuTe_Tensors/BUILD.md
- **Activities**:
  - Study tensor creation and slicing operations
  - Implement tensor_basics.cu
  - Experiment with different tensor operations
- **Deliverable**: Successfully implement and run tensor basics

#### Week 8: CuTE Tiled Copy
- **Objective**: Master efficient memory movement patterns
- **Materials**:
  - CuTE/Module_03_Tiled_Copy/README.md
  - CuTE/Module_03_Tiled_Copy/tiled_copy_basics.cu
  - CuTE/Module_03_Tiled_Copy/BUILD.md
- **Activities**:
  - Study tiled copy mechanisms
  - Implement tiled_copy_basics.cu
  - Experiment with vectorized loads and async operations
- **Deliverable**: Successfully implement and run tiled copy examples

#### Week 9: CUTLASS 3.x Introduction
- **Objective**: Begin CUTLASS 3.x modules
- **Materials**:
  - Cutlass3.x/module1-Layouts and Tensors/README.md
  - Cutlass3.x/module1-Layouts and Tensors/main.cu
  - Cutlass3.x/CMakeLists.txt
- **Activities**:
  - Study CUTLASS layout and tensor concepts
  - Implement module 1 examples
  - Compare with CuTE approaches
- **Deliverable**: Successfully compile and run CUTLASS module 1

#### Week 10: CUTLASS Tiled Copy
- **Objective**: Master CUTLASS tiled copy operations
- **Materials**:
  - Cutlass3.x/module2-Tiled Copy/README.md
  - Cutlass3.x/module2-Tiled Copy/main.cu
  - Cutlass3.x/module2-Tiled Copy/CMakeLists.txt
- **Activities**:
  - Study CUTLASS tiled copy mechanisms
  - Implement module 2 examples
  - Compare with CuTE implementations
- **Deliverable**: Successfully compile and run CUTLASS module 2

### Phase 3: Advanced GPU Programming (Weeks 11-16)

#### Week 11: MMA Atoms & Tensor Cores
- **Objective**: Master matrix multiplication using Tensor Cores
- **Materials**:
  - CuTE/Module_04_MMA_Atoms/README.md
  - CuTE/Module_04_MMA_Atoms/mma_atom_basics.cu
  - CuTE/Module_04_MMA_Atoms/BUILD.md
- **Activities**:
  - Study MMA atom operations
  - Implement mma_atom_basics.cu
  - Experiment with different MMA configurations
- **Deliverable**: Successfully implement and run MMA atom examples

#### Week 12: CUTLASS Tiled MMA
- **Objective**: Master CUTLASS MMA operations
- **Materials**:
  - Cutlass3.x/module3-Tiled MMA/README.md
  - Cutlass3.x/module3-Tiled MMA/main.cu
  - Cutlass3.x/module3-Tiled MMA/CMakeLists.txt
- **Activities**:
  - Study CUTLASS MMA implementations
  - Implement module 3 examples
  - Compare with CuTE MMA approaches
- **Deliverable**: Successfully compile and run CUTLASS module 3

#### Week 13: Shared Memory Optimization
- **Objective**: Master shared memory and swizzling techniques
- **Materials**:
  - CuTE/Module_05_Shared_Memory_Swizzling/README.md
  - CuTE/Module_05_Shared_Memory_Swizzling/shared_memory_layouts.cu
  - CuTE/Module_05_Shared_Memory_Swizzling/BUILD.md
- **Activities**:
  - Study shared memory layouts and swizzling
  - Implement shared memory optimization examples
  - Experiment with bank conflict resolution
- **Deliverable**: Successfully implement and run shared memory examples

#### Week 14: CUTLASS Epilogues
- **Objective**: Master fused operations and epilogues
- **Materials**:
  - Cutlass3.x/module4-Fused Bias-Add/README.md
  - Cutlass3.x/module4-Fused Bias-Add/main.cu
  - Cutlass3.x/module4-Fused Bias-Add/CMakeLists.txt
- **Activities**:
  - Study fused bias-add and activation functions
  - Implement module 4 examples
  - Experiment with different epilogue operations
- **Deliverable**: Successfully compile and run CUTLASS module 4

#### Week 15: Collective Mainloops
- **Objective**: Master collective operations and mainloop design
- **Materials**:
  - CuTE/Module_06_Collective_Mainloops/README.md
  - CuTE/Module_06_Collective_Mainloops/producer_consumer_pipeline.cu
  - CuTE/Module_06_Collective_Mainloops/BUILD.md
- **Activities**:
  - Study producer-consumer pipeline design
  - Implement collective mainloop examples
  - Experiment with coordination patterns
- **Deliverable**: Successfully implement and run collective mainloop examples

#### Week 16: CUTLASS Mainloop Pipelining
- **Objective**: Master advanced pipelining techniques
- **Materials**:
  - Cutlass3.x/module5-Mainloop Pipelining/README.md
  - Cutlass3.x/module5-Mainloop Pipelining/main.cu
  - Cutlass3.x/module5-Mainloop Pipelining/CMakeLists.txt
- **Activities**:
  - Study mainloop pipelining concepts
  - Implement module 5 examples
  - Experiment with temporal overlap techniques
- **Deliverable**: Successfully compile and run CUTLASS module 5

### Phase 4: Production-Level Skills (Weeks 17-24)

#### Week 17: Advanced Template Metaprogramming
- **Objective**: Master advanced C++ template techniques
- **Materials**:
  - Temp-Meta/module_4_advanced.md
  - Temp-Meta/module_7_cutlass_patterns.md
- **Activities**:
  - Complete advanced template metaprogramming
  - Study CUTLASS template patterns
  - Implement complex template examples
- **Deliverable**: Complete advanced template exercises

#### Week 18: CUTLASS Fused Operations
- **Objective**: Master advanced fused operations
- **Materials**:
  - Cutlass3.x/module6-Fused Epilogues/README.md
  - Cutlass3.x/module6-Fused Epilogues/main.cu
  - Cutlass3.x/module6-Fused Epilogues/CMakeLists.txt
- **Activities**:
  - Study advanced fused epilogue techniques
  - Implement module 6 examples
  - Experiment with VRAM roundtrip avoidance
- **Deliverable**: Successfully compile and run CUTLASS module 6

#### Week 19: CUTLASS Architecture Deep Dive
- **Objective**: Understand CUTLASS architecture internals
- **Materials**:
  - Temp-Meta/module_6_cutlass_architecture.md
  - Temp-Meta/module_8_advanced_customization.md
- **Activities**:
  - Study CUTLASS architecture overview
  - Explore advanced customization techniques
  - Implement custom epilogue operations
- **Deliverable**: Complete architecture exercises

#### Week 20: Performance Optimization Fundamentals
- **Objective**: Learn performance measurement and optimization
- **Materials**:
  - Temp-Meta/module_10_performance_optimization.md (sections 1-3)
  - Profiling tools documentation
- **Activities**:
  - Study performance measurement techniques
  - Learn to use Nsight Compute profiler
  - Profile existing implementations
- **Deliverable**: Complete profiling exercises

#### Week 21: Memory Bandwidth Optimization
- **Objective**: Master memory optimization techniques
- **Materials**:
  - Temp-Meta/module_10_performance_optimization.md (sections 4-6)
  - Memory optimization examples
- **Activities**:
  - Study memory bandwidth optimization
  - Optimize existing kernels for memory
  - Measure performance improvements
- **Deliverable**: Optimized kernel implementations

#### Week 22: Occupancy & Register Optimization
- **Objective**: Master occupancy and register optimization
- **Materials**:
  - Temp-Meta/module_10_performance_optimization.md (sections 7-9)
  - Occupancy optimization examples
- **Activities**:
  - Study occupancy optimization techniques
  - Optimize register usage
  - Balance occupancy vs register usage
- **Deliverable**: Optimized kernel implementations

#### Week 23: Real-World Applications
- **Objective**: Apply knowledge to realistic scenarios
- **Materials**:
  - Temp-Meta/module_9_real_world_applications.md
  - Integration examples
- **Activities**:
  - Study real-world applications
  - Implement integration examples
  - Work on complex optimization problems
- **Deliverable**: Complete integration project

#### Week 24: Capstone Project
- **Objective**: Synthesize all learning in a comprehensive project
- **Materials**:
  - All previous modules
  - Capstone project specification
- **Activities**:
  - Design and implement a complete kernel
  - Apply all learned optimization techniques
  - Profile and optimize the implementation
- **Deliverable**: Complete capstone project with documentation

## Assessment & Milestones

### Weekly Assessments
Each week includes practical exercises that reinforce the concepts covered. Complete all exercises before moving to the next week.

### Monthly Reviews
At the end of each month, conduct a self-assessment:
- Can you explain the key concepts from the past month?
- Have you completed all practical exercises?
- Are you comfortable with the code examples?

### Mid-Program Checkpoint (Week 12)
- Successfully implement a complete CuTE-based kernel
- Demonstrate understanding of layout algebra, tensors, and MMA operations
- Show proficiency with profiling and optimization tools

### Final Assessment (Week 24)
- Complete the capstone project implementing a high-performance kernel
- Demonstrate mastery of all concepts covered
- Prepare a presentation explaining your implementation choices and optimizations

## Resources & Support

### Internal Documentation
- README.md: Main repository overview
- COMPLETE_LEARNING_GUIDE.md: Extended learning materials
- HANDS_ON_IMPLEMENTATION_GUIDE.md: Practical implementation strategies

### External Resources
- NVIDIA CUDA Documentation
- CUTLASS GitHub Repository
- CuTe Documentation
- GPU Architecture Whitepapers

## Tips for Success

1. **Practice Daily**: Spend at least 2-3 hours daily on the material
2. **Code Along**: Don't just read - implement every example
3. **Experiment**: Modify examples to understand how changes affect performance
4. **Take Notes**: Document your learning and insights
5. **Join Communities**: Engage with GPU computing communities for support
6. **Profile Everything**: Always measure performance impact of changes

## Next Steps After Completion

Upon completing this learning path, you'll be prepared for:
- Senior GPU Kernel Engineer positions
- High-Performance Computing specialist roles
- AI Infrastructure engineering roles
- Deep Learning compiler development
- Independent GPU optimization consulting

Consider contributing back to the community by:
- Creating educational content
- Contributing to open-source GPU libraries
- Mentoring other learners
- Publishing performance optimization techniques