# C++ Systems Programming for CUDA Engineers

Accelerated curriculum for engineers who know CUDA deeply but need C++ template fluency to read CUTLASS.

## Prerequisites

- Strong C instincts
- CUDA expert: shared memory, occupancy, warp divergence
- Built LLM inference systems (vLLM, SGLang, PagedAttention)
- RAII and constexpr concepts understood

## Critical Gaps This Curriculum Fills

1. **lvalue vs rvalue + named rvalue rule** — why `std::move` exists
2. **unique_ptr is move-only** — ownership transfer, not sharing
3. **std::atomic two-layer model** — atomicity vs memory ordering
4. **Type aliases in templates** — `using value_type = T` pattern
5. **Undefined Behavior taxonomy** — recognize and fix
6. **std::forward and forwarding references** — perfect forwarding
7. **Template specialization vs overloading** — type dispatch

## Directory Structure

```
phase0_foundations/     # lvalue/rvalue — DO NOT SKIP
phase1_ownership/       # move semantics, unique_ptr, Rule of Five
phase2_undefined_behavior/  # DEBUG exercises — broken code to fix
phase3_templates/       # type aliases, specialization, CUTLASS patterns
phase4_atomics_memory_model/  # atomicity vs ordering, two-layer model
phase5_concurrency/     # thread pools, lock-free, barriers
phase6_modern_patterns/ # CRTP, expression templates, Rust FFI
phase7_interview_sim/   # timed exercises, verbal checklists
```

## Build Instructions

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug  # Debug enables sanitizers
cmake --build . --target list_exercises
```

### Compile Individual Exercise

```bash
# C++ exercise
g++ -std=c++20 -O2 -Wall -Wextra -fsanitize=address,undefined \
    -o ex exercise.cpp && ./ex

# CUDA exercise
nvcc -std=c++17 -O2 -arch=sm_89 -o ex exercise.cu && ./ex
```

## How to Use

1. **Start at phase0** — lvalue/rvalue foundation is non-negotiable
2. **Read exercise.cpp first** — understand the task
3. **Try hints.md** — 3 hints, progressively specific
4. **Implement** — compile and run until VERIFY passes
5. **Check solution.cpp** — only after completion or if stuck
6. **Move on** — each exercise builds on the last

## Exercise Formats

- **SCAFFOLD** — fill-in-the-blank, specific TODOs
- **IMPLEMENT** — write from memory, timed
- **DEBUG** — broken code, name the UB, fix it
- **ANNOTATE** — read real CUTLASS, explain opaque sections

## Time Commitment

- Phase 0: 45 min (foundation — do not rush)
- Phase 1: 90 min (ownership)
- Phase 2: 60 min (UB recognition)
- Phase 3: 120 min (templates)
- Phase 4: 60 min (atomics)
- Phase 5: 90 min (concurrency)
- Phase 6: 90 min (modern patterns)
- Phase 7: 60 min (interview simulation)

**Total: ~10 hours** for complete fluency.

## CUTLASS Mapping

Every exercise connects to CUTLASS 3.x patterns:
- Type aliases → `cutlass::TensorRef`, policy structs
- Move semantics → `cutlass::Kernel`, collective builders
- Specialization → `__half` vs `float` dispatch
- Atomics → reduction, epilogue fusion
- CRTP → `cutlass::gemm::Collective`

---

**Rule:** Complete phase0 before phase1. Everything collapses without lvalue/rvalue fluency.
