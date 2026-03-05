# Module 08: MMA Atom Internals — Thread and Value Layouts

## What This Module Teaches

This module exposes the internal structure of MMA atoms — the distinction between **thread layout** and **value layout** inside a MMA operation. This is the foundation for correctly implementing FlashAttention-2's online softmax rescale.

1. **Thread Layout**: Which of the 32 warp threads participates in which row/column of the output tile?
2. **Value Layout**: Which registers does each thread own, and what (row, col) do they correspond to?
3. **Fragment Rescale**: Implement the FlashAttention-2 rescale step using the value layout

## Why MMA Atom Internals Matter for LLM Inference

**Job mapping:**

- **NVIDIA DL Software Engineer** — "GPU architecture and compilation stack" and "kernel-level performance" — The FlashAttention-2 online softmax rescale requires knowing exactly which registers each thread owns in the C fragment.
- **Cerebras Performance Engineer** — "Production-quality kernel optimization" — You cannot correctly implement `acc_s = acc_s * scale` without knowing which fragment elements belong to the same row.

**The problem:** In FlashAttention-2, when a new max is found in a KV block, you must rescale the accumulator:
```
acc_o[i] *= alpha[row_of(i)]  // for each fragment element
```

But threads don't own elements in row-major order! The MMA atom uses a specific layout for bank conflict avoidance and Tensor Core efficiency. Without understanding this layout, you'll scale the wrong elements.

## Prerequisites

- Module 04: TiledMMA (`make_tiled_mma`, `partition_fragment_A/B/C`, `gemm`)
- Module 01: Layouts (`make_layout`, `shape`, `stride`, `print_layout`)
- Module 02.5: Layout Algebra (`composition`, `complement`)

## The MMA Atom Mental Model

```
SM80_16x8x16_F16F16F16F16_TN produces a 16×8 output tile

Thread Layout (which thread owns which output position):
┌────────────────────────────────┐
│ T0  T1  T4  T5  T8  T9  T12 T13│  ← Row 0
│ T2  T3  T6  T7  T10 T11 T14 T15│  ← Row 1
│ T0  T1  T4  T5  T8  T9  T12 T13│  ← Row 2
│ ...                            │
│ T16 T17 T20 T21 T24 T25 T28 T29│  ← Row 14
│ T18 T19 T22 T23 T26 T27 T30 T31│  ← Row 15
└────────────────────────────────┘

Key observations:
- Threads are NOT in row-major order!
- Each thread owns 2 elements (16×8 = 128 elements / 32 threads = 4 elements per thread... wait, that's 4)
- Actually for this atom: 16×8 = 128 elements, 32 threads → 4 elements per thread

Value Layout (which register index corresponds to which position):
For thread 0:
  fragment[0] → (row=0, col=0)
  fragment[1] → (row=0, col=2)
  fragment[2] → (row=2, col=0)
  fragment[3] → (row=2, col=2)

This is NOT sequential! You must use the value layout to map fragment index → (row, col).
```

## Exercises

| Exercise | Concept | LLM Pattern |
|----------|---------|-------------|
| ex01_atom_thread_layout.cu | Thread layout of `SM80_16x8x16_F16F16F16F16_TN` | Understanding warp participation |
| ex02_atom_value_layout.cu | Value layout — register → (row, col) mapping | Fragment ownership |
| ex03_fragment_rescale.cu | FlashAttention-2 rescale using value layout | Online softmax correction |

## Exit Criteria

Before starting the FlashAttention-2 capstone, you must be able to:

1. Print and visualize the thread layout of `SM80_16x8x16_F16F16F16F16_TN`
2. Manually compute which output element `fragment[0]` of thread 0 corresponds to
3. Implement `rescale_output_fragment` that correctly maps fragment index → row index → scale factor
4. Explain why the thread layout is NOT row-major (bank conflict avoidance)

## Nsight Metrics

After each exercise, check:

- **nsight compute**: `sm__warps_per_sm` — verify warp occupancy
- **Stall**: `sm__thread_issue_rate` — check for divergence in rescale loops
- **Register**: `gpu__sm_register_space_usage` — verify fragment register allocation

## Common Mistakes

1. **Assuming row-major thread layout:** The thread layout is specifically designed to avoid bank conflicts. Thread 0 does NOT own elements (0,0), (0,1), (0,2), (0,3) in order.

2. **Iterating fragment sequentially:** You cannot do `for i in range(num_elements): fragment[i] *= scale`. You must use the value layout to find which row each fragment element belongs to.

3. **Wrong scale factor indexing:** The scale factor is per-row, not per-element. Two fragment elements in the same row share the same scale factor.

4. **Confusing thread layout and value layout:** Thread layout answers "which thread owns which output position." Value layout answers "which registers does a thread own, and what positions do they map to?"

---

Next: **ex01_atom_thread_layout.cu** — visualize the thread layout of an MMA atom.
