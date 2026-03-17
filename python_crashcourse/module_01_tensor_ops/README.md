# Module 01 — Tensor Ops and Indexing

## Motivation

Tensors are the universal data structure of ML systems: every model weight, every
activation map, every batch of inputs is a tensor. As an ML or systems engineer you
will spend a disproportionate amount of time reshaping, slicing, and combining tensors
— often under tight latency budgets where a misplaced `.contiguous()` call or an
accidental data copy can silently double your memory footprint. PyTorch and NumPy share
a nearly identical mental model (n-dimensional arrays with strides), but differ in
device placement, autograd, and broadcasting edge cases. Mastering both APIs — tensor
creation, dtype promotion, reshape vs view, fancy indexing, boolean masks, broadcasting,
in-place ops, einsum, and the underlying stride/contiguity machinery — gives you the
vocabulary to read model code fluently, debug shape errors in seconds, and write
inference kernels that stay on the fast path.

---

## Concept Map

```
tensor creation
  (torch.tensor / torch.zeros / torch.randn / np.array / np.zeros)
        │
        ▼
shapes & dtypes
  (.shape, .dtype, .device, type promotion rules)
        │
        ▼
reshape / view
  (.reshape() copies if needed  ←→  .view() requires contiguous)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
slicing                               fancy indexing
  (basic [i, j], ranges [a:b])         (index with a tensor/list)
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
               boolean masks
         (tensor[mask] → 1-D gather)
                       │
                       ▼
               broadcasting
         (implicit shape expansion rules:
          align right, size-1 dims stretch)
                       │
                       ▼
           in-place ops  vs  out-of-place ops
         (tensor.add_()  vs  tensor + other;
          in-place breaks autograd — know when to avoid)
                       │
                       ▼
                   einsum
         (torch.einsum / np.einsum:
          express matmul, outer, trace, batch-matmul
          with a single index string)
                       │
                       ▼
           strides & contiguity
         (.stride(), .is_contiguous(),
          .contiguous() — when a view becomes a copy)
```

---

## How to Run

Run each file from the `module_01_tensor_ops/` directory (or prefix the path):

```bash
# Exercise 1 — tensor creation, shapes, dtypes, reshape/view
python python_crashcourse/module_01_tensor_ops/01_basics.py

# Exercise 2 — slicing, fancy indexing, boolean masks
python python_crashcourse/module_01_tensor_ops/02_indexing.py

# Exercise 3 — broadcasting, in-place vs out-of-place ops, einsum
python python_crashcourse/module_01_tensor_ops/03_operations.py

# Exercise 4 — challenge: strides, contiguity, mini inference pipeline
python python_crashcourse/module_01_tensor_ops/04_challenge.py
```

Each file prints `✓ Section N passed` for every check you complete correctly, and
`✗ Section N failed: <reason>` when something is wrong. You can re-run at any time —
the files are idempotent.

---

## What to Do If Stuck

### Solution files

Complete, fully-commented reference implementations live in `solutions/`:

| Exercise file      | Solution file                         |
| ------------------ | ------------------------------------- |
| `01_basics.py`     | `solutions/01_basics_solution.py`     |
| `02_indexing.py`   | `solutions/02_indexing_solution.py`   |
| `03_operations.py` | `solutions/03_operations_solution.py` |
| `04_challenge.py`  | `solutions/04_challenge_solution.py`  |

Try to work through the `# TODO:` markers yourself first — the self-checks will tell
you exactly which section is failing. Peek at the solution only after you have made a
genuine attempt.

### Official documentation

- **PyTorch tensors**: <https://pytorch.org/docs/stable/tensors.html>
- **PyTorch `torch` namespace** (creation ops, math ops): <https://pytorch.org/docs/stable/torch.html>
- **PyTorch indexing semantics**: <https://pytorch.org/docs/stable/tensor_view.html>
- **`torch.einsum`**: <https://pytorch.org/docs/stable/generated/torch.einsum.html>
- **NumPy array creation**: <https://numpy.org/doc/stable/user/basics.creation.html>
- **NumPy indexing**: <https://numpy.org/doc/stable/user/basics.indexing.html>
- **NumPy broadcasting**: <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- **`numpy.einsum`**: <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
