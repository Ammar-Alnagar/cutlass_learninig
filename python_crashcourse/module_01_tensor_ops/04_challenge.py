"""
Challenge: Mini Inference Pipeline

Scenario: You are building a lightweight inference pipeline for a small
transformer encoder.  Raw token IDs arrive as a flat 1-D buffer from a
C++ tokenizer.  You must:

  1. Understand the memory layout (strides) of the tensors you create.
  2. Detect when an operation breaks contiguity and fix it cheaply.
  3. Run a single forward pass: embed → project → bias → threshold mask.
  4. Hand the result off to a NumPy-based post-processor without copying data.

This challenge integrates every topic from Module 01:
  - Tensor creation and dtypes          (Section 1 of 01_basics.py)
  - Reshape / view                      (Section 4 of 01_basics.py)
  - Slicing and fancy indexing          (02_indexing.py)
  - Broadcasting                        (Section 1 of 03_operations.py)
  - einsum                              (Sections 3-4 of 03_operations.py)
  - Strides and contiguity              (this file, Sections 1-2)
  - NumPy bridge                        (this file, Section 4)
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Section 1: Strides and memory layout
# ---------------------------------------------------------------------------
# WHY: A tensor's "stride" describes how many elements to skip in the
# underlying flat storage to move one step along each dimension.
#
# For a freshly created (rows, cols) tensor stored in row-major order:
#   stride(0) = cols   (moving one row forward skips `cols` elements)
#   stride(1) = 1      (moving one column forward skips 1 element)
#
# After .transpose(0, 1), the strides are SWAPPED but the storage is
# unchanged — the tensor is now non-contiguous because the logical layout
# no longer matches the physical layout.

# A (4, 6) weight matrix — imagine a small projection layer.
torch.manual_seed(7)
W = torch.randn(4, 6)   # shape: (out_features=4, in_features=6)

# TODO: Inspect the strides of W.
# Assign W.stride() to `w_strides`.
# Expected: (6, 1) — row-major, 6 elements per row.
w_strides = None  # stub — replace with your implementation

# TODO: Transpose W to get shape (6, 4).
# Assign to `W_T`.
# WHY: We'll need W^T for the matmul in Section 3 (input @ W^T).
W_T = torch.zeros(6, 4)  # stub

# TODO: Inspect the strides of W_T.
# Assign W_T.stride() to `wt_strides`.
# Expected: (1, 6) — strides are swapped after transpose; storage unchanged.
wt_strides = None  # stub

# TODO: Check whether W_T is contiguous.
# Assign W_T.is_contiguous() to `wt_is_contiguous`.
# Expected: False — transpose does NOT rearrange memory.
wt_is_contiguous = True  # stub


def check_1():
    try:
        assert w_strides == (6, 1), (
            f"Expected w_strides == (6, 1), got {w_strides}"
        )
        assert W_T.shape == torch.Size([6, 4]), (
            f"Expected W_T shape (6, 4), got {W_T.shape}"
        )
        assert wt_strides == (1, 6), (
            f"Expected wt_strides == (1, 6), got {wt_strides}"
        )
        assert wt_is_contiguous is False, (
            "W_T should NOT be contiguous after transpose"
        )
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: Contiguity checks and .contiguous()
# ---------------------------------------------------------------------------
# WHY: Many PyTorch ops (e.g., .view()) require a contiguous tensor.
# Calling .contiguous() on a non-contiguous tensor allocates new memory
# and copies data so that strides become "normal" (row-major).
# On a contiguous tensor, .contiguous() is a no-op (returns self).
#
# Rule: call .contiguous() only when you need it — it costs a memory
# allocation.  Prefer .reshape() over .view() when contiguity is uncertain,
# because reshape() calls .contiguous() internally if needed.

# TODO: Make W_T contiguous.
# Assign W_T.contiguous() to `W_T_contig`.
W_T_contig = torch.zeros(6, 4)  # stub

# TODO: Check that W_T_contig IS contiguous.
# Assign W_T_contig.is_contiguous() to `wt_contig_flag`.
# Expected: True
wt_contig_flag = False  # stub

# TODO: Inspect the strides of W_T_contig.
# Assign W_T_contig.stride() to `wt_contig_strides`.
# Expected: (4, 1) — now row-major for a (6, 4) matrix.
wt_contig_strides = None  # stub

# TODO: Verify that .contiguous() on an already-contiguous tensor is a no-op.
# Call W.contiguous() and check whether the result shares storage with W
# using the `data_ptr()` method.
# Assign True/False to `same_storage` — True if they share the same pointer.
# WHY: data_ptr() returns the address of the first element; if it's the same
# for both tensors, no copy was made.
same_storage = False  # stub


def check_2():
    try:
        assert wt_contig_flag is True, (
            "W_T_contig should be contiguous"
        )
        assert wt_contig_strides == (4, 1), (
            f"Expected wt_contig_strides == (4, 1), got {wt_contig_strides}"
        )
        assert same_storage is True, (
            ".contiguous() on an already-contiguous tensor should return self (same data_ptr)"
        )
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: Mini inference pipeline
# ---------------------------------------------------------------------------
# WHY: This section chains every Module 01 skill into a realistic forward pass:
#
#   flat token IDs  →  reshape to (batch, seq)
#                   →  embedding lookup (fancy index into an embedding table)
#                   →  linear projection via einsum (batch matmul)
#                   →  add bias via broadcasting
#                   →  threshold mask to find "active" output neurons
#
# Each step mirrors what a real transformer encoder does, just at toy scale.

torch.manual_seed(42)

# Embedding table: vocab_size=20, embed_dim=6.
# WHY: In production this is a learned nn.Embedding; here it's random.
embedding_table = torch.randn(20, 6)   # shape: (vocab=20, embed_dim=6)

# Flat token IDs from the tokenizer — 2 sequences of 5 tokens each.
flat_ids = torch.tensor([3, 7, 1, 15, 9,   # sequence 0
                          2, 8, 4, 11, 0])  # sequence 1

# TODO: Reshape flat_ids into shape (batch=2, seq=5).
# Assign to `token_ids`.
# WHY: The model expects a 2-D (batch, seq) input, not a flat 1-D buffer.
token_ids = torch.zeros(2, 5, dtype=torch.long)  # stub

# TODO: Use fancy indexing to look up embeddings for every token in token_ids.
# Index embedding_table with token_ids to get shape (2, 5, 6).
# Assign to `token_embeddings`.
# WHY: embedding_table[token_ids] gathers one 6-dim row per token ID.
token_embeddings = torch.zeros(2, 5, 6)  # stub

# TODO: Project token_embeddings from embed_dim=6 to out_dim=4 using einsum.
# Use the weight matrix W (shape 4×6) defined in Section 1.
# einsum string: "bse,oe->bso"  (b=batch, s=seq, e=embed_dim, o=out_dim)
# Assign to `projected`.  Shape should be (2, 5, 4).
# WHY: This is a linear layer without bias — one matrix multiply per token.
projected = torch.zeros(2, 5, 4)  # stub

# TODO: Add a bias vector of shape (4,) to `projected` via broadcasting.
# Create the bias as torch.ones(4) * 0.1 and assign to `bias_vec`.
# Then add it to projected and assign to `output`.  Shape: (2, 5, 4).
# WHY: Broadcasting aligns (4,) to (1, 1, 4) and stretches over batch & seq.
bias_vec = torch.zeros(4)  # stub
output = torch.zeros(2, 5, 4)  # stub

# TODO: Create a boolean mask `active_mask` that is True wherever output > 0.
# Shape should be (2, 5, 4).
# WHY: This simulates a ReLU-like activation check — finding which neurons fired.
active_mask = torch.zeros(2, 5, 4, dtype=torch.bool)  # stub

# TODO: Use active_mask to gather all "active" output values into a 1-D tensor.
# Assign to `active_values`.  Shape will be (N,) where N = active_mask.sum().
# WHY: Downstream sparse ops (e.g., top-k, scatter) often work on flat lists
# of active values rather than the full dense tensor.
active_values = torch.zeros(1)  # stub


def check_3():
    try:
        assert token_ids.shape == torch.Size([2, 5]), (
            f"Expected token_ids shape (2, 5), got {token_ids.shape}"
        )
        assert token_ids[0, 0].item() == 3, (
            f"Expected token_ids[0,0] == 3, got {token_ids[0,0].item()}"
        )
        assert token_ids[1, 4].item() == 0, (
            f"Expected token_ids[1,4] == 0, got {token_ids[1,4].item()}"
        )

        assert token_embeddings.shape == torch.Size([2, 5, 6]), (
            f"Expected token_embeddings shape (2, 5, 6), got {token_embeddings.shape}"
        )
        # Spot-check: first token of first sequence is ID 3
        assert torch.allclose(token_embeddings[0, 0], embedding_table[3]), (
            "token_embeddings[0,0] should equal embedding_table[3]"
        )

        assert projected.shape == torch.Size([2, 5, 4]), (
            f"Expected projected shape (2, 5, 4), got {projected.shape}"
        )
        # Verify against torch.matmul reference
        expected_proj = torch.matmul(token_embeddings, W.T)
        assert torch.allclose(projected, expected_proj, atol=1e-5), (
            "projected should equal token_embeddings @ W.T"
        )

        assert output.shape == torch.Size([2, 5, 4]), (
            f"Expected output shape (2, 5, 4), got {output.shape}"
        )
        assert torch.allclose(output, projected + bias_vec, atol=1e-5), (
            "output should equal projected + bias_vec"
        )

        assert active_mask.shape == torch.Size([2, 5, 4]), (
            f"Expected active_mask shape (2, 5, 4), got {active_mask.shape}"
        )
        assert active_mask.dtype == torch.bool, (
            f"Expected dtype bool, got {active_mask.dtype}"
        )

        assert active_values.ndim == 1, (
            f"Expected active_values to be 1-D, got {active_values.ndim}-D"
        )
        assert active_values.shape[0] == active_mask.sum().item(), (
            f"Expected {active_mask.sum().item()} active values, got {active_values.shape[0]}"
        )
        assert torch.all(active_values > 0), (
            "All active_values should be > 0 (they passed the mask)"
        )
        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: NumPy bridge
# ---------------------------------------------------------------------------
# WHY: Many post-processing steps (metrics, serialisation, visualisation)
# live in NumPy-land.  PyTorch tensors and NumPy arrays can share memory
# when the tensor is on CPU and contiguous — no copy needed.
#
# Rules:
#   tensor.numpy()          → zero-copy if tensor is CPU + contiguous
#   np.from_dlpack(tensor)  → zero-copy alternative (newer API)
#   tensor.detach().numpy() → needed if tensor has requires_grad=True
#
# Modifying the NumPy array WILL modify the tensor (shared memory).
# This is a feature, not a bug — but be aware of it.

# TODO: Convert `output` to a NumPy array using .numpy().
# Assign to `output_np`.
# WHY: output is CPU + contiguous (freshly computed), so this is zero-copy.
output_np = np.zeros((2, 5, 4))  # stub

# TODO: Verify that output_np and output share the same memory.
# Check output_np.ctypes.data == output.data_ptr() and assign to `shares_memory`.
# WHY: ctypes.data gives the raw pointer of the NumPy array's buffer;
# data_ptr() gives the same for the PyTorch tensor.  Equal → shared memory.
shares_memory = False  # stub

# TODO: Compute the mean of output_np along the last axis (axis=-1) using NumPy.
# Assign to `output_mean_np`.  Shape should be (2, 5).
# WHY: Downstream code might want per-token scalar summaries rather than
# full 4-dim vectors.
output_mean_np = np.zeros((2, 5))  # stub


def check_4():
    try:
        assert output_np.shape == (2, 5, 4), (
            f"Expected output_np shape (2, 5, 4), got {output_np.shape}"
        )
        assert np.allclose(output_np, output.numpy()), (
            "output_np values should match output.numpy()"
        )

        assert shares_memory is True, (
            "output_np and output should share memory (zero-copy .numpy())"
        )

        assert output_mean_np.shape == (2, 5), (
            f"Expected output_mean_np shape (2, 5), got {output_mean_np.shape}"
        )
        expected_mean = output.numpy().mean(axis=-1)
        assert np.allclose(output_mean_np, expected_mean), (
            "output_mean_np should equal output.numpy().mean(axis=-1)"
        )
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
