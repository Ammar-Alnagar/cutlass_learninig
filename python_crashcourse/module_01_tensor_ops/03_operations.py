"""
Scenario: Fused attention score computation for a transformer inference server.

In a multi-head self-attention layer, the core operation is:

    scores = softmax( (Q @ K^T) / sqrt(d_k) ) @ V

where Q, K, V are (batch, heads, seq_len, d_k) tensors.  Before you can
implement that efficiently you need three building blocks:

  1. Broadcasting -- adding a per-head bias to every position without loops
  2. In-place vs out-of-place ops -- knowing when .add_() is safe vs dangerous
  3. einsum -- expressing matmul, batch matmul, and outer products concisely
  4. einsum for attention -- computing scaled dot-product scores in one line

This file covers:
  - Section 1: Broadcasting -- bias addition to a batch of embeddings
  - Section 2: In-place vs out-of-place ops -- .add_() vs +, autograd hazards
  - Section 3: einsum basics -- matmul, batch matmul, outer product
  - Section 4: einsum for attention -- scaled dot-product attention scores
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Section 1: Broadcasting
# ---------------------------------------------------------------------------
# WHY: Broadcasting lets PyTorch (and NumPy) apply an operation between
# tensors of *different* shapes without copying data.  The rule is:
#
#   Align shapes from the RIGHT.  For each dimension pair:
#     - If sizes match -> fine.
#     - If one size is 1 -> that dimension is "stretched" to match the other.
#     - If sizes differ and neither is 1 -> error.
#
# Example:
#   embeddings : (batch=8, seq=12, d_model=64)
#   bias       :              (d_model=64,)   <- aligned right, broadcast over batch & seq
#
# This is exactly how transformer bias terms work: one bias vector is added
# to every (batch, position) pair without a Python loop.

torch.manual_seed(0)
np.random.seed(0)

# A batch of token embeddings: 8 sequences, 12 tokens each, 64-dim vectors.
# WHY: float32 is the standard dtype for model activations.
embeddings = torch.randn(8, 12, 64)       # shape: (batch=8, seq=12, d_model=64)

# A learned bias vector -- one value per embedding dimension.
# WHY: Shape (64,) broadcasts over the batch and seq dimensions automatically.
bias = torch.randn(64)                    # shape: (d_model=64,)

# A per-head scale vector -- one scale per attention head (8 heads).
# Shape (8, 1, 1) broadcasts over seq and d_model.
head_scales = torch.rand(8, 1, 1) + 0.5  # shape: (heads=8, 1, 1)

# TODO: Add `bias` to `embeddings` using the + operator (out-of-place).
# Assign the result to `biased_embeddings`.
# WHY: Broadcasting aligns (64,) to (1, 1, 64) then stretches over batch & seq.
# Result shape should be (8, 12, 64).
biased_embeddings = torch.zeros(8, 12, 64)  # stub -- replace with your implementation

# TODO: Scale `embeddings` by `head_scales` using the * operator.
# Assign the result to `scaled_embeddings`.
# WHY: (8, 1, 1) * (8, 12, 64) -> each of the 8 "head slices" gets its own
# scalar multiplier, broadcast over all 12 positions and 64 dims.
# Result shape should be (8, 12, 64).
scaled_embeddings = torch.zeros(8, 12, 64)  # stub

# TODO: Using NumPy, add a bias vector of shape (64,) to a (8, 12, 64) array.
# Use `embeddings_np` (already created below) and `bias_np` (already created below).
# Assign the result to `biased_embeddings_np`.  Shape should be (8, 12, 64).
# WHY: NumPy follows the same broadcasting rules as PyTorch.
embeddings_np = embeddings.numpy()           # convert to NumPy (shares memory)
bias_np = bias.numpy()                       # same broadcasting rules apply in NumPy
biased_embeddings_np = np.zeros((8, 12, 64))  # stub


def check_1():
    """Verify Section 1: broadcasting produces correct shapes and values."""
    try:
        assert biased_embeddings.shape == torch.Size([8, 12, 64]), (
            f"Expected (8, 12, 64), got {biased_embeddings.shape}"
        )
        # Every position should have the bias added -- spot-check two elements.
        # WHY: allclose handles floating-point rounding; exact == is fragile.
        assert torch.allclose(biased_embeddings[0, 0], embeddings[0, 0] + bias), (
            "biased_embeddings[0, 0] should equal embeddings[0, 0] + bias"
        )
        assert torch.allclose(biased_embeddings[7, 11], embeddings[7, 11] + bias), (
            "biased_embeddings[7, 11] should equal embeddings[7, 11] + bias"
        )

        assert scaled_embeddings.shape == torch.Size([8, 12, 64]), (
            f"Expected (8, 12, 64), got {scaled_embeddings.shape}"
        )
        # Head 0's scale should multiply every position in head 0.
        assert torch.allclose(scaled_embeddings[0], embeddings[0] * head_scales[0]), (
            "scaled_embeddings[0] should equal embeddings[0] * head_scales[0]"
        )

        assert biased_embeddings_np.shape == (8, 12, 64), (
            f"Expected (8, 12, 64), got {biased_embeddings_np.shape}"
        )
        # NumPy result should match the PyTorch result numerically.
        assert np.allclose(biased_embeddings_np, biased_embeddings.numpy()), (
            "NumPy biased_embeddings_np doesn't match PyTorch biased_embeddings"
        )
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: In-place vs out-of-place ops
# ---------------------------------------------------------------------------
# WHY: PyTorch has two flavors of every arithmetic op:
#
#   Out-of-place (default):  c = a + b   -- allocates a NEW tensor for c
#   In-place:                a.add_(b)   -- modifies a's memory directly
#
# In-place ops save memory (no allocation) but have two important hazards:
#
#   1. Autograd: if `a` is a leaf variable that requires_grad, modifying it
#      in-place corrupts the computation graph.  PyTorch will raise a
#      RuntimeError to protect you -- but only at runtime, not at definition.
#
#   2. Shared views: if `b` is a view of `a` (e.g., b = a[:, 0]), an in-place
#      op on `a` silently changes `b` too, causing hard-to-debug bugs.
#
# Rule of thumb: use in-place ops only on intermediate tensors that are NOT
# part of the autograd graph and have NO views pointing to them.

# A simple (3, 4) activation tensor -- imagine post-linear-layer activations.
activations = torch.randn(3, 4)

# A dropout mask: 1.0 where we keep, 0.0 where we drop.
# WHY: Multiplying by a mask is a common in-place use case in custom layers.
dropout_mask = (torch.rand(3, 4) > 0.3).float()  # ~70% keep rate

# TODO: Compute `out_of_place_result` by adding 1.0 to `activations` using
# the + operator (out-of-place).  `activations` must NOT be modified.
# Assign to `out_of_place_result`.  Shape should be (3, 4).
out_of_place_result = torch.zeros(3, 4)  # stub

# TODO: Make a COPY of `activations` called `activations_copy` using .clone().
# WHY: .clone() creates a new tensor with the same data but independent memory,
# so in-place ops on the copy don't affect the original.
activations_copy = torch.zeros(3, 4)  # stub

# TODO: Apply the dropout mask IN-PLACE to `activations_copy` using .mul_().
# Assign nothing -- .mul_() modifies `activations_copy` directly.
# WHY: In-place mul_ avoids allocating a new tensor for the masked result.
# (Your code here should call activations_copy.mul_(dropout_mask))

# TODO: Demonstrate the autograd hazard.  Create a leaf tensor `leaf` with
# requires_grad=True, then try to do an in-place add on it inside a
# try/except RuntimeError block.  Assign the caught error message (str(e))
# to `inplace_error_msg`.  If no error is raised, assign an empty string.
# WHY: PyTorch forbids in-place ops on leaf tensors that require grad because
# it would destroy the gradient information needed for backprop.
leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
inplace_error_msg = ""  # stub -- replace with try/except block


def check_2():
    """Verify Section 2: in-place vs out-of-place behavior."""
    try:
        # Out-of-place: original activations must be unchanged.
        original_sum = activations.sum().item()
        result_sum = out_of_place_result.sum().item()
        assert out_of_place_result.shape == torch.Size([3, 4]), (
            f"Expected (3, 4), got {out_of_place_result.shape}"
        )
        # out_of_place_result should be activations + 1
        assert torch.allclose(out_of_place_result, activations + 1.0), (
            "out_of_place_result should equal activations + 1.0"
        )
        # activations itself must be unchanged
        assert torch.allclose(activations.sum(), torch.tensor(original_sum)), (
            "activations was modified -- use out-of-place + instead of +="
        )

        # In-place: activations_copy should now have zeros where mask is 0.
        assert activations_copy.shape == torch.Size([3, 4]), (
            f"Expected (3, 4), got {activations_copy.shape}"
        )
        # Wherever dropout_mask is 0, activations_copy must also be 0.
        zero_positions = (dropout_mask == 0)
        assert torch.all(activations_copy[zero_positions] == 0.0), (
            "In-place mul_ should zero out positions where dropout_mask == 0"
        )

        # Autograd hazard: in-place on a leaf with requires_grad must raise.
        assert len(inplace_error_msg) > 0, (
            "Expected a RuntimeError when doing in-place op on a requires_grad leaf"
        )
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: einsum basics
# ---------------------------------------------------------------------------
# WHY: torch.einsum() lets you express any tensor contraction using Einstein
# summation notation -- a compact string that describes which dimensions are
# summed over and which are kept.  The format is:
#
#   torch.einsum("ij,jk->ik", A, B)
#
# Read it as: "for each i and k, sum over j: A[i,j] * B[j,k]".
# This is exactly matrix multiplication.  einsum generalises to any number
# of tensors and dimensions, making it ideal for attention computations.
#
# Common patterns:
#   "ij,jk->ik"    -- matrix multiply (2-D x 2-D)
#   "bij,bjk->bik" -- batch matrix multiply (3-D x 3-D)
#   "i,j->ij"      -- outer product (two 1-D vectors -> 2-D matrix)
#   "bij->bi"      -- sum over last dim (reduce)

torch.manual_seed(1)

# Two matrices for basic matmul.
A = torch.randn(4, 6)   # shape: (m=4, k=6)
B = torch.randn(6, 5)   # shape: (k=6, n=5)

# Two batches of matrices for batch matmul.
# WHY: In attention, Q and K are both (batch, seq, d_k) -- batch matmul
# computes all pairwise dot products in one call.
Q_batch = torch.randn(2, 3, 6)  # shape: (batch=2, seq=3, d_k=6)
K_batch = torch.randn(2, 3, 6)  # shape: (batch=2, seq=3, d_k=6)

# Two vectors for outer product.
u = torch.tensor([1.0, 2.0, 3.0])  # shape: (3,)
v = torch.tensor([4.0, 5.0])       # shape: (2,)

# TODO: Use torch.einsum to compute the matrix product of A and B.
# Assign to `matmul_result`.  Shape should be (4, 5).
# Hint: the einsum string for matmul is "ij,jk->ik".
matmul_result = torch.zeros(4, 5)  # stub

# TODO: Use torch.einsum to compute the batch matrix product of Q_batch and
# K_batch transposed (i.e., Q @ K^T for each batch item).
# Assign to `batch_matmul_result`.  Shape should be (2, 3, 3).
# Hint: "bij,bkj->bik" -- note K is indexed as bkj (j is the shared dim).
batch_matmul_result = torch.zeros(2, 3, 3)  # stub

# TODO: Use torch.einsum to compute the outer product of u and v.
# Assign to `outer_result`.  Shape should be (3, 2).
# Hint: "i,j->ij" -- no shared dims, so nothing is summed.
outer_result = torch.zeros(3, 2)  # stub


def check_3():
    """Verify Section 3: einsum produces correct shapes and values."""
    try:
        assert matmul_result.shape == torch.Size([4, 5]), (
            f"Expected (4, 5), got {matmul_result.shape}"
        )
        # einsum matmul must match torch.mm exactly.
        assert torch.allclose(matmul_result, torch.mm(A, B)), (
            "matmul_result should equal torch.mm(A, B)"
        )

        assert batch_matmul_result.shape == torch.Size([2, 3, 3]), (
            f"Expected (2, 3, 3), got {batch_matmul_result.shape}"
        )
        # einsum batch matmul must match torch.bmm(Q, K^T).
        expected_bmm = torch.bmm(Q_batch, K_batch.transpose(1, 2))
        assert torch.allclose(batch_matmul_result, expected_bmm), (
            "batch_matmul_result should equal torch.bmm(Q_batch, K_batch.transpose(1,2))"
        )

        assert outer_result.shape == torch.Size([3, 2]), (
            f"Expected (3, 2), got {outer_result.shape}"
        )
        # Spot-check: outer[i, j] == u[i] * v[j]
        # u[0]=1.0, v[0]=4.0 -> outer[0,0] = 4.0
        assert abs(outer_result[0, 0].item() - 4.0) < 1e-5, (
            f"outer_result[0, 0] should be 4.0 (u[0]=1 * v[0]=4), got {outer_result[0, 0].item()}"
        )
        # u[2]=3.0, v[1]=5.0 -> outer[2,1] = 15.0
        assert abs(outer_result[2, 1].item() - 15.0) < 1e-5, (
            f"outer_result[2, 1] should be 15.0 (u[2]=3 * v[1]=5), got {outer_result[2, 1].item()}"
        )
        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: einsum for attention
# ---------------------------------------------------------------------------
# WHY: Scaled dot-product attention is the heart of every transformer.
# The formula is:
#
#   scores[b, h, i, j] = sum_k( Q[b, h, i, k] * K[b, h, j, k] ) / sqrt(d_k)
#
# where:
#   b = batch index
#   h = head index
#   i = query position (row of the score matrix)
#   j = key position   (column of the score matrix)
#   k = head dimension (summed over -- the "dot product" axis)
#
# einsum string: "bhik,bhjk->bhij"
#   - b, h, i, j are kept (appear on both sides of ->)
#   - k is summed (appears in inputs but not in output)
#
# Dividing by sqrt(d_k) prevents the dot products from growing so large that
# softmax saturates and gradients vanish.

torch.manual_seed(2)

# Typical small attention setup: batch=2, heads=4, seq=6, d_k=8.
batch, heads, seq, d_k = 2, 4, 6, 8

# Q, K, V tensors -- in practice these come from linear projections of the
# input embeddings.  Here we use random values for the exercise.
Q = torch.randn(batch, heads, seq, d_k)  # shape: (batch=2, heads=4, seq=6, d_k=8)
K = torch.randn(batch, heads, seq, d_k)  # shape: (batch=2, heads=4, seq=6, d_k=8)
V = torch.randn(batch, heads, seq, d_k)  # shape: (batch=2, heads=4, seq=6, d_k=8)

# TODO: Use torch.einsum to compute the raw (unscaled) attention scores.
# The einsum string is "bhik,bhjk->bhij".
# Assign to `raw_scores`.  Shape should be (2, 4, 6, 6).
# WHY: Each score[b, h, i, j] is the dot product of query i with key j,
# measuring how much position i should "attend to" position j.
raw_scores = torch.zeros(batch, heads, seq, seq)  # stub

# TODO: Scale `raw_scores` by dividing by sqrt(d_k).
# Assign to `scaled_scores`.  Shape should be (2, 4, 6, 6).
# WHY: Without scaling, large d_k causes dot products to grow as O(d_k),
# pushing softmax into saturation where gradients are near zero.
# Use d_k ** 0.5 or math.sqrt(d_k) -- both work.
scaled_scores = torch.zeros(batch, heads, seq, seq)  # stub

# TODO: Apply softmax over the LAST dimension (dim=-1) of `scaled_scores`.
# Assign to `attention_weights`.  Shape should be (2, 4, 6, 6).
# WHY: Softmax converts raw scores into a probability distribution over
# key positions, so each row sums to 1.0.
attention_weights = torch.zeros(batch, heads, seq, seq)  # stub

# TODO: Use torch.einsum to compute the attended output by weighting V with
# the attention weights.  The einsum string is "bhij,bhjk->bhik".
# Assign to `attended_output`.  Shape should be (2, 4, 6, 8).
# WHY: Each output position i is a weighted sum of all value vectors,
# where the weights come from how much i attends to each position j.
attended_output = torch.zeros(batch, heads, seq, d_k)  # stub


def check_4():
    """Verify Section 4: einsum attention scores and output."""
    try:
        assert raw_scores.shape == torch.Size([batch, heads, seq, seq]), (
            f"Expected ({batch}, {heads}, {seq}, {seq}), got {raw_scores.shape}"
        )
        # Verify against torch.matmul as a reference.
        # WHY: torch.matmul on 4-D tensors does batch matmul over the last two dims.
        expected_raw = torch.matmul(Q, K.transpose(-2, -1))
        assert torch.allclose(raw_scores, expected_raw, atol=1e-5), (
            "raw_scores should equal torch.matmul(Q, K.transpose(-2, -1))"
        )

        assert scaled_scores.shape == torch.Size([batch, heads, seq, seq]), (
            f"Expected ({batch}, {heads}, {seq}, {seq}), got {scaled_scores.shape}"
        )
        assert torch.allclose(scaled_scores, raw_scores / (d_k ** 0.5), atol=1e-5), (
            "scaled_scores should equal raw_scores / sqrt(d_k)"
        )

        assert attention_weights.shape == torch.Size([batch, heads, seq, seq]), (
            f"Expected ({batch}, {heads}, {seq}, {seq}), got {attention_weights.shape}"
        )
        # Each row of attention_weights must sum to 1.0 (it's a probability dist).
        row_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            "Each row of attention_weights should sum to 1.0 (softmax output)"
        )
        # All weights must be non-negative (softmax output is always >= 0).
        assert torch.all(attention_weights >= 0), (
            "attention_weights should be non-negative"
        )

        assert attended_output.shape == torch.Size([batch, heads, seq, d_k]), (
            f"Expected ({batch}, {heads}, {seq}, {d_k}), got {attended_output.shape}"
        )
        # Verify against torch.matmul reference.
        expected_output = torch.matmul(attention_weights, V)
        assert torch.allclose(attended_output, expected_output, atol=1e-5), (
            "attended_output should equal torch.matmul(attention_weights, V)"
        )
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
