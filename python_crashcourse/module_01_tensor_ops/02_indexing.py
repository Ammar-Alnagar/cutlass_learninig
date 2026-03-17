"""
Scenario: Extracting token embeddings from a batch for an NLP inference server.

Your transformer model produces a 3-D embedding tensor of shape
(batch=4, seq_len=8, embed_dim=16) — one embedding vector per token, per
sequence, per batch item.  Before passing embeddings downstream (e.g., to a
classification head or a retrieval index), you often need to:

  - Pull out specific sequences or token positions (slicing)
  - Gather embeddings for a hand-picked list of token indices (fancy indexing)
  - Mask out padding tokens or low-confidence positions (boolean masks)
  - Combine all three techniques in a single pipeline step

This file covers:
  - Section 1: Basic slicing — rows, columns, and ranges
  - Section 2: Fancy indexing — selecting specific token positions
  - Section 3: Boolean masks — thresholding and padding masks
  - Section 4: Combined indexing — slicing + fancy indexing + masks together
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Shared data: a (batch=4, seq_len=8, embed_dim=16) embedding tensor
# ---------------------------------------------------------------------------
# WHY: torch.manual_seed() makes the random values reproducible so the
# self-checks can assert exact shapes and, where needed, exact values.
torch.manual_seed(42)
np.random.seed(42)

# embeddings[b, t, :] is the 16-dim embedding for token t in batch item b.
embeddings = torch.randn(4, 8, 16)   # shape: (batch=4, seq_len=8, embed_dim=16)

# A NumPy mirror — useful for practicing the same indexing patterns in NumPy.
embeddings_np = embeddings.numpy()   # shares memory with embeddings (no copy)


# ---------------------------------------------------------------------------
# Section 1: Basic slicing
# ---------------------------------------------------------------------------
# WHY: Slicing uses Python's start:stop:step syntax extended to multiple
# dimensions.  It always returns a *view* (no data copy), so it's O(1) and
# memory-efficient — critical when working with large embedding matrices.
#
# Key rules:
#   tensor[i]        → removes dimension i (scalar index)
#   tensor[i:j]      → keeps dimension, selects rows i..j-1
#   tensor[:, i:j]   → selects columns i..j-1 across all rows
#   tensor[..., i:j] → Ellipsis (...) means "all preceding dimensions"

# TODO: Extract the embedding matrix for the FIRST batch item only.
# Result shape should be (8, 16) — all 8 tokens, all 16 dims.
# Hint: index the batch dimension with a scalar (removes that dimension).
first_item_embeddings = torch.zeros(8, 16)  # stub — replace with your implementation

# TODO: Extract the embeddings for the FIRST TWO batch items.
# Result shape should be (2, 8, 16).
# Hint: use a slice 0:2 on the batch dimension.
first_two_items = torch.zeros(2, 8, 16)  # stub

# TODO: Extract the embeddings for tokens 2 through 5 (inclusive) across ALL
# batch items.  Result shape should be (4, 4, 16).
# Hint: slice the seq_len dimension with 2:6.
middle_tokens = torch.zeros(4, 4, 16)  # stub

# TODO: Extract only the FIRST 4 embedding dimensions for ALL tokens in ALL
# batch items.  Result shape should be (4, 8, 4).
# Hint: use the Ellipsis (...) to skip batch and seq_len, then slice embed_dim.
first_four_dims = torch.zeros(4, 8, 4)  # stub

# TODO: Do the same "first 4 embedding dims" extraction using NumPy on
# embeddings_np.  Assign to `first_four_dims_np`.
# Result shape should be (4, 8, 4).
first_four_dims_np = np.zeros((4, 8, 4))  # stub


def check_1():
    """Verify Section 1: basic slicing produces correct shapes and values."""
    try:
        assert first_item_embeddings.shape == torch.Size([8, 16]), (
            f"Expected (8, 16), got {first_item_embeddings.shape}"
        )
        # Scalar indexing should give the same values as embeddings[0]
        assert torch.allclose(first_item_embeddings, embeddings[0]), (
            "first_item_embeddings values don't match embeddings[0]"
        )

        assert first_two_items.shape == torch.Size([2, 8, 16]), (
            f"Expected (2, 8, 16), got {first_two_items.shape}"
        )

        assert middle_tokens.shape == torch.Size([4, 4, 16]), (
            f"Expected (4, 4, 16), got {middle_tokens.shape}"
        )
        # Token index 2 in the original should be index 0 in the slice
        assert torch.allclose(middle_tokens[:, 0, :], embeddings[:, 2, :]), (
            "middle_tokens[:, 0, :] should equal embeddings[:, 2, :]"
        )

        assert first_four_dims.shape == torch.Size([4, 8, 4]), (
            f"Expected (4, 8, 4), got {first_four_dims.shape}"
        )
        assert torch.allclose(first_four_dims, embeddings[..., :4]), (
            "first_four_dims values don't match embeddings[..., :4]"
        )

        assert first_four_dims_np.shape == (4, 8, 4), (
            f"Expected (4, 8, 4), got {first_four_dims_np.shape}"
        )
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: Fancy indexing
# ---------------------------------------------------------------------------
# WHY: Fancy indexing (also called advanced indexing) lets you select an
# *arbitrary* set of positions using an index tensor or list — not just
# contiguous ranges.  This is essential for:
#   - Gathering the [CLS] token and specific named entities from a sequence
#   - Selecting the top-k attended tokens after computing attention scores
#   - Reordering a batch by confidence score
#
# Unlike slicing, fancy indexing always returns a COPY (new memory), because
# the selected elements may not be contiguous in memory.

# Suppose the model flagged these token positions as "important" in each
# sequence (e.g., named entities or high-attention tokens).
important_positions = torch.tensor([0, 3, 7])  # token indices to extract

# TODO: Use fancy indexing to extract the embeddings at `important_positions`
# for ALL batch items.  Result shape should be (4, 3, 16).
# Hint: index the seq_len dimension with the tensor `important_positions`.
important_embeddings = torch.zeros(4, 3, 16)  # stub

# TODO: Extract the embeddings for batch items 1 and 3 ONLY (not 0 or 2).
# Result shape should be (2, 8, 16).
# Hint: index the batch dimension with a Python list [1, 3].
selected_batch_items = torch.zeros(2, 8, 16)  # stub

# TODO: Use fancy indexing to extract a specific (batch_item, token) pair for
# each entry in a "gather list".  Given:
#   batch_indices  = [0, 1, 2, 3]   (one per batch item)
#   token_indices  = [0, 3, 5, 7]   (different token per batch item)
# Extract the 16-dim embedding for each (batch_item, token) pair.
# Result shape should be (4, 16).
# Hint: index with two lists/tensors simultaneously: embeddings[batch_idx, token_idx]
batch_indices = [0, 1, 2, 3]
token_indices = [0, 3, 5, 7]
paired_embeddings = torch.zeros(4, 16)  # stub

# TODO: Repeat the `important_positions` fancy index using NumPy on
# embeddings_np.  Assign to `important_embeddings_np`.
# Result shape should be (4, 3, 16).
important_embeddings_np = np.zeros((4, 3, 16))  # stub


def check_2():
    """Verify Section 2: fancy indexing selects the right elements."""
    try:
        assert important_embeddings.shape == torch.Size([4, 3, 16]), (
            f"Expected (4, 3, 16), got {important_embeddings.shape}"
        )
        # The first selected token (position 0) should match embeddings[:, 0, :]
        assert torch.allclose(important_embeddings[:, 0, :], embeddings[:, 0, :]), (
            "important_embeddings[:, 0, :] should equal embeddings[:, 0, :]"
        )
        # The last selected token (position 7) should match embeddings[:, 7, :]
        assert torch.allclose(important_embeddings[:, 2, :], embeddings[:, 7, :]), (
            "important_embeddings[:, 2, :] should equal embeddings[:, 7, :]"
        )

        assert selected_batch_items.shape == torch.Size([2, 8, 16]), (
            f"Expected (2, 8, 16), got {selected_batch_items.shape}"
        )
        assert torch.allclose(selected_batch_items[0], embeddings[1]), (
            "selected_batch_items[0] should equal embeddings[1]"
        )
        assert torch.allclose(selected_batch_items[1], embeddings[3]), (
            "selected_batch_items[1] should equal embeddings[3]"
        )

        assert paired_embeddings.shape == torch.Size([4, 16]), (
            f"Expected (4, 16), got {paired_embeddings.shape}"
        )
        # Spot-check: first pair is (batch=0, token=0)
        assert torch.allclose(paired_embeddings[0], embeddings[0, 0, :]), (
            "paired_embeddings[0] should equal embeddings[0, 0, :]"
        )
        # Spot-check: last pair is (batch=3, token=7)
        assert torch.allclose(paired_embeddings[3], embeddings[3, 7, :]), (
            "paired_embeddings[3] should equal embeddings[3, 7, :]"
        )

        assert important_embeddings_np.shape == (4, 3, 16), (
            f"Expected (4, 3, 16), got {important_embeddings_np.shape}"
        )
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: Boolean masks
# ---------------------------------------------------------------------------
# WHY: Boolean masks let you select elements based on a *condition* rather
# than explicit indices.  Two common ML use cases:
#
#   1. Threshold masking: keep only embeddings whose L2 norm exceeds a
#      threshold (e.g., filter out near-zero "dead" embeddings).
#
#   2. Padding masks: transformer inputs are padded to a fixed length with a
#      special PAD token (id=0).  A boolean mask marks which positions are
#      real tokens vs. padding so the model can ignore padding in attention.
#
# Important: boolean indexing on a multi-dim tensor *flattens* the selected
# dimension.  embeddings[mask] where mask.shape==(4,8) returns shape (N, 16)
# where N is the number of True entries — not (4, 8, 16).

# --- 3a: Threshold mask on embedding norms ---
# Compute the L2 norm of each token embedding: shape (4, 8)
# WHY: torch.norm(..., dim=-1) reduces the last dimension (embed_dim),
# giving one scalar norm per (batch, token) pair.
token_norms = torch.norm(embeddings, dim=-1)   # shape: (4, 8)

# TODO: Create a boolean mask `high_norm_mask` that is True wherever
# token_norms > 4.0.  Shape should be (4, 8).
# Hint: use a simple comparison operator on token_norms.
high_norm_mask = torch.zeros(4, 8, dtype=torch.bool)  # stub

# TODO: Use `high_norm_mask` to select the embeddings of high-norm tokens.
# Assign to `high_norm_embeddings`.  Shape will be (N, 16) where N is the
# number of True entries in high_norm_mask.
# Hint: index embeddings directly with the 2-D boolean mask.
high_norm_embeddings = torch.zeros(1, 16)  # stub

# --- 3b: Padding mask ---
# Simulate a batch of token IDs where 0 means PAD.
# Shape: (batch=4, seq_len=8)
token_ids = torch.tensor([
    [101, 2054, 2003,    0,    0,    0,    0,    0],  # 3 real tokens, 5 PAD
    [101, 1045, 2293, 4715,  102,    0,    0,    0],  # 5 real tokens, 3 PAD
    [101, 7592, 1010, 2026, 2171, 2003, 2026,  102],  # 8 real tokens, 0 PAD
    [101, 2054,    0,    0,    0,    0,    0,    0],  # 2 real tokens, 6 PAD
])

# TODO: Create a boolean mask `real_token_mask` that is True wherever
# token_ids != 0 (i.e., real tokens, not padding).  Shape: (4, 8).
real_token_mask = torch.zeros(4, 8, dtype=torch.bool)  # stub

# TODO: Use `real_token_mask` to extract only the real-token embeddings.
# Assign to `real_token_embeddings`.  Shape will be (M, 16) where M is the
# total number of non-PAD tokens across the whole batch.
# WHY: This is how you compute loss only over real tokens — you gather them
# first, then pass the flat (M, 16) tensor to your loss function.
real_token_embeddings = torch.zeros(1, 16)  # stub

# TODO: Count the number of real tokens per batch item using the mask.
# Assign to `tokens_per_item` — a 1-D tensor of shape (4,) with integer counts.
# Hint: .sum(dim=1) sums along the seq_len dimension.
tokens_per_item = torch.zeros(4, dtype=torch.long)  # stub


def check_3():
    """Verify Section 3: boolean masks select the right elements."""
    try:
        assert high_norm_mask.shape == torch.Size([4, 8]), (
            f"Expected high_norm_mask shape (4, 8), got {high_norm_mask.shape}"
        )
        assert high_norm_mask.dtype == torch.bool, (
            f"Expected dtype bool, got {high_norm_mask.dtype}"
        )
        # Every selected embedding should actually have norm > 4.0
        selected_norms = torch.norm(high_norm_embeddings, dim=-1)
        assert torch.all(selected_norms > 4.0), (
            "Some selected embeddings have norm <= 4.0"
        )

        assert real_token_mask.shape == torch.Size([4, 8]), (
            f"Expected real_token_mask shape (4, 8), got {real_token_mask.shape}"
        )
        assert real_token_mask.dtype == torch.bool, (
            f"Expected dtype bool, got {real_token_mask.dtype}"
        )
        # PAD positions (token_id == 0) must be False
        assert not real_token_mask[0, 3], (
            "Position (0, 3) is PAD (id=0) but mask is True"
        )
        # Real token positions must be True
        assert real_token_mask[0, 0], (
            "Position (0, 0) is a real token but mask is False"
        )

        # Total real tokens: 3 + 5 + 8 + 2 = 18
        assert real_token_embeddings.shape == torch.Size([18, 16]), (
            f"Expected real_token_embeddings shape (18, 16), got {real_token_embeddings.shape}"
        )

        assert tokens_per_item.shape == torch.Size([4]), (
            f"Expected tokens_per_item shape (4,), got {tokens_per_item.shape}"
        )
        expected_counts = torch.tensor([3, 5, 8, 2])
        assert torch.equal(tokens_per_item, expected_counts), (
            f"Expected token counts [3, 5, 8, 2], got {tokens_per_item.tolist()}"
        )
        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: Combined indexing
# ---------------------------------------------------------------------------
# WHY: Real inference pipelines rarely use just one indexing technique.  A
# typical pattern is:
#   1. Slice to a subset of batch items (slicing)
#   2. Gather specific token positions from that subset (fancy indexing)
#   3. Filter out low-quality embeddings using a threshold (boolean mask)
#
# Chaining these operations is idiomatic PyTorch and avoids materialising
# large intermediate tensors.

# We'll work with the first two batch items only.
# WHY: Slicing first reduces the tensor size before the more expensive
# fancy-index and mask operations.
sub_batch = embeddings[:2]   # shape: (2, 8, 16)  — a view, no copy

# TODO: From `sub_batch`, use fancy indexing to extract tokens at positions
# [1, 4, 6].  Assign to `sub_selected`.  Shape should be (2, 3, 16).
sub_selected = torch.zeros(2, 3, 16)  # stub

# TODO: Compute the L2 norm of each embedding in `sub_selected`.
# Assign to `sub_norms`.  Shape should be (2, 3).
# Hint: torch.norm(..., dim=-1) reduces the embed_dim axis.
sub_norms = torch.zeros(2, 3)  # stub

# TODO: Create a boolean mask `quality_mask` that is True wherever
# sub_norms > 3.5.  Shape should be (2, 3).
quality_mask = torch.zeros(2, 3, dtype=torch.bool)  # stub

# TODO: Apply `quality_mask` to `sub_selected` to keep only high-quality
# embeddings.  Assign to `quality_embeddings`.  Shape will be (K, 16) where
# K is the number of True entries in quality_mask.
quality_embeddings = torch.zeros(1, 16)  # stub

# TODO: Using NumPy, replicate the full pipeline on embeddings_np:
#   a) Slice to the first two batch items → shape (2, 8, 16)
#   b) Fancy-index tokens [1, 4, 6]       → shape (2, 3, 16)
#   c) Compute L2 norms (np.linalg.norm with axis=-1) → shape (2, 3)
#   d) Build a boolean mask where norms > 3.5
#   e) Apply the mask to get quality embeddings → shape (K, 16)
# Assign the final result to `quality_embeddings_np`.
quality_embeddings_np = np.zeros((1, 16))  # stub


def check_4():
    """Verify Section 4: combined slicing + fancy indexing + boolean mask."""
    try:
        assert sub_selected.shape == torch.Size([2, 3, 16]), (
            f"Expected sub_selected shape (2, 3, 16), got {sub_selected.shape}"
        )
        # Token position 1 in sub_batch should be index 0 in sub_selected
        assert torch.allclose(sub_selected[:, 0, :], sub_batch[:, 1, :]), (
            "sub_selected[:, 0, :] should equal sub_batch[:, 1, :]"
        )

        assert sub_norms.shape == torch.Size([2, 3]), (
            f"Expected sub_norms shape (2, 3), got {sub_norms.shape}"
        )
        # Norms must be non-negative
        assert torch.all(sub_norms >= 0), "Norms should be non-negative"

        assert quality_mask.shape == torch.Size([2, 3]), (
            f"Expected quality_mask shape (2, 3), got {quality_mask.shape}"
        )
        assert quality_mask.dtype == torch.bool, (
            f"Expected dtype bool, got {quality_mask.dtype}"
        )

        # quality_embeddings should have embed_dim=16
        assert quality_embeddings.shape[-1] == 16, (
            f"Expected last dim 16, got {quality_embeddings.shape[-1]}"
        )
        # Every kept embedding must satisfy the norm threshold
        kept_norms = torch.norm(quality_embeddings, dim=-1)
        assert torch.all(kept_norms > 3.5), (
            "Some quality_embeddings have norm <= 3.5"
        )
        # Count should match the number of True entries in quality_mask
        assert quality_embeddings.shape[0] == quality_mask.sum().item(), (
            f"Expected {quality_mask.sum().item()} embeddings, "
            f"got {quality_embeddings.shape[0]}"
        )

        # NumPy result should have the same count and last dim
        assert quality_embeddings_np.shape[-1] == 16, (
            f"Expected last dim 16 for NumPy result, got {quality_embeddings_np.shape[-1]}"
        )
        assert quality_embeddings_np.shape[0] == quality_mask.sum().item(), (
            f"NumPy result count {quality_embeddings_np.shape[0]} doesn't match "
            f"expected {quality_mask.sum().item()}"
        )
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
