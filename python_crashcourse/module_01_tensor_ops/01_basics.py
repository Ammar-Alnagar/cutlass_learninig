"""
Scenario: Preparing a batch of model inputs for an NLP inference server.

You're building a preprocessing step for a transformer-based text classifier.
Raw token IDs arrive as Python lists; you need to convert them into tensors,
inspect their shapes and dtypes, and reshape them so they match the model's
expected input format: (batch_size, seq_len).

This file covers:
  - Section 1: Creating tensors from Python lists (torch.tensor, np.array)
  - Section 2: Tensor shapes and dtypes (.shape, .dtype, .to(), .astype())
  - Section 3: Tensor creation ops (zeros, ones, randn, np equivalents)
  - Section 4: Reshape and view (.reshape() vs .view())
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Section 1: Creating tensors from Python lists
# ---------------------------------------------------------------------------
# WHY: Model inputs almost always start life as Python data structures (lists
# of token IDs, floats from a feature store, etc.).  torch.tensor() and
# np.array() are the two primary on-ramps from Python → tensor world.
# Knowing both matters because many data pipelines mix NumPy preprocessing
# with PyTorch model execution.

# A "batch" here is a list of sequences; each inner list is one sample's
# token IDs.  Shape will be (batch_size=2, seq_len=5).
raw_token_ids = [
    [101, 2054, 2003, 2026, 3793],   # "what is my text" + [CLS]
    [101, 1045, 2293, 4715, 102],    # "I love python" + [CLS]/[SEP]
]

# TODO: Create a PyTorch tensor from raw_token_ids.
# Use torch.tensor() and assign the result to `token_tensor`.
# Hint: torch.tensor([[1,2],[3,4]]) creates a 2-D int64 tensor.
token_tensor = torch.zeros(1)  # stub — replace with your implementation

# TODO: Create a NumPy array from raw_token_ids.
# Use np.array() and assign the result to `token_array`.
token_array = np.zeros(1)  # stub — replace with your implementation


def check_1():
    """Verify Section 1: tensor and array creation."""
    try:
        # Both objects should have the right shape
        assert token_tensor.shape == torch.Size([2, 5]), (
            f"Expected token_tensor shape (2, 5), got {token_tensor.shape}"
        )
        assert token_array.shape == (2, 5), (
            f"Expected token_array shape (2, 5), got {token_array.shape}"
        )
        # Values should match the original Python list
        assert token_tensor[0, 0].item() == 101, (
            f"Expected token_tensor[0,0] == 101, got {token_tensor[0,0].item()}"
        )
        assert token_array[1, 2] == 2293, (
            f"Expected token_array[1,2] == 2293, got {token_array[1,2]}"
        )
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: Tensor shapes and dtypes
# ---------------------------------------------------------------------------
# WHY: Models are strict about dtypes.  Embedding layers expect Long (int64);
# attention masks expect Float (float32); some ops require matching dtypes
# between operands.  Inspecting .shape and .dtype before feeding data into a
# model saves hours of cryptic runtime errors.

# TODO: Inspect the shape of token_tensor.
# Assign token_tensor.shape to `t_shape`.
t_shape = None  # stub

# TODO: Inspect the dtype of token_tensor.
# Assign token_tensor.dtype to `t_dtype`.
t_dtype = None  # stub

# TODO: Cast token_tensor to float32 using .to(torch.float32).
# Assign the result to `token_tensor_float`.
# WHY: Some downstream ops (e.g., matrix multiply) require float, not int.
token_tensor_float = torch.zeros(1)  # stub

# TODO: Cast token_array to float32 using .astype(np.float32).
# Assign the result to `token_array_float`.
token_array_float = np.zeros(1)  # stub


def check_2():
    """Verify Section 2: shapes, dtypes, and casting."""
    try:
        assert t_shape == torch.Size([2, 5]), (
            f"Expected t_shape == torch.Size([2, 5]), got {t_shape}"
        )
        # torch.tensor() infers int64 from Python ints by default
        assert t_dtype == torch.int64, (
            f"Expected t_dtype == torch.int64, got {t_dtype}"
        )
        assert token_tensor_float.dtype == torch.float32, (
            f"Expected float32, got {token_tensor_float.dtype}"
        )
        assert token_array_float.dtype == np.float32, (
            f"Expected np.float32, got {token_array_float.dtype}"
        )
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: Tensor creation ops
# ---------------------------------------------------------------------------
# WHY: You rarely build every tensor from raw data.  Padding masks, attention
# masks, and weight initializations all use creation ops like zeros/ones/randn.
# Knowing the NumPy equivalents matters when bridging data-loading code
# (often NumPy-based) with model code (PyTorch-based).

# TODO: Create a (2, 5) tensor of all zeros using torch.zeros().
# Assign to `padding_mask`.
# WHY: A zero padding mask tells the model to ignore those positions.
padding_mask = torch.zeros(1)  # stub

# TODO: Create a (2, 5) tensor of all ones using torch.ones().
# Assign to `attention_mask`.
# WHY: An all-ones attention mask means "attend to every token".
attention_mask = torch.zeros(1)  # stub

# TODO: Create a (2, 5) tensor of random normal values using torch.randn().
# Assign to `noise`.
# WHY: Random noise is used in data augmentation and dropout simulation.
noise = torch.zeros(1)  # stub

# TODO: Create a (2, 5) NumPy array of zeros using np.zeros().
# Assign to `np_zeros`.
np_zeros = np.zeros(1)  # stub

# TODO: Create a (2, 5) NumPy array of ones using np.ones().
# Assign to `np_ones`.
np_ones = np.zeros(1)  # stub


def check_3():
    """Verify Section 3: creation ops produce correct shapes and values."""
    try:
        assert padding_mask.shape == torch.Size([2, 5]), (
            f"Expected padding_mask shape (2,5), got {padding_mask.shape}"
        )
        assert torch.all(padding_mask == 0), "padding_mask should be all zeros"

        assert attention_mask.shape == torch.Size([2, 5]), (
            f"Expected attention_mask shape (2,5), got {attention_mask.shape}"
        )
        assert torch.all(attention_mask == 1), "attention_mask should be all ones"

        assert noise.shape == torch.Size([2, 5]), (
            f"Expected noise shape (2,5), got {noise.shape}"
        )
        # randn values are random; just check dtype and shape
        assert noise.dtype == torch.float32, (
            f"Expected float32 from randn, got {noise.dtype}"
        )

        assert np_zeros.shape == (2, 5), (
            f"Expected np_zeros shape (2,5), got {np_zeros.shape}"
        )
        assert np.all(np_zeros == 0), "np_zeros should be all zeros"

        assert np_ones.shape == (2, 5), (
            f"Expected np_ones shape (2,5), got {np_ones.shape}"
        )
        assert np.all(np_ones == 1), "np_ones should be all ones"

        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: Reshape and view
# ---------------------------------------------------------------------------
# WHY: Models expect specific tensor layouts.  A flat sequence of 10 token IDs
# might need to become (2, 5) for a batch of 2 sequences, or (1, 2, 5) to add
# a batch dimension.  .reshape() and .view() both change shape without copying
# data — but .view() requires the tensor to be *contiguous* in memory, while
# .reshape() handles non-contiguous tensors by copying if needed.
#
# Rule of thumb:
#   - Use .view() when you know the tensor is contiguous (e.g., freshly created).
#   - Use .reshape() when you're unsure or after ops like .transpose() that
#     break contiguity.

# A flat sequence of 10 token IDs (one long row)
flat_ids = torch.tensor([101, 2054, 2003, 2026, 3793, 101, 1045, 2293, 4715, 102])
# flat_ids.shape == (10,)

# TODO: Use .reshape() to turn flat_ids into shape (2, 5).
# Assign to `reshaped`.
# WHY: reshape() works even if the tensor is non-contiguous.
reshaped = torch.zeros(1)  # stub

# TODO: Use .view() to turn flat_ids into shape (2, 5).
# Assign to `viewed`.
# WHY: view() is slightly faster than reshape() for contiguous tensors because
# it never copies data — it just reinterprets the existing memory layout.
viewed = torch.zeros(1)  # stub

# TODO: Use .reshape() with -1 as a wildcard dimension to turn flat_ids into
# shape (5, -1), letting PyTorch infer the second dimension automatically.
# Assign to `auto_reshaped`.
# WHY: -1 is useful when you know one dimension but want PyTorch to compute
# the other, avoiding manual arithmetic that can silently go wrong.
auto_reshaped = torch.zeros(1)  # stub


def check_4():
    """Verify Section 4: reshape and view produce correct shapes."""
    try:
        assert reshaped.shape == torch.Size([2, 5]), (
            f"Expected reshaped shape (2,5), got {reshaped.shape}"
        )
        assert viewed.shape == torch.Size([2, 5]), (
            f"Expected viewed shape (2,5), got {viewed.shape}"
        )
        # Values should be the same as flat_ids, just re-arranged in 2-D
        assert reshaped[0, 0].item() == 101, (
            f"Expected reshaped[0,0] == 101, got {reshaped[0,0].item()}"
        )
        assert viewed[1, 4].item() == 102, (
            f"Expected viewed[1,4] == 102, got {viewed[1,4].item()}"
        )
        assert auto_reshaped.shape == torch.Size([5, 2]), (
            f"Expected auto_reshaped shape (5,2), got {auto_reshaped.shape}"
        )
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
