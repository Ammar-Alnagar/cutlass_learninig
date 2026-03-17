"""
Module 06 — Tensor Indexing
Exercise 01 — Slicing

WHAT YOU'RE BUILDING:
  Tensor slicing is fundamental for kernel work — extracting tiles,
  selecting batches, slicing attention heads. Understanding strides
  and views vs copies is essential for efficient kernels.

OBJECTIVE:
  - Master basic tensor slicing syntax
  - Understand view vs copy semantics
  - Practice with multi-dimensional indexing
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between t[0] and t[0:1]?
# Q2: Does t[:, :10] create a copy or a view?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch

# Create a test tensor
t = torch.arange(24).reshape(4, 6)
print(f"Original tensor shape: {t.shape}")
print(f"Original tensor:\n{t}\n")

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Extract the first row (shape: [6]).
# HINT: t[0] or t[0, :]


def get_first_row(tensor: torch.Tensor) -> torch.Tensor:
    """Extract first row."""
    # TODO: implement
    return tensor[0]


# TODO [EASY]: Extract the first column (shape: [4]).
# HINT: t[:, 0]


def get_first_column(tensor: torch.Tensor) -> torch.Tensor:
    """Extract first column."""
    return tensor[:, 0]


# TODO [EASY]: Extract a 2x3 submatrix from top-left.
# HINT: t[0:2, 0:3] or t[:2, :3]


def get_top_left_submatrix(tensor: torch.Tensor) -> torch.Tensor:
    """Extract 2x3 submatrix from top-left."""
    return tensor[:2, :3]


# TODO [MEDIUM]: Extract every other row (rows 0, 2).
# HINT: t[::2, :] or t[0::2]


def get_even_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Extract every other row."""
    return tensor[::2, :]


# TODO [EASY]: Extract last 2 columns.
# HINT: t[:, -2:]


def get_last_two_columns(tensor: torch.Tensor) -> torch.Tensor:
    """Extract last 2 columns."""
    return tensor[:, -2:]


# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What's the shape of t[0] vs t[0:1]?
# C2: How would you extract a tile for a blocked matmul kernel?

if __name__ == "__main__":
    print("Testing slicing operations...\n")

    print(f"First row: {get_first_row(t)}\n")
    print(f"First column: {get_first_column(t)}\n")
    print(f"Top-left 2x3:\n{get_top_left_submatrix(t)}\n")
    print(f"Even rows:\n{get_even_rows(t)}\n")
    print(f"Last 2 columns:\n{get_last_two_columns(t)}\n")

    print("Done!")
