"""
Module 15 — PyTorch Patterns
Exercise 03 — Memory Management and Inference Modes

WHAT YOU'RE BUILDING:
  Inference optimization requires understanding PyTorch memory:
  inference_mode vs no_grad, memory caching, KV cache patterns.
  vLLM's performance depends on these patterns.

OBJECTIVE:
  - Use inference_mode, no_grad, eval() correctly
  - Manage CUDA memory (cache, empty_cache)
  - Understand memory layout (contiguous, strides)
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between inference_mode and no_grad?
# Q2: When does PyTorch cache CUDA memory? When should you clear it?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
from typing import Tuple

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Compare inference_mode, no_grad, and regular forward.
#              Measure memory and speed differences.
# HINT: 
#   - with torch.inference_mode(): fastest, no grad tracking
#   - with torch.no_grad(): no grad, but allows some ops
#   - regular: full grad tracking

def benchmark_inference_modes(model: nn.Module, x: torch.Tensor) -> Tuple[float, float, float]:
    """Benchmark different inference modes."""
    import time
    
    # TODO: Time regular forward (with grad)
    # TODO: Time with torch.no_grad()
    # TODO: Time with torch.inference_mode()
    # Return times as tuple
    pass

# TODO [MEDIUM]: Understand CUDA memory management.
#              Track memory before/after operations.
# HINT: torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

def analyze_memory_usage():
    """Analyze CUDA memory usage patterns."""
    if not torch.cuda.is_available():
        return
    
    # TODO: Check initial memory
    # HINT: torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0)
    
    # TODO: Allocate tensor, check memory again
    
    # TODO: Delete tensor, check memory (is it freed?)
    
    # TODO: Call empty_cache(), check memory
    pass

# TODO [EASY]: Understand tensor memory layout.
#              Contiguous tensors are faster for kernels.
# HINT: x.is_contiguous(), x.contiguous(), x.stride()

def analyze_tensor_layout():
    """Analyze tensor memory layout."""
    # Create a contiguous tensor
    x = torch.randn(4, 4, 4)
    
    # TODO: Check if contiguous, print strides
    # HINT: x.is_contiguous(), x.stride()
    
    # TODO: Transpose and check again
    # HINT: y = x.transpose(0, 1); y.is_contiguous()
    
    # TODO: Make contiguous again
    # HINT: z = y.contiguous(); z.is_contiguous()
    pass

# TODO [MEDIUM]: Implement a simple KV cache pattern.
#              This is how vLLM stores attention keys/values.
# HINT: Pre-allocate cache, update with indexing

class SimpleKVCache:
    """Simple KV cache for transformer inference."""

    def __init__(self, max_seq_len: int, hidden_dim: int, num_layers: int):
        # TODO: Pre-allocate key and value caches
        # HINT: self.keys = torch.zeros(num_layers, max_seq_len, hidden_dim)
        pass

    def update(self, layer: int, seq_pos: int, 
               key: torch.Tensor, value: torch.Tensor):
        """Update cache at position seq_pos for given layer."""
        # TODO: Store key and value at the right position
        pass

    def get(self, layer: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys/values up to seq_len."""
        # TODO: Return keys and values up to seq_len
        pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How much faster is inference_mode vs no_grad? When use each?
# C2: Why doesn't memory get freed immediately when deleting tensors?

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required for this exercise")
        exit(1)

    print("Benchmarking inference modes...")
    model = nn.Linear(1024, 1024).cuda()
    x = torch.randn(32, 1024, device='cuda')
    times = benchmark_inference_modes(model, x)
    print(f"  Regular: {times[0]*1000:.2f}ms")
    print(f"  no_grad: {times[1]*1000:.2f}ms")
    print(f"  inference_mode: {times[2]*1000:.2f}ms")

    print("\nAnalyzing memory usage...")
    analyze_memory_usage()

    print("\nAnalyzing tensor layout...")
    analyze_tensor_layout()

    print("\nTesting KV cache...")
    cache = SimpleKVCache(max_seq_len=128, hidden_dim=768, num_layers=12)
    for i in range(5):
        key = torch.randn(12, 1, 768)  # (layers, batch, hidden)
        value = torch.randn(12, 1, 768)
        cache.update(0, i, key[0], value[0])  # Simplified
    keys, values = cache.get(0, 5)
    print(f"  Cache shape: keys={keys.shape}, values={values.shape}")

    print("\nDone!")
