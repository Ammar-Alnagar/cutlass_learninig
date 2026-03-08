"""
Module 15 — PyTorch Patterns
Exercise 04 — Distributed Basics (NCCL, Tensor Parallelism)

WHAT YOU'RE BUILDING:
  vLLM and SGLang use distributed training for large models.
  Understanding NCCL, all-reduce, and tensor parallelism is essential
  for multi-GPU serving.

OBJECTIVE:
  - Understand torch.distributed API
  - Implement all-reduce pattern
  - Know tensor parallelism vs pipeline parallelism
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does all-reduce do in distributed training?
# Q2: How does tensor parallelism split a linear layer across GPUs?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List

# Note: This exercise is conceptual — full distributed testing
# requires multiple GPUs/processes. Understand the patterns.

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Understand the distributed initialization pattern.
#              This is how vLLM initializes multi-GPU serving.
# HINT: dist.init_process_group(backend='nccl')

def init_distributed_example():
    """Example distributed initialization pattern."""
    # TODO: Fill in the initialization pattern
    # HINT:
    # dist.init_process_group(backend='nccl')
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # local_rank = int(os.environ.get('LOCAL_RANK', 0))
    pass

# TODO [MEDIUM]: Implement all-reduce for gradient averaging.
#              This is how gradients are synced across GPUs.
# HINT: dist.all_reduce(tensor, op=ReduceOp.SUM)

def all_reduce_gradients(model: nn.Module):
    """Average gradients across all processes."""
    # TODO: Iterate over model parameters, all_reduce each gradient
    # HINT:
    # for param in model.parameters():
    #     if param.grad is not None:
    #         dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    #         param.grad /= dist.get_world_size()
    pass

# TODO [EASY]: Understand tensor parallelism for linear layers.
#              Split weight matrix across GPUs, compute in parallel.
# HINT: 
#   - Column parallel: split output dimension
#   - Row parallel: split input dimension

class TensorParallelLinear(nn.Module):
    """Linear layer split across GPUs (conceptual)."""

    def __init__(self, in_features: int, out_features: int, 
                 world_size: int, rank: int):
        super().__init__()
        # TODO: Split weight matrix across GPUs
        # Each GPU has out_features // world_size output features
        # HINT: 
        # local_out = out_features // world_size
        # self.weight = nn.Parameter(torch.randn(local_out, in_features))
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Local computation, then all-gather results
        # HINT:
        # local_out = F.linear(x, self.weight)
        # Gather local outputs from all GPUs
        pass

# TODO [MEDIUM]: Understand pipeline parallelism pattern.
#              Different layers on different GPUs.
# HINT: Send activations between stages via P2P communication

class PipelineStage(nn.Module):
    """Single stage of a pipelined model."""

    def __init__(self, stage_layers: List[nn.Module], 
                 is_first: bool, is_last: bool):
        super().__init__()
        self.layers = nn.Sequential(*stage_layers)
        self.is_first = is_first
        self.is_last = is_last
        
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO: If first stage, receive or create input
        # TODO: Run through layers
        # TODO: If last stage, return; else send to next stage
        pass

# TODO [EASY]: Know when to use each parallelism strategy.
#              Fill in:
#              - Data parallelism: ______
#              - Tensor parallelism: ______
#              - Pipeline parallelism: ______

def choose_parallelism(model_size: str, num_gpus: int) -> str:
    """Recommend parallelism strategy."""
    # TODO: Return strategy based on model size and GPU count
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What's the communication pattern for tensor vs pipeline parallelism?
# C2: Why does vLLM use tensor parallelism for large model inference?

if __name__ == "__main__":
    print("Distributed patterns (conceptual)...")
    
    print("\nInitialization pattern...")
    init_distributed_example()
    
    print("\nAll-reduce pattern...")
    print("  Pattern: each GPU computes grad, all-reduce sums, divide by world_size")
    
    print("\nTensor parallelism pattern...")
    print("  Pattern: split weight matrix, compute in parallel, all-gather results")
    
    print("\nPipeline parallelism pattern...")
    print("  Pattern: different layers on different GPUs, pass activations")
    
    print("\nStrategy recommendations...")
    print(f"  Small model (7B), 8 GPUs: {choose_parallelism('7B', 8)}")
    print(f"  Large model (70B), 8 GPUs: {choose_parallelism('70B', 8)}")
    print(f"  Huge model (175B), 64 GPUs: {choose_parallelism('175B', 64)}")

    print("\nDone!")
