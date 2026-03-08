"""
Project 02 — MoE Grouped GEMM
File: moe_gemm_FILL_IN.py

Implement Mixture of Experts (MoE) with Grouped GEMM for efficient expert routing.
"""

import torch
import cutlass
import time
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class MoEConfig:
    num_experts: int
    hidden_size: int
    expert_width: int
    top_k: int
    total_tokens: int
    dtype: torch.dtype


config = MoEConfig(
    num_experts=8,
    hidden_size=1024,
    expert_width=2048,
    top_k=2,  # Each token goes to top-2 experts
    total_tokens=4096,
    dtype=torch.float16,
)

device = torch.device("cuda")

print("=" * 60)
print("MoE Grouped GEMM Implementation")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  Experts: {config.num_experts}")
print(f"  Hidden: {config.hidden_size}, Expert width: {config.expert_width}")
print(f"  Top-K: {config.top_k}")
print(f"  Total tokens: {config.total_tokens}")


# ==============================================================================
# EXPERT ROUTING
# ==============================================================================

def router_network(hidden: torch.Tensor, num_experts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple router network: linear + softmax → expert scores.
    
    Returns:
        expert_scores: [tokens, num_experts]
        expert_weights: [tokens, top_k]
    """
    # Linear projection to expert logits
    router_weights = torch.randn(hidden.shape[-1], num_experts, 
                                  dtype=hidden.dtype, device=hidden.device)
    logits = torch.matmul(hidden, router_weights)
    
    # Softmax → expert probabilities
    scores = torch.softmax(logits, dim=-1)
    
    # Top-K selection
    topk_values, topk_indices = torch.topk(scores, config.top_k, dim=-1)
    
    return topk_indices, topk_values


def group_tokens_by_expert(tokens: torch.Tensor, 
                           expert_indices: torch.Tensor,
                           num_experts: int) -> Tuple[list, list]:
    """
    Group tokens by their assigned experts.
    
    Returns:
        grouped_inputs: List of tensors, one per expert
        expert_counts: Number of tokens per expert
    """
    grouped_inputs = []
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=tokens.device)
    
    for expert_id in range(num_experts):
        # Find tokens assigned to this expert
        mask = (expert_indices == expert_id)
        assigned_tokens = mask.sum().item()
        expert_counts[expert_id] = assigned_tokens
        
        # Gather tokens (simplified - in practice need proper indexing)
        if assigned_tokens > 0:
            # Get indices of tokens assigned to this expert
            token_indices = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            grouped_inputs.append(tokens[token_indices])
        else:
            grouped_inputs.append(torch.zeros(0, tokens.shape[-1], 
                                               dtype=tokens.dtype, device=tokens.device))
    
    return grouped_inputs, expert_counts


# ==============================================================================
# CREATE TEST DATA
# ==============================================================================

# Input tokens
tokens = torch.randn(config.total_tokens, config.hidden_size, 
                     dtype=config.dtype, device=device)

# Run router
expert_indices, expert_weights = router_network(tokens, config.num_experts)

print(f"\nExpert assignment:")
for i in range(config.num_experts):
    count = (expert_indices == i).sum().item()
    print(f"  Expert {i}: {count} tokens")

# Group tokens by expert
grouped_inputs, expert_counts = group_tokens_by_expert(
    tokens, expert_indices, config.num_experts
)

# Create expert weights
expert_weights = torch.randn(config.num_experts, config.hidden_size, config.expert_width,
                              dtype=config.dtype, device=device)


# ==============================================================================
# FILL IN: MoE with Grouped GEMM
# ==============================================================================

print("\n" + "=" * 60)
print("MoE with Grouped GEMM")
print("=" * 60)

# TODO [HARD]: Implement MoE forward pass with Grouped GEMM
#
# Steps:
# 1. Group tokens by expert (done above)
# 2. Create problem descriptors for each expert's GEMM
# 3. Run GroupedGEMM to process all experts in parallel
# 4. Scatter outputs back to original token positions
# 5. Combine with expert weights

# TODO: Create problem descriptors
# problems = []
# for i in range(config.num_experts):
#     M = grouped_inputs[i].shape[0]  # tokens for this expert
#     K = config.hidden_size
#     N = config.expert_width
#     problems.append(cutlass.GemmCoord(M, K, N))

# TODO: Run GroupedGEMM
# grouped_plan = cutlass.op.GroupedGemm(
#     element=cutlass.float16,
#     layout=cutlass.LayoutType.RowMajor,
# )
# expert_outputs = grouped_plan.run(grouped_inputs, expert_weights, problems)

# Placeholder: naive loop (replace with GroupedGEMM)
expert_outputs = []
for i in range(config.num_experts):
    if grouped_inputs[i].shape[0] > 0:
        output = torch.mm(grouped_inputs[i], expert_weights[i])
        expert_outputs.append(output)
    else:
        expert_outputs.append(torch.zeros(0, config.expert_width, 
                                           dtype=config.dtype, device=device))

print(f"\nProcessed {config.num_experts} experts")
print(f"Total output tokens: {sum(o.shape[0] for o in expert_outputs)}")


# ==============================================================================
# SCATTER OUTPUTS
# ==============================================================================

def scatter_outputs(expert_outputs: list, expert_indices: torch.Tensor,
                    expert_weights: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """
    Scatter expert outputs back to original token positions.
    
    Each token's output is weighted sum of its assigned experts' outputs.
    """
    final_output = torch.zeros(total_tokens, expert_outputs[0].shape[-1],
                               dtype=expert_outputs[0].dtype, device=expert_outputs[0].device)
    
    # For each token, combine outputs from its top-K experts
    for token_idx in range(total_tokens):
        for k in range(config.top_k):
            expert_id = expert_indices[token_idx, k].item()
            weight = expert_weights[token_idx, k]
            # Add weighted output (simplified - needs proper indexing)
    
    return final_output


final_output = scatter_outputs(expert_outputs, expert_indices, expert_weights, 
                                config.total_tokens)
print(f"\nFinal output shape: {final_output.shape}")


# ==============================================================================
# BENCHMARK: GroupedGEMM vs Naive Loop
# ==============================================================================

def benchmark_naive_moe(grouped_inputs, expert_weights, num_warmup=10, num_iters=50):
    """Benchmark naive MoE (loop over experts)."""
    # Warmup
    for _ in range(num_warmup):
        outputs = []
        for i in range(len(grouped_inputs)):
            if grouped_inputs[i].shape[0] > 0:
                outputs.append(torch.mm(grouped_inputs[i], expert_weights[i]))
        torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        outputs = []
        for i in range(len(grouped_inputs)):
            if grouped_inputs[i].shape[0] > 0:
                outputs.append(torch.mm(grouped_inputs[i], expert_weights[i]))
        torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_grouped_moe(grouped_inputs, expert_weights, num_warmup=10, num_iters=50):
    """Benchmark GroupedGEMM MoE."""
    # Placeholder (implement with GroupedGEMM)
    return benchmark_naive_moe(grouped_inputs, expert_weights, num_warmup, num_iters)


print("\n" + "=" * 60)
print("Benchmark")
print("=" * 60)

naive_latency = benchmark_naive_moe(grouped_inputs, expert_weights)
grouped_latency = benchmark_grouped_moe(grouped_inputs, expert_weights)

print(f"\nNaive MoE (loop):    {naive_latency:.3f} ms")
print(f"GroupedGEMM MoE:     {grouped_latency:.3f} ms")

if grouped_latency > 0:
    speedup = naive_latency / grouped_latency
    print(f"\nSpeedup: {speedup:.2f}×")
    print(f"Target: 2× speedup vs naive loop")

# Tokens per second
total_token_expert_pairs = config.total_tokens * config.top_k
tokens_per_sec = total_token_expert_pairs / (grouped_latency * 1e-3)
print(f"\nThroughput: {tokens_per_sec/1e6:.2f}M token-expert pairs/sec")


# ==============================================================================
# NEXT STEPS
# ==============================================================================

print("\n" + "=" * 60)
print("Next Steps")
print("=" * 60)
print("""
To complete the MoE implementation:

1. Implement proper token gathering
   - Use advanced indexing for token assignment
   - Handle variable tokens per expert

2. Use GroupedGEMM
   - Create problem descriptors
   - Run cutlass.op.GroupedGemm

3. Implement scattering
   - Combine expert outputs with weights
   - Handle top-K routing

4. Add load balancing
   - Auxiliary loss for balanced routing
   - Capacity factor handling

5. Optimize
   - FP8 quantization for experts
   - Kernel fusion (router + gather)
""")
