"""
Module 07 — Blackwell Features
Exercise 03 — Blockscaled FP4 GEMM (SM103 Thor)

LEVEL: 2 (CuTe DSL custom kernel)

WHAT YOU'RE BUILDING:
  Blockscaled FP4 GEMM for SM103 (Thor GPU) — the cutting-edge 4-bit 
  quantization pattern for extreme inference optimization. SM103 adds 
  microscaling format support (MXFP4) for better accuracy.

OBJECTIVE:
  - Understand SM100 vs SM103 differences
  - Implement blockscaled FP4 GEMM
  - Compare NVFP4 vs MXFP4 formats
  - Learn microscaling quantization

NOTE: Requires CUDA 13.0+ and SM103 (Thor, RTX 5090)
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time
import math
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What's the difference between SM100 and SM103?

# Q2: What is MXFP4? How does it differ from NVFP4?

# Q3: What's the accuracy vs speed trade-off for FP4?


# ==============================================================================
# SETUP
# ==============================================================================

# Check GPU capability
def check_sm103_support():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability()
        sm = compute_cap[0] * 10 + compute_cap[1]
        is_sm103 = sm >= 103
        return is_sm103, gpu_name, sm
    return False, "No GPU", 0


sm103_supported, gpu_name, sm_version = check_sm103_support()

print("=" * 60)
print("Blockscaled FP4 GEMM (SM103 Thor)")
print("=" * 60)
print(f"\nGPU: {gpu_name} (SM{sm_version})")
print(f"SM103 Support: {'✓ Yes' if sm103_supported else '✗ No (requires SM103+)'}")

if not sm103_supported:
    print("\n⚠️  SM103 features require Thor (SM103) or later.")
    print("   This exercise will show placeholder results.")
    print("   Target: RTX 5090, Thor (automotive)")

# Matrix dimensions (LLM inference)
M, K, N = 128, 8192, 8192  # Decode phase: small batch, large hidden
dtype = torch.float16
device = torch.device("cuda")

print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"Use case: LLM decode (token generation)")


# ==============================================================================
# FP4 FORMAT COMPARISON
# ==============================================================================

@dataclass
class FP4Format:
    name: str
    bits: int
    exp_bits: int
    mantissa_bits: int
    max_value: float
    description: str


fp4_formats = {
    "nvfp4": FP4Format(
        name="NVFP4",
        bits=4,
        exp_bits=2,
        mantissa_bits=1,
        max_value=6.0,
        description="NVIDIA FP4 (2 exp, 1 mantissa), block-scaled"
    ),
    "mxfp4": FP4Format(
        name="MXFP4",
        bits=4,
        exp_bits=3,
        mantissa_bits=0,
        max_value=28.0,
        description="OCP Microscaling FP4 (3 exp, 0 mantissa)"
    ),
}

print(f"\nFP4 Format Comparison:")
print(f"  {'Format':<10} {'Exp':<6} {'Mantissa':<10} {'Max':<8} {'Description'}")
print(f"  {'-'*60}")
for key, fmt in fp4_formats.items():
    print(f"  {fmt.name:<10} {fmt.exp_bits:<6} {fmt.mantissa_bits:<10} {fmt.max_value:<8.1f} {fmt.description[:35]}")

print(f"\nKey differences:")
print(f"  NVFP4: Better precision (1 mantissa bit)")
print(f"  MXFP4: Better range (3 exp bits, no mantissa)")
print(f"  SM103: Supports both formats natively")


# ==============================================================================
# BLOCKSCALED QUANTIZATION
# ==============================================================================

def quantize_blockscaled_nvfp4(tensor: torch.Tensor, 
                                block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize to NVFP4 with block scaling.
    
    Each block gets its own scale factor.
    Values quantized to 4-bit: [-8, 7] range.
    """
    original_shape = tensor.shape
    flat = tensor.flatten()
    
    # Pad to multiple of block_size
    padded_size = (len(flat) + block_size - 1) // block_size * block_size
    padded = torch.zeros(padded_size, dtype=torch.float32, device=tensor.device)
    padded[:len(flat)] = flat
    
    # Reshape into blocks
    num_blocks = padded_size // block_size
    blocks = padded.reshape(num_blocks, block_size)
    
    # Compute scale per block
    block_max = blocks.abs().max(dim=1).values
    scales = block_max / 7.0  # NVFP4 max = 7
    scales = scales.clamp(min=1e-8)
    
    # Quantize
    quantized = (blocks / scales.unsqueeze(1)).round().clamp(-8, 7)
    
    # Pack 2 values per byte (4-bit)
    quantized = (quantized + 8).to(torch.uint8)  # Shift to [0, 15]
    
    # Reshape back
    quantized_flat = quantized.flatten()[:len(flat)]
    quantized_tensor = quantized_flat.reshape(original_shape)
    
    return quantized_tensor, scales


def quantize_blockscaled_mxfp4(tensor: torch.Tensor,
                                block_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize to MXFP4 with block scaling.
    
    MXFP4 uses 3 exp bits, 0 mantissa bits.
    Values are essentially exponents only.
    """
    original_shape = tensor.shape
    flat = tensor.abs().flatten()
    
    # Pad
    padded_size = (len(flat) + block_size - 1) // block_size * block_size
    padded = torch.zeros(padded_size, dtype=torch.float32, device=tensor.device)
    padded[:len(flat)] = flat
    
    # Reshape
    num_blocks = padded_size // block_size
    blocks = padded.reshape(num_blocks, block_size)
    
    # MXFP4: quantize to exponent only (log scale)
    # Values represented as: sign × 2^exp
    block_max = blocks.max(dim=1).values
    # Compute exponent (log2)
    exponents = torch.log2(block_max.clamp(min=1e-8)).round().clamp(-8, 7)
    scales = 2.0 ** exponents
    
    # Quantize
    quantized = blocks / scales.unsqueeze(1)
    quantized = quantized.round().clamp(-15, 15)
    
    # Store as 4-bit
    quantized = (quantized + 8).to(torch.uint8)
    
    # Reshape
    quantized_flat = quantized.flatten()[:len(flat)]
    quantized_tensor = quantized_flat.reshape(original_shape)
    
    return quantized_tensor, scales


def dequantize_blockscaled(quantized: torch.Tensor, scales: torch.Tensor,
                           block_size: int = 128) -> torch.Tensor:
    """Dequantize blockscaled FP4."""
    original_shape = quantized.shape
    flat_q = (quantized.flatten().to(torch.float32) - 8)  # Shift back
    flat_s = scales
    
    # Pad
    padded_size = (len(flat_q) + block_size - 1) // block_size * block_size
    padded_q = torch.zeros(padded_size, dtype=torch.float32, device=quantized.device)
    padded_q[:len(flat_q)] = flat_q
    
    # Reshape
    num_blocks = padded_size // block_size
    blocks = padded_q.reshape(num_blocks, block_size)
    
    # Dequantize
    dequantized = blocks * flat_s.unsqueeze(1)
    
    # Reshape
    dequantized_flat = dequantized.flatten()[:len(flat_q)]
    return dequantized_flat.reshape(original_shape)


# ==============================================================================
# CREATE AND QUANTIZE TENSORS
# ==============================================================================

# Create FP32 reference tensors
A_fp32 = torch.randn(M, K, dtype=torch.float32, device=device) * 0.5
B_fp32 = torch.randn(K, N, dtype=torch.float32, device=device) * 0.1

# Reference: FP32 GEMM
C_fp32_ref = torch.mm(A_fp32, B_fp32)

# Quantize weights to NVFP4
block_size = 128
B_nvfp4, B_scales_nvfp4 = quantize_blockscaled_nvfp4(B_fp32, block_size)
B_nvfp4_dq = dequantize_blockscaled(B_nvfp4, B_scales_nvfp4, block_size)
C_nvfp4_sim = torch.mm(A_fp32, B_nvfp4_dq)

# Quantize weights to MXFP4
B_mxfp4, B_scales_mxfp4 = quantize_blockscaled_mxfp4(B_fp32, block_size=32)
B_mxfp4_dq = dequantize_blockscaled(B_mxfp4, B_scales_mxfp4, block_size=32)
C_mxfp4_sim = torch.mm(A_fp32, B_mxfp4_dq)

# Compute errors
nvfp4_error = (C_nvfp4_sim - C_fp32_ref).abs().div(C_fp32_ref.abs() + 1e-8).max().item()
mxfp4_error = (C_mxfp4_sim - C_fp32_ref).abs().div(C_fp32_ref.abs() + 1e-8).max().item()

print(f"\nBlockscaled Quantization (block_size={block_size}):")
print(f"  NVFP4 max relative error: {nvfp4_error*100:.2f}%")
print(f"  MXFP4 max relative error: {mxfp4_error*100:.2f}%")


# ==============================================================================
# FILL IN: Level 2 — CuTe DSL Blockscaled FP4 GEMM
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 2: CuTe DSL Blockscaled FP4 GEMM")
print("=" * 60)

# TODO [HARD]: Implement blockscaled FP4 GEMM kernel
# HINT:
#   - Use cute.MMA_Atom(cute.GMMA_FP4) for FP4 MMA
#   - Load FP4 weights (packed 2 per byte)
#   - Load and apply block scales in epilogue
#   - Use tcgen05 for SM100+ or tcgen05_thor for SM103
# REF: cutlass/examples/python/CuTeDSL/blockscaled_gemm.py

# TODO: Define blockscaled FP4 GEMM kernel
# @cutlass.cute.kernel
# def blockscaled_fp4_gemm(A: cute.Tensor, B_fp4: cute.Tensor,
#                          scales: cute.Tensor, C: cute.Tensor,
#                          block_size: int):
#     """
#     Blockscaled FP4 GEMM for SM103.
#     
#     - A: FP16 activations
#     - B_fp4: FP4 weights (packed)
#     - scales: FP16 block scales
#     - C: FP32 output
#     """
#     # Load A tile (FP16)
#     # Load B tile (FP4, unpack)
#     # Load scales for this block
#     # Compute FP4 × FP16 → FP32 MMA
#     # Apply scales in epilogue
#     # Store C
#     ...

# Placeholder kernel
@cutlass.cute.kernel
def blockscaled_fp4_gemm(A: cute.Tensor, B_fp4: cute.Tensor,
                         scales: cute.Tensor, C: cute.Tensor,
                         block_size: int):
    """Placeholder blockscaled FP4 GEMM kernel."""
    pass

print(f"\nBlockscaled FP4 GEMM kernel defined (placeholder)")
print(f"  Format: NVFP4/MXFP4")
print(f"  Block size: {block_size}")
print(f"  Target: SM103 (Thor)")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

# Compare NVFP4 to FP32
nvfp4_correct = torch.allclose(C_nvfp4_sim, C_fp32_ref, rtol=0.1, atol=0.1)
print(f"\nNVFP4 correctness: {'✓ PASS' if nvfp4_correct else '✗ FAIL'}")
print(f"  Max relative error: {nvfp4_error*100:.2f}%")

# Compare MXFP4 to FP32
mxfp4_correct = torch.allclose(C_mxfp4_sim, C_fp32_ref, rtol=0.1, atol=0.1)
print(f"\nMXFP4 correctness: {'✓ PASS' if mxfp4_correct else '✗ FAIL'}")
print(f"  Max relative error: {mxfp4_error*100:.2f}%")

print(f"\nRecommendation: {'NVFP4' if nvfp4_error < mxfp4_error else 'MXFP4'} for this workload")


# ==============================================================================
# BENCHMARK
# ==============================================================================

def benchmark_fp4_gemm(A, B_fp4, scales, block_size,
                       num_warmup=10, num_iters=100) -> float:
    """Benchmark FP4 GEMM (simulated)."""
    # Dequantize for simulation
    B_dq = dequantize_blockscaled(B_fp4, scales, block_size)
    
    # Warmup
    C = torch.zeros(A.shape[0], B_dq.shape[1], dtype=torch.float32, device=A.device)
    for _ in range(num_warmup):
        C.copy_(torch.mm(A, B_dq))
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A, B_dq))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_fp16_gemm(A_fp16, B_fp16, num_warmup=10, num_iters=100) -> float:
    """Benchmark FP16 GEMM."""
    C = torch.zeros(A_fp16.shape[0], B_fp16.shape[1], dtype=torch.float16, device=A_fp16.device)
    
    for _ in range(num_warmup):
        C.copy_(torch.mm(A_fp16, B_fp16))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A_fp16, B_fp16))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


print("\n" + "=" * 60)
print("Performance (Estimated)")
print("=" * 60)

# Prepare tensors
A_fp16 = A_fp32.to(torch.float16)
B_fp16 = B_fp32.to(torch.float16)

fp16_latency = benchmark_fp16_gemm(A_fp16, B_fp16)
fp4_latency = benchmark_fp4_gemm(A_fp32, B_nvfp4, B_scales_nvfp4, block_size)

print(f"\nResults:")
print(f"  FP16 GEMM: {fp16_latency:.3f} ms")
print(f"  FP4 GEMM:  {fp4_latency:.3f} ms (estimated)")

if fp4_latency > 0 and fp16_latency > 0:
    speedup = fp16_latency / fp4_latency
    print(f"\n  Speedup: {speedup:.2f}×")
    
    # Memory bandwidth
    fp16_bytes = K * N * 2  # FP16 weights
    fp4_bytes = K * N * 0.5  # FP4 weights (4 bits)
    
    fp16_bw = fp16_bytes / (fp16_latency * 1e-3) / 1e9
    fp4_bw = fp4_bytes / (fp4_latency * 1e-3) / 1e9
    
    print(f"\n  FP16 effective bandwidth: {fp16_bw:.0f} GB/s")
    print(f"  FP4 effective bandwidth:  {fp4_bw:.0f} GB/s")
    print(f"  Memory savings: {100 * (1 - fp4_bytes/fp16_bytes):.0f}%")


# ==============================================================================
# SM100 vs SM103 COMPARISON
# ==============================================================================

print("\n" + "=" * 60)
print("SM100 vs SM103 Comparison")
print("=" * 60)

print("""
SM100 (B100, B200, GB200):
  - Data center GPUs
  - NVFP4 support
  - High FP4 throughput (20,000+ TFLOPS)
  - Large memory (180GB HBM3e)
  - Focus: LLM training/inference

SM103 (RTX 5090, Thor):
  - Consumer/Automotive GPUs
  - NVFP4 + MXFP4 support
  - Good FP4 throughput (2,000+ TFLOPS)
  - GDDR7 memory (32GB typical)
  - Focus: Gaming, edge AI, automotive

Key SM103 additions:
  - MXFP4 format (OCP microscaling standard)
  - Better FP4 quantization options
  - Optimized for edge deployment
""")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: SM100 vs SM103 difference?")
print("        SM100: Data center (B100, B200)")
print("        SM103: Consumer/Automotive (RTX 5090, Thor)")
print("        SM103 adds MXFP4 support")

print("\n    Q2: NVFP4 vs MXFP4?")
print(f"        NVFP4 error: {nvfp4_error*100:.2f}%")
print(f"        MXFP4 error: {mxfp4_error*100:.2f}%")
print("        NVFP4: Better precision (1 mantissa bit)")
print("        MXFP4: Better range (3 exp bits)")

print("\n    Q3: FP4 accuracy vs speed trade-off?")
print("        FP4: 4× memory savings, 2-4× speedup")
print("        Accuracy loss: 5-15% typical")
print("        Acceptable for: inference, not training")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command (on SM103):")
print(f"    ncu --metrics sm__inst_executed_pipe_tensor.sum,\\")
print(f"                dram__throughput.sum \\")
print(f"        python ex03_blockscaled_gemm_FILL_IN.py")
print("\n    Look for:")
print("      - FP4 Tensor Core utilization")
print("      - Memory bandwidth efficiency")
print("      - Scale application overhead")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: When would you use SM103 vs SM100?")
print("    A: SM100 (B100/B200):")
print("       - Data center deployment")
print("       - Maximum throughput needed")
print("       - Large model (> 100B params)")
print("       SM103 (Thor/RTX 5090):")
print("       - Edge deployment")
print("       - Cost-sensitive deployment")
print("       - Automotive applications")

print("\n    Q: What's the FP4 quantization strategy?")
print("    A: Recommended approach:")
print("       1. Weights: Blockscaled NVFP4 (block=128)")
print("       2. Activations: FP8 or FP16")
print("       3. Calibrate on 1000+ samples")
print("       4. Consider QAT for accuracy-critical models")
print("       5. Use MXFP4 for high dynamic range data")

# C4: Production guidance
print("\nC4: Production FP4 Tips")
print("    Use FP4 when:")
print("      - Deploying on Blackwell GPUs")
print("      - Memory bandwidth bound")
print("      - Throughput is critical")
print("      - Accuracy tolerance > 5%")
print("\n    Format selection:")
print("      - NVFP4: General purpose (better precision)")
print("      - MXFP4: High dynamic range data")
print("      - Block size: 64-128 (smaller = better accuracy)")

print("\n" + "=" * 60)
print("Exercise 03 Complete!")
print("=" * 60)
