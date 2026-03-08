"""
Module 05 — Mixed Precision
Exercise 01 — FP8 GEMM with E4M3 (TensorRT-LLM Pattern)

LEVEL: 1 (High-level op API)

WHAT YOU'RE BUILDING:
  FP8 E4M3 GEMM — the exact pattern used in TensorRT-LLM's FP8 inference 
  for Hopper GPUs. E4M3 provides 2× memory bandwidth savings vs FP16 with 
  minimal accuracy loss for inference workloads.

OBJECTIVE:
  - Configure GEMM for FP8 E4M3 inputs with FP32 accumulation
  - Understand FP8 quantization/dequantization
  - Compare FP8 vs FP16 performance and accuracy
  - Learn when FP8 is appropriate (inference vs training)
"""

import torch
import cutlass
import time
from dataclasses import dataclass
from typing import Tuple


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What's the expected speedup of FP8 vs FP16 on Hopper?
#     Consider: memory bandwidth vs tensor core throughput

# Q2: What's the maximum relative error you expect from FP8 E4M3 vs FP32?
#     Hint: E4M3 has 3 mantissa bits = ~1 decimal digit precision

# Q3: Can you train in FP8? What are the challenges?


# ==============================================================================
# SETUP
# ==============================================================================

# Check GPU capability (FP8 requires Hopper SM90+)
def check_fp8_support():
    """Check if GPU supports FP8 Tensor Cores."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability()
        sm = compute_cap[0] * 10 + compute_cap[1]
        
        if sm >= 90:
            return True, gpu_name, sm
        else:
            return False, gpu_name, sm
    return False, "No GPU", 0


fp8_supported, gpu_name, sm_version = check_fp8_support()

print("=" * 60)
print("FP8 GEMM with E4M3 (TensorRT-LLM Pattern)")
print("=" * 60)
print(f"\nGPU: {gpu_name} (SM{sm_version})")
print(f"FP8 Support: {'✓ Yes' if fp8_supported else '✗ No (requires Hopper SM90+)'}")

if not fp8_supported:
    print("\n⚠️  FP8 requires Hopper (SM90) or later.")
    print("   This exercise will show placeholder results.")
    print("   Consider running on H100/H200 or RTX 4090 (partial support)")

# Matrix dimensions (typical transformer layer)
M, K, N = 1024, 4096, 4096
dtype_fp8 = torch.float8_e4m3fn
dtype_fp16 = torch.float16
dtype_fp32 = torch.float32
device = torch.device("cuda")

# Create FP32 reference tensors
A_fp32 = torch.randn(M, K, dtype=dtype_fp32, device=device)
B_fp32 = torch.randn(K, N, dtype=dtype_fp32, device=device)

# Reference: FP32 GEMM
C_fp32_ref = torch.mm(A_fp32, B_fp32)

# Quantize to FP8 E4M3
def quantize_to_fp8_e4m3(tensor_fp32: torch.Tensor) -> torch.Tensor:
    """Quantize FP32 tensor to FP8 E4M3."""
    # Clamp to FP8 E4M3 range (approximately ±448)
    tensor_clamped = tensor_fp32.clamp(-448, 448)
    return tensor_clamped.to(dtype_fp8)


def dequantize_from_fp8(tensor_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor to FP32 for comparison."""
    return tensor_fp8.to(dtype_fp32)


A_fp8 = quantize_to_fp8_e4m3(A_fp32)
B_fp8 = quantize_to_fp8_e4m3(B_fp32)

# Reference: FP8 GEMM (quantize → compute → dequantize)
# Note: PyTorch doesn't natively support FP8 GEMM, so we simulate
A_fp8_dequant = dequantize_from_fp8(A_fp8)
B_fp8_dequant = dequantize_from_fp8(B_fp8)
C_fp8_simulated = torch.mm(A_fp8_dequant, B_fp8_dequant)

print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"\nFP8 E4M3 characteristics:")
print(f"  Exponent bits: 4")
print(f"  Mantissa bits: 3")
print(f"  Range: ~±448")
print(f"  Precision: ~1 decimal digit")


# ==============================================================================
# FILL IN: Level 1 — FP8 GEMM with cutlass.op
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1: High-Level Op API (cutlass.op.Gemm FP8)")
print("=" * 60)

# TODO [HARD]: Configure FP8 GEMM
# HINT:
#   - Use cutlass.float8_e4m3fn for element type
#   - Use cutlass.float32 for accumulator (important!)
#   - FP8 requires SM90+ (Hopper)
# REF: cutlass/examples/python/CuTeDSL/fp8_gemm.py

# TODO: Create FP8 GEMM plan
# plan_fp8 = cutlass.op.Gemm(
#     element=cutlass.float8_e4m3fn,
#     accumulator_type=cutlass.float32,  # FP32 accumulation for accuracy
#     layout=cutlass.LayoutType.RowMajor,
# )

# TODO: Allocate output (FP32 for accumulation)
# C_fp8 = torch.zeros(M, N, dtype=dtype_fp32, device=device)

# TODO: Run FP8 GEMM
# A_cutlass = from_dlpack(A_fp8)
# B_cutlass = from_dlpack(B_fp8)
# plan_fp8.run(A_cutlass, B_cutlass, C_fp8)

# Placeholder (replace with implementation)
plan_fp8 = None
C_fp8 = C_fp8_simulated.clone()

print(f"\nCUTLASS FP8 GEMM completed")
print(f"Output dtype: {C_fp8.dtype}")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

# Compare FP8 result to FP32 reference
max_abs_error = (C_fp8 - C_fp32_ref).abs().max().item()
max_rel_error = (C_fp8 - C_fp32_ref).abs().div(C_fp32_ref.abs() + 1e-8).max().item()

print(f"\nFP8 vs FP32 Reference:")
print(f"  Max Absolute Error: {max_abs_error:.6f}")
print(f"  Max Relative Error: {max_rel_error:.4f} ({max_rel_error*100:.2f}%)")

# Also compare to simulated FP8
if not fp8_supported:
    print(f"\n⚠️  Using simulated FP8 (PyTorch fallback)")
    print(f"   For real FP8 performance, run on Hopper GPU")


# ==============================================================================
# BENCHMARK: FP8 vs FP16
# ==============================================================================

def benchmark_fp8_gemm(A_fp8, B_fp8, C_fp8, plan, 
                       num_warmup=10, num_iters=100) -> float:
    """Benchmark FP8 GEMM."""
    if plan is None:
        # Fallback to simulated FP8
        A_dequant = A_fp8.to(dtype_fp32)
        B_dequant = B_fp8.to(dtype_fp32)
        for _ in range(num_warmup):
            _ = torch.mm(A_dequant, B_dequant)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = torch.mm(A_dequant, B_dequant)
        torch.cuda.synchronize()
        
        return (time.perf_counter() - start) / num_iters * 1000
    
    # Warmup
    for _ in range(num_warmup):
        plan.run(from_dlpack(A_fp8), from_dlpack(B_fp8), C_fp8)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        plan.run(from_dlpack(A_fp8), from_dlpack(B_fp8), C_fp8)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_fp16_gemm(A_fp16, B_fp16, C_fp16, 
                        num_warmup=10, num_iters=100) -> float:
    """Benchmark FP16 GEMM."""
    for _ in range(num_warmup):
        C_fp16.copy_(torch.mm(A_fp16, B_fp16))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        C_fp16.copy_(torch.mm(A_fp16, B_fp16))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


# Prepare FP16 tensors
A_fp16 = A_fp32.to(dtype_fp16)
B_fp16 = B_fp32.to(dtype_fp16)
C_fp16 = torch.zeros(M, N, dtype=dtype_fp16, device=device)

print("\n" + "=" * 60)
print("Performance: FP8 vs FP16")
print("=" * 60)

fp8_latency = benchmark_fp8_gemm(A_fp8, B_fp8, C_fp8, plan_fp8)
fp16_latency = benchmark_fp16_gemm(A_fp16, B_fp16, C_fp16)

print(f"\nResults:")
print(f"  FP8 E4M3: {fp8_latency:.3f} ms")
print(f"  FP16:     {fp16_latency:.3f} ms")

if fp8_latency > 0 and fp16_latency > 0:
    speedup = fp16_latency / fp8_latency
    print(f"\n  Speedup: {speedup:.2f}×")
    
    # Compute TFLOPS
    flops = 2 * M * N * K
    fp8_tflops = flops / (fp8_latency * 1e-3) / 1e12
    fp16_tflops = flops / (fp16_latency * 1e-3) / 1e12
    
    print(f"\n  FP8 TFLOPS:  {fp8_tflops:.1f}")
    print(f"  FP16 TFLOPS: {fp16_tflops:.1f}")
    
    # Theoretical speedup (FP8 has 2× tensor core throughput on Hopper)
    print(f"\n  Theoretical FP8 speedup: up to 2× (Hopper Tensor Cores)")


# ==============================================================================
# MEMORY BANDWIDTH ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("Memory Bandwidth Analysis")
print("=" * 60)

bytes_fp8 = 1   # FP8 = 1 byte
bytes_fp16 = 2  # FP16 = 2 bytes

# Memory traffic for GEMM (read A, read B, write C)
memory_fp8 = (M * K + K * N) * bytes_fp8 + M * N * 4  # FP32 output
memory_fp16 = (M * K + K * N) * bytes_fp16 + M * N * 2

print(f"\nMemory Traffic:")
print(f"  FP8:  {memory_fp8 / 1e6:.1f} MB (inputs FP8, output FP32)")
print(f"  FP16: {memory_fp16 / 1e6:.1f} MB (inputs FP16, output FP16)")
print(f"  Savings: {(1 - memory_fp8 / memory_fp16) * 100:.1f}%")

# Bandwidth utilization
if fp8_latency > 0:
    bw_fp8 = memory_fp8 / (fp8_latency * 1e-3) / 1e9
    print(f"\n  FP8 Effective Bandwidth: {bw_fp8:.0f} GB/s")

if fp16_latency > 0:
    bw_fp16 = memory_fp16 / (fp16_latency * 1e-3) / 1e9
    print(f"  FP16 Effective Bandwidth: {bw_fp16:.0f} GB/s")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: FP8 vs FP16 speedup?")
if fp8_latency > 0 and fp16_latency > 0:
    print(f"        Actual: {speedup:.2f}×")
print("        Theoretical: up to 2× on Hopper")
print("        Limited by: memory bandwidth, quantization overhead")

print("\n    Q2: FP8 E4M3 relative error?")
print(f"        Actual: {max_rel_error*100:.2f}%")
print("        Expected: 1-5% for typical LLM weights")
print("        Acceptable for: inference, not training")

print("\n    Q3: Can you train in FP8?")
print("        Answer: Challenging. Issues:")
print("        - Limited dynamic range (gradient explosion/vanishing)")
print("        - Low precision (gradient noise)")
print("        - Requires: loss scaling, gradient clipping")
print("        Research: 'FP8 Formats for Neural Network Training'")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics sm__inst_executed_pipe_tensor.sum,\\")
print(f"                dram__throughput.sum \\")
print(f"        python ex01_fp8_gemm_e4m3_FILL_IN.py")
print("\n    Look for:")
print("      - Higher tensor core throughput for FP8")
print("      - Lower memory bandwidth for FP8")
print("      - Same or better SM utilization")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: When should you use FP8 E4M3 vs E5M2?")
print("    A: E4M3 (4 exp, 3 mantissa):")
print("       - Better precision (3 bits vs 2)")
print("       - Smaller range (±448 vs ±57344)")
print("       - Use for: weights, activations")
print("       E5M2 (5 exp, 2 mantissa):")
print("       - Better range (similar to FP16)")
print("       - Less precision")
print("       - Use for: gradients, high-dynamic-range data")

print("\n    Q: How does TensorRT-LLM use FP8?")
print("    A: FP8 inference pipeline:")
print("       1. Quantize weights offline to FP8 E4M3")
print("       2. Quantize activations online (per-token)")
print("       3. FP8 GEMM with FP32 accumulation")
print("       4. Dequantize output if needed")
print("       Result: 2× memory savings, minimal accuracy loss")

# C4: Production guidance
print("\nC4: Production FP8 Tips")
print("    Use FP8 for:")
print("      - Inference on Hopper GPUs")
print("      - Memory-bandwidth-bound workloads")
print("      - Large models (memory capacity limited)")
print("\n    Avoid FP8 for:")
print("      - Training (use BF16/FP16)")
print("      - High-accuracy requirements")
print("      - Pre-Hopper GPUs (no native FP8 Tensor Core)")
print("\n    Quantization tips:")
print("      - Per-tensor for weights (offline)")
print("      - Per-token for activations (online)")
print("      - Calibrate on representative data")

print("\n" + "=" * 60)
print("Exercise 01 Complete!")
print("=" * 60)
