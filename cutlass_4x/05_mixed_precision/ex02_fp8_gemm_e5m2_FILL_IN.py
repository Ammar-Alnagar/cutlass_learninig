"""
Module 05 — Mixed Precision
Exercise 02 — FP8 GEMM with E5M2 (Extended Range)

LEVEL: 1 (High-level op API)

WHAT YOU'RE BUILDING:
  FP8 E5M2 GEMM — extended range FP8 format used for gradients and 
  high-dynamic-range data. E5M2 matches FP16's exponent range, making it 
  suitable for data that needs wider dynamic range over precision.

OBJECTIVE:
  - Configure GEMM for FP8 E5M2 inputs
  - Compare E5M2 vs E4M3 trade-offs (range vs precision)
  - Understand when to use each FP8 format
  - Learn FP8 format selection for different tensor types
"""

import torch
import cutlass
import time
from dataclasses import dataclass
from typing import Tuple


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: E5M2 has 5 exponent bits vs E4M3's 4 bits. What does this mean?
#     How does it affect range and precision?

# Q2: Which FP8 format is better for gradients: E4M3 or E5M2?
#     Why?

# Q3: Can you mix E4M3 and E5M2 in the same model?
#     Example: E4M3 for weights, E5M2 for activations?


# ==============================================================================
# SETUP
# ==============================================================================

# Check GPU capability
def check_fp8_support():
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
print("FP8 GEMM with E5M2 (Extended Range)")
print("=" * 60)
print(f"\nGPU: {gpu_name} (SM{sm_version})")
print(f"FP8 Support: {'✓ Yes' if fp8_supported else '✗ No (requires Hopper SM90+)'}")

# Matrix dimensions
M, K, N = 1024, 4096, 4096
dtype_fp8_e5m2 = torch.float8_e5m2
dtype_fp8_e4m3 = torch.float8_e4m3fn
dtype_fp32 = torch.float32
device = torch.device("cuda")

# Create FP32 reference tensors
A_fp32 = torch.randn(M, K, dtype=dtype_fp32, device=device)
B_fp32 = torch.randn(K, N, dtype=dtype_fp32, device=device)

# Reference: FP32 GEMM
C_fp32_ref = torch.mm(A_fp32, B_fp32)

# FP8 format characteristics
@dataclass
class FP8Format:
    name: str
    dtype: torch.dtype
    exp_bits: int
    mantissa_bits: int
    max_value: float
    description: str


fp8_formats = {
    "e4m3": FP8Format(
        name="E4M3",
        dtype=dtype_fp8_e4m3fn,
        exp_bits=4,
        mantissa_bits=3,
        max_value=448.0,
        description="Better precision, smaller range (weights, activations)"
    ),
    "e5m2": FP8Format(
        name="E5M2",
        dtype=dtype_fp8_e5m2,
        exp_bits=5,
        mantissa_bits=2,
        max_value=57344.0,
        description="Better range, less precision (gradients, high dynamic range)"
    ),
}

print(f"\nFP8 Format Comparison:")
print(f"  {'Format':<10} {'Exp Bits':<10} {'Mantissa':<10} {'Max Value':<12} {'Use Case'}")
print(f"  {'-'*55}")
for key, fmt in fp8_formats.items():
    print(f"  {fmt.name:<10} {fmt.exp_bits:<10} {fmt.mantissa_bits:<10} {fmt.max_value:<12.0f} {fmt.description[:30]}")


# ==============================================================================
# QUANTIZATION FUNCTIONS
# ==============================================================================

def quantize_to_fp8(tensor_fp32: torch.Tensor, fmt: FP8Format) -> torch.Tensor:
    """Quantize FP32 tensor to specified FP8 format."""
    # Clamp to FP8 range
    tensor_clamped = tensor_fp32.clamp(-fmt.max_value, fmt.max_value)
    return tensor_clamped.to(fmt.dtype)


def dequantize_from_fp8(tensor_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor to FP32."""
    return tensor_fp8.to(dtype_fp32)


# Quantize to both formats
A_e4m3 = quantize_to_fp8(A_fp32, fp8_formats["e4m3"])
B_e4m3 = quantize_to_fp8(B_fp32, fp8_formats["e4m3"])

A_e5m2 = quantize_to_fp8(A_fp32, fp8_formats["e5m2"])
B_e5m2 = quantize_to_fp8(B_fp32, fp8_formats["e5m2"])


# ==============================================================================
# FILL IN: Level 1 — FP8 E5M2 GEMM
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1: High-Level Op API (cutlass.op.Gemm FP8 E5M2)")
print("=" * 60)

# TODO [HARD]: Configure FP8 E5M2 GEMM
# HINT:
#   - Use cutlass.float8_e5m2 for element type
#   - Use cutlass.float32 for accumulator
#   - Same API as E4M3, just different element type
# REF: cutlass/examples/python/CuTeDSL/fp8_gemm.py

# TODO: Create FP8 E5M2 GEMM plan
# plan_e5m2 = cutlass.op.Gemm(
#     element=cutlass.float8_e5m2,
#     accumulator_type=cutlass.float32,
#     layout=cutlass.LayoutType.RowMajor,
# )

# Placeholder (replace with implementation)
plan_e5m2 = None
plan_e4m3 = None

# Simulated results (dequantize and compute in FP32)
A_e4m3_dequant = dequantize_from_fp8(A_e4m3)
B_e4m3_dequant = dequantize_from_fp8(B_e4m3)
C_e4m3_sim = torch.mm(A_e4m3_dequant, B_e4m3_dequant)

A_e5m2_dequant = dequantize_from_fp8(A_e5m2)
B_e5m2_dequant = dequantize_from_fp8(B_e5m2)
C_e5m2_sim = torch.mm(A_e5m2_dequant, B_e5m2_dequant)

print(f"\nCUTLASS FP8 E5M2 GEMM configured")


# ==============================================================================
# VERIFICATION: E4M3 vs E5M2 Accuracy
# ==============================================================================

print("\n" + "=" * 60)
print("Verification: E4M3 vs E5M2 Accuracy")
print("=" * 60)

# E4M3 error
e4m3_max_abs_error = (C_e4m3_sim - C_fp32_ref).abs().max().item()
e4m3_max_rel_error = (C_e4m3_sim - C_fp32_ref).abs().div(C_fp32_ref.abs() + 1e-8).max().item()

# E5M2 error
e5m2_max_abs_error = (C_e5m2_sim - C_fp32_ref).abs().max().item()
e5m2_max_rel_error = (C_e5m2_sim - C_fp32_ref).abs().div(C_fp32_ref.abs() + 1e-8).max().item()

print(f"\nE4M3 vs FP32:")
print(f"  Max Absolute Error: {e4m3_max_abs_error:.6f}")
print(f"  Max Relative Error: {e4m3_max_rel_error:.4f} ({e4m3_max_rel_error*100:.2f}%)")

print(f"\nE5M2 vs FP32:")
print(f"  Max Absolute Error: {e5m2_max_abs_error:.6f}")
print(f"  Max Relative Error: {e5m2_max_rel_error:.4f} ({e5m2_max_rel_error*100:.2f}%)")

print(f"\nComparison:")
if e4m3_max_rel_error < e5m2_max_rel_error:
    print(f"  E4M3 is more accurate (better for precision-sensitive data)")
else:
    print(f"  E5M2 is more accurate (data has high dynamic range)")


# Test with high dynamic range data
print("\n" + "=" * 60)
print("High Dynamic Range Test")
print("=" * 60)

# Create data with large values (simulating gradients)
A_large = torch.randn(M, K, dtype=dtype_fp32, device=device) * 100  # Large values
B_large = torch.randn(K, N, dtype=dtype_fp32, device=device) * 100

C_large_ref = torch.mm(A_large, B_large)

# Quantize to both formats
A_large_e4m3 = quantize_to_fp8(A_large, fp8_formats["e4m3"])
B_large_e4m3 = quantize_to_fp8(B_large, fp8_formats["e4m3"])

A_large_e5m2 = quantize_to_fp8(A_large, fp8_formats["e5m2"])
B_large_e5m2 = quantize_to_fp8(B_large, fp8_formats["e5m2"])

# Check clamping
e4m3_clamped = (A_large.abs() > fp8_formats["e4m3"].max_value).float().mean().item() * 100
e5m2_clamped = (A_large.abs() > fp8_formats["e5m2"].max_value).float().mean().item() * 100

print(f"\nLarge value quantization:")
print(f"  E4M3: {e4m3_clamped:.1f}% of values clamped to ±{fp8_formats['e4m3'].max_value}")
print(f"  E5M2: {e5m2_clamped:.1f}% of values clamped to ±{fp8_formats['e5m2'].max_value}")

# Compute errors
A_large_e4m3_dequant = dequantize_from_fp8(A_large_e4m3)
B_large_e4m3_dequant = dequantize_from_fp8(B_large_e4m3)
C_large_e4m3 = torch.mm(A_large_e4m3_dequant, B_large_e4m3_dequant)

A_large_e5m2_dequant = dequantize_from_fp8(A_large_e5m2)
B_large_e5m2_dequant = dequantize_from_fp8(B_large_e5m2)
C_large_e5m2 = torch.mm(A_large_e5m2_dequant, B_large_e5m2_dequant)

large_e4m3_error = (C_large_e4m3 - C_large_ref).abs().div(C_large_ref.abs() + 1e-8).max().item()
large_e5m2_error = (C_large_e5m2 - C_large_ref).abs().div(C_large_ref.abs() + 1e-8).max().item()

print(f"\nLarge value relative error:")
print(f"  E4M3: {large_e4m3_error*100:.2f}%")
print(f"  E5M2: {large_e5m2_error*100:.2f}%")

if large_e5m2_error < large_e4m3_error:
    print(f"\n  ✓ E5M2 better for high dynamic range data!")
else:
    print(f"\n  E4M3 still adequate (depends on value distribution)")


# ==============================================================================
# BENCHMARK
# ==============================================================================

print("\n" + "=" * 60)
print("Performance: E4M3 vs E5M2")
print("=" * 60)

def benchmark_fp8_gemm(A_fp8, B_fp8, num_warmup=10, num_iters=100) -> float:
    """Benchmark FP8 GEMM (simulated)."""
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


e4m3_latency = benchmark_fp8_gemm(A_e4m3, B_e4m3)
e5m2_latency = benchmark_fp8_gemm(A_e5m2, B_e5m2)

print(f"\nResults (simulated):")
print(f"  FP8 E4M3: {e4m3_latency:.3f} ms")
print(f"  FP8 E5M2: {e5m2_latency:.3f} ms")

# On real Hopper hardware, both should have similar performance
print(f"\nNote: On Hopper GPU, E4M3 and E5M2 have similar performance.")
print(f"      Choice should be based on accuracy requirements.")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: E5M2 vs E4M3 difference?")
print("        Answer:")
print(f"        E4M3: {fp8_formats['e4m3'].exp_bits} exp, {fp8_formats['e4m3'].mantissa_bits} mantissa, range ±{fp8_formats['e4m3'].max_value}")
print(f"        E5M2: {fp8_formats['e5m2'].exp_bits} exp, {fp8_formats['e5m2'].mantissa_bits} mantissa, range ±{fp8_formats['e5m2'].max_value}")
print("        E5M2 = 2× range, ½ precision")

print("\n    Q2: Which format for gradients?")
print("        Answer: E5M2 (better range for gradient dynamics)")
print("                Gradients can have large spikes during training")

print("\n    Q3: Can you mix formats?")
print("        Answer: Yes! Common pattern:")
print("        - E4M3 for weights (precision matters)")
print("        - E5M2 for activations (range matters)")
print("        - E5M2 for gradients (large dynamic range)")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics sm__inst_executed_pipe_tensor.sum \\")
print(f"        python ex02_fp8_gemm_e5m2_FILL_IN.py")
print("\n    Look for:")
print("      - Same tensor core utilization for both formats")
print("      - Similar instruction counts")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: How do you choose between E4M3 and E5M2?")
print("    A: Decision tree:")
print("       1. Is data range > 448? → E5M2")
print("       2. Is precision critical? → E4M3")
print("       3. Weights → E4M3 (static, precision matters)")
print("       4. Activations → E4M3 (usually bounded)")
print("       5. Gradients → E5M2 (dynamic range)")
print("       6. When in doubt → E4M3 (better accuracy)")

print("\n    Q: What's the FP8 quantization strategy for LLMs?")
print("    A: Common approach:")
print("       1. Weights: E4M3 per-channel (offline)")
print("       2. Activations: E4M3 per-token (online)")
print("       3. Gradients: E5M2 (if training in FP8)")
print("       4. Master weights: FP32 (for weight updates)")

# C4: Production guidance
print("\nC4: Production FP8 Format Selection")
print("    Use E4M3 for:")
print("      - Weights (quantized offline)")
print("      - Bounded activations")
print("      - Accuracy-critical paths")
print("\n    Use E5M2 for:")
print("      - Gradients (training)")
print("      - High dynamic range data")
print("      - Unbounded activations (with large outliers)")
print("\n    Mixed precision strategy:")
print("      - E4M3 weights + E5M2 activations = good balance")
print("      - Calibrate on representative data")
print("      - Monitor accuracy degradation")

print("\n" + "=" * 60)
print("Exercise 02 Complete!")
print("=" * 60)
