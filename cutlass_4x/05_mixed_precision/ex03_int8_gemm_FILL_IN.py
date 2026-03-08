"""
Module 05 — Mixed Precision
Exercise 03 — INT8 GEMM (Quantized Inference)

LEVEL: 1 (High-level op API)

WHAT YOU'RE BUILDING:
  INT8 GEMM for quantized inference — the production pattern used in 
  TensorRT, DeepLearning, and neural network accelerators. INT8 provides 
  4× memory savings vs FP32 and 2× vs FP16 with careful quantization.

OBJECTIVE:
  - Configure GEMM for INT8 inputs with INT32 accumulation
  - Understand symmetric vs asymmetric quantization
  - Learn per-tensor vs per-channel quantization
  - Compare INT8 vs FP16 accuracy and performance
"""

import torch
import cutlass
import time
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What's the range of INT8? How does quantization map FP32 to INT8?

# Q2: Why use INT32 accumulation instead of INT8?

# Q3: What's the difference between per-tensor and per-channel quantization?
#     Which is more accurate?


# ==============================================================================
# SETUP
# ==============================================================================

# Matrix dimensions (typical quantized linear layer)
M, K, N = 512, 2048, 8192
dtype_int8 = torch.int8
dtype_int32 = torch.int32
dtype_fp32 = torch.float32
device = torch.device("cuda")

print("=" * 60)
print("INT8 GEMM (Quantized Inference)")
print("=" * 60)
print(f"\nMatrix: M={M}, K={K}, N={N}")
print(f"INT8 range: [-128, 127]")
print(f"INT32 accumulation: [-2^31, 2^31-1]")


# ==============================================================================
# QUANTIZATION FUNCTIONS
# ==============================================================================

@dataclass
class QuantParams:
    """Quantization parameters."""
    scale: float
    zero_point: int  # For asymmetric quantization
    quant_min: int = -128
    quant_max: int = 127


def quantize_per_tensor_symmetric(tensor_fp32: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Symmetric per-tensor quantization.
    
    scale = max(|tensor|) / 127
    quantized = round(tensor / scale)
    """
    scale = tensor_fp32.abs().max() / 127.0
    quantized = (tensor_fp32 / scale).round().clamp(-128, 127).to(dtype_int8)
    return quantized, scale.item()


def quantize_per_tensor_asymmetric(tensor_fp32: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """
    Asymmetric per-tensor quantization.
    
    scale = (max - min) / 255
    zero_point = round(-min / scale) - 128
    quantized = round(tensor / scale) + zero_point
    """
    min_val = tensor_fp32.min()
    max_val = tensor_fp32.max()
    scale = (max_val - min_val) / 255.0
    zero_point = round(-min_val / scale).item() - 128
    quantized = (tensor_fp32 / scale).round().add(zero_point).clamp(-128, 127).to(dtype_int8)
    return quantized, scale.item(), zero_point


def quantize_per_channel_symmetric(tensor_fp32: torch.Tensor, dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-channel quantization.
    
    For weight tensors (K, N), quantize each output channel separately.
    """
    # Move quantization dim to last
    if dim != -1 and dim != tensor_fp32.dim() - 1:
        tensor_fp32 = tensor_fp32.transpose(dim, -1)
    
    # Compute scale per channel
    shape = tensor_fp32.shape
    flat = tensor_fp32.reshape(-1, shape[-1])  # [*, channels]
    scales = flat.abs().max(dim=0).values / 127.0
    
    # Quantize
    quantized = (tensor_fp32 / scales).round().clamp(-128, 127).to(dtype_int8)
    
    return quantized, scales


def dequantize(quantized: torch.Tensor, scale: float, zero_point: int = 0) -> torch.Tensor:
    """Dequantize INT8 tensor to FP32."""
    return (quantized.to(dtype_fp32) - zero_point) * scale


# Create FP32 reference tensors
A_fp32 = torch.randn(M, K, dtype=dtype_fp32, device=device) * 0.5  # activations
B_fp32 = torch.randn(K, N, dtype=dtype_fp32, device=device) * 0.1  # weights

# Reference: FP32 GEMM
C_fp32_ref = torch.mm(A_fp32, B_fp32)


# ==============================================================================
# QUANTIZE TENSORS
# ==============================================================================

print("\n" + "=" * 60)
print("Quantization Methods")
print("=" * 60)

# Method 1: Per-tensor symmetric
A_int8_pts, A_scale_pts = quantize_per_tensor_symmetric(A_fp32)
B_int8_pts, B_scale_pts = quantize_per_tensor_symmetric(B_fp32)

# Dequantize and compute
A_dequant_pts = dequantize(A_int8_pts, A_scale_pts)
B_dequant_pts = dequantize(B_int8_pts, B_scale_pts)
C_pts_sim = torch.mm(A_dequant_pts, B_dequant_pts)

pts_error = (C_pts_sim - C_fp32_ref).abs().div(C_fp32_ref.abs() + 1e-8).max().item()
print(f"\n1. Per-Tensor Symmetric:")
print(f"   A scale: {A_scale_pts:.6f}, B scale: {B_scale_pts:.6f}")
print(f"   Max relative error: {pts_error*100:.2f}%")

# Method 2: Per-tensor asymmetric
A_int8_pta, A_scale_pta, A_zp_pta = quantize_per_tensor_asymmetric(A_fp32)
B_int8_pta, B_scale_pta, B_zp_pta = quantize_per_tensor_asymmetric(B_fp32)

A_dequant_pta = dequantize(A_int8_pta, A_scale_pta, A_zp_pta)
B_dequant_pta = dequantize(B_int8_pta, B_scale_pta, B_zp_pta)
C_pta_sim = torch.mm(A_dequant_pta, B_dequant_pta)

pta_error = (C_pta_sim - C_fp32_ref).abs().div(C_fp32_ref.abs() + 1e-8).max().item()
print(f"\n2. Per-Tensor Asymmetric:")
print(f"   A scale: {A_scale_pta:.6f}, zp: {A_zp_pta}")
print(f"   B scale: {B_scale_pta:.6f}, zp: {B_zp_pta}")
print(f"   Max relative error: {pta_error*100:.2f}%")

# Method 3: Per-channel symmetric (recommended for weights)
B_int8_pc, B_scales_pc = quantize_per_channel_symmetric(B_fp32, dim=0)

# For per-channel, dequantization is slightly different
B_dequant_pc = B_int8_pc.to(dtype_fp32) * B_scales_pc
A_dequant_pc = dequantize(A_int8_pts, A_scale_pts)  # activation still per-tensor
C_pc_sim = torch.mm(A_dequant_pc, B_dequant_pc)

pc_error = (C_pc_sim - C_fp32_ref).abs().div(C_fp32_ref.abs() + 1e-8).max().item()
print(f"\n3. Per-Channel Symmetric (weights):")
print(f"   A scale: {A_scale_pts:.6f} (per-tensor)")
print(f"   B scales: {B_scales_pc.min().item():.6f} - {B_scales_pc.max().item():.6f} (per-channel)")
print(f"   Max relative error: {pc_error*100:.2f}%")

print(f"\n{'='*50}")
print(f"Best accuracy: {'Per-Channel' if pc_error < min(pts_error, pta_error) else 'Per-Tensor'}")
print(f"{'='*50}")


# ==============================================================================
# FILL IN: Level 1 — INT8 GEMM with cutlass.op
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1: High-Level Op API (cutlass.op.Gemm INT8)")
print("=" * 60)

# TODO [HARD]: Configure INT8 GEMM
# HINT:
#   - Use cutlass.int8 for element type
#   - Use cutlass.int32 for accumulator (important!)
#   - Consider per-channel quantization for weights
# REF: cutlass/examples/python/CuTeDSL/int8_gemm.py

# TODO: Create INT8 GEMM plan
# plan_int8 = cutlass.op.Gemm(
#     element=cutlass.int8,
#     accumulator_type=cutlass.int32,
#     layout=cutlass.LayoutType.RowMajor,
# )

# TODO: Run INT8 GEMM
# C_int32 = torch.zeros(M, N, dtype=torch.int32, device=device)
# plan_int8.run(from_dlpack(A_int8), from_dlpack(B_int8), C_int32)

# TODO: Dequantize output
# output_scale = A_scale * B_scale  # or per-channel scales
# C_fp32 = C_int32.to(torch.float32) * output_scale

# Placeholder (replace with implementation)
plan_int8 = None
C_int32 = torch.zeros(M, N, dtype=dtype_int32, device=device)

# Simulated INT8 GEMM (using FP32 emulation)
output_scale = A_scale_pts * B_scale_pts
C_int32_sim = (C_pts_sim / output_scale).round().to(dtype_int32)
C_fp32_from_int8 = C_int32_sim.to(dtype_fp32) * output_scale

print(f"\nCUTLASS INT8 GEMM configured")
print(f"Output dtype: INT32 (accumulation)")


# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

# Compare INT8 result to FP32 reference
max_abs_error = (C_fp32_from_int8 - C_fp32_ref).abs().max().item()
max_rel_error = (C_fp32_from_int8 - C_fp32_ref).abs().div(C_fp32_ref.abs() + 1e-8).max().item()

print(f"\nINT8 vs FP32 Reference:")
print(f"  Max Absolute Error: {max_abs_error:.6f}")
print(f"  Max Relative Error: {max_rel_error:.4f} ({max_rel_error*100:.2f}%)")

# Typical INT8 accuracy
print(f"\nTypical INT8 accuracy:")
print(f"  < 1% error: Excellent (well-quantized model)")
print(f"  1-5% error: Good (acceptable for most inference)")
print(f"  > 5% error: May need quantization-aware training")


# ==============================================================================
# BENCHMARK: INT8 vs FP16
# ==============================================================================

def benchmark_int8_gemm(A_int8, B_int8, C_int32, plan,
                        num_warmup=10, num_iters=100) -> float:
    """Benchmark INT8 GEMM."""
    if plan is None:
        # Fallback to simulated
        A_fp = A_int8.to(dtype_fp32)
        B_fp = B_int8.to(dtype_fp32)
        for _ in range(num_warmup):
            _ = torch.mm(A_fp, B_fp)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = torch.mm(A_fp, B_fp)
        torch.cuda.synchronize()
        
        return (time.perf_counter() - start) / num_iters * 1000
    
    # Warmup
    for _ in range(num_warmup):
        plan.run(from_dlpack(A_int8), from_dlpack(B_int8), C_int32)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        plan.run(from_dlpack(A_int8), from_dlpack(B_int8), C_int32)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def benchmark_fp16_gemm(A_fp16, B_fp16, num_warmup=10, num_iters=100) -> float:
    """Benchmark FP16 GEMM."""
    C = torch.zeros(A_fp16.shape[0], B_fp16.shape[1], dtype=A_fp16.dtype, device=A_fp16.device)
    for _ in range(num_warmup):
        C.copy_(torch.mm(A_fp16, B_fp16))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A_fp16, B_fp16))
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


# Prepare FP16 tensors
A_fp16 = A_fp32.to(torch.float16)
B_fp16 = B_fp32.to(torch.float16)

print("\n" + "=" * 60)
print("Performance: INT8 vs FP16")
print("=" * 60)

int8_latency = benchmark_int8_gemm(A_int8_pts, B_int8_pts, C_int32, plan_int8)
fp16_latency = benchmark_fp16_gemm(A_fp16, B_fp16)

print(f"\nResults:")
print(f"  INT8:  {int8_latency:.3f} ms")
print(f"  FP16:  {fp16_latency:.3f} ms")

if int8_latency > 0 and fp16_latency > 0:
    speedup = fp16_latency / int8_latency
    print(f"\n  Speedup: {speedup:.2f}×")
    
    # Compute TFLOPS (INT8 ops are counted differently)
    # INT8 Tensor Core: 2× FP16 throughput on Ampere+
    flops = 2 * M * N * K
    int8_tflops = flops / (int8_latency * 1e-3) / 1e12
    fp16_tflops = flops / (fp16_latency * 1e-3) / 1e12
    
    print(f"\n  INT8 OP/s:  {int8_tflops:.1f} TOPS (theoretical 2× FP16)")
    print(f"  FP16 TFLOPS: {fp16_tflops:.1f}")


# ==============================================================================
# MEMORY ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("Memory Analysis")
print("=" * 60)

bytes_int8 = 1
bytes_fp16 = 2
bytes_fp32 = 4

memory_int8 = M * K * bytes_int8 + K * N * bytes_int8  # inputs only
memory_fp16 = M * K * bytes_fp16 + K * N * bytes_fp16

print(f"\nWeight + Activation Storage:")
print(f"  INT8:  {memory_int8 / 1e6:.1f} MB")
print(f"  FP16:  {memory_fp16 / 1e6:.1f} MB")
print(f"  FP32:  {(M * K + K * N) * bytes_fp32 / 1e6:.1f} MB")
print(f"\n  INT8 vs FP16: {(1 - memory_int8 / memory_fp16) * 100:.0f}% savings")
print(f"  INT8 vs FP32: {(1 - memory_int8 / ((M * K + K * N) * bytes_fp32)) * 100:.0f}% savings")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: INT8 range?")
print("        Answer: [-128, 127] (signed 8-bit)")
print("        Quantization: scale = max(|x|) / 127")
print("        dequant: x_fp32 = x_int8 * scale")

print("\n    Q2: Why INT32 accumulation?")
print("        Answer: Prevent overflow!")
print("        INT8 × INT8 = INT16, but sum of many products")
print("        can exceed INT16 range. INT32 provides headroom.")
print("        Max possible: 128 × 128 × K (for K up to ~131K)")

print("\n    Q3: Per-tensor vs per-channel?")
print(f"        Per-tensor error: {pts_error*100:.2f}%")
print(f"        Per-channel error: {pc_error*100:.2f}%")
print("        Per-channel is more accurate (handles weight scale")
print("        variation across output channels)")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics sm__inst_executed_pipe_tensor.sum,\\")
print(f"                dram__throughput.sum \\")
print(f"        python ex03_int8_gemm_FILL_IN.py")
print("\n    Look for:")
print("      - Higher tensor core throughput for INT8")
print("      - Lower memory bandwidth for INT8")
print("      - INT32 store at output")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: When should you use INT8 vs FP8?")
print("    A: INT8 advantages:")
print("       - Mature tooling (TensorRT, ONNX Runtime)")
print("       - Better for bounded, symmetric data")
print("       - Integer arithmetic (some GPUs faster)")
print("       FP8 advantages:")
print("       - Better dynamic range (exponent)")
print("       - No quantization params needed")
print("       - Native on Hopper")
print("       Rule: FP8 for new Hopper deployments,")
print("             INT8 for existing Ampere/legacy")

print("\n    Q: What is quantization-aware training (QAT)?")
print("    A: QAT simulates quantization during training:")
print("       1. Insert fake quantize/dequantize ops")
print("       2. Train with quantization noise")
print("       3. Model learns to be robust to quantization")
print("       Result: Much better INT8 accuracy (< 1% loss)")

# C4: Production guidance
print("\nC4: Production INT8 Tips")
print("    Quantization strategy:")
print("      1. Weights: Per-channel symmetric (offline)")
print("      2. Activations: Per-tensor dynamic (online)")
print("      3. Calibrate on 100-1000 representative samples")
print("\n    When INT8 may not work:")
print("      - Models with outliers (LLaMA attention)")
print("      - Very small models (< 10M params)")
print("      - High-accuracy requirements (> 99% FP32)")
print("      Consider: FP8, mixed precision, or QAT")

print("\n" + "=" * 60)
print("Exercise 03 Complete!")
print("=" * 60)
