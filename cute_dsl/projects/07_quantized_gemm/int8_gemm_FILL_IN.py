"""
Project 07 — INT8 GEMM

ALGORITHM: INT8 GEMM with INT32 accumulation
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch
import time

M, N, K = 4096, 4096, 4096


@cutlass.jit
def kernel_int8_gemm(
    A_int8: cute.Tensor,
    B_int8: cute.Tensor,
    C_int32: cute.Tensor,
    alpha: float,
    beta: float,
):
    """
    INT8 GEMM with INT32 accumulation.
    
    FILL IN [HARD]: Implement INT8×INT8→INT32 GEMM.
    
    HINT:
      1. Use INT8 MMA atom
      2. Accumulate in INT32
      3. Handle quantization scaling
    """
    pass


def quantize_to_int8(x):
    """Quantize FP16 tensor to INT8."""
    x_abs_max = x.abs().max()
    scale = x_abs_max / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale.item()


def run():
    print("\n" + "=" * 60)
    print("  Project 07 — INT8 GEMM")
    print("=" * 60)
    
    print(f"\n  Matrix sizes: A[{M},{K}], B[{K},{N}], C[{M},{N}]")
    
    torch.manual_seed(42)
    
    A_fp16 = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B_fp16 = torch.randn((K, N), dtype=torch.float16, device='cuda')
    
    A_int8, scale_a = quantize_to_int8(A_fp16)
    B_int8, scale_b = quantize_to_int8(B_fp16)
    
    C_int32 = torch.zeros((M, N), dtype=torch.int32, device='cuda')
    
    print(f"\n  Quantization scales: A={scale_a:.6f}, B={scale_b:.6f}")
    
    A_cute = from_dlpack(A_int8)
    B_cute = from_dlpack(B_int8)
    C_cute = from_dlpack(C_int32)
    
    kernel_int8_gemm[64, 128](A_cute, B_cute, C_cute, 1.0, 0.0)
    torch.cuda.synchronize()
    
    print("\n  ✓ Kernel launched (placeholder)")
    print("=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    run()
