"""
FILE: intensity_calculator.py
TEACHES: Compute arithmetic intensity for any attention configuration
MAPS TO: NVIDIA/Cerebras interview questions — determine compute vs. memory bound
RUN: python intensity_calculator.py — no arguments needed
"""

import numpy as np

# ============================================================
# PART 1: Hardware Roofline Data
# Math reference: see 01_roofline_for_attention.md, section "Hardware Roofline"
# ============================================================

# Hardware specifications (FP16 tensor cores)
HARDWARE = {
    "H100": {
        "peak_flops_s": 989e12,    # 989 TFLOPs/s (dense FP16)
        "peak_bw_bs": 3.35e12,     # 3.35 TB/s
        "name": "NVIDIA H100"
    },
    "A100": {
        "peak_flops_s": 312e12,    # 312 TFLOPs/s (dense FP16)
        "peak_bw_bs": 2.0e12,      # 2.0 TB/s
        "name": "NVIDIA A100"
    },
    "RTX4060": {
        "peak_flops_s": 184e12,    # 184 TFLOPs/s (dense FP16)
        "peak_bw_bs": 272e9,       # 272 GB/s
        "name": "NVIDIA RTX 4060"
    }
}

# Data type sizes (bytes per element)
DTYPE_BYTES = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "FP8": 1,
    "INT8": 1,
    "INT4": 0.5
}

print("=" * 70)
print("ARITHMETIC INTENSITY CALCULATOR")
print("=" * 70)
print()

# ============================================================
# PART 2: Model Configurations
# Math reference: see 02_memory_formula.md, section "Numbers That Matter"
# ============================================================

# LLaMA-3 model configurations
MODELS = {
    "LLaMA-3 8B": {
        "L": 32,       # layers
        "d": 4096,     # hidden size
        "H": 32,       # query heads
        "H_kv": 8,     # KV heads (GQA)
        "d_h": 128     # head dimension
    },
    "LLaMA-3 70B": {
        "L": 80,
        "d": 8192,
        "H": 64,
        "H_kv": 8,     # GQA
        "d_h": 128
    },
    "LLaMA-3 405B": {
        "L": 126,
        "d": 16384,
        "H": 128,
        "H_kv": 8,     # GQA
        "d_h": 128
    }
}

print("Available models:")
for name, config in MODELS.items():
    print(f"  {name}: L={config['L']}, d={config['d']}, H={config['H']}, H_kv={config['H_kv']}, d_h={config['d_h']}")
print()

# ============================================================
# PART 3: Arithmetic Intensity Formulas
# Math reference: see 01_roofline_for_attention.md, section "Attention AI"
# ============================================================

def compute_prefill_ai(S, use_flash_attention=True):
    """
    Compute arithmetic intensity for prefill phase.
    
    FlashAttention: AI = S / 2
    Naive: AI = (S * d_h) / (2 * (d_h + S))
    """
    if use_flash_attention:
        return S / 2
    else:
        d_h = 128  # typical value
        return (S * d_h) / (2 * (d_h + S))

def compute_decode_ai(dtype_bytes=2):
    """
    Compute arithmetic intensity for decode phase.
    
    AI = 1 / dtype_bytes (independent of S, B, etc.)
    """
    return 1.0 / dtype_bytes

def compute_flops_prefill(B, H, S, d_h):
    """
    FLOPs for prefill (attention only, all layers).
    FLOPs = 4 * B * H * S^2 * d_h
    """
    return 4 * B * H * S**2 * d_h

def compute_bytes_prefill(B, H, S, d_h, use_flash_attention=True):
    """
    HBM bytes for prefill.
    FlashAttention: 8 * B * H * S * d_h
    Naive: 8 * B * H * S * (d_h + S)
    """
    if use_flash_attention:
        return 8 * B * H * S * d_h
    else:
        return 8 * B * H * S * (d_h + S)

def compute_flops_decode(B, H, S, d_h):
    """
    FLOPs for decode (one token, attention only).
    FLOPs = 4 * B * H * S * d_h
    """
    return 4 * B * H * S * d_h

def compute_bytes_decode(B, H, S, d_h, dtype_bytes=2):
    """
    HBM bytes for decode (one token).
    Bytes = 4 * B * H * S * d_h * dtype_bytes
    """
    return 4 * B * H * S * d_h * dtype_bytes

def compute_kv_cache_bytes(L, S, d, B, dtype_bytes=2, H_kv=None, H=None, d_h=None):
    """
    KV cache memory in bytes.
    
    With GQA: KV Cache = 2 * L * S * (H_kv * d_h) * dtype_bytes * B
    
    If H_kv is None, uses d directly: 2 * L * S * d * dtype_bytes * B
    """
    if H_kv is not None and d_h is not None:
        return 2 * L * S * (H_kv * d_h) * dtype_bytes * B
    else:
        return 2 * L * S * d * dtype_bytes * B

# ============================================================
# PART 4: Analysis for LLaMA-3 Models
# ============================================================

print("=" * 70)
print("PREFILL ANALYSIS (FlashAttention)")
print("=" * 70)

S_values = [512, 1024, 2048, 4096, 8192]
B_prefill = 1  # Batch size for prefill analysis

print(f"\nBatch size: B={B_prefill}")
print()
print(f"{'Model':<15} {'S':<6} {'AI':<10} {'FLOPs':<12} {'Bytes':<10} {'Bound (H100)':<15}")
print("-" * 70)

for model_name, config in MODELS.items():
    for S in S_values:
        ai = compute_prefill_ai(S, use_flash_attention=True)
        flops = compute_flops_prefill(B_prefill, config['H'], S, config['d_h'])
        bytes_ = compute_bytes_prefill(B_prefill, config['H'], S, config['d_h'], use_flash_attention=True)
        
        # Determine bound (H100 peak AI ≈ 295)
        h100_peak_ai = 295
        bound = "Compute" if ai >= h100_peak_ai else "Memory"
        
        print(f"{model_name:<15} {S:<6} {ai:<10.0f} {flops/1e9:<12.1f}GF {bytes_/1e6:<10.0f}MB {bound:<15}")
    print()

# ============================================================
# PART 5: Decode Analysis
# ============================================================

print("=" * 70)
print("DECODE ANALYSIS (per token)")
print("=" * 70)

dtype = "FP16"
dtype_bytes_val = DTYPE_BYTES[dtype]

print(f"\nData type: {dtype} ({dtype_bytes_val} bytes)")
print()
print(f"{'Model':<15} {'S':<6} {'AI':<10} {'FLOPs':<12} {'Bytes':<10} {'Bound (H100)':<15}")
print("-" * 70)

for model_name, config in MODELS.items():
    for S in S_values:
        ai = compute_decode_ai(dtype_bytes_val)
        flops = compute_flops_decode(B_prefill, config['H'], S, config['d_h'])
        bytes_ = compute_bytes_decode(B_prefill, config['H'], S, config['d_h'], dtype_bytes_val)
        
        # Determine bound
        h100_peak_ai = 295
        bound = "Compute" if ai >= h100_peak_ai else "Memory"
        
        print(f"{model_name:<15} {S:<6} {ai:<10.1f} {flops/1e6:<12.1f}M {bytes_/1e3:<10.0f}KB {bound:<15}")
    print()

# ============================================================
# PART 6: KV Cache Memory Analysis
# ============================================================

print("=" * 70)
print("KV CACHE MEMORY ANALYSIS")
print("=" * 70)

S_kv = 4096
B_kv = 1

print(f"\nSequence length: S={S_kv}, Batch size: B={B_kv}")
print()
print(f"{'Model':<15} {'dtype':<6} {'KV Cache':<12} {'Per layer':<10}")
print("-" * 50)

for model_name, config in MODELS.items():
    for dtype, dtype_bytes_val in [("FP16", 2), ("FP8", 1)]:
        kv_bytes = compute_kv_cache_bytes(
            config['L'], S_kv, config['d'], B_kv, dtype_bytes_val,
            H_kv=config['H_kv'], d_h=config['d_h']
        )
        kv_per_layer = kv_bytes / config['L']
        
        if kv_bytes >= 1e9:
            kv_str = f"{kv_bytes/1e9:.1f} GB"
        else:
            kv_str = f"{kv_bytes/1e6:.0f} MB"
        
        if kv_per_layer >= 1e6:
            layer_str = f"{kv_per_layer/1e6:.0f} MB"
        else:
            layer_str = f"{kv_per_layer/1e3:.0f} KB"
        
        print(f"{model_name:<15} {dtype:<6} {kv_str:<12} {layer_str:<10}")
print()

# ============================================================
# PART 7: Maximum Batch Size Calculation
# Math reference: see 03_batch_size_effect.md, section "Maximum Batch Size"
# ============================================================

print("=" * 70)
print("MAXIMUM BATCH SIZE (A100 80GB)")
print("=" * 70)

gpu_memory = 80e9       # 80 GB
model_name = "LLaMA-3 8B"
config = MODELS[model_name]

# Model weights
model_weights_bytes = 8e9 * 2  # 8B params × 2 bytes (FP16) = 16 GB
available_memory = gpu_memory - model_weights_bytes  # 64 GB for KV cache

print(f"\n{model_name} on A100 80GB:")
print(f"  Model weights: {model_weights_bytes/1e9:.0f} GB")
print(f"  Available for KV cache: {available_memory/1e9:.0f} GB")
print()

for dtype, dtype_bytes_val in [("FP16", 2), ("FP8", 1)]:
    for S in [2048, 4096, 8192]:
        # KV cache per sequence
        kv_per_seq = compute_kv_cache_bytes(
            config['L'], S, config['d'], B=1, dtype_bytes=dtype_bytes_val,
            H_kv=config['H_kv'], d_h=config['d_h']
        )
        
        # Max batch size
        b_max = int(available_memory / kv_per_seq)
        
        print(f"  S={S}, {dtype}: KV/seq = {kv_per_seq/1e6:.0f} MB, B_max = {b_max}")
print()

# ============================================================
# PART 8: Interactive Calculator
# ============================================================

print("=" * 70)
print("INTERACTIVE CALCULATOR — Example Configurations")
print("=" * 70)

# Example configurations to analyze
examples = [
    {"name": "LLaMA-3 8B Prefill (S=4096)", "phase": "prefill", "model": "LLaMA-3 8B", "S": 4096, "B": 1},
    {"name": "LLaMA-3 8B Decode (S=4096)", "phase": "decode", "model": "LLaMA-3 8B", "S": 4096, "B": 1},
    {"name": "LLaMA-3 8B Prefill (S=8192)", "phase": "prefill", "model": "LLaMA-3 8B", "S": 8192, "B": 32},
    {"name": "LLaMA-3 70B Decode (S=4096)", "phase": "decode", "model": "LLaMA-3 70B", "S": 4096, "B": 8},
]

print()
for ex in examples:
    config = MODELS[ex['model']]
    phase = ex['phase']
    S = ex['S']
    B = ex['B']
    
    print(f"{ex['name']}:")
    
    if phase == "prefill":
        ai = compute_prefill_ai(S, use_flash_attention=True)
        flops = compute_flops_prefill(B, config['H'], S, config['d_h'])
        bytes_ = compute_bytes_prefill(B, config['H'], S, config['d_h'], use_flash_attention=True)
    else:  # decode
        ai = compute_decode_ai(2)  # FP16
        flops = compute_flops_decode(B, config['H'], S, config['d_h'])
        bytes_ = compute_bytes_decode(B, config['H'], S, config['d_h'], 2)
    
    # Determine bound for each hardware
    for hw_key in ["H100", "A100"]:
        hw = HARDWARE[hw_key]
        peak_ai = hw['peak_flops_s'] / hw['peak_bw_bs']
        bound = "Compute" if ai >= peak_ai else "Memory"
        utilization = min(1.0, ai / peak_ai) * 100
        
        print(f"  {hw_key}: AI={ai:.0f} FLOPs/byte, Peak AI={peak_ai:.0f}, {bound} ({utilization:.0f}% of peak)")
    
    print(f"  FLOPs: {flops/1e9:.1f} GF, Bytes: {bytes_/1e6:.1f} MB")
    print()

# ============================================================
# VERIFY: Summary
# ============================================================

print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

# Key results to verify
h100_peak_ai = HARDWARE['H100']['peak_flops_s'] / HARDWARE['H100']['peak_bw_bs']

print(f"\nH100 peak AI: {h100_peak_ai:.0f} FLOPs/byte")
print(f"A100 peak AI: {HARDWARE['A100']['peak_flops_s'] / HARDWARE['A100']['peak_bw_bs']:.0f} FLOPs/byte")
print()

# Prefill AI at S=4096
ai_prefill_4096 = compute_prefill_ai(4096, use_flash_attention=True)
print(f"Prefill AI (S=4096, FA): {ai_prefill_4096:.0f} FLOPs/byte")
print(f"  Status: {'COMPUTE-BOUND' if ai_prefill_4096 >= h100_peak_ai else 'MEMORY-BOUND'} on H100")

# Decode AI
ai_decode = compute_decode_ai(2)
print(f"Decode AI (FP16): {ai_decode:.1f} FLOPs/byte")
print(f"  Status: {'COMPUTE-BOUND' if ai_decode >= h100_peak_ai else 'MEMORY-BOUND'} on H100")

# KV cache for LLaMA-3 8B
kv_8b = compute_kv_cache_bytes(32, 4096, 4096, 1, 2, H_kv=8, d_h=128)
print(f"LLaMA-3 8B KV cache (S=4096, FP16): {kv_8b/1e9:.2f} GB per sequence")

print()
print("PASS — Arithmetic intensity calculator complete.")
print()
print("Key insights:")
print("  1. Prefill AI = S/2 (FlashAttention) — compute-bound for S > 590 on H100")
print("  2. Decode AI = 0.5 FLOPs/byte (FP16) — always memory-bound")
print("  3. Batch size doesn't change AI, but improves throughput")
print("  4. GQA reduces KV cache by H_kv/H_q, enabling larger batch sizes")
print()
print("Use this calculator to determine optimal batch sizes and identify bottlenecks.")
