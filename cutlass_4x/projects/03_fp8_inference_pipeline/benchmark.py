"""
Project 03 — FP8 Inference Pipeline
File: benchmark.py

Benchmark FP8 inference pipeline vs FP16 baseline with accuracy evaluation.
"""

import torch
import time
import math
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class BenchmarkConfig:
    batch_size: int
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    dtype: torch.dtype
    fp8_dtype: torch.dtype
    num_warmup: int
    num_iters: int


config = BenchmarkConfig(
    batch_size=32,
    seq_len=512,
    hidden_size=1024,
    intermediate_size=4096,
    num_layers=4,
    dtype=torch.float16,
    fp8_dtype=torch.float8_e4m3fn,
    num_warmup=10,
    num_iters=50,
)


class FP8Linear(torch.nn.Module):
    """Simplified FP8 linear layer for benchmark."""
    
    def __init__(self, in_features, out_features, dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype))
        self.fp8_max = 448.0
    
    def forward(self, x):
        # Simulate FP8 quantization
        weight_max = self.weight.abs().max()
        scale = weight_max / self.fp8_max
        weight_fp8 = (self.weight / scale.clamp(min=1e-8)).clamp(-128, 127) / 128
        
        x_fp8 = x  # Skip activation quantization for benchmark
        
        # Compute with quantized weights
        output = torch.nn.functional.linear(x_fp8, weight_fp8, self.bias)
        return output


class TransformerBlock(torch.nn.Module):
    """Simple transformer block for accuracy testing."""
    
    def __init__(self, hidden_size, intermediate_size, use_fp8=False):
        super().__init__()
        linear_cls = FP8Linear if use_fp8 else torch.nn.Linear
        
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.mlp1 = linear_cls(hidden_size, intermediate_size)
        self.mlp2 = linear_cls(intermediate_size, hidden_size)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = torch.nn.functional.gelu(self.mlp1(x))
        mlp_out = self.mlp2(mlp_out)
        x = self.norm2(x + mlp_out)
        
        return x


def benchmark_model(model, input_tensor, num_warmup=10, num_iters=50):
    """Benchmark model forward pass."""
    model.eval()
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / num_iters * 1000


def evaluate_accuracy(fp16_model, fp8_model, input_tensor, criterion):
    """Evaluate FP8 model accuracy vs FP16."""
    fp16_model.eval()
    fp8_model.eval()
    
    with torch.no_grad():
        fp16_output = fp16_model(input_tensor)
        fp8_output = fp8_model(input_tensor)
    
    # Compute error metrics
    mse = criterion(fp8_output, fp16_output)
    max_error = (fp8_output - fp16_output).abs().max().item()
    
    return {
        'mse': mse.item(),
        'max_error': max_error,
        'correlation': torch.corrcoef(fp8_output.flatten(), fp16_output.flatten())[0, 1].item(),
    }


def run_benchmark():
    print("=" * 70)
    print("FP8 Inference Pipeline Benchmark")
    print("=" * 70)
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability()
    sm = compute_cap[0] * 10 + compute_cap[1]
    
    print(f"\nGPU: {gpu_name} (SM{sm})")
    print(f"Configuration:")
    print(f"  Batch: {config.batch_size}, Seq: {config.seq_len}")
    print(f"  Hidden: {config.hidden_size}, Intermediate: {config.intermediate_size}")
    print(f"  Layers: {config.num_layers}")
    
    # Create input
    input_tensor = torch.randn(config.batch_size, config.seq_len, config.hidden_size,
                                dtype=config.dtype, device=device)
    
    # Create models
    def create_model(use_fp8):
        layers = []
        for _ in range(config.num_layers):
            layers.append(TransformerBlock(
                config.hidden_size, config.intermediate_size, use_fp8=use_fp8
            ))
        return torch.nn.Sequential(*layers).to(device)
    
    fp16_model = create_model(use_fp8=False)
    fp8_model = create_model(use_fp8=True)
    
    # Copy weights for fair comparison
    for fp16_block, fp8_block in zip(fp16_model, fp8_model):
        fp8_block.mlp1.weight.data = fp16_block.mlp1.weight.data.clone()
        fp8_block.mlp1.bias.data = fp16_block.mlp1.bias.data.clone()
        fp8_block.mlp2.weight.data = fp16_block.mlp2.weight.data.clone()
        fp8_block.mlp2.bias.data = fp16_block.mlp2.bias.data.clone()
    
    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    
    fp16_latency = benchmark_model(fp16_model, input_tensor, config.num_warmup, config.num_iters)
    fp8_latency = benchmark_model(fp8_model, input_tensor, config.num_warmup, config.num_iters)
    
    print(f"\nFP16 Model: {fp16_latency:.3f} ms")
    print(f"FP8 Model:  {fp8_latency:.3f} ms")
    
    if fp8_latency > 0:
        speedup = fp16_latency / fp8_latency
        print(f"\nSpeedup: {speedup:.2f}×")
        print(f"Target:  >1.5× on Hopper")
    
    # Tokens per second
    tokens_per_iter = config.batch_size * config.seq_len
    tokens_per_sec = tokens_per_iter / (fp8_latency * 1e-3)
    print(f"\nTokens/sec: {tokens_per_sec/1000:.1f}K")
    
    print("\n" + "=" * 70)
    print("Accuracy Evaluation")
    print("=" * 70)
    
    criterion = torch.nn.MSELoss()
    accuracy = evaluate_accuracy(fp16_model, fp8_model, input_tensor, criterion)
    
    print(f"\nMSE:           {accuracy['mse']:.6f}")
    print(f"Max Error:     {accuracy['max_error']:.6f}")
    print(f"Correlation:   {accuracy['correlation']:.4f}")
    
    # Accuracy assessment
    if accuracy['correlation'] > 0.99:
        print(f"\n✓ Excellent accuracy (correlation > 0.99)")
    elif accuracy['correlation'] > 0.95:
        print(f"✓ Good accuracy (correlation > 0.95)")
    else:
        print(f"⚠ Accuracy may need improvement")
    
    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"{'Metric':<20} {'FP16':<15} {'FP8':<15} {'Change'}")
    print("-" * 60)
    print(f"{'Latency (ms)':<20} {fp16_latency:<15.3f} {fp8_latency:<15.3f} {100*(fp8_latency-fp16_latency)/fp16_latency:+.1f}%")
    print(f"{'Tokens/sec':<20} {tokens_per_iter/(fp16_latency*1e-3)/1000:<15.1f}K {tokens_per_sec/1000:<15.1f}K {100*(tokens_per_sec/(tokens_per_iter/(fp16_latency*1e-3))-1):+.1f}%")
    print(f"{'Accuracy':<20} {'100%':<15} {accuracy['correlation']*100:<15.1f}% {100*(accuracy['correlation']-1):+.1f}%")


if __name__ == "__main__":
    run_benchmark()
