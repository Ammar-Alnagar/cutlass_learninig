# Benchmark Results

## Environment

- **GPU**: [Your GPU here]
- **CUDA**: 12.x
- **CUTLASS**: 3.x

## Roofline Analysis

```
[Insert roofline chart here]
```

## CUTLASS 3.x vs cuBLAS

| Kernel | cuBLAS TFLOPS | CUTLASS TFLOPS | % cuBLAS |
|--------|---------------|----------------|----------|
| Dense GEMM FP16 | - | - | - |
| Dense GEMM FP8 | - | - | - |
| Warp-spec GEMM | - | - | - |
| FA2 Attention | - | - | - |
| FA3 Attention | - | - | - |

## CUTLASS 3.x vs ThunderKittens

| Kernel | ThunderKittens TFLOPS | CUTLASS TFLOPS | Speedup |
|--------|----------------------|----------------|---------|
| Dense GEMM FP16 | - | - | - |
| Dense GEMM FP8 | - | - | - |
| Warp-spec GEMM | N/A | - | - |
| FA2 Attention | - | - | - |
| FA3 Attention | N/A | - | - |

## Module Benchmarks

### Module 01: CollectiveBuilder

| Config | Problem | Time (ms) | TFLOPS |
|--------|---------|-----------|--------|
| Small tiles | 1024³ | - | - |
| Medium tiles | 4096³ | - | - |
| Large tiles | 8192³ | - | - |

### Module 02: EVT Fusion

| Fusion | Time (ms) | Speedup |
|--------|-----------|---------|
| Unfused | - | 1.0× |
| Scale+Bias | - | - |
| ReLU | - | - |
| GELU | - | - |

### Module 03: Warp Specialization

| Config | Time (ms) | TFLOPS | Speedup |
|--------|-----------|--------|---------|
| Standard | - | - | 1.0× |
| Warp-spec | - | - | - |
| Ping-pong 8-stage | - | - | - |

### Module 04: StreamK

| Problem | Traditional (ms) | StreamK (ms) | Speedup |
|---------|-----------------|--------------|---------|
| Square aligned | - | - | - |
| Square misaligned | - | - | - |
| Tall/skinny | - | - | - |

### Module 05: Grouped GEMM

| Config | Sequential (ms) | Grouped (ms) | Speedup |
|--------|-----------------|--------------|---------|
| 8 experts | - | - | - |
| Variable tokens | - | - | - |

## Notes

- Fill in benchmark results after running on target hardware
- Use `ncu --set full` for detailed profiling
- Report average of 50+ runs for stability
