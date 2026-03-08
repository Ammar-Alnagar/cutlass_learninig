# Project 08 — Benchmarks Master

## Files
- `roofline.py` — Auto roofline chart generator
- `compare_cutlass_cpp.py` — CuTe DSL vs C++ perf table
- `results/` — CSV + PNG benchmark outputs

## Roofline Model

```
                    ┌────────────────────────────────────────┐
                    │            Roofline Chart              │
                    │                                        │
                    │  1000 │          ● H100 Peak           │
                    │       │        ╱                       │
                    │   500 │      ╱  ● FA3 (700 TFLOPS)     │
                    │       │    ╱    ● FA2 (550 TFLOPS)     │
                    │   100 │  ╱      ● GEMM (234 TFLOPS)    │
                    │       │╱────────────────────────────   │
                    │    10 │────────────────────────────    │
                    │       └────────┴─────────┴─────────────│
                    │            0.01    0.1      1    100   │
                    │          Arithmetic Intensity (FLOP/B) │
                    └────────────────────────────────────────┘
```

## Metrics to Collect

| Kernel | Metric | Target |
|--------|--------|--------|
| GEMM | TFLOPS | >75% peak |
| Softmax | GB/s | >85% BW |
| FA2 | TFLOPS | >150 (A100) |
| FA3 | TFLOPS | >700 (H100) |
