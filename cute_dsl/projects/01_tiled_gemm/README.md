# Project 01 — Tiled GEMM

## Target Performance

| GPU | Peak FP16 TFLOPS | Target (>75%) |
|-----|------------------|---------------|
| A100 (SM80) | 312 | 234 TFLOPS |
| H100 (SM90) | 989 | 742 TFLOPS |
| B200 (SM100) | 2250 | 1688 TFLOPS |

## Algorithm

```
// Standard GEMM: C = A @ B
// C[M,N] = sum_k A[M,K] * B[K,N]

for m_tile in range(ceil(M / TILE_M)):
    for n_tile in range(ceil(N / TILE_N)):
        accum = zeros(TILE_M, TILE_N)
        for k_tile in range(ceil(K / TILE_K)):
            # Load tiles
            A_tile = load(A[m_tile*TILE_M : (m_tile+1)*TILE_M, 
                          k_tile*TILE_K : (k_tile+1)*TILE_K])
            B_tile = load(B[k_tile*TILE_K : (k_tile+1)*TILE_K,
                          n_tile*TILE_N : (n_tile+1)*TILE_N])
            # MMA
            accum += A_tile @ B_tile
        store(C[m_tile*TILE_M : (m_tile+1)*TILE_M,
                n_tile*TILE_N : (n_tile+1)*TILE_N], accum)
```

## Tiling Diagram

```
┌─────────────────────────────────┐
│            C (M x N)            │
│  ┌───────┬───────┬───────┐      │
│  │Tile(0,│Tile(0,│ ...   │      │
│  │  0)   │  1)   │       │      │
│  ├───────┼───────┼───────┤      │
│  │Tile(1,│Tile(1,│ ...   │      │
│  │  0)   │  1)   │       │      │
│  └───────┴───────┴───────┘      │
└─────────────────────────────────┘
```

## Files

- `gemm_ampere.py` — SM80 implementation with double-buffer
- `gemm_hopper.py` — SM90 implementation with TMA and warp specialization
- `gemm_blackwell.py` — SM100 implementation with tcgen05
- `benchmark.py` — TFLOPS measurement vs cuBLAS

## Job Relevance

This project directly maps to:
- **NVIDIA DL Software Engineer**: GEMM optimization is core interview topic
- **Cerebras Performance Engineer**: Wafer-scale matrix multiply
- **vLLM/TensorRT-LLM**: Foundation for all linear layers

## Nsight Compute Metrics

```bash
ncu --metrics tensor__pipe_tensor_op_hmma.sum,\
            smsp__inst_executed_pipe_tensor.sum,\
            gpu__time_duration.sum \
    --set full --target-processes all \
    python gemm_ampere.py
```
