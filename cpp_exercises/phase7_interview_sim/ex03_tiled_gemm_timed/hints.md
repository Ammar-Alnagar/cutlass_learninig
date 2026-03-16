# Hints for ex03_tiled_gemm_timed

## H1 — Concept Direction
Each thread block computes one TILE_SIZE x TILE_SIZE tile of the output matrix C. Threads cooperate to load tiles of A and B into shared memory, then each thread computes one element of the output tile.

## H2 — Names the Tool
Shared memory: `__shared__ float As[TILE_SIZE][TILE_SIZE]`. Synchronization: `__syncthreads()` after loading, before reusing. Boundary check: `if (row < M && col < N)`.

## H3 — Minimal Usage (Unrelated Context)
```cpp
__shared__ float tile[16][16];

// Load
tile[threadIdx.y][threadIdx.x] = global[row * K + col];

// Sync before using
__syncthreads();

// Compute
sum += tile[threadIdx.y][k] * tile[k][threadIdx.x];

// Sync before overwriting
__syncthreads();
```
