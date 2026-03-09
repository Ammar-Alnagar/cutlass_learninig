#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

__global__ void async_pipeline_kernel(float *gmem_data, float *gmem_out) {
  constexpr int TILE_SIZE = 256;
  constexpr int NUM_TILES = 4;
  constexpr int TOTAL_SIZE = TILE_SIZE * NUM_TILES;

  auto gmem_ptr = make_gmem_ptr<float>(gmem_data);
  auto gmem_out_ptr = make_gmem_ptr<float>(gmem_out);

  __shared__ float smem_buffer[2 * TILE_SIZE];
  auto smem_ptr = make_smem_ptr<float>(smem_buffer);

  auto smem_0 = make_tensor(smem_ptr, make_layout(Int<TILE_SIZE>{}));
  auto smem_1 =
      make_tensor(smem_ptr + TILE_SIZE, make_layout(Int<TILE_SIZE>{}));

  // TODO 1: async copy atom (128-bit, float)
  using CopyAtom = /* YOUR CODE HERE */;

  // TODO 2: tiled_copy — same pattern as ex02 (128 threads, 4 values)
  auto tiled_copy = /* YOUR CODE HERE */;

  // TODO 3: get per-thread slice — same as ex02
  auto thr_copy = /* YOUR CODE HERE */;

  int write_stage = 0;
  int read_stage = 0;

  // Prologue: load first tile
  auto full_gmem = make_tensor(gmem_ptr, make_layout(Int<TOTAL_SIZE>{}));
  auto gmem_tile =
      local_tile(full_gmem, make_layout(Int<TILE_SIZE>{}), make_coord(0));

  // TODO 4: partition and copy gmem_tile -> smem_0
  auto src = /* YOUR CODE HERE */;
  auto dst = /* YOUR CODE HERE */;
  copy(tiled_copy, src, dst);

  // TODO 5: commit the async copy
  /* YOUR CODE HERE */;

  // Main loop
  for (int tile_idx = 1; tile_idx < NUM_TILES; tile_idx++) {
    // TODO 6: wait for previous load to complete
    /* YOUR CODE HERE */;

    __syncthreads();

    auto smem_current = (read_stage == 0) ? smem_0 : smem_1;

    for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
      float val = smem_current(i);
      val = val * 2.0f + 1.0f;
      smem_current(i) = val;
    }

    // Load next tile
    gmem_tile = local_tile(full_gmem, make_layout(Int<TILE_SIZE>{}),
                           make_coord(tile_idx));
    auto smem_next = (write_stage == 0) ? smem_0 : smem_1;

    // TODO 7: partition and copy gmem_tile -> smem_next
    auto src2 = /* YOUR CODE HERE */;
    auto dst2 = /* YOUR CODE HERE */;
    copy(tiled_copy, src2, dst2);

    // TODO 8: commit async copy for next tile
    /* YOUR CODE HERE */;

    write_stage = 1 - write_stage;
    read_stage = 1 - read_stage;
  }

  // Epilogue
  // TODO 9: wait for all pending async copies
  /* YOUR CODE HERE */;

  __syncthreads();

  auto smem_current = (read_stage == 0) ? smem_0 : smem_1;
  for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
    float val = smem_current(i);
    val = val * 2.0f + 1.0f;
    smem_current(i) = val;
  }

  // Write back — only first tile for verification
  auto gmem_dst =
      local_tile(make_tensor(gmem_out_ptr, make_layout(Int<TOTAL_SIZE>{})),
                 make_layout(Int<TILE_SIZE>{}), make_coord(0));

  if (threadIdx.x == 0) {
    for (int i = 0; i < TILE_SIZE; i++)
      gmem_dst(i) = smem_0(i);
  }
}

void cpu_reference_pipeline(float *input, float *output, int tile_size,
                            int num_tiles) {
  for (int t = 0; t < num_tiles; t++)
    for (int i = 0; i < tile_size; i++)
      output[t * tile_size + i] = input[t * tile_size + i] * 2.0f + 1.0f;
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("=== Async Copy Pipeline ===\nGPU: %s\n", prop.name);

  if (prop.major < 8) {
    printf("ERROR: cp.async requires sm_80+\n");
    return 1;
  }

  constexpr int TILE_SIZE = 256;
  constexpr int NUM_TILES = 4;
  constexpr int TOTAL_SIZE = TILE_SIZE * NUM_TILES;
  constexpr size_t TOTAL_BYTES = TOTAL_SIZE * sizeof(float);

  float *d_in, *d_out;
  cudaMalloc(&d_in, TOTAL_BYTES);
  cudaMalloc(&d_out, TOTAL_BYTES);

  std::vector<float> h_in(TOTAL_SIZE);
  for (int i = 0; i < TOTAL_SIZE; i++)
    h_in[i] = static_cast<float>(i);
  cudaMemcpy(d_in, h_in.data(), TOTAL_BYTES, cudaMemcpyHostToDevice);

  for (int i = 0; i < 10; i++)
    async_pipeline_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in,
                                                                     d_out);
  cudaDeviceSynchronize();

  nvtxRangePush("async_pipeline_kernel");
  async_pipeline_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in, d_out);
  nvtxRangePop();
  cudaDeviceSynchronize();

  std::vector<float> h_out(TOTAL_SIZE);
  cudaMemcpy(h_out.data(), d_out, TOTAL_BYTES, cudaMemcpyDeviceToHost);

  std::vector<float> h_expected(TOTAL_SIZE);
  cpu_reference_pipeline(h_in.data(), h_expected.data(), TILE_SIZE, NUM_TILES);

  bool pass = true;
  for (int i = 0; i < TILE_SIZE; i++) { // only first tile written back
    if (h_out[i] != h_expected[i]) {
      pass = false;
      printf("Mismatch at %d: expected %.1f got %.1f\n", i, h_expected[i],
             h_out[i]);
      break;
    }
  }
  printf("\n[%s] Async pipeline verified\n", pass ? "PASS" : "FAIL");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; i++)
    async_pipeline_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in,
                                                                     d_out);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < 100; i++)
    async_pipeline_kernel<<<1, 128, 2 * TILE_SIZE * sizeof(float)>>>(d_in,
                                                                     d_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  // TODO 10: fix timing — one call, two events
  /* YOUR CODE HERE */;
  elapsed_ms /= 100.0f;

  printf("[Timing] Average kernel time: %.6f ms\n", elapsed_ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_out);
  return pass ? 0 : 1;
}
