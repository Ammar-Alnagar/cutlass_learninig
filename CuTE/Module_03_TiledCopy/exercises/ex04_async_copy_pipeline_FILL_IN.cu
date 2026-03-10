#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <iostream>
using namespace cute;

// WHAT THIS IS:
// Two tiles in gmem. Load tile 0 async while "computing" on nothing,
// then load tile 1 async, process both, write back.
// Pure cp.async syntax practice — no ping-pong complexity.

__global__ void cp_async_practice(float *gmem_in, float *gmem_out) {
  constexpr int TILE = 512; // 512 floats = 2KB smem, fits easily

  // Two smem buffers, one per tile
  __shared__ float buf0[TILE];
  __shared__ float buf1[TILE];

  auto smem_0 = make_tensor(make_smem_ptr(buf0), make_layout(Int<TILE>{}));
  auto smem_1 = make_tensor(make_smem_ptr(buf1), make_layout(Int<TILE>{}));

  auto full = make_tensor(make_gmem_ptr(gmem_in), make_layout(Int<TILE * 2>{}));

  // Slice gmem into two tiles — no make_coord, just scalar index
  auto tile0 = local_tile(full, make_layout(Int<TILE>{}), 0);
  auto tile1 = local_tile(full, make_layout(Int<TILE>{}), 1);

  // TODO 1: define async copy atom, 128-bit, float
  using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>;

  // TODO 2: make tiled_copy — 128 threads, 4 values
  auto tiled_copy = make_tiled_copy(CopyAtom{}, make_layout(Int<128>{}),
                                    make_layout(Int<4>{}));

  // TODO 3: get per-thread slice
  auto thr_copy = tiled_copy.get_slice(threadIdx.x);

  // TODO 4: partition tile0 (src) and smem_0 (dst)
  auto src0 = thr_copy.partition_S(tile0);
  auto dst0 = thr_copy.partition_D(smem_0);

  // TODO 5: issue async copy tile0 -> smem_0
  /* YOUR CODE HERE */;
  copy(tiled_copy, src0, dst0);
  // TODO 6: partition tile1 (src) and smem_1 (dst)
  auto src1 = thr_copy.partition_S(tile1);
  auto dst1 = thr_copy.partition_D(smem_1);

  // TODO 7: issue async copy tile1 -> smem_1
  /* YOUR CODE HERE */;
  copy(tiled_copy, src1, dst1);

  // TODO 8: fence — commit both in-flight copies
  /* YOUR CODE HERE */;
  cp_async_fence();

  // TODO 9: wait — block until all copies done
  /* YOUR CODE HERE */;
  cp_async_wait<0>();

  __syncthreads();

  // Simple compute: multiply every element by 2
  for (int i = threadIdx.x; i < TILE; i += blockDim.x) {
    buf0[i] *= 2.0f;
    buf1[i] *= 2.0f;
  }
  __syncthreads();

  // Write back
  auto out_full =
      make_tensor(make_gmem_ptr(gmem_out), make_layout(Int<TILE * 2>{}));
  auto out0 = local_tile(out_full, make_layout(Int<TILE>{}), 0);
  auto out1 = local_tile(out_full, make_layout(Int<TILE>{}), 1);
  // Add this before TODO 10
  auto tiled_copy_out =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                      make_layout(Int<128>{}), make_layout(Int<4>{}));
  auto thr_copy_out = tiled_copy_out.get_slice(threadIdx.x);

  // TODO 10
  auto src_s0 = thr_copy_out.partition_S(smem_0);
  auto dst_o0 = thr_copy_out.partition_D(out0);
  copy(tiled_copy_out, src_s0, dst_o0);

  // TODO 11
  auto src_s1 = thr_copy_out.partition_S(smem_1);
  auto dst_o1 = thr_copy_out.partition_D(out1);
  copy(tiled_copy_out, src_s1, dst_o1);
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s\n", prop.name);
  if (prop.major < 8) {
    printf("Needs sm_80+\n");
    return 1;
  }

  constexpr int TILE = 512;
  constexpr int SIZE = TILE * 2;
  constexpr size_t BYTES = SIZE * sizeof(float);

  float *d_in, *d_out;
  cudaMalloc(&d_in, BYTES);
  cudaMalloc(&d_out, BYTES);

  std::vector<float> h_in(SIZE);
  for (int i = 0; i < SIZE; i++)
    h_in[i] = static_cast<float>(i);
  cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice);

  cp_async_practice<<<1, 128>>>(d_in, d_out);
  cudaDeviceSynchronize();

  std::vector<float> h_out(SIZE);
  cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost);

  bool pass = true;
  for (int i = 0; i < SIZE; i++) {
    if (h_out[i] != h_in[i] * 2.0f) {
      pass = false;
      printf("Mismatch at %d: expected %.1f got %.1f\n", i, h_in[i] * 2.0f,
             h_out[i]);
      break;
    }
  }
  printf("[%s] cp.async practice\n", pass ? "PASS" : "FAIL");

  cudaFree(d_in);
  cudaFree(d_out);
  return pass ? 0 : 1;
}
