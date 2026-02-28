/**
 * Exercise 10: Real-World Layout Patterns
 *
 * Objective: Apply layout algebra to real-world GPU programming scenarios
 *            including matrix multiplication, convolution, and data formats
 *
 * Tasks:
 * 1. Design layouts for matrix multiplication tiles
 * 2. Create layouts for NHWC and NCHW tensor formats
 * 3. Model thread block hierarchies for 2D convolutions
 * 4. Design layouts for batched operations
 *
 * Key Concepts:
 * - Real-world layout patterns
 * - Thread-to-data mapping
 * - Multi-dimensional data organization
 * - Performance-oriented layout design
 */

#include "cute/layout.hpp"
#include "cute/util/print.hpp"
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

// =========================================================================
// Helper Functions
// =========================================================================

template <typename Layout>
void print_2d_layout(const char* name, Layout const& layout, int rows,
                     int cols) {
  std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%4d ", layout(i, j));
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  std::cout << "=== Exercise 10: Real-World Layout Patterns ===" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 1: Matrix Multiplication Thread Layout
  // =========================================================================
  std::cout << "--- Task 1: Matrix Multiplication Thread Layout ---" << std::endl;

  // For a 16x16 output tile with 32 threads (1 warp)
  // Each thread computes 8 elements (16*16/32 = 8)

  // Thread layout: 4 rows x 8 columns of threads
  auto thread_layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                   make_stride(Int<8>{}, Int<1>{}));

  std::cout << "Thread layout (4x8 threads):" << std::endl;
  print(thread_layout);
  std::cout << std::endl;

  // Output tile layout: 16x16 elements
  auto output_tile = make_layout(make_shape(Int<16>{}, Int<16>{}),
                                 make_stride(Int<16>{}, Int<1>{}));

  std::cout << "Output tile (16x16 elements):" << std::endl;
  print_2d_layout("Output", output_tile, 16, 16);

  // Each thread computes a 2x4 block of the output
  std::cout << "Thread-to-output mapping:" << std::endl;
  std::cout << "  Each thread computes 2x4 = 8 elements" << std::endl;
  std::cout << "  Thread (0,0) computes elements at:" << std::endl;
  for (int di = 0; di < 2; ++di) {
    for (int dj = 0; dj < 4; ++dj) {
      int row = di * 4;  // 4 row stride between thread rows
      int col = dj * 2;  // 2 col stride between thread cols
      std::cout << "    (" << row << "," << col << ") -> offset "
                << output_tile(row, col) << std::endl;
    }
  }
  std::cout << std::endl;

  // =========================================================================
  // TASK 2: NHWC vs NCHW Tensor Formats
  // =========================================================================
  std::cout << "--- Task 2: NHWC vs NCHW Tensor Formats ---" << std::endl;

  // Tensor dimensions: N=4 (batch), C=3 (channels), H=8 (height), W=8 (width)
  constexpr int N = 4, C = 3, H = 8, W = 8;

  // NHWC format (TensorFlow, cuDNN default)
  // Memory order: batch -> height -> width -> channel (channel is contiguous)
  auto nhwc_layout = make_layout(
      make_shape(Int<N>{}, Int<H>{}, Int<W>{}, Int<C>{}),
      make_stride(Int<H * W * C>{}, Int<W * C>{}, Int<C>{}, Int<1>{}));

  std::cout << "NHWC Layout (N=4, H=8, W=8, C=3):" << std::endl;
  std::cout << "  Shape:  " << nhwc_layout.shape() << std::endl;
  std::cout << "  Stride: " << nhwc_layout.stride() << std::endl;
  std::cout << "  Total elements: " << size(nhwc_layout) << std::endl;
  std::cout << "  Channel stride = 1 (contiguous channels)" << std::endl;
  std::cout << std::endl;

  // NCHW format (PyTorch default)
  // Memory order: batch -> channel -> height -> width (width is contiguous)
  auto nchw_layout = make_layout(
      make_shape(Int<N>{}, Int<C>{}, Int<H>{}, Int<W>{}),
      make_stride(Int<C * H * W>{}, Int<H * W>{}, Int<W>{}, Int<1>{}));

  std::cout << "NCHW Layout (N=4, C=3, H=8, W=8):" << std::endl;
  std::cout << "  Shape:  " << nchw_layout.shape() << std::endl;
  std::cout << "  Stride: " << nchw_layout.stride() << std::endl;
  std::cout << "  Total elements: " << size(nchw_layout) << std::endl;
  std::cout << "  Width stride = 1 (contiguous width)" << std::endl;
  std::cout << std::endl;

  // Compare access patterns
  std::cout << "Access pattern comparison:" << std::endl;
  std::cout << "  NHWC(0, 0, 0, 0) = " << nhwc_layout(0, 0, 0, 0)
            << " (first element)" << std::endl;
  std::cout << "  NHWC(0, 0, 0, 1) = " << nhwc_layout(0, 0, 0, 1)
            << " (next channel)" << std::endl;
  std::cout << "  NHWC(0, 0, 1, 0) = " << nhwc_layout(0, 0, 1, 0)
            << " (next width)" << std::endl;
  std::cout << std::endl;

  std::cout << "  NCHW(0, 0, 0, 0) = " << nchw_layout(0, 0, 0, 0)
            << " (first element)" << std::endl;
  std::cout << "  NCHW(0, 0, 0, 1) = " << nchw_layout(0, 0, 0, 1)
            << " (next width)" << std::endl;
  std::cout << "  NCHW(0, 0, 1, 0) = " << nchw_layout(0, 0, 1, 0)
            << " (next height)" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 3: 2D Convolution Thread Block Layout
  // =========================================================================
  std::cout << "--- Task 3: 2D Convolution Thread Block Layout ---" << std::endl;

  // For a 2D convolution with 16x16 output tile
  // Using 16x16 thread block (256 threads)

  // Thread block layout: 16x16 threads
  auto block_threads = make_layout(make_shape(Int<16>{}, Int<16>{}),
                                   make_stride(Int<16>{}, Int<1>{}));

  std::cout << "Thread block (16x16 threads):" << std::endl;
  print(block_threads);
  std::cout << std::endl;

  // Each thread computes one output element
  // Input access pattern depends on filter size (e.g., 3x3 filter)

  // For a 3x3 filter, each thread needs to load a 3x3 region of input
  auto filter_region = make_layout(make_shape(Int<3>{}, Int<3>{}),
                                   make_stride(Int<3>{}, Int<1>{}));

  std::cout << "Filter region per thread (3x3):" << std::endl;
  print(filter_region);
  std::cout << std::endl;

  // Shared memory layout for input tile (with padding)
  // 16 (output) + 3 - 1 = 18 input elements per dimension
  auto smem_input = make_layout(make_shape(Int<18>{}, Int<18>{}),
                                make_stride(Int<18>{}, Int<1>{}));

  std::cout << "Shared memory input tile (18x18 with padding):" << std::endl;
  print(smem_input);
  std::cout << std::endl;

  // =========================================================================
  // TASK 4: Batched Matrix Multiplication Layout
  // =========================================================================
  std::cout << "--- Task 4: Batched Matrix Multiplication Layout ---" << std::endl;

  // Batched GEMM: multiple independent matrix multiplications
  // Batch size: 8, Matrix size: 64x64

  constexpr int BATCH = 8;
  constexpr int M = 64, N = 64, K = 64;

  // Batch-major layout (each batch is contiguous)
  auto batch_layout = make_layout(
      make_shape(Int<BATCH>{}, Int<M>{}, Int<N>{}),
      make_stride(Int<M * N>{}, Int<N>{}, Int<1>{})  // Row-major within batch
  );

  std::cout << "Batched matrix layout (8 batches of 64x64):" << std::endl;
  std::cout << "  Shape:  " << batch_layout.shape() << std::endl;
  std::cout << "  Stride: " << batch_layout.stride() << std::endl;
  std::cout << "  Batch stride: " << get<0>(batch_layout.stride())
            << " (elements per batch)" << std::endl;
  std::cout << std::endl;

  // Access specific batches
  std::cout << "Batch access examples:" << std::endl;
  std::cout << "  First element of batch 0: " << batch_layout(0, 0, 0)
            << std::endl;
  std::cout << "  First element of batch 1: " << batch_layout(1, 0, 0)
            << std::endl;
  std::cout << "  First element of batch 7: " << batch_layout(7, 0, 0)
            << std::endl;
  std::cout << "  Batch spacing: " << batch_layout(1, 0, 0) - batch_layout(0, 0, 0)
            << " elements" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 5: Strided Batched Layout (Batch Stride)
  // =========================================================================
  std::cout << "--- Task 5: Strided Batched Layout ---" << std::endl;

  // Strided batch: batches are not contiguous in memory
  // Useful when processing slices of a larger tensor

  auto strided_batch = make_layout(
      make_shape(Int<BATCH>{}, Int<M>{}, Int<N>{}),
      make_stride(Int<M * N * 2>{}, Int<N>{}, Int<1>{})  // 2x spacing between batches
  );

  std::cout << "Strided batched layout (2x batch spacing):" << std::endl;
  std::cout << "  Shape:  " << strided_batch.shape() << std::endl;
  std::cout << "  Stride: " << strided_batch.stride() << std::endl;
  std::cout << "  Batch stride: " << get<0>(strided_batch.stride())
            << " (2x normal for interleaving)" << std::endl;
  std::cout << std::endl;

  std::cout << "Strided batch access:" << std::endl;
  std::cout << "  Batch 0 start: " << strided_batch(0, 0, 0) << std::endl;
  std::cout << "  Batch 1 start: " << strided_batch(1, 0, 0) << std::endl;
  std::cout << "  Batch 2 start: " << strided_batch(2, 0, 0) << std::endl;
  std::cout << "  Spacing: " << strided_batch(1, 0, 0) - strided_batch(0, 0, 0)
            << " elements" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 6: Multi-Head Attention Layout
  // =========================================================================
  std::cout << "--- Task 6: Multi-Head Attention Layout ---" << std::endl;

  // Transformer attention: (batch, heads, seq_len, hidden)
  constexpr int SEQ_LEN = 128;
  constexpr int HEADS = 8;
  constexpr int HIDDEN = 64;

  // Attention score layout: (batch, heads, seq, seq)
  auto attention_layout = make_layout(
      make_shape(Int<N>{}, Int<HEADS>{}, Int<SEQ_LEN>{}, Int<SEQ_LEN>{}),
      GenRowMajor{});

  std::cout << "Attention score layout (4x8x128x128):" << std::endl;
  std::cout << "  Shape:  " << attention_layout.shape() << std::endl;
  std::cout << "  Stride: " << attention_layout.stride() << std::endl;
  std::cout << "  Total scores: " << size(attention_layout)
            << " (" << (size(attention_layout) / 1e6) << "M)" << std::endl;
  std::cout << std::endl;

  // Access pattern for a specific token's attention
  std::cout << "Attention for token 0 (all heads):" << std::endl;
  std::cout << "  Head 0, token 0 -> all targets: offsets "
            << attention_layout(0, 0, 0, 0) << " to "
            << attention_layout(0, 0, 0, SEQ_LEN - 1) << std::endl;
  std::cout << "  Head 1, token 0 -> all targets: offsets "
            << attention_layout(0, 1, 0, 0) << " to "
            << attention_layout(0, 1, 0, SEQ_LEN - 1) << std::endl;
  std::cout << "  Head spacing: "
            << attention_layout(0, 1, 0, 0) - attention_layout(0, 0, 0, 0)
            << " elements" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 7: GEMM Operand Layouts
  // =========================================================================
  std::cout << "--- Task 7: GEMM Operand Layouts ---" << std::endl;

  // For C = A × B where:
  // A: M×K, B: K×N, C: M×N
  // Using 16×16 tiles

  constexpr int TILE_M = 16, TILE_N = 16, TILE_K = 16;

  // Matrix A layout (row-major for coalesced load)
  auto A_layout = make_layout(make_shape(Int<M>{}, Int<K>{}), GenRowMajor{});

  // Matrix B layout (column-major for coalesced load in some algorithms)
  auto B_layout = make_layout(make_shape(Int<K>{}, Int<N>{}), GenColMajor{});

  // Matrix C layout (row-major for coalesced store)
  auto C_layout = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});

  std::cout << "GEMM operand layouts:" << std::endl;
  std::cout << "  A (M×K, row-major):  " << A_layout << std::endl;
  std::cout << "  B (K×N, col-major):  " << B_layout << std::endl;
  std::cout << "  C (M×N, row-major):  " << C_layout << std::endl;
  std::cout << std::endl;

  // Shared memory layouts for tiles
  auto As_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                               make_stride(Int<TILE_K + 1>{}, Int<1>{}));
  auto Bs_layout = make_layout(make_shape(Int<TILE_K>{}, Int<TILE_N>{}),
                               make_stride(Int<TILE_N + 1>{}, Int<1>{}));

  std::cout << "Shared memory tile layouts (with padding):" << std::endl;
  std::cout << "  As (16×16, padded): " << As_layout << std::endl;
  std::cout << "  Bs (16×16, padded): " << Bs_layout << std::endl;
  std::cout << "  Padding avoids bank conflicts during MMA" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // TASK 8: Complete Thread Hierarchy for GEMM
  // =========================================================================
  std::cout << "--- Task 8: Complete Thread Hierarchy for GEMM ---" << std::endl;

  // Grid: multiple thread blocks for large matrices
  // Block: 128 threads (4 warps)
  // Each block computes a 64×64 tile of C

  constexpr int BLOCK_M = 64, BLOCK_N = 64;
  constexpr int THREADS_PER_BLOCK = 128;

  // Grid layout (number of blocks in each dimension)
  int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
  int grid_n = (N + BLOCK_N - 1) / BLOCK_N;

  auto grid_layout = make_layout(make_shape(grid_m, grid_n), GenRowMajor{});

  std::cout << "Grid layout (" << grid_m << "×" << grid_n << " blocks):"
            << std::endl;
  std::cout << "  Shape:  " << grid_layout.shape() << std::endl;
  std::cout << "  Stride: " << grid_layout.stride() << std::endl;
  std::cout << "  Total blocks: " << size(grid_layout) << std::endl;
  std::cout << std::endl;

  // Block layout (threads within a block)
  auto block_layout = make_layout(make_shape(Int<4>{}, Int<32>{}),
                                  make_stride(Int<32>{}, Int<1>{}));

  std::cout << "Block layout (4 warps × 32 lanes):" << std::endl;
  print(block_layout);
  std::cout << std::endl;

  // Each thread's output responsibility
  int elements_per_thread = (BLOCK_M * BLOCK_N) / THREADS_PER_BLOCK;
  std::cout << "Work distribution:" << std::endl;
  std::cout << "  Block tile: " << BLOCK_M << "×" << BLOCK_N
            << " = " << BLOCK_M * BLOCK_N << " elements" << std::endl;
  std::cout << "  Threads: " << THREADS_PER_BLOCK << std::endl;
  std::cout << "  Elements per thread: " << elements_per_thread << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // CHALLENGE: Design a Layout for Depthwise Convolution
  // =========================================================================
  std::cout << "=== Challenge: Depthwise Convolution Layout ===" << std::endl;

  // Depthwise convolution: each channel is filtered independently
  // Input: (N, C, H, W), Filter: (C, 3, 3), Output: (N, C, H', W')

  std::cout << "Design a layout for depthwise convolution:" << std::endl;
  std::cout << "  Input:  N=4, C=32, H=64, W=64" << std::endl;
  std::cout << "  Filter: C=32, 3×3 per channel" << std::endl;
  std::cout << "  Output: N=4, C=32, H=62, W=62" << std::endl;
  std::cout << std::endl;

  // TODO: Design the input layout for coalesced access
  auto depthwise_input = make_layout(
      make_shape(Int<4>{}, Int<32>{}, Int<64>{}, Int<64>{}),
      GenRowMajor{});

  std::cout << "Your input layout:" << std::endl;
  std::cout << "  Shape:  " << depthwise_input.shape() << std::endl;
  std::cout << "  Stride: " << depthwise_input.stride() << std::endl;
  std::cout << std::endl;

  // TODO: Design the filter layout for efficient channel access
  auto depthwise_filter = make_layout(
      make_shape(Int<32>{}, Int<3>{}, Int<3>{}),
      GenRowMajor{});

  std::cout << "Your filter layout:" << std::endl;
  std::cout << "  Shape:  " << depthwise_filter.shape() << std::endl;
  std::cout << "  Stride: " << depthwise_filter.stride() << std::endl;
  std::cout << std::endl;

  std::cout << "Considerations:" << std::endl;
  std::cout << "1. Coalesced access for input loading" << std::endl;
  std::cout << "2. Channel independence (can process channels in parallel)"
            << std::endl;
  std::cout << "3. Filter access pattern (same filter reused for all spatial)"
            << std::endl;
  std::cout << "4. Output store pattern" << std::endl;
  std::cout << std::endl;

  std::cout << "=== Exercise Complete ===" << std::endl;
  std::cout << "Key Learnings:" << std::endl;
  std::cout << "1. Matrix multiplication needs careful thread-to-data mapping"
            << std::endl;
  std::cout << "2. NHWC vs NCHW have different contiguous dimensions"
            << std::endl;
  std::cout << "3. Convolution requires shared memory tiling" << std::endl;
  std::cout << "4. Batched operations can be contiguous or strided" << std::endl;
  std::cout << "5. Attention has 4D layout with specific access patterns"
            << std::endl;
  std::cout << "6. GEMM operands benefit from different layouts (row/col)"
            << std::endl;
  std::cout << "7. Complete kernels need grid->block->thread hierarchy"
            << std::endl;
  std::cout << "8. Depthwise convolution has channel-independent parallelism"
            << std::endl;

  return 0;
}
