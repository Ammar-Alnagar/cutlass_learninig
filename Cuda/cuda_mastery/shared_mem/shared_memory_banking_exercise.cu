/*
 * Shared Memory Banking Hands-On Exercise
 *
 * Fix bank conflicts in the shared memory transpose operation.
 * Fill in the missing code sections marked with TODO comments.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Shared Memory Transpose - STUDENT EXERCISE
__global__ void sharedMemoryTranspose(float *input, float *output, int width) {
  // TODO: Modify shared memory declaration to avoid bank conflicts

  __shared__ float tile[32][33]; // Add padding to avoid bank conflicts

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int col = threadIdx.x;
  int row = threadIdx.y;
  // Load data into shared memory (coalesced read)
  if (x < width && y < width) {
    tile[threadIdx.y][threadIdx.x] = input[y * width + x];
  }
  __syncthreads();

  if (x < width && y < width) {

    output[x * width + y] = tile[col][row];
    // output[y * width + x] = /* YOUR CORRECTED ACCESS HERE */;
  }
}

// Utility function to initialize matrix
void initMatrix(float *mat, int width, float start_val = 1.0f) {
  for (int i = 0; i < width * width; i++) {
    mat[i] = start_val + i * 0.01f;
  }
}

// Utility function to print first few elements of a matrix
void printMatrix(float *mat, int width, int rows = 3, int cols = 3) {
  printf("First %d×%d elements:\n", rows, cols);
  for (int i = 0; i < rows && i < width; i++) {
    for (int j = 0; j < cols && j < width; j++) {
      printf("%.2f ", mat[i * width + j]);
    }
    printf("\n");
  }
}

int main() {
  printf("=== Shared Memory Banking Hands-On Exercise ===\n");
  printf("Complete the missing code sections in the sharedMemoryTranspose "
         "kernel.\n\n");

  // Setup for shared memory banking exercise
  const int WIDTH = 32;
  const int N = WIDTH * WIDTH;
  size_t size = N * sizeof(float);

  // Allocate host memory
  float *h_input, *h_output;
  h_input = (float *)malloc(size);
  h_output = (float *)malloc(size);

  // Initialize input matrix
  initMatrix(h_input, WIDTH, 1.0f);

  // Allocate device memory
  float *d_input, *d_output;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size);

  // Copy input to device
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  // Launch shared memory transpose kernel
  dim3 blockSize(32, 32); // 32x32 threads per block
  dim3 gridSize((WIDTH + 31) / 32,
                (WIDTH + 31) / 32); // One block for 32x32 tile

  // This will fail to compile until you complete the kernel
  sharedMemoryTranspose<<<gridSize, blockSize>>>(d_input, d_output, WIDTH);
  cudaDeviceSynchronize();

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // Copy result back to host
  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

  printf("Result matrix after completing sharedMemoryTranspose kernel:\n");
  printMatrix(h_output, WIDTH);

  // Cleanup
  free(h_input);
  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);

  printf("\nExercise completed! Try the other hands-on exercises.\n");

  return 0;
}
