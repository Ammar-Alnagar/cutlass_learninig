/*
 * Vector Addition Hands-On Exercise
 *
 * Complete the vector addition kernel that computes C = A + B.
 * Fill in the missing code sections marked with TODO comments.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Vector Addition - STUDENT EXERCISE
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
  // TODO: Calculate the global thread index
  // Hint: Use blockIdx.x, blockDim.x, and threadIdx.x
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: Add bounds checking to prevent out-of-bounds access
  if (i < N) {
    // TODO: Perform the vector addition: C[i] = A[i] + B[i]
    /* YOUR CODE HERE */
    C[i] = A[i] + B[i];
  }
}

// Utility function to initialize vectors
void initVector(float *vec, int n, float start_val = 1.0f) {
  for (int i = 0; i < n; i++) {
    vec[i] = start_val + i * 0.1f;
  }
}

// Utility function to print first few elements of a vector
void printVector(float *vec, int n, int count = 10000) {
  printf("First %d elements: ", count > n ? n : count);
  for (int i = 0; i < (count > n ? n : count); i++) {
    printf("%.2f ", vec[i]);
  }
  printf("\n");
}

int main() {
  printf("=== Vector Addition Hands-On Exercise ===\n");
  printf("Complete the missing code sections in the vectorAdd kernel.\n\n");

  // Setup for vector addition exercise
  const int N = 1 << 26;
  size_t size = N * sizeof(float);

  // Allocate host memory
  float *h_A, *h_B, *h_C;
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  // Initialize input vectors
  initVector(h_A, N, 1.0f);
  initVector(h_B, N, 2.0f);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // Copy input vectors to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Launch vector addition kernel
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // This will fail to compile until you complete the kernel
  vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  printf("Result vector after completing vectorAdd kernel:\n");
  printVector(h_C, N);

  // Cleanup
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  printf("\nExercise completed! Try the other hands-on exercises.\n");

  return 0;
}
