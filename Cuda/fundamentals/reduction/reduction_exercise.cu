/*
 * Reduction Operation Hands-On Exercise
 *
 * Complete the reduction kernel that computes the sum of array elements.
 * Fill in the missing code sections marked with TODO comments.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Reduction Sum - STUDENT EXERCISE
__global__ void reductionSum(float *input, float *output, int n) {
  // TODO: Declare shared memory for this block
  // Hint: Use __shared__ keyword and size it appropriately
  /* YOUR DECLARATION HERE */;
  __shared__ sdata[256];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load input into shared memory
  if (i < n) {
    sdata[tid] = input[i];
  } else {
    sdata[tid] = 0.0f; // Pad with zeros
  }
  __syncthreads();

  // Perform reduction in shared memory
  // TODO: Complete the reduction loop
  // Hint: Each iteration reduces the number of active elements by half
  for (int s = 1; s < blockDim.x; s *= 2) {
    // TODO: Check bounds and perform reduction
    if (/* YOUR CONDITION HERE */) {
      // TODO: Add element at tid+s to element at tid
      /* YOUR CODE HERE */;
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

// Utility function to initialize vector
void initVector(float *vec, int n, float start_val = 1.0f) {
  for (int i = 0; i < n; i++) {
    vec[i] = start_val + i * 0.1f;
  }
}

int main() {
  printf("=== Reduction Operation Hands-On Exercise ===\n");
  printf("Complete the missing code sections in the reductionSum kernel.\n\n");

  // Setup for reduction exercise
  const int N = 1024;
  size_t size = N * sizeof(float);
  size_t blockSize = 256;
  size_t gridSize = (N + blockSize - 1) / blockSize;

  // Allocate host memory
  float *h_input, *h_output;
  h_input = (float *)malloc(size);
  h_output = (float *)malloc(gridSize * sizeof(float)); // One output per block

  // Initialize input vector
  initVector(h_input, N, 1.0f);

  // Allocate device memory
  float *d_input, *d_output;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, gridSize * sizeof(float));

  // Copy input to device
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  // Launch reduction kernel
  reductionSum<<<gridSize, blockSize>>>(d_input, d_output, N);
  cudaDeviceSynchronize();

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // Copy partial results back to host
  cudaMemcpy(h_output, d_output, gridSize * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Compute final sum on CPU
  float final_sum = 0.0f;
  for (int i = 0; i < gridSize; i++) {
    final_sum += h_output[i];
    printf("Block %d partial sum: %.2f\n", i, h_output[i]);
  }

  printf("Final reduced sum: %.2f\n", final_sum);

  // Cleanup
  free(h_input);
  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);

  printf("\nExercise completed! Try the other hands-on exercises.\n");

  return 0;
}
