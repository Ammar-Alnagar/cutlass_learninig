/*
 * Memory Coalescing Hands-On Exercise
 *
 * Fix the memory access pattern to ensure coalesced access for optimal
 * performance. Fill in the missing code sections marked with TODO comments.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Coalesced Memory Access - STUDENT EXERCISE
__global__ void coalescedCopy(float *input, float *output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: Implement coalesced memory access
  // Current implementation has poor coalescing - fix it
  if (tid < n) {
    // Instead of this potentially inefficient access:
    // output[tid] = input[tid * stride];  // This is just an example of bad
    // access

    // Implement proper coalesced access where consecutive threads
    // access consecutive memory locations
    /* YOUR CODE HERE */
    output[tid] = input[tid]; // This is the correct coalesced access
  }
}

// Kernel: Uncoalesced Memory Access - For comparison
__global__ void uncoalescedCopy(float *input, float *output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Uncoalesced access pattern - threads access memory with stride
  if (tid < n / 2) {
    output[tid] = input[tid * 2]; // Every other element - uncoalesced
  }
}

// Utility function to initialize vector
void initVector(float *vec, int n, float start_val = 1.0f) {
  for (int i = 0; i < n; i++) {
    vec[i] = start_val + i * 0.1f;
  }
}

// Utility function to print first few elements of a vector
void printVector(float *vec, int n, int count = 10) {
  printf("First %d elements: ", count > n ? n : count);
  for (int i = 0; i < (count > n ? n : count); i++) {
    printf("%.2f ", vec[i]);
  }
  printf("\n");
}

int main() {
  printf("=== Memory Coalescing Hands-On Exercise ===\n");
  printf("Complete the missing code sections in the coalescedCopy kernel.\n\n");

  // Setup for memory coalescing exercise
  const int N = 1024;
  size_t size = N * sizeof(float);

  // Allocate host memory
  float *h_input, *h_output_coalesced, *h_output_uncoalesced;
  h_input = (float *)malloc(size);
  h_output_coalesced = (float *)malloc(size);
  h_output_uncoalesced = (float *)malloc(size);

  // Initialize input vector
  initVector(h_input, N, 1.0f);

  // Allocate device memory
  float *d_input, *d_output_coalesced, *d_output_uncoalesced;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output_coalesced, size);
  cudaMalloc(&d_output_uncoalesced, size);

  // Copy input to device
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  // Launch coalesced copy kernel
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // This will fail to compile until you complete the kernel
  coalescedCopy<<<gridSize, blockSize>>>(d_input, d_output_coalesced, N);
  cudaDeviceSynchronize();

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Coalesced kernel execution failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // Also run the uncoalesced version for comparison
  uncoalescedCopy<<<gridSize, blockSize>>>(d_input, d_output_uncoalesced, N);
  cudaDeviceSynchronize();

  // Copy results back to host
  cudaMemcpy(h_output_coalesced, d_output_coalesced, size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output_uncoalesced, d_output_uncoalesced, size,
             cudaMemcpyDeviceToHost);

  printf("Result after completing coalescedCopy kernel:\n");
  printVector(h_output_coalesced, N);

  printf("Result from uncoalescedCopy kernel (for comparison):\n");
  printVector(h_output_uncoalesced, N);

  // Cleanup
  free(h_input);
  free(h_output_coalesced);
  free(h_output_uncoalesced);
  cudaFree(d_input);
  cudaFree(d_output_coalesced);
  cudaFree(d_output_uncoalesced);

  printf("\nExercise completed! Try the other hands-on exercises.\n");

  return 0;
}
