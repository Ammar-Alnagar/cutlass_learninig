/*
 * Atomic Operations Hands-On Exercise
 *
 * Complete the atomic operations kernel to handle race conditions properly.
 * Fill in the missing code sections marked with TODO comments.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Atomic Operations - STUDENT EXERCISE
__global__ void atomicHistogram(unsigned char *input, unsigned int *histogram,
                                int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    unsigned char value = input[tid];

    // TODO: Use atomic operation to increment histogram bin safely
    // Without atomics, multiple threads might update the same bin
    // simultaneously Hint: Use atomicAdd function
    /* YOUR CODE HERE */;
    atomicAdd(&histogram[value], 1);
  }
}

// Kernel: Non-atomic version for comparison (will have race conditions)
__global__ void nonAtomicHistogram(unsigned char *input,
                                   unsigned int *histogram, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    unsigned char value = input[tid];

    // Non-atomic version - will have race conditions
    histogram[value]++;
  }
}

// Utility function to initialize data
void initByteArray(unsigned char *arr, int n) {
  for (int i = 0; i < n; i++) {
    arr[i] = i % 256; // Values from 0 to 255
  }
}

int main() {
  printf("=== Atomic Operations Hands-On Exercise ===\n");
  printf(
      "Complete the missing code sections in the atomicHistogram kernel.\n\n");

  // Setup for atomic operations exercise
  const int N = 10000;
  const int HIST_SIZE = 256; // For values 0-255

  size_t input_size = N * sizeof(unsigned char);
  size_t hist_size = HIST_SIZE * sizeof(unsigned int);

  // Allocate host memory
  unsigned char *h_input;
  unsigned int *h_histogram_atomic, *h_histogram_non_atomic;

  h_input = (unsigned char *)malloc(input_size);
  h_histogram_atomic = (unsigned int *)calloc(HIST_SIZE, sizeof(unsigned int));
  h_histogram_non_atomic =
      (unsigned int *)calloc(HIST_SIZE, sizeof(unsigned int));

  // Initialize input array
  initByteArray(h_input, N);

  // Allocate device memory
  unsigned char *d_input;
  unsigned int *d_histogram_atomic, *d_histogram_non_atomic;

  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_histogram_atomic, hist_size);
  cudaMalloc(&d_histogram_non_atomic, hist_size);

  // Initialize histograms to zero on device
  cudaMemset(d_histogram_atomic, 0, hist_size);
  cudaMemset(d_histogram_non_atomic, 0, hist_size);

  // Copy input to device
  cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

  // Launch atomic histogram kernel
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // This will fail to compile until you complete the kernel
  atomicHistogram<<<gridSize, blockSize>>>(d_input, d_histogram_atomic, N);
  cudaDeviceSynchronize();

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Atomic kernel execution failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // Also run non-atomic version for comparison
  nonAtomicHistogram<<<gridSize, blockSize>>>(d_input, d_histogram_non_atomic,
                                              N);
  cudaDeviceSynchronize();

  // Copy results back to host
  cudaMemcpy(h_histogram_atomic, d_histogram_atomic, hist_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_histogram_non_atomic, d_histogram_non_atomic, hist_size,
             cudaMemcpyDeviceToHost);

  // Print some histogram values
  printf("Atomic histogram - First 10 bins: ");
  for (int i = 0; i < 10; i++) {
    printf("%u ", h_histogram_atomic[i]);
  }
  printf("\n");

  printf("Non-atomic histogram - First 10 bins: ");
  for (int i = 0; i < 10; i++) {
    printf("%u ", h_histogram_non_atomic[i]);
  }
  printf("\n");

  // Cleanup
  free(h_input);
  free(h_histogram_atomic);
  free(h_histogram_non_atomic);
  cudaFree(d_input);
  cudaFree(d_histogram_atomic);
  cudaFree(d_histogram_non_atomic);

  printf("\nExercise completed! Try the other hands-on exercises.\n");

  return 0;
}
