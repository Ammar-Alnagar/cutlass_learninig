/*
 * Matrix Multiplication Hands-On Exercise
 *
 * Complete the matrix multiplication kernel that computes C = A × B.
 * Fill in the missing code sections marked with TODO comments.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Matrix Multiplication - STUDENT EXERCISE
__global__ void matrixMul(float *A, float *B, float *C, int width) {
  // TODO: Calculate row and column indices for this thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Only compute if within matrix bounds
  if (row < width && col < width) {
    float sum = 0.0f;

    // TODO: Compute dot product of row from A and column from B
    // Hint: Loop from 0 to width and accumulate the products
    for (int k = 0; k < width; k++) {
      // YOUR CODE HERE: multiply A[row][k] by B[k][col] and add to sum
      /* YOUR CODE HERE */
      sum += A[row * width + k] * B[k * width + col];
    }

    // Store result in C[row][col]
    C[row * width + col] = sum;
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
  printf("=== Matrix Multiplication Hands-On Exercise ===\n");
  printf("Complete the missing code sections in the matrixMul kernel.\n\n");

  // Setup for matrix multiplication exercise
  const int WIDTH = 8192;
  const int N = WIDTH * WIDTH;
  size_t size = N * sizeof(float);

  // Allocate host memory
  float *h_A, *h_B, *h_C;
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  // Initialize input matrices
  initMatrix(h_A, WIDTH, 1.0f);
  initMatrix(h_B, WIDTH, 2.0f);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // Copy input matrices to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Launch matrix multiplication kernel
  dim3 blockSize(32, 8); // 16x16 threads per block
  dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                (WIDTH + blockSize.y - 1) / blockSize.y);

  // This will fail to compile until you complete the kernel
  matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, WIDTH);
  cudaDeviceSynchronize();

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  printf("Result matrix after completing matrixMul kernel:\n");
  printMatrix(h_C, WIDTH);

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
