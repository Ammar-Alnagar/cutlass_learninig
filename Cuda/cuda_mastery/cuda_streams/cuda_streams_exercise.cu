/*
 * CUDA Streams Hands-On Exercise
 *
 * Complete the asynchronous execution kernel using CUDA streams.
 * Fill in the missing code sections marked with TODO comments.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Simple computation to run asynchronously
__global__ void asyncComputation(float *data, int n, float factor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    data[tid] = data[tid] * factor + 1.0f;
  }
}

int main() {
  printf("=== CUDA Streams Hands-On Exercise ===\n");
  printf("Complete the missing code sections to implement asynchronous "
         "execution.\n\n");

  // Setup for streams exercise
  const int N = 1000000;
  const int CHUNK_SIZE = N / 4; // Divide work into 4 chunks
  size_t chunk_size_bytes = CHUNK_SIZE * sizeof(float);

  // Allocate host memory for chunks
  float *h_chunk1, *h_chunk2, *h_chunk3, *h_chunk4;
  h_chunk1 = (float *)malloc(chunk_size_bytes);
  h_chunk2 = (float *)malloc(chunk_size_bytes);
  h_chunk3 = (float *)malloc(chunk_size_bytes);
  h_chunk4 = (float *)malloc(chunk_size_bytes);

  // Initialize host data
  for (int i = 0; i < CHUNK_SIZE; i++) {
    h_chunk1[i] = i * 1.0f;
    h_chunk2[i] = i * 2.0f;
    h_chunk3[i] = i * 3.0f;
    h_chunk4[i] = i * 4.0f;
  }

  // Allocate device memory for chunks
  float *d_chunk1, *d_chunk2, *d_chunk3, *d_chunk4;
  cudaMalloc(&d_chunk1, chunk_size_bytes);
  cudaMalloc(&d_chunk2, chunk_size_bytes);
  cudaMalloc(&d_chunk3, chunk_size_bytes);
  cudaMalloc(&d_chunk4, chunk_size_bytes);

  // TODO: Create CUDA streams for asynchronous execution
  // Hint: Use cudaStreamCreate()
  /* YOUR STREAM DECLARATIONS AND CREATION HERE */;
  cudaStream_t stream1, stream2, stream3, stream4;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  cudaStreamCreate(&stream4);
  int blockSize = 256;
  int gridSize = (CHUNK_SIZE + blockSize - 1) / blockSize;

  // TODO: Launch asynchronous memory copies and kernel executions
  // Hint: Use cudaMemcpyAsync() and kernel<<<...>>>() with stream parameter
  // Example: cudaMemcpyAsync(d_chunk1, h_chunk1, chunk_size_bytes,
  // cudaMemcpyHostToDevice, stream1);
  //          asyncComputation<<<blocks, threads, 0, stream1>>>(d_chunk1,
  //          CHUNK_SIZE, 2.0f);
  // Chunk 1
  cudaMemcpyAsync(d_chunk1, h_chunk1, chunk_size_bytes, cudaMemcpyHostToDevice,
                  stream1);
  asyncComputation<<<gridSize, blockSize, 0, stream1>>>(d_chunk1, CHUNK_SIZE,
                                                        2.0f);

  // Chunk 2
  cudaMemcpyAsync(d_chunk2, h_chunk2, chunk_size_bytes, cudaMemcpyHostToDevice,
                  stream2);
  asyncComputation<<<gridSize, blockSize, 0, stream2>>>(d_chunk2, CHUNK_SIZE,
                                                        3.0f);

  // Chunk 3
  cudaMemcpyAsync(d_chunk3, h_chunk3, chunk_size_bytes, cudaMemcpyHostToDevice,
                  stream3);
  asyncComputation<<<gridSize, blockSize, 0, stream3>>>(d_chunk3, CHUNK_SIZE,
                                                        4.0f);

  // Chunk 4
  cudaMemcpyAsync(d_chunk4, h_chunk4, chunk_size_bytes, cudaMemcpyHostToDevice,
                  stream4);
  asyncComputation<<<gridSize, blockSize, 0, stream4>>>(d_chunk4, CHUNK_SIZE,
                                                        5.0f);

  // TODO: Synchronize all streams
  // Hint: Use cudaStreamSynchronize() for each stream
  /* YOUR SYNCHRONIZATION CODE */;

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  cudaStreamSynchronize(stream3);
  cudaStreamSynchronize(stream4);

  // Copy results back to host
  cudaMemcpy(h_chunk1, d_chunk1, chunk_size_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_chunk2, d_chunk2, chunk_size_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_chunk3, d_chunk3, chunk_size_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_chunk4, d_chunk4, chunk_size_bytes, cudaMemcpyDeviceToHost);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);
  cudaStreamDestroy(stream4);
  // Print some results
  printf("Results after async computation - Chunk 1 first 5 elements: ");
  for (int i = 0; i < 5; i++) {
    printf("%.2f ", h_chunk1[i]);
  }
  printf("\n");

  printf("Results after async computation - Chunk 2 first 5 elements: ");
  for (int i = 0; i < 5; i++) {
    printf("%.2f ", h_chunk2[i]);
  }
  printf("\n");

  // Cleanup
  free(h_chunk1);
  free(h_chunk2);
  free(h_chunk3);
  free(h_chunk4);
  cudaFree(d_chunk1);
  cudaFree(d_chunk2);
  cudaFree(d_chunk3);
  cudaFree(d_chunk4);

  // TODO: Destroy streams
  // Hint: Use cudaStreamDestroy()
  /* YOUR STREAM DESTRUCTION CODE */;

  printf("\nExercise completed! Try the other hands-on exercises.\n");

  return 0;
}
