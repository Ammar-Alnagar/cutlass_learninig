/*
 * CUDA Memory Coalescing Tutorial
 * 
 * This tutorial demonstrates memory coalescing concepts and techniques.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: Coalesced access pattern (GOOD)
__global__ void coalescedAccess(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Coalesced: consecutive threads access consecutive memory addresses
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel 2: Strided access pattern (BAD - uncoalesced)
__global__ void stridedAccess(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Uncoalesced: threads access memory with stride, not consecutive
        int accessIdx = idx * stride;
        if (accessIdx < n * stride) {
            output[idx] = input[accessIdx] * 2.0f;
        }
    }
}

// Kernel 3: Reverse access pattern (BAD - uncoalesced)
__global__ void reverseAccess(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Uncoalesced: threads access memory in reverse order
        output[idx] = input[n - 1 - idx] * 2.0f;
    }
}

// Kernel 4: Matrix transpose with coalesced access using shared memory
#define TILE_SIZE 32
__global__ void transposeCoalesced(float* input, float* output, int n) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read input in coalesced pattern
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Calculate output coordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Write output in coalesced pattern
    if (x < n && y < n) {
        output[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Kernel 5: Array-of-Structures vs Structure-of-Arrays comparison
struct ParticleAoS {
    float x, y, z;
    float vx, vy, vz;
};

struct ParticlesSoA {
    float* x; float* y; float* z;
    float* vx; float* vy; float* vz;
};

__global__ void processParticlesAoS(ParticleAoS* particles, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Uncoalesced: each thread accesses different struct members
        particles[idx].x += particles[idx].vx;
        particles[idx].y += particles[idx].vy;
        particles[idx].z += particles[idx].vz;
    }
}

__global__ void processParticlesSoA(ParticlesSoA p, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Coalesced: consecutive threads access consecutive elements in each array
        p.x[idx] += p.vx[idx];
        p.y[idx] += p.vy[idx];
        p.z[idx] += p.vz[idx];
    }
}

// Helper function to measure execution time
float measureKernelTime(void (*kernel)(float*, float*, int), float* input, float* output, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kernel<<<(n + 255) / 256, 256>>>(input, output, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

int main() {
    printf("=== CUDA Memory Coalescing Tutorial ===\n\n");
    
    const int N = 1024 * 1024;  // 1M elements
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_output_coalesced, *h_output_strided, *h_output_reverse;
    h_input = (float*)malloc(size);
    h_output_coalesced = (float*)malloc(size);
    h_output_strided = (float*)malloc(size);
    h_output_reverse = (float*)malloc(size);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output_coalesced, *d_output_strided, *d_output_reverse;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output_coalesced, size);
    cudaMalloc(&d_output_strided, size);
    cudaMalloc(&d_output_reverse, size);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Example 1: Coalesced access
    printf("1. Coalesced Access Pattern:\n");
    float time_coalesced = measureKernelTime(coalescedAccess, d_input, d_output_coalesced, N);
    cudaMemcpy(h_output_coalesced, d_output_coalesced, size, cudaMemcpyDeviceToHost);
    printf("   Time: %.3f ms\n", time_coalesced);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output_coalesced[i]);
    }
    printf("\n\n");
    
    // Example 2: Strided access
    printf("2. Strided Access Pattern (Uncoalesced):\n");
    float time_strided = measureKernelTime(stridedAccess, d_input, d_output_strided, N/2);
    cudaMemcpy(h_output_strided, d_output_strided, size, cudaMemcpyDeviceToHost);
    printf("   Time: %.3f ms\n", time_strided);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output_strided[i]);
    }
    printf("\n\n");
    
    // Example 3: Reverse access
    printf("3. Reverse Access Pattern (Uncoalesced):\n");
    float time_reverse = measureKernelTime(reverseAccess, d_input, d_output_reverse, N);
    cudaMemcpy(h_output_reverse, d_output_reverse, size, cudaMemcpyDeviceToHost);
    printf("   Time: %.3f ms\n", time_reverse);
    printf("   First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_output_reverse[i]);
    }
    printf("\n\n");
    
    // Performance comparison
    printf("4. Performance Comparison:\n");
    printf("   Coalesced access time: %.3f ms\n", time_coalesced);
    printf("   Strided access time:   %.3f ms (%.2fx slower)\n", time_strided, time_strided/time_coalesced);
    printf("   Reverse access time:   %.3f ms (%.2fx slower)\n", time_reverse, time_reverse/time_coalesced);
    printf("\n");
    
    // Example 5: Matrix transpose with coalescing
    printf("5. Matrix Transpose with Coalesced Access:\n");
    const int MATRIX_SIZE = 1024;
    const int MATRIX_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;
    size_t matrix_size = MATRIX_ELEMENTS * sizeof(float);
    
    float *h_matrix_in, *h_matrix_out;
    float *d_matrix_in, *d_matrix_out;
    
    h_matrix_in = (float*)malloc(matrix_size);
    h_matrix_out = (float*)malloc(matrix_size);
    cudaMalloc(&d_matrix_in, matrix_size);
    cudaMalloc(&d_matrix_out, matrix_size);
    
    // Initialize matrix
    for (int i = 0; i < MATRIX_ELEMENTS; i++) {
        h_matrix_in[i] = i * 1.0f;
    }
    cudaMemcpy(d_matrix_in, h_matrix_in, matrix_size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE, 
                  (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    
    transposeCoalesced<<<gridSize, blockSize>>>(d_matrix_in, d_matrix_out, MATRIX_SIZE);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_matrix_out, d_matrix_out, matrix_size, cudaMemcpyDeviceToHost);
    printf("   Transpose completed successfully.\n");
    printf("   First few transposed elements: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_matrix_out[i]);
    }
    printf("\n\n");
    
    // Example 6: AoS vs SoA comparison
    printf("6. Array-of-Structures vs Structure-of-Arrays:\n");
    
    // AoS example
    ParticleAoS* h_particles_aos = (ParticleAoS*)malloc(N * sizeof(ParticleAoS));
    ParticleAoS* d_particles_aos;
    cudaMalloc(&d_particles_aos, N * sizeof(ParticleAoS));
    
    // Initialize AoS particles
    for (int i = 0; i < N; i++) {
        h_particles_aos[i].x = i * 1.0f;
        h_particles_aos[i].y = i * 2.0f;
        h_particles_aos[i].z = i * 3.0f;
        h_particles_aos[i].vx = 1.0f;
        h_particles_aos[i].vy = 2.0f;
        h_particles_aos[i].vz = 3.0f;
    }
    cudaMemcpy(d_particles_aos, h_particles_aos, N * sizeof(ParticleAoS), cudaMemcpyHostToDevice);
    
    processParticlesAoS<<<(N + 255) / 256, 256>>>(d_particles_aos, N);
    cudaDeviceSynchronize();
    
    // SoA example
    ParticlesSoA h_particles_soa, d_particles_soa;
    cudaMalloc(&d_particles_soa.x, N * sizeof(float));
    cudaMalloc(&d_particles_soa.y, N * sizeof(float));
    cudaMalloc(&d_particles_soa.z, N * sizeof(float));
    cudaMalloc(&d_particles_soa.vx, N * sizeof(float));
    cudaMalloc(&d_particles_soa.vy, N * sizeof(float));
    cudaMalloc(&d_particles_soa.vz, N * sizeof(float));
    
    // Initialize SoA particles
    float *temp_x, *temp_y, *temp_z, *temp_vx, *temp_vy, *temp_vz;
    temp_x = (float*)malloc(N * sizeof(float));
    temp_y = (float*)malloc(N * sizeof(float));
    temp_z = (float*)malloc(N * sizeof(float));
    temp_vx = (float*)malloc(N * sizeof(float));
    temp_vy = (float*)malloc(N * sizeof(float));
    temp_vz = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        temp_x[i] = i * 1.0f;
        temp_y[i] = i * 2.0f;
        temp_z[i] = i * 3.0f;
        temp_vx[i] = 1.0f;
        temp_vy[i] = 2.0f;
        temp_vz[i] = 3.0f;
    }
    
    cudaMemcpy(d_particles_soa.x, temp_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_soa.y, temp_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_soa.z, temp_z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_soa.vx, temp_vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_soa.vy, temp_vy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_soa.vz, temp_vz, N * sizeof(float), cudaMemcpyHostToDevice);
    
    processParticlesSoA<<<(N + 255) / 256, 256>>>(d_particles_soa, N);
    cudaDeviceSynchronize();
    
    printf("   AoS (uncoalesced) vs SoA (coalesced) demonstrated.\n\n");
    
    // Cleanup
    free(h_input);
    free(h_output_coalesced);
    free(h_output_strided);
    free(h_output_reverse);
    free(h_matrix_in);
    free(h_matrix_out);
    free(h_particles_aos);
    free(temp_x);
    free(temp_y);
    free(temp_z);
    free(temp_vx);
    free(temp_vy);
    free(temp_vz);
    
    cudaFree(d_input);
    cudaFree(d_output_coalesced);
    cudaFree(d_output_strided);
    cudaFree(d_output_reverse);
    cudaFree(d_matrix_in);
    cudaFree(d_matrix_out);
    cudaFree(d_particles_aos);
    cudaFree(d_particles_soa.x);
    cudaFree(d_particles_soa.y);
    cudaFree(d_particles_soa.z);
    cudaFree(d_particles_soa.vx);
    cudaFree(d_particles_soa.vy);
    cudaFree(d_particles_soa.vz);
    
    printf("Tutorial completed!\n");
    return 0;
}