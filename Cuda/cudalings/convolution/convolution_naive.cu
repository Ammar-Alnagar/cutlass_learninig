#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Naive 2D convolution kernel
__global__ void convolution_naive(float *input, float *output, float *kernel, 
                                  int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int ix = x + kx - half_kernel;
                int iy = y + ky - half_kernel;
                
                // Handle boundary conditions with zero-padding
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    sum += input[iy * width + ix] * kernel[ky * kernel_size + kx];
                }
            }
        }
        
        output[y * width + x] = sum;
    }
}

// Utility function to initialize a 2D Gaussian kernel
void init_gaussian_kernel(float *kernel, int size, float sigma) {
    int center = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center;
            float dy = y - center;
            float val = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            kernel[y * size + x] = val;
            sum += val;
        }
    }
    
    // Normalize the kernel
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

// Utility function to check CUDA errors
void checkCudaError(cudaError_t error, const char* function) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", function, cudaGetErrorString(error));
        exit(1);
    }
}

// Timing utility
double get_time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

int main(int argc, char **argv) {
    int width = 1024, height = 1024;  // Default image dimensions
    int kernel_size = 5;              // Default kernel size
    
    if (argc >= 3) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
    if (argc >= 4) {
        kernel_size = atoi(argv[3]);
    }
    
    printf("Convolution: %dx%d image with %dx%d kernel\n", width, height, kernel_size, kernel_size);
    
    size_t input_size = width * height * sizeof(float);
    size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);
    size_t output_size = width * height * sizeof(float);
    
    // Host arrays
    float *h_input = (float*)malloc(input_size);
    float *h_kernel = (float*)malloc(kernel_size_bytes);
    float *h_output = (float*)malloc(output_size);
    
    // Initialize input with random values
    for (int i = 0; i < width * height; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    
    // Initialize Gaussian kernel
    init_gaussian_kernel(h_kernel, kernel_size, 1.0f);
    
    // Device arrays
    float *d_input, *d_kernel, *d_output;
    checkCudaError(cudaMalloc(&d_input, input_size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_kernel, kernel_size_bytes), "cudaMalloc d_kernel");
    checkCudaError(cudaMalloc(&d_output, output_size), "cudaMalloc d_output");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "cudaMemcpy h_input to d_input");
    checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_kernel to d_kernel");
    
    // Kernel configuration
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    printf("Launching kernel with grid (%d,%d) and block (%d,%d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    convolution_naive<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, width, height, kernel_size);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    convolution_naive<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, width, height, kernel_size);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "cudaMemcpy d_output to h_output");
    
    // Verification: Check a few random points
    bool success = true;
    int num_checks = min(100, width * height / 10);
    int half_kernel = kernel_size / 2;
    
    for (int check = 0; check < num_checks; check++) {
        int x = half_kernel + rand() % (width - 2 * half_kernel);
        int y = half_kernel + rand() % (height - 2 * half_kernel);
        
        float expected = 0.0f;
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int ix = x + kx - half_kernel;
                int iy = y + ky - half_kernel;
                expected += h_input[iy * width + ix] * h_kernel[ky * kernel_size + kx];
            }
        }
        
        if (abs(h_output[y * width + x] - expected) > 1e-3) {
            printf("Verification failed at [%d][%d]: expected %f, got %f\n", x, y, expected, h_output[y * width + x]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Verification PASSED (checked %d random elements)\n", num_checks);
        
        // Calculate performance metrics
        long long ops = (long long)width * height * kernel_size * kernel_size; // Multiply-add operations
        double gflops = (ops / 1e9) / (kernel_time_ms / 1000.0);
        double bytes_moved = (width * height * sizeof(float)) + (kernel_size * kernel_size * sizeof(float)) + (width * height * sizeof(float)); // Input + kernel + output
        double bandwidth_gbs = (bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbs);
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    free(h_input); free(h_kernel); free(h_output);
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}