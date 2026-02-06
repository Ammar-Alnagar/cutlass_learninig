// gaussian_blur_initial.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Naive implementation of Gaussian blur
__global__ void gaussian_blur_naive(float *input, float *output, int width, int height, float *kernel, int kernel_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply convolution with Gaussian kernel
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            int px = x + kx;
            int py = y + ky;
            
            // Handle boundary conditions
            if (px >= 0 && px < width && py >= 0 && py < height) {
                int pixel_idx = py * width + px;
                int kernel_idx = (ky + kernel_radius) * (2 * kernel_radius + 1) + (kx + kernel_radius);
                
                sum += input[pixel_idx] * kernel[kernel_idx];
                weight_sum += kernel[kernel_idx];
            }
        }
    }
    
    int idx = y * width + x;
    output[idx] = sum / weight_sum;
}