// gaussian_blur_optimized.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16
#define KERNEL_RADIUS 2
#define FILTER_SIZE (2 * KERNEL_RADIUS + 1)

__global__ void gaussian_blur_optimized(float *input, float *output, int width, int height, float *kernel) {
    // Shared memory with halo for convolution
    __shared__ float tile[TILE_SIZE + 2*KERNEL_RADIUS][TILE_SIZE + 2*KERNEL_RADIUS];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    
    // Load center tile
    if (x < width && y < height) {
        tile[ty + KERNEL_RADIUS][tx + KERNEL_RADIUS] = input[y * width + x];
    } else {
        tile[ty + KERNEL_RADIUS][tx + KERNEL_RADIUS] = 0.0f;
    }
    
    // Load halo regions
    // Top and bottom halos
    if (ty < KERNEL_RADIUS) {
        int load_y_top = y - KERNEL_RADIUS;
        int load_y_bottom = y + TILE_SIZE;
        
        if (load_y_top >= 0 && x < width) {
            tile[ty][tx + KERNEL_RADIUS] = input[load_y_top * width + x];
        } else {
            tile[ty][tx + KERNEL_RADIUS] = 0.0f;
        }
        
        if (load_y_bottom < height && x < width) {
            tile[ty + TILE_SIZE + KERNEL_RADIUS][tx + KERNEL_RADIUS] = input[load_y_bottom * width + x];
        } else {
            tile[ty + TILE_SIZE + KERNEL_RADIUS][tx + KERNEL_RADIUS] = 0.0f;
        }
    }
    
    // Left and right halos
    if (tx < KERNEL_RADIUS) {
        int load_x_left = x - KERNEL_RADIUS;
        int load_x_right = x + TILE_SIZE;
        
        if (load_x_left >= 0 && y < height) {
            tile[ty + KERNEL_RADIUS][tx] = input[y * width + load_x_left];
        } else {
            tile[ty + KERNEL_RADIUS][tx] = 0.0f;
        }
        
        if (load_x_right < width && y < height) {
            tile[ty + KERNEL_RADIUS][tx + TILE_SIZE + KERNEL_RADIUS] = input[y * width + load_x_right];
        } else {
            tile[ty + KERNEL_RADIUS][tx + TILE_SIZE + KERNEL_RADIUS] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Apply convolution using shared memory
    if (x < width && y < height) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int ky = 0; ky < FILTER_SIZE; ky++) {
            for (int kx = 0; kx < FILTER_SIZE; kx++) {
                int shared_x = tx + kx;
                int shared_y = ty + ky;
                
                int kernel_idx = ky * FILTER_SIZE + kx;
                sum += tile[shared_y][shared_x] * kernel[kernel_idx];
                weight_sum += kernel[kernel_idx];
            }
        }
        
        int idx = y * width + x;
        output[idx] = sum / weight_sum;
    }
}