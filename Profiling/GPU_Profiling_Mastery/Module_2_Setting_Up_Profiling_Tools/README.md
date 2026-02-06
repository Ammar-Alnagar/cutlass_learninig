# Module 2: Setting Up Profiling Tools and Environment

## Overview

In this module, we'll set up the necessary tools for GPU profiling. We'll cover the most popular GPU profiling tools and how to configure your environment for effective profiling.

## Learning Objectives

By the end of this module, you will:
- Install and configure major GPU profiling tools
- Understand the differences between various profiling tools
- Set up a proper development environment for GPU profiling
- Run your first profiling session

## Popular GPU Profiling Tools

### NVIDIA Tools
- **Nsight Compute**: Detailed kernel profiling for CUDA applications
- **Nsight Systems**: System-wide performance analysis
- **nvprof**: Legacy profiler (still useful for learning)

### AMD Tools
- **ROCm Profiler (rocprof)**: For AMD GPU profiling
- **CodeXL**: AMD's comprehensive profiling suite

### Vendor-Neutral Tools
- **gDEBugger**: OpenGL debugging and profiling
- **RenderDoc**: Frame capture and analysis tool

For this course, we'll focus primarily on NVIDIA tools since they're widely used and well-documented.

## Installing NVIDIA Profiling Tools

### Prerequisites
1. NVIDIA GPU with compute capability 3.5 or higher
2. Compatible driver (450.80.02 or later)
3. CUDA Toolkit installed

### Installing CUDA Toolkit
```bash
# On Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# On CentOS/RHEL
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo yum install -y cuda-toolkit-11-8
```

### Verifying Installation
```bash
nvidia-smi
nvcc --version
```

## Setting Up a Sample CUDA Project

Let's create a simple CUDA program to practice profiling:

```cuda
// simple_kernel.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Host arrays
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    printf("Vector addition completed successfully!\n");
    return 0;
}
```

### Compiling the CUDA Program
```bash
nvcc -o simple_kernel simple_kernel.cu
```

## Using nvprof (Legacy but Educational)

Let's start with nvprof to understand basic profiling concepts:

```bash
# Basic profiling
nvprof ./simple_kernel

# More detailed profiling
nvprof --print-gpu-trace ./simple_kernel

# Profile specific metrics
nvprof --metrics achieved_occupancy,warps_launched ./simple_kernel
```

## Using Nsight Compute

Nsight Compute provides more detailed kernel analysis:

```bash
# Profile the kernel and save results
ncu --set full ./simple_kernel

# Profile specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed, dram__throughput.avg.bytes_per_second ./simple_kernel

# Interactive profiling
ncu --gui ./simple_kernel  # Opens GUI for detailed analysis
```

## Using Nsight Systems

Nsight Systems provides system-wide profiling:

```bash
# Capture timeline
nsys profile --trace=cuda,nvtx ./simple_kernel

# View results
nsys-ui report1.nsys-rep  # Opens GUI
```

## Environment Configuration Tips

### Setting Up Environment Variables
```bash
# Add CUDA to PATH and LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Creating a Profiling Script
Create a script to run common profiling commands:

```bash
#!/bin/bash
# profile_script.sh

PROGRAM=$1

echo "=== Basic Profiling ==="
nvprof $PROGRAM

echo -e "\n=== Memory Throughput ==="
nvprof --metrics dram_read_throughput,dram_write_throughput $PROGRAM

echo -e "\n=== Occupancy Metrics ==="
nvprof --metrics achieved_occupancy,active_warps_per_active_cycle $PROGRAM

echo -e "\n=== Warp Execution Efficiency ==="
nvprof --metrics warp_execution_efficiency,branch_efficiency $PROGRAM
```

## Hands-On Exercise

1. Install the CUDA toolkit and profiling tools
2. Compile and run the sample CUDA program
3. Profile it using nvprof with different metrics
4. Try using Nsight Compute for more detailed analysis
5. Document your observations about the profiling output

## Key Takeaways

- Proper tool setup is crucial for effective GPU profiling
- Different tools provide different levels of detail
- Start with basic profiling and gradually move to more advanced tools
- Always profile with realistic workloads

## Next Steps

In the next module, we'll dive deeper into basic profiling techniques and learn how to interpret profiling results.