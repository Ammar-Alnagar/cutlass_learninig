# Profiling with Nsight Compute

## Concept Overview

Nsight Compute is NVIDIA's premier profiler for CUDA applications, providing detailed insights into kernel performance, memory access patterns, and hardware utilization. It helps identify bottlenecks and optimization opportunities by collecting and analyzing a wide range of performance metrics.

## Introduction to Nsight Compute

### What is Nsight Compute?
- NVIDIA's command-line and GUI profiler for CUDA applications
- Collects detailed metrics on kernel execution, memory access, and hardware utilization
- Provides guided analysis to highlight performance bottlenecks
- Offers metric comparison against theoretical peaks

### Key Features
- **Detailed Metrics**: Hundreds of performance counters
- **Guided Analysis**: Automatic bottleneck detection
- **Source Correlation**: Links metrics to specific source code lines
- **Comparison Views**: Compare different kernel executions
- **Export Capabilities**: Generate reports and visualizations

## Installing and Setting Up Nsight Compute

### Installation
```bash
# Download from NVIDIA Developer website
wget https://developer.download.nvidia.com/compute/nsight-compute/Windows/win-x64/NsightCompute-windows-x86_64-2022.2.1.zip

# Or use package manager
sudo apt-get install nsight-compute
```

### Basic Usage
```bash
# Profile a CUDA application
ncu ./my_cuda_app

# Profile specific kernels
ncu --kernel-name "my_kernel" ./my_cuda_app

# Collect specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gmem__throughput.avg.pct_of_peak_sustained_elapsed ./my_cuda_app
```

## Essential Metrics to Monitor

### SM Utilization Metrics
```bash
# Occupancy metrics
sm__maximum_warps_per_active_cycle.pct  # Warp launch rate
smsp__warps_launched.avg_per_second     # Warps launched per second
sm__warps_launched_rate.avg             # Warp launch rate
```

### Memory Bandwidth Metrics
```bash
# Global memory metrics
gmem__throughput.avg.pct_of_peak_sustained_elapsed  # Global memory utilization
gmem__bytes_per_second.sum                         # Global memory bandwidth
dram__bytes_per_second.sum                         # DRAM bandwidth
```

### Compute Utilization Metrics
```bash
# Compute metrics
smsp__sass_thread_inst_executed_op_fadd_pred_on.thread_pct  # FADD utilization
smsp__sass_thread_inst_executed_op_fmul_pred_on.thread_pct  # FMUL utilization
smsp__sass_thread_inst_executed_op_ffma_pred_on.thread_pct  # FFMA utilization
sm__pipe_fma_cycles.avg                                  # FMA pipe utilization
```

### Warp Stall Reasons
```bash
# Stall reason metrics
smsp__warp_issue_stalled_imc_miss.pct_of_issue_slot_cycles_elapsed  # IMC miss stalls
smsp__warp_issue_stalled_barriers.pct_of_issue_slot_cycles_elapsed  # Barrier stalls
smsp__warp_issue_stalled_membar.pct_of_issue_slot_cycles_elapsed    # Memory barrier stalls
smsp__warp_issue_stalled_short_scoreboard.pct_of_issue_slot_cycles_elapsed  # Scoreboard stalls
```

## Profiling Workflow

### Step 1: Basic Profiling
```bash
# Profile with default metrics
ncu --target-processes all ./my_application

# Profile only specific kernels
ncu --kernel-name "my_kernel_name" ./my_application
```

### Step 2: Metric Selection
```bash
# Focus on specific metric categories
ncu --metrics sm__throughput,achieved_occupancy,gmem__throughput ./my_application

# Custom metric selection
ncu --metrics smsp__thread_inst_executed_per_inst_executed,smsp__inst_executed_per_warp_initiated ./my_application
```

### Step 3: Guided Analysis
```bash
# Run guided analysis to identify bottlenecks
ncu --set full ./my_application

# Focus on specific analysis areas
ncu --set memory ./my_application  # Memory-focused analysis
ncu --set compute ./my_application  # Compute-focused analysis
```

## Interpreting Profiling Results

### Sample Output Analysis
```
Kernel: vectorAdd(float*, float*, float*, int)
Section: Speed Of Light
------------------------------
gm__throughput.avg.pct_of_peak_sustained_elapsed  65.23%  # Global memory utilization
sm__throughput.avg.pct_of_peak_sustained_elapsed  23.45%  # SM throughput
------------------------------

Section: Memory Workload Analysis
---------------------------------
gld_efficiency                             87.12%  # Global load efficiency
gst_efficiency                             92.34%  # Global store efficiency
---------------------------------
```

### Performance Indicators
- **High Memory Utilization** (>80%): Memory-bound kernel
- **Low Memory Efficiency** (<60%): Coalescing issues
- **Low SM Throughput** (<50%): Underutilized compute resources
- **High Stall Rates**: Execution pipeline inefficiencies

## Common Performance Bottlenecks

### 1. Memory-Bound Issues
```cuda
// Identify with: low gld_efficiency, gst_efficiency
// Solution: improve coalescing, use shared memory

// Bad: uncoalesced access
__global__ void bad_access(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Strided access pattern
    data[tid * 2] = tid;  // Uncoalesced
}

// Good: coalesced access
__global__ void good_access(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid] = tid;  // Coalesced
}
```

### 2. Compute-Bound Issues
```cuda
// Identify with: high ALU utilization, low memory utilization
// Solution: optimize arithmetic, use specialized instructions

// Potentially compute-bound kernel
__global__ void compute_heavy(float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float x = input[tid];
        // Many arithmetic operations
        for (int i = 0; i < 100; i++) {
            x = x * x + 0.1f;
            x = sqrtf(x);
        }
        output[tid] = x;
    }
}
```

### 3. Occupancy Issues
```bash
# Identify with: low achieved_occupancy
# Solutions: reduce register usage, adjust block size

# Check occupancy
ncu --metrics achieved_occupancy,sm__warps_per_active_cycle.max ./my_app
```

## Optimization Strategies Based on Profiling

### Memory Optimization
```bash
# If gld_efficiency < 60%
# 1. Check for coalescing issues
ncu --metrics dram__sectors_read_per_request.dyn,gld_transactions_per_request ./my_app

# 2. Consider shared memory usage
ncu --metrics sm__shared_utilization ./my_app
```

### Compute Optimization
```bash
# If sm__pipe_fma_cycles.avg < 50%
# 1. Check for arithmetic intensity
ncu --metrics smsp__sass_thread_inst_executed_op_ffma_pred_on.thread_pct ./my_app

# 2. Consider using tensor cores for matrix operations
ncu --metrics smsp__sass_thread_inst_executed_op_hmma_pred_on.thread_pct ./my_app
```

### Occupancy Optimization
```bash
# If achieved_occupancy < 50%
# 1. Check register usage
ncu --metrics smsp__inst_executed_per_warp_initiated,sm__max_warps_per_active_cycle ./my_app

# 2. Use launch bounds
__global__ 
__launch_bounds__(256, 4)  // 256 threads/block, 4 blocks/SM minimum
void optimized_kernel(float* data) {
    // Kernel code
}
```

## Advanced Profiling Techniques

### Source Code Correlation
```bash
# Profile with source correlation
ncu --source ./my_app

# Highlight specific source lines
ncu --source --section SpeedOfLight --page detail ./my_app
```

### Timeline Analysis
```bash
# Generate timeline for multiple kernels
ncu --export timeline_report ./my_app

# Analyze kernel launch patterns
ncu --metrics smsp__cycles_elapsed_per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed ./my_app
```

### Custom Configuration Files
```bash
# Create custom metric collection
cat > custom_config.ncu << EOF
sm__throughput.avg.pct_of_peak_sustained_elapsed
gmem__throughput.avg.pct_of_peak_sustained_elapsed
achieved_occupancy
gld_efficiency
gst_efficiency
EOF

# Use custom config
ncu --profile-from-config custom_config.ncu ./my_app
```

## Profiling Best Practices

### 1. Profile Representative Workloads
- Use realistic data sizes
- Profile steady-state behavior, not initialization
- Average results over multiple runs

### 2. Iterative Optimization
- Profile → Optimize → Profile again
- Focus on biggest bottlenecks first
- Measure impact of each change

### 3. Compare Against Baseline
- Establish baseline performance
- Track improvements quantitatively
- Use comparison views in Nsight Compute

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Use Nsight Compute to profile CUDA kernels and identify bottlenecks
- Interpret profiling metrics to understand performance characteristics
- Apply profiling insights to guide optimization efforts
- Measure and validate performance improvements quantitatively

## Hands-on Tutorial

See the `profiling_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.