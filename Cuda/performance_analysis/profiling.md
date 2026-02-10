# Profiling with Nsight Compute

## Concept Overview

Nsight Compute is NVIDIA's premier profiler for CUDA applications, providing detailed analysis of kernel performance, memory access patterns, and hardware utilization. It offers comprehensive metrics and guided analysis to help identify performance bottlenecks and optimization opportunities.

## Understanding Nsight Compute

### What is Nsight Compute?

Nsight Compute is a command-line and GUI-based profiler that:
- Collects detailed GPU kernel metrics
- Provides guided analysis of performance bottlenecks
- Offers comparison capabilities between different runs
- Generates detailed reports with recommendations

### Key Features
- **Detailed Metrics**: Hundreds of hardware counters and derived metrics
- **Guided Analysis**: Automated identification of bottlenecks
- **Source Correlation**: Links metrics to specific source code lines
- **Comparison View**: Compare different kernel implementations
- **Export Capabilities**: Generate reports in various formats

## Installation and Setup

### Installing Nsight Compute
```bash
# Download from NVIDIA Developer website
wget https://developer.download.nvidia.com/compute/nsight-compute/Windows/<version>/nsight-compute-<version>.exe

# Or install via package manager on Linux
sudo apt-get install nsight-compute
```

### Basic Usage
```bash
# Profile a CUDA application
ncu --set full ./your_cuda_application

# Profile specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./your_app

# Profile specific kernels
ncu --kernel-name "your_kernel_name" ./your_app
```

## Essential Metrics and Categories

### SM (Streaming Multiprocessor) Utilization
```bash
# SM utilization metrics
ncu --metrics sm__utilization ./your_app
# Shows percentage of time SMs are active

ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./your_app
# Shows how much of peak throughput is being achieved
```

### Memory Bandwidth Metrics
```bash
# Global memory metrics
ncu --metrics dram__bytes_read.throughput,dram__bytes_write.throughput ./your_app
# Shows read/write bandwidth utilization

ncu --metrics gpc__cycles_elapsed.sm/gpc__cycles_elapsed.all ./your_app
# Shows SM active cycles vs total cycles
```

### Warp Stall Reasons
```bash
# Identify why warps are stalling
ncu --metrics smsp__warps_launched.avg_per_active_cycle ./your_app
# Shows warp launch efficiency

ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./your_app
# Shows thread execution efficiency
```

## Command-Line Profiling Examples

### Basic Profiling
```bash
# Full profiling (collects all metrics)
ncu --set full ./vector_add

# Targeted profiling for specific metrics
ncu --metrics flop_count_sp,achieved_occupancy,sm__throughput.avg.pct_of_peak_sustained_elapsed ./vector_add

# Profile only specific kernels
ncu --kernel-name "vectorAdd" ./vector_add
```

### Advanced Profiling Options
```bash
# Profile with specific sampling intervals
ncu --profile-from-start off --sampling-interval 100 ./your_app

# Profile with memory tracing
ncu --page-size 4K --clock-control none --target-processes all ./your_app

# Export results to file
ncu --export profile_results ./your_app
```

## Interpreting Profiling Results

### Sample Output Analysis
```
==PROF== Disconnected from process 12345
==PROF== Analyzing 'vectorAdd'
-------------------------------------------------------------------------------
Nvtx Range Statistics

Time(%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name
-------  ---------------  ---------  --------  --------  --------  --------  -----------  ----
100.0    1,234,567      1          1,234,567 1,234,567 1,234,567 1,234,567 0           vectorAdd

Kernel Statistics

Time(%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name
-------  ---------------  ---------  --------  --------  --------  --------  -----------  ----
100.0    1,234,567      1          1,234,567 1,234,567 1,234,567 1,234,567 0           vectorAdd

Avg. Est. achieved GMEM BW (% of peak): 15.2%
```

### Key Interpretations
- **Achieved Bandwidth**: 15.2% suggests room for improvement in memory access patterns
- **Utilization**: Low utilization might indicate insufficient parallelism
- **Stall Reasons**: High stall rates indicate specific bottlenecks

## Guided Analysis

### Using Presets
```bash
# Memory-focused analysis
ncu --set memory ./your_app

# Compute-focused analysis  
ncu --set compute ./your_app

# Launch configuration analysis
ncu --set launch ./your_app

# All metrics (comprehensive but slow)
ncu --set full ./your_app
```

### Custom Metric Sets
```bash
# Create custom metric collection
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,achieved_occupancy,branch_efficiency ./your_app
```

## Source Code Correlation

### Line-Level Profiling
```bash
# Profile with source correlation
ncu --source ./your_app

# Show source with metrics
ncu --source --metrics flop_count_sp,branch_efficiency ./your_app
```

### Example Output with Source Correlation
```
Line  SM_Flops  Branch_Efficiency  Description
----  --------  -----------------  -----------
  23    100000        95.2%        C[i] = A[i] + B[i];
  24         0        100%         if (i < n) {
  25     50000        95.2%           C[i] *= scale;
  26         0        100%         }
```

## Performance Bottleneck Identification

### Common Bottlenecks and Solutions

#### 1. Low Occupancy
```bash
# Check occupancy
ncu --metrics achieved_occupancy ./your_app

# If low occupancy detected:
# - Reduce register usage
# - Adjust block size
# - Use __launch_bounds__
```

#### 2. Memory Bandwidth Issues
```bash
# Check memory bandwidth
ncu --metrics dram__throughput ./your_app

# If bandwidth is low:
# - Check coalescing: gld_efficiency, gst_efficiency
# - Consider shared memory usage
# - Optimize access patterns
```

#### 3. Warp Divergence
```bash
# Check branch efficiency
ncu --metrics branch_efficiency ./your_app

# If branch efficiency is low:
# - Minimize conditional statements
# - Use predication where possible
# - Restructure algorithms to reduce divergence
```

#### 4. Shared Memory Bank Conflicts
```bash
# Check shared memory efficiency
ncu --metrics sm__shared_utilization ./your_app

# Check for bank conflicts
ncu --metrics shared_replay_overhead ./your_app
```

## Advanced Profiling Techniques

### Iterative Profiling
```bash
# Step 1: Overall performance
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,achieved_occupancy ./your_app

# Step 2: Memory access patterns
ncu --metrics gld_efficiency,gst_efficiency,shared_efficiency ./your_app

# Step 3: Detailed bottleneck analysis
ncu --metrics smsp__stall_pipe_busy,smsp__stall_exec_dependency ./your_app
```

### Comparing Optimizations
```bash
# Profile original version
ncu --export original_profile ./original_version

# Profile optimized version  
ncu --export optimized_profile ./optimized_version

# Compare results using GUI or command line
ncu --compare original_profile.ncu-rep optimized_profile.ncu-rep
```

## Automated Analysis Script

```bash
#!/bin/bash
# automated_profiling.sh

APP_NAME=$1
KERNEL_NAME=${2:-".*"}  # Default to all kernels

echo "Profiling application: $APP_NAME"
echo "Targeting kernels: $KERNEL_NAME"

# Basic performance metrics
echo "=== Basic Performance ==="
ncu --kernel-name "$KERNEL_NAME" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,achieved_occupancy,branch_efficiency \
    --print-summary per-kernel \
    $APP_NAME

# Memory analysis
echo "=== Memory Analysis ==="
ncu --kernel-name "$KERNEL_NAME" \
    --metrics dram__bytes_throughput,gld_efficiency,gst_efficiency,shared_efficiency \
    --print-summary per-kernel \
    $APP_NAME

# Warp stall analysis
echo "=== Warp Stall Analysis ==="
ncu --kernel-name "$KERNEL_NAME" \
    --metrics smsp__stall_pipe_busy,smsp__stall_exec_dependency,smsp__stall_misc \
    --print-summary per-kernel \
    $APP_NAME

echo "Profiling complete. See detailed results with: ncu --kernel-name '$KERNEL_NAME' --set full $APP_NAME"
```

## Integration with Development Workflow

### Continuous Profiling
```bash
# Add to your Makefile or build script
profile: build
	ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,achieved_occupancy,gld_efficiency \
	    --export $(PROFILE_OUTPUT) ./$(TARGET)
	@echo "Profile saved to $(PROFILE_OUTPUT)"
```

### Regression Testing
```bash
# Compare current performance to baseline
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --export current.ncu-rep ./current_version
ncu --compare baseline.ncu-rep current.ncu-rep
```

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Use Nsight Compute to collect detailed GPU kernel metrics
- Interpret profiler data to identify performance bottlenecks
- Apply guided analysis sections to prioritize optimization opportunities
- Compare different kernel implementations using profiling data

## Hands-on Tutorial

See the `profiling_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.