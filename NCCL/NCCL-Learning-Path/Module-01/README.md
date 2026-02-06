# Module 1: Introduction to NCCL - Concepts and Setup

## What is NCCL?

NCCL (pronounced "Nickel") stands for NVIDIA Collective Communications Library. It is a library of multi-GPU collective communication primitives that are useful for parallel computing applications, particularly in deep learning and HPC (High Performance Computing).

### Key Features of NCCL:
- Optimized for NVIDIA GPUs and interconnects (NVLink, PCIe)
- Supports multiple collective operations (AllReduce, AllGather, Broadcast, etc.)
- Handles complex topologies automatically
- Provides high bandwidth and low latency
- Works across multiple processes and nodes

## Why Use NCCL?

When training deep learning models on multiple GPUs, you often need to synchronize gradients or parameters across devices. Traditional approaches like CPU-based implementations are too slow for this. NCCL provides GPU-accelerated collective operations that are essential for distributed training.

## Prerequisites

Before diving into NCCL, ensure you have:
- A system with NVIDIA GPUs
- CUDA toolkit installed
- NVIDIA drivers compatible with your CUDA version
- Basic understanding of C/C++ programming
- Familiarity with GPU programming concepts (optional but helpful)

## Installing NCCL

### Option 1: Using Package Managers

For Ubuntu/Debian:
```bash
sudo apt-get install libnccl2 libnccl-dev
```

For CentOS/RHEL:
```bash
sudo yum install nccl
```

### Option 2: Building from Source

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build
```

## Basic Architecture

NCCL operates at multiple levels:
1. **Host level**: Manages multiple processes and threads
2. **Device level**: Manages multiple GPUs within a process
3. **Collective level**: Executes coordinated operations across GPUs

## Key Concepts

### Collectives
A collective operation involves multiple participating GPUs working together to perform a computation. Common examples include:
- **AllReduce**: Reduces data across all GPUs and broadcasts the result
- **Broadcast**: Sends data from one GPU to all others
- **AllGather**: Gathers data from all GPUs and distributes to all
- **Reduce**: Reduces data from all GPUs to one

### Communicator (comm)
A communicator represents a group of participating GPUs and contains all the state needed for collective operations.

### Root
In operations like broadcast or reduce, the "root" is the GPU that sends data (broadcast) or receives the result (reduce).

## Getting Started with NCCL Programming

NCCL follows a typical workflow:
1. Initialize the library
2. Create a communicator
3. Perform collective operations
4. Destroy the communicator
5. Clean up the library

## Learning Objectives for This Module

By the end of this module, you should:
- Understand what NCCL is and why it's important
- Know how to install and set up NCCL
- Understand basic NCCL concepts and terminology
- Be familiar with the typical NCCL programming workflow
- Have a working NCCL environment for the next modules

## Next Steps

After completing this module, proceed to Module 2 where you'll learn about basic collective operations like AllReduce and Broadcast with hands-on examples.