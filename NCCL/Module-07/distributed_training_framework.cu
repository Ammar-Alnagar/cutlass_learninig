/*
 * Module 7: Real-world Applications and Case Studies
 * 
 * This example implements a simplified distributed training framework
 * that demonstrates how NCCL is used in real-world deep learning systems.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <math.h>
#include <vector>
#include <chrono>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Simple linear algebra operations for neural network simulation
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void add_bias(float* input, float* bias, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;
    
    if (idx < total_elements) {
        int col = idx % cols;
        output[idx] = input[idx] + bias[col];
    }
}

__global__ void relu_activation(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void compute_loss_and_gradient(float* predictions, float* targets, float* loss, 
                                        float* grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        grad[idx] = diff;  // Simple MSE gradient
        
        // Accumulate loss (will need reduction later)
        loss[idx] = diff * diff;
    }
}

// Simulated neural network layer
struct Layer {
    float *weights, *biases, *activations, *gradients;
    int input_size, output_size;
    
    Layer(int in_size, int out_size) : input_size(in_size), output_size(out_size) {
        // Allocate GPU memory for layer parameters
        CUDACHECK(cudaMalloc(&weights, sizeof(float) * input_size * output_size));
        CUDACHECK(cudaMalloc(&biases, sizeof(float) * output_size));
        CUDACHECK(cudaMalloc(&activations, sizeof(float) * output_size));
        CUDACHECK(cudaMalloc(&gradients, sizeof(float) * output_size));
        
        // Initialize weights randomly
        float* h_weights = (float*)malloc(sizeof(float) * input_size * output_size);
        float* h_biases = (float*)malloc(sizeof(float) * output_size);
        
        for (int i = 0; i < input_size * output_size; i++) {
            h_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;  // Small random values
        }
        for (int i = 0; i < output_size; i++) {
            h_biases[i] = 0.0f;
        }
        
        CUDACHECK(cudaMemcpy(weights, h_weights, sizeof(float) * input_size * output_size, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(biases, h_biases, sizeof(float) * output_size, cudaMemcpyHostToDevice));
        
        free(h_weights);
        free(h_biases);
    }
    
    ~Layer() {
        CUDACHECK(cudaFree(weights));
        CUDACHECK(cudaFree(biases));
        CUDACHECK(cudaFree(activations));
        CUDACHECK(cudaFree(gradients));
    }
};

// Neural network model
struct NeuralNetwork {
    std::vector<Layer*> layers;
    float *input, *output, *loss_buffer, *gradient_buffer;
    int batch_size, input_size, output_size;
    
    NeuralNetwork(int batch_sz, int in_size, int out_size) : 
        batch_size(batch_sz), input_size(in_size), output_size(out_size) {
        
        // Create a simple 2-layer network
        layers.push_back(new Layer(input_size, 128));  // Hidden layer
        layers.push_back(new Layer(128, output_size)); // Output layer
        
        // Allocate memory for input, output, and temporary buffers
        CUDACHECK(cudaMalloc(&input, sizeof(float) * batch_size * input_size));
        CUDACHECK(cudaMalloc(&output, sizeof(float) * batch_size * output_size));
        CUDACHECK(cudaMalloc(&loss_buffer, sizeof(float) * batch_size * output_size));
        CUDACHECK(cudaMalloc(&gradient_buffer, sizeof(float) * batch_size * output_size));
    }
    
    ~NeuralNetwork() {
        for (auto layer : layers) {
            delete layer;
        }
        CUDACHECK(cudaFree(input));
        CUDACHECK(cudaFree(output));
        CUDACHECK(cudaFree(loss_buffer));
        CUDACHECK(cudaFree(gradient_buffer));
    }
    
    void forward_pass(cudaStream_t stream) {
        // Initialize with input data
        float* current_activations = input;
        
        for (size_t i = 0; i < layers.size(); i++) {
            Layer* layer = layers[i];
            
            // Matrix multiply: activations * weights
            dim3 block(16, 16);
            dim3 grid((layer->output_size + block.x - 1) / block.x, 
                     (batch_size + block.y - 1) / block.y);
            
            matrix_multiply<<<grid, block, 0, stream>>>(
                current_activations, layer->weights, layer->activations, 
                batch_size, layer->output_size, 
                (i == 0) ? input_size : layers[i-1]->output_size);
            
            // Add bias
            int total_elements = batch_size * layer->output_size;
            int threads = 256;
            int blocks = (total_elements + threads - 1) / threads;
            
            add_bias<<<blocks, threads, 0, stream>>>(
                layer->activations, layer->biases, layer->activations, 
                batch_size, layer->output_size);
            
            // Apply activation function (except for last layer)
            if (i < layers.size() - 1) {
                relu_activation<<<blocks, threads, 0, stream>>>(
                    layer->activations, layer->activations, total_elements);
            }
            
            current_activations = layer->activations;
        }
        
        // Copy final output
        CUDACHECK(cudaMemcpyAsync(output, current_activations, 
                                 sizeof(float) * batch_size * output_size, 
                                 cudaMemcpyDeviceToDevice, stream));
    }
    
    void backward_pass(float* targets, cudaStream_t stream) {
        // Compute loss and initial gradients
        int total_elements = batch_size * output_size;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        
        compute_loss_and_gradient<<<blocks, threads, 0, stream>>>(
            output, targets, loss_buffer, gradient_buffer, total_elements);
        
        // Propagate gradients backwards through layers
        float* current_gradients = gradient_buffer;
        
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer* layer = layers[i];
            int prev_output_size = (i > 0) ? layers[i-1]->output_size : input_size;
            
            // For simplicity, we'll just store gradients in the layer's gradient buffer
            // In a real implementation, we would compute weight gradients and update weights
            CUDACHECK(cudaMemcpyAsync(layer->gradients, current_gradients,
                                     sizeof(float) * batch_size * layer->output_size,
                                     cudaMemcpyDeviceToDevice, stream));
            
            // In a real implementation, we would compute gradients for previous layer
            // and update weights using the computed gradients
        }
    }
};

// Distributed trainer that uses NCCL for gradient synchronization
class DistributedTrainer {
private:
    int world_size;  // Total number of processes/GPUs
    int rank;        // Current process rank
    ncclComm_t comm; // NCCL communicator
    cudaStream_t stream;
    
public:
    NeuralNetwork* model;
    
    DistributedTrainer(int world_sz, int rk, int batch_size, int input_size, int output_size) 
        : world_size(world_sz), rank(rk) {
        
        // Set device for this rank
        CUDACHECK(cudaSetDevice(rank));
        
        // Create CUDA stream
        CUDACHECK(cudaStreamCreate(&stream));
        
        // Initialize NCCL communicator
        NCCLCHECK(ncclCommInitRank(&comm, world_size, (ncclUniqueId){0}, rank));
        
        // Create neural network model
        model = new NeuralNetwork(batch_size, input_size, output_size);
        
        printf("Initialized trainer on GPU %d (rank %d) with %d total GPUs\n", 
               rank, rank, world_size);
    }
    
    ~DistributedTrainer() {
        delete model;
        ncclCommDestroy(comm);
        CUDACHECK(cudaStreamDestroy(stream));
    }
    
    void synchronize_gradients() {
        // In a real implementation, we would synchronize gradients across all GPUs
        // For this example, we'll just demonstrate the concept
        
        printf("Synchronizing gradients across all %d GPUs...\n", world_size);
        
        // Example: synchronize a dummy gradient buffer
        // In reality, this would be done for each parameter tensor
        float* dummy_grads;
        CUDACHECK(cudaMalloc(&dummy_grads, sizeof(float) * 1024));
        CUDACHECK(cudaMemset(dummy_grads, rank * 10.0f, sizeof(float) * 1024));
        
        // Perform AllReduce to average gradients across all GPUs
        NCCLCHECK(ncclAllReduce((const void*)dummy_grads, (void*)dummy_grads, 1024, 
                               ncclFloat32, ncclSum, comm, stream));
        
        // Divide by world_size to get average (this division would normally happen on GPU)
        // In a real implementation, we'd do this with a kernel
        
        CUDACHECK(cudaStreamSynchronize(stream));
        CUDACHECK(cudaFree(dummy_grads));
        
        printf("Gradient synchronization completed\n");
    }
    
    float train_step(float* input_data, float* target_data) {
        // Set device
        CUDACHECK(cudaSetDevice(rank));
        
        // Copy input and target data to GPU
        CUDACHECK(cudaMemcpyAsync(model->input, input_data, 
                                sizeof(float) * model->batch_size * model->input_size,
                                cudaMemcpyHostToDevice, stream));
        
        CUDACHECK(cudaMemcpyAsync(target_data, target_data,  // Note: this is a simplification
                                sizeof(float) * model->batch_size * model->output_size,
                                cudaMemcpyHostToDevice, stream));
        
        // Forward pass
        model->forward_pass(stream);
        
        // Backward pass
        model->backward_pass(target_data, stream);
        
        // Synchronize gradients across all GPUs
        synchronize_gradients();
        
        // In a real implementation, we would update model parameters here
        // using the synchronized gradients
        
        // For this example, just return a dummy loss value
        CUDACHECK(cudaStreamSynchronize(stream));
        return 0.1f; // Dummy loss
    }
    
    void save_checkpoint(const char* filename) {
        printf("Saving checkpoint for rank %d to %s\n", rank, filename);
        // In a real implementation, we would save model weights to file
    }
    
    void load_checkpoint(const char* filename) {
        printf("Loading checkpoint for rank %d from %s\n", rank, filename);
        // In a real implementation, we would load model weights from file
    }
};

int main(int argc, char* argv[]) {
    int nGPUs = 4;  // Number of GPUs to simulate
    
    // Check available GPUs
    int gpu_count;
    CUDACHECK(cudaGetDeviceCount(&gpu_count));
    if (gpu_count < nGPUs) {
        nGPUs = gpu_count;
        printf("Only %d GPUs available, using %d\n", gpu_count, nGPUs);
    }
    
    printf("Starting distributed training simulation with %d GPUs\n", nGPUs);

    // Initialize NCCL unique ID on rank 0, broadcast to others
    // In a real multi-process setup, this would be handled by the process launcher
    ncclUniqueId id;
    if (getenv("OMPI_COMM_WORLD_RANK") || getenv("RANK")) {
        // Running under MPI or similar - would get unique ID from environment
        printf("Running in distributed environment\n");
    } else {
        // Standalone simulation - create our own ID
        if (getpid() % 2 == 0) {  // Simulate rank 0 creating the ID
            NCCLCHECK(ncclGetUniqueId(&id));
        }
        // In a real scenario, the ID would be shared between processes
    }

    // Create trainers for each simulated GPU/rank
    std::vector<DistributedTrainer*> trainers;
    
    for (int rank = 0; rank < nGPUs; rank++) {
        // In a real distributed setup, each process would only create its own trainer
        // For this simulation, we'll create all trainers in one process
        trainers.push_back(new DistributedTrainer(nGPUs, rank, 32, 784, 10));
    }

    // Simulate training process
    printf("\nStarting distributed training simulation...\n");
    
    const int num_epochs = 2;
    const int steps_per_epoch = 5;
    
    // Generate dummy training data
    std::vector<float> dummy_input(32 * 784);  // batch_size * input_size
    std::vector<float> dummy_target(32 * 10);  // batch_size * output_size
    
    for (size_t i = 0; i < dummy_input.size(); i++) {
        dummy_input[i] = (float)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < dummy_target.size(); i++) {
        dummy_target[i] = (float)(rand() % 10);  // Class labels 0-9
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("\nEpoch %d/%d\n", epoch + 1, num_epochs);
        
        float epoch_loss = 0.0f;
        
        for (int step = 0; step < steps_per_epoch; step++) {
            printf("  Step %d/%d - ", step + 1, steps_per_epoch);
            
            // Each trainer performs a training step
            for (int rank = 0; rank < nGPUs; rank++) {
                float loss = trainers[rank]->train_step(dummy_input.data(), dummy_target.data());
                epoch_loss += loss;
                
                if (rank == 0) {  // Just print from rank 0 to avoid clutter
                    printf("GPU %d: loss=%.4f ", rank, loss);
                }
            }
            
            if (epoch == 0 && step == 0) {
                printf("[Demo: Showing all GPUs for first step]");
            }
            printf("\n");
        }
        
        printf("  Average epoch loss: %.4f\n", epoch_loss / (steps_per_epoch * nGPUs));
        
        // Save checkpoint periodically
        if (epoch % 1 == 0) {
            for (int rank = 0; rank < nGPUs; rank++) {
                char filename[256];
                snprintf(filename, sizeof(filename), "checkpoint_rank_%d_epoch_%d.bin", rank, epoch);
                trainers[rank]->save_checkpoint(filename);
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    printf("\nTraining completed in %ld ms\n", duration.count());

    // Cleanup
    for (auto trainer : trainers) {
        delete trainer;
    }

    printf("\nReal-world distributed training simulation completed!\n");
    printf("This example demonstrated:\n");
    printf("- A simplified neural network implementation\n");
    printf("- Gradient synchronization using NCCL\n");
    printf("- Distributed training loop with checkpointing\n");
    printf("- Real-world patterns used in production systems\n");

    return 0;
}