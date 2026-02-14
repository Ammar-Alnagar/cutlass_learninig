#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// CSR (Compressed Sparse Row) format structures
typedef struct {
    float *values;      // Non-zero values
    int *col_indices;   // Column indices of non-zeros
    int *row_ptr;       // Pointer to start of each row
    int num_rows;
    int num_cols;
    int nnz;            // Number of non-zero elements
} csr_matrix;

// Naive SPMV kernel - one thread per row
__global__ void spmv_naive(csr_matrix d_mat, float *vec, float *result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < d_mat.num_rows) {
        float sum = 0.0f;
        int row_start = d_mat.row_ptr[row];
        int row_end = d_mat.row_ptr[row + 1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += d_mat.values[j] * vec[d_mat.col_indices[j]];
        }
        result[row] = sum;
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

// Generate a sparse matrix in CSR format
void generate_sparse_matrix(csr_matrix *mat, int rows, int cols, float density) {
    mat->num_rows = rows;
    mat->num_cols = cols;
    
    // Count non-zero elements
    int total_elements = rows * cols;
    mat->nnz = (int)(total_elements * density);
    
    // Allocate host memory
    float *temp_values = (float*)malloc(total_elements * sizeof(float));
    int *temp_col_indices = (int*)malloc(total_elements * sizeof(int));
    int *temp_row_counts = (int*)calloc(rows, sizeof(int));
    
    // Initialize with zeros
    for (int i = 0; i < total_elements; i++) {
        temp_values[i] = 0.0f;
    }
    
    // Randomly place non-zero values
    int nnz_count = 0;
    while (nnz_count < mat->nnz) {
        int pos = rand() % total_elements;
        int row_idx = pos / cols;
        int col_idx = pos % cols;
        
        if (temp_values[pos] == 0.0f) {
            temp_values[pos] = (float)(rand()) / RAND_MAX;
            temp_col_indices[nnz_count] = col_idx;
            temp_row_counts[row_idx]++;
            nnz_count++;
        }
    }
    
    // Convert row counts to row pointers
    mat->row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    mat->row_ptr[0] = 0;
    for (int i = 0; i < rows; i++) {
        mat->row_ptr[i + 1] = mat->row_ptr[i] + temp_row_counts[i];
    }
    
    // Allocate and fill values and column indices arrays
    mat->values = (float*)malloc(mat->nnz * sizeof(float));
    mat->col_indices = (int*)malloc(mat->nnz * sizeof(int));
    
    // Reset row counts to use as current position tracker
    for (int i = 0; i < rows; i++) {
        temp_row_counts[i] = 0;
    }
    
    // Fill the actual sparse matrix arrays
    for (int pos = 0; pos < total_elements; pos++) {
        if (temp_values[pos] != 0.0f) {
            int row_idx = pos / cols;
            int idx_in_row = temp_row_counts[row_idx]++;
            int dest_idx = mat->row_ptr[row_idx] + idx_in_row;
            
            mat->values[dest_idx] = temp_values[pos];
            mat->col_indices[dest_idx] = pos % cols;
        }
    }
    
    // Clean up temporary arrays
    free(temp_values);
    free(temp_col_indices);
    free(temp_row_counts);
}

// CPU SPMV for verification
void cpu_spmv(csr_matrix *mat, float *vec, float *result) {
    for (int i = 0; i < mat->num_rows; i++) {
        result[i] = 0.0f;
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
            result[i] += mat->values[j] * vec[mat->col_indices[j]];
        }
    }
}

int main(int argc, char **argv) {
    int rows = 1024, cols = 1024;
    float density = 0.01f; // 1% non-zero elements
    
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    if (argc >= 4) {
        density = atof(argv[3]);
    }
    
    printf("Sparse matrix-vector multiplication: %dx%d with %.2f%% density\n", 
           rows, cols, density * 100);
    
    // Generate sparse matrix
    csr_matrix h_mat;
    generate_sparse_matrix(&h_mat, rows, cols, density);
    
    // Host vectors
    float *h_vec = (float*)malloc(cols * sizeof(float));
    float *h_result = (float*)malloc(rows * sizeof(float));
    float *h_expected = (float*)malloc(rows * sizeof(float));
    
    // Initialize vector with random values
    for (int i = 0; i < cols; i++) {
        h_vec[i] = (float)(rand()) / RAND_MAX;
    }
    
    // Calculate expected result on CPU
    cpu_spmv(&h_mat, h_vec, h_expected);
    
    // Device matrix structure
    csr_matrix d_mat;
    checkCudaError(cudaMalloc(&d_mat.values, h_mat.nnz * sizeof(float)), "cudaMalloc d_mat.values");
    checkCudaError(cudaMalloc(&d_mat.col_indices, h_mat.nnz * sizeof(int)), "cudaMalloc d_mat.col_indices");
    checkCudaError(cudaMalloc(&d_mat.row_ptr, (h_mat.num_rows + 1) * sizeof(int)), "cudaMalloc d_mat.row_ptr");
    
    // Device vectors
    float *d_vec, *d_result;
    checkCudaError(cudaMalloc(&d_vec, cols * sizeof(float)), "cudaMalloc d_vec");
    checkCudaError(cudaMalloc(&d_result, rows * sizeof(float)), "cudaMalloc d_result");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_mat.values, h_mat.values, h_mat.nnz * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy values");
    checkCudaError(cudaMemcpy(d_mat.col_indices, h_mat.col_indices, h_mat.nnz * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy col_indices");
    checkCudaError(cudaMemcpy(d_mat.row_ptr, h_mat.row_ptr, (h_mat.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy row_ptr");
    checkCudaError(cudaMemcpy(d_vec, h_vec, cols * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_vec");
    
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel with %d blocks of %d threads each\n", blocksPerGrid, threadsPerBlock);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    spmv_naive<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_vec, d_result);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    spmv_naive<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_vec, d_result);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_result, d_result, rows * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_result");
    
    // Verification: Check a few random points
    bool success = true;
    int num_checks = min(100, rows);
    
    for (int check = 0; check < num_checks; check++) {
        int idx = (check * rows) / num_checks;  // Evenly distributed indices
        
        if (abs(h_result[idx] - h_expected[idx]) > 1e-3) {
            printf("Verification failed at index %d: expected %f, got %f\n", idx, h_expected[idx], h_result[idx]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Verification PASSED (checked %d elements)\n", num_checks);
        
        // Calculate performance metrics
        double flops = 2.0 * h_mat.nnz; // 1 multiply + 1 add per non-zero
        double gflops = (flops / 1e9) / (kernel_time_ms / 1000.0);
        double bytes_moved = (h_mat.nnz * sizeof(float)) + (cols * sizeof(float)) + (rows * sizeof(float)); // Values + vector + result
        double bandwidth_gbs = (bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbs);
        printf("Non-zeros: %d (%.2f%% sparsity)\n", h_mat.nnz, (float)h_mat.nnz/(rows*cols)*100.0f);
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    free(h_mat.values); free(h_mat.col_indices); free(h_mat.row_ptr);
    free(h_vec); free(h_result); free(h_expected);
    cudaFree(d_mat.values); cudaFree(d_mat.col_indices); cudaFree(d_mat.row_ptr);
    cudaFree(d_vec); cudaFree(d_result);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}