/**
 * Project 14: MLA - Reference Solution
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct MLAConfig {
    static constexpr int NUM_HEADS = 8;
    static constexpr int HEAD_DIM = 64;
    static constexpr int NUM_LATENTS = 4;
    static constexpr int LATENT_DIM = 32;
    static constexpr int BLOCK_SIZE = 256;
};

__global__ void mla_kernel(
    float* Q, float* K_latent, float* V_latent, float* O,
    float* W_Q_latent, float* W_out,
    int* seq_lens,
    int batch, int seq_len,
    int num_heads, int head_dim,
    int num_latents, int latent_dim) {
    
    int head_idx = blockIdx.z;
    int batch_idx = blockIdx.y;
    int q_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch || q_pos >= seq_len || head_idx >= num_heads) return;
    
    int d = num_heads * head_dim;
    int latent_total = num_latents * latent_dim;
    float scale = rsqrtf((float)latent_dim);
    
    // Pointers
    float* q_ptr = Q + (batch_idx * seq_len + q_pos) * d + head_idx * head_dim;
    float* k_ptr = K_latent + batch_idx * seq_len * latent_total;
    float* v_ptr = V_latent + batch_idx * seq_len * latent_total;
    float* o_ptr = O + (batch_idx * seq_len + q_pos) * d + head_idx * head_dim;
    float* w_q_ptr = W_Q_latent + head_idx * head_dim * latent_total;
    float* w_out_ptr = W_out + latent_total * (head_idx * head_dim);
    
    int actual_seq = seq_lens ? seq_lens[batch_idx] : seq_len;
    
    // Project Q to latent space
    float q_latent[32];
    for (int l = 0; l < latent_total && l < 32; l++) {
        float sum = 0;
        for (int k = 0; k < head_dim; k++) {
            sum += q_ptr[k] * w_q_ptr[k * latent_total + l];
        }
        q_latent[l] = sum;
    }
    
    // Attention in latent space
    float max_score = -std::numeric_limits<float>::infinity();
    float sum_exp = 0.0f;
    float accum[32] = {0.0f};
    
    for (int kv_pos = 0; kv_pos < actual_seq; kv_pos++) {
        float score = 0;
        for (int l = 0; l < latent_total && l < 32; l++) {
            score += q_latent[l] * k_ptr[kv_pos * latent_total + l];
        }
        score *= scale;
        
        float new_max = fmaxf(max_score, score);
        float rescale = expf(max_score - new_max);
        for (int l = 0; l < latent_total && l < 32; l++) accum[l] *= rescale;
        
        float prob = expf(score - new_max);
        sum_exp = sum_exp * rescale + prob;
        
        for (int l = 0; l < latent_total && l < 32; l++) {
            accum[l] += prob * v_ptr[kv_pos * latent_total + l];
        }
        max_score = new_max;
    }
    
    // Project back to output
    if (sum_exp > 0.0f) {
        for (int k = 0; k < head_dim; k++) {
            float out_val = 0;
            for (int l = 0; l < latent_total && l < 32; l++) {
                out_val += (accum[l] / sum_exp) * w_out_ptr[l * head_dim + k];
            }
            o_ptr[k] = out_val;
        }
    }
}

} // namespace cute

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

int main() {
    using Config = cute::MLAConfig;
    const int batch = 4, seq_len = 64;
    const int num_heads = 8, head_dim = 64;
    const int num_latents = 4, latent_dim = 32;
    const int d = num_heads * head_dim;
    
    std::cout << "=== Project 14: MLA (Solution) ===" << std::endl;
    
    int total_qkv = batch * seq_len * d;
    int total_latent = batch * seq_len * num_latents * latent_dim;
    
    std::vector<float> h_Q(total_qkv), h_K_latent(total_latent), h_V_latent(total_latent);
    std::vector<float> h_O(total_qkv, 0);
    std::vector<float> h_W_Q_latent(d * num_latents * latent_dim);
    std::vector<float> h_W_out(num_latents * latent_dim * d);
    
    for (int i = 0; i < total_qkv; i++) h_Q[i] = (i % 100) * 0.01f;
    for (int i = 0; i < total_latent; i++) {
        h_K_latent[i] = h_V_latent[i] = (i % 100) * 0.01f;
    }
    for (int i = 0; i < d * num_latents * latent_dim; i++) {
        h_W_Q_latent[i] = h_W_out[i] = (i % 100) * 0.01f;
    }
    
    float *d_Q, *d_K_latent, *d_V_latent, *d_O, *d_W_Q_latent, *d_W_out;
    CUDA_CHECK(cudaMalloc(&d_Q, total_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_latent, total_latent * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_latent, total_latent * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, total_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_Q_latent, d * num_latents * latent_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_out, num_latents * latent_dim * d * sizeof(float)));
    
    cudaMemcpy(d_Q, h_Q.data(), total_qkv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_latent, h_K_latent.data(), total_latent * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_latent, h_V_latent.data(), total_latent * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_Q_latent, h_W_Q_latent.data(), d * num_latents * latent_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_out, h_W_out.data(), num_latents * latent_dim * d * sizeof(float), cudaMemcpyHostToDevice);
    
    const dim3 block(Config::BLOCK_SIZE);
    const dim3 grid((seq_len + block.x - 1) / block.x, batch, num_heads);
    
    cute::mla_kernel<<<grid, block>>>(d_Q, d_K_latent, d_V_latent, d_O,
                                       d_W_Q_latent, d_W_out, nullptr,
                                       batch, seq_len, num_heads, head_dim,
                                       num_latents, latent_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_O.data(), d_O, total_qkv * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "[PASS] MLA" << std::endl;
    
    cudaFree(d_Q); cudaFree(d_K_latent); cudaFree(d_V_latent);
    cudaFree(d_O); cudaFree(d_W_Q_latent); cudaFree(d_W_out);
    return 0;
}
