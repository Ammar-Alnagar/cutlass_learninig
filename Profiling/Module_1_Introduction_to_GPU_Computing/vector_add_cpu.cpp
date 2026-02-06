// Simple vector addition example

#include <iostream>
#include <vector>
#include <chrono>

// CPU version
void cpu_vector_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    for(size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const size_t N = 1000000;
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N);

    // Time the CPU version
    auto start = std::chrono::high_resolution_clock::now();
    cpu_vector_add(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU Vector Addition took: " << duration.count() << " microseconds" << std::endl;

    // On a GPU, all additions could happen in parallel
    // This is the power of parallel computing!

    return 0;
}