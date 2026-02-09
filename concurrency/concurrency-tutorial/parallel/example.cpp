// Parallel Algorithms - Hands-on Example
// This example demonstrates parallel execution of standard algorithms

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <chrono>
#include <random>
#include <functional>

// Function to generate a large dataset for testing
std::vector<int> generateDataset(size_t size) {
    std::vector<int> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    
    return data;
}

// Function to perform a computationally intensive operation
int heavyComputation(int value) {
    // Simulate a computation that takes some time
    volatile int result = value;
    for (int i = 0; i < 100; ++i) {
        result = (result * result + value) % 10007;
    }
    return result;
}

void sequentialVsParallelSort() {
    std::cout << "\n=== Sequential vs Parallel Sort ===" << std::endl;
    
    const size_t dataSize = 10000000; // 10 million elements
    
    // Test sequential sort
    auto dataSeq = generateDataset(dataSize);
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::seq, dataSeq.begin(), dataSeq.end());
    auto end = std::chrono::high_resolution_clock::now();
    auto seqDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Test parallel sort
    auto dataPar = generateDataset(dataSize);
    start = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::par, dataPar.begin(), dataPar.end());
    end = std::chrono::high_resolution_clock::now();
    auto parDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Sequential sort: " << seqDuration.count() << " ms" << std::endl;
    std::cout << "Parallel sort: " << parDuration.count() << " ms" << std::endl;
    std::cout << "Speedup: " << (double)seqDuration.count() / parDuration.count() << "x" << std::endl;
}

void parallelTransformExample() {
    std::cout << "\n=== Parallel Transform Example ===" << std::endl;
    
    const size_t dataSize = 5000000; // 5 million elements
    auto data = generateDataset(dataSize);
    std::vector<int> result(dataSize);
    
    // Sequential transform
    auto start = std::chrono::high_resolution_clock::now();
    std::transform(std::execution::seq, data.begin(), data.end(), result.begin(), heavyComputation);
    auto end = std::chrono::high_resolution_clock::now();
    auto seqDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Parallel transform
    start = std::chrono::high_resolution_clock::now();
    std::transform(std::execution::par, data.begin(), data.end(), result.begin(), heavyComputation);
    end = std::chrono::high_resolution_clock::now();
    auto parDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Sequential transform: " << seqDuration.count() << " ms" << std::endl;
    std::cout << "Parallel transform: " << parDuration.count() << " ms" << std::endl;
    std::cout << "Speedup: " << (double)seqDuration.count() / parDuration.count() << "x" << std::endl;
}

void parallelReduceExample() {
    std::cout << "\n=== Parallel Reduce Example ===" << std::endl;
    
    const size_t dataSize = 10000000; // 10 million elements
    auto data = generateDataset(dataSize);
    
    // Sequential reduce
    auto start = std::chrono::high_resolution_clock::now();
    long long seqSum = std::reduce(std::execution::seq, data.begin(), data.end(), 0LL);
    auto end = std::chrono::high_resolution_clock::now();
    auto seqDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Parallel reduce
    start = std::chrono::high_resolution_clock::now();
    long long parSum = std::reduce(std::execution::par, data.begin(), data.end(), 0LL);
    end = std::chrono::high_resolution_clock::now();
    auto parDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Sequential reduce: " << seqDuration.count() << " ms, sum: " << seqSum << std::endl;
    std::cout << "Parallel reduce: " << parDuration.count() << " ms, sum: " << parSum << std::endl;
    std::cout << "Speedup: " << (double)seqDuration.count() / parDuration.count() << "x" << std::endl;
}

void parallelFindExample() {
    std::cout << "\n=== Parallel Find Example ===" << std::endl;
    
    const size_t dataSize = 5000000; // 5 million elements
    auto data = generateDataset(dataSize);
    
    // Modify one element to search for
    const int target = 9999;
    if (data.size() > dataSize/2) {
        data[dataSize/2] = target;
    }
    
    // Sequential find
    auto start = std::chrono::high_resolution_clock::now();
    auto seqIt = std::find(std::execution::seq, data.begin(), data.end(), target);
    auto end = std::chrono::high_resolution_clock::now();
    auto seqDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Parallel find
    start = std::chrono::high_resolution_clock::now();
    auto parIt = std::find(std::execution::par, data.begin(), data.end(), target);
    end = std::chrono::high_resolution_clock::now();
    auto parDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Sequential find: " << seqDuration.count() << " ms";
    if (seqIt != data.end()) {
        std::cout << ", found at index: " << std::distance(data.begin(), seqIt);
    } else {
        std::cout << ", not found";
    }
    std::cout << std::endl;
    
    std::cout << "Parallel find: " << parDuration.count() << " ms";
    if (parIt != data.end()) {
        std::cout << ", found at index: " << std::distance(data.begin(), parIt);
    } else {
        std::cout << ", not found";
    }
    std::cout << std::endl;
    
    std::cout << "Speedup: " << (double)seqDuration.count() / parDuration.count() << "x" << std::endl;
}

void parallelTransformReduceExample() {
    std::cout << "\n=== Parallel Transform-Reduce Example ===" << std::endl;
    
    const size_t dataSize = 5000000; // 5 million elements
    auto data = generateDataset(dataSize);
    
    // Sequential transform-reduce
    auto start = std::chrono::high_resolution_clock::now();
    long long seqResult = std::transform_reduce(
        std::execution::seq, 
        data.begin(), data.end(), 
        0LL, 
        std::plus<long long>(), 
        heavyComputation
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto seqDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Parallel transform-reduce
    start = std::chrono::high_resolution_clock::now();
    long long parResult = std::transform_reduce(
        std::execution::par, 
        data.begin(), data.end(), 
        0LL, 
        std::plus<long long>(), 
        heavyComputation
    );
    end = std::chrono::high_resolution_clock::now();
    auto parDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Sequential transform-reduce: " << seqDuration.count() << " ms, result: " << seqResult << std::endl;
    std::cout << "Parallel transform-reduce: " << parDuration.count() << " ms, result: " << parResult << std::endl;
    std::cout << "Speedup: " << (double)seqDuration.count() / parDuration.count() << "x" << std::endl;
}

int main() {
    std::cout << "Parallel Algorithms - Hands-on Example" << std::endl;
    
    sequentialVsParallelSort();
    parallelTransformExample();
    parallelReduceExample();
    parallelFindExample();
    parallelTransformReduceExample();
    
    std::cout << "\nAll parallel algorithm examples completed!" << std::endl;
    
    return 0;
}