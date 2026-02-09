// Introduction to Concurrency - Hands-on Example
// This example demonstrates the difference between sequential and concurrent execution

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

// Function that simulates work by sleeping for a duration
void simulateWork(int id, int duration_ms) {
    std::cout << "Task " << id << " starting..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    std::cout << "Task " << id << " completed." << std::endl;
}

int main() {
    std::cout << "=== Sequential Execution ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Sequential execution - tasks run one after another
    simulateWork(1, 1000);  // 1 second task
    simulateWork(2, 1000);  // 1 second task
    simulateWork(3, 1000);  // 1 second task
    
    auto end = std::chrono::high_resolution_clock::now();
    auto sequential_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Sequential execution took: " << sequential_duration.count() << " ms" << std::endl;

    std::cout << "\n=== Concurrent Execution ===" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    // Concurrent execution - tasks run in parallel
    std::vector<std::thread> threads;
    threads.emplace_back(simulateWork, 1, 1000);  // 1 second task
    threads.emplace_back(simulateWork, 2, 1000);  // 1 second task
    threads.emplace_back(simulateWork, 3, 1000);  // 1 second task
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto concurrent_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Concurrent execution took: " << concurrent_duration.count() << " ms" << std::endl;
    
    std::cout << "\nSpeedup: " << static_cast<double>(sequential_duration.count()) / concurrent_duration.count() << "x" << std::endl;
    
    return 0;
}