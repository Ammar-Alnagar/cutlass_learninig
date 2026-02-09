// Asynchronous Programming - Hands-on Example
// This example demonstrates futures, promises, and async execution

#include <iostream>
#include <future>
#include <chrono>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>

// Function that simulates an async operation using packaged_task
int simulateAsyncOperation(int id, int delay_ms) {
    std::cout << "Async operation " << id << " starting..." << std::endl;
    
    // Simulate work with sleep
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    
    // Return result
    int result = id * id; // Just a simple computation
    std::cout << "Async operation " << id << " completed with result: " << result << std::endl;
    
    return result;
}

// Example using packaged_task and future
void packagedTaskExample() {
    std::cout << "\n=== Packaged Task Example ===" << std::endl;
    
    // Create packaged tasks that wrap callable objects
    std::packaged_task<int()> task1([]() { 
        return simulateAsyncOperation(1, 800); 
    });
    
    std::packaged_task<int()> task2([]() { 
        return simulateAsyncOperation(2, 600); 
    });
    
    // Get futures from the packaged tasks
    std::future<int> future1 = task1.get_future();
    std::future<int> future2 = task2.get_future();
    
    // Launch tasks asynchronously
    std::thread t1(std::move(task1));
    std::thread t2(std::move(task2));
    
    // Do other work while tasks are running
    std::cout << "Main thread doing other work..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    // Wait for and get results
    int result1 = future1.get();
    int result2 = future2.get();
    
    std::cout << "Results: " << result1 << ", " << result2 << std::endl;
    
    // Join threads
    t1.join();
    t2.join();
}

// Example using async function
void asyncFunctionExample() {
    std::cout << "\n=== Async Function Example ===" << std::endl;
    
    // Launch async operations
    auto future1 = std::async(std::launch::async, simulateAsyncOperation, 3, 1000);
    auto future2 = std::async(std::launch::async, simulateAsyncOperation, 4, 700);
    auto future3 = std::async(std::launch::async, []() {
        std::cout << "Lambda async operation starting..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "Lambda async operation completed." << std::endl;
        return 42;
    });
    
    // Do other work
    std::cout << "Main thread working while async operations run..." << std::endl;
    
    // Wait for and collect results
    int result1 = future1.get();
    int result2 = future2.get();
    int result3 = future3.get();
    
    std::cout << "Async results: " << result1 << ", " << result2 << ", " << result3 << std::endl;
}

// Example demonstrating promise usage
void promiseExample() {
    std::cout << "\n=== Promise Example ===" << std::endl;
    
    std::promise<int> promise;
    std::future<int> future = promise.get_future();
    
    // Launch thread that will set the promise value
    std::thread setter_thread([&promise]() {
        std::cout << "Setter thread doing work..." << std::endl;
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        
        // Set the promise value
        try {
            int computed_value = 100 + 200;
            std::cout << "Setter thread setting promise value: " << computed_value << std::endl;
            promise.set_value(computed_value);
        } catch (...) {
            // Handle exceptions
            promise.set_exception(std::current_exception());
        }
    });
    
    // Do other work while waiting
    std::cout << "Main thread waiting for promise..." << std::endl;
    
    // Wait for the future to be ready and get the value
    int result = future.get();
    std::cout << "Main thread received value: " << result << std::endl;
    
    setter_thread.join();
}

// Example demonstrating exception handling in futures
void exceptionHandlingExample() {
    std::cout << "\n=== Exception Handling Example ===" << std::endl;
    
    std::promise<int> promise;
    std::future<int> future = promise.get_future();
    
    std::thread exception_thread([&promise]() {
        try {
            std::cout << "Thread doing work that might fail..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            
            // Simulate an error condition
            throw std::runtime_error("Something went wrong!");
        } catch (...) {
            // Capture the exception in the promise
            promise.set_exception(std::current_exception());
        }
    });
    
    try {
        // This will re-throw the exception from the thread
        int result = future.get();
        std::cout << "Unexpected success: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
    
    exception_thread.join();
}

// Example showing async pipeline
void pipelineExample() {
    std::cout << "\n=== Pipeline Example ===" << std::endl;
    
    // Stage 1: Generate data
    auto stage1 = std::async(std::launch::async, []() {
        std::vector<int> data = {1, 2, 3, 4, 5};
        std::cout << "Stage 1: Generated data" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return data;
    });
    
    // Stage 2: Process data (runs concurrently with stage 1)
    auto stage2 = std::async(std::launch::async, [data_future = stage1.share()]() {
        auto data = data_future.get(); // Wait for stage 1
        std::cout << "Stage 2: Processing data" << std::endl;
        std::transform(data.begin(), data.end(), data.begin(), [](int x) { 
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate processing
            return x * 2; 
        });
        std::cout << "Stage 2: Data processed" << std::endl;
        return data;
    });
    
    // Stage 3: Finalize data (runs concurrently with stage 2, after stage 1)
    auto stage3 = std::async(std::launch::async, [processed_data_future = stage2.share()]() {
        auto processed_data = processed_data_future.get(); // Wait for stage 2
        std::cout << "Stage 3: Finalizing data" << std::endl;
        int sum = 0;
        for (int val : processed_data) {
            sum += val;
        }
        std::cout << "Stage 3: Sum calculated" << std::endl;
        return sum;
    });
    
    // Get final result
    int final_result = stage3.get();
    std::cout << "Pipeline final result: " << final_result << std::endl;
}

int main() {
    std::cout << "Asynchronous Programming - Hands-on Example" << std::endl;
    
    packagedTaskExample();
    asyncFunctionExample();
    promiseExample();
    exceptionHandlingExample();
    pipelineExample();
    
    std::cout << "\nAll async examples completed!" << std::endl;
    
    return 0;
}