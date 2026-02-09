// Thread Basics - Hands-on Example
// This example demonstrates thread creation, lifecycle, and basic operations

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <functional>

// Function to demonstrate basic thread creation and joining
void basicThreadExample() {
    std::cout << "\n=== Basic Thread Creation ===" << std::endl;
    
    // Create a thread that executes a function
    std::thread t([]() {
        std::cout << "Hello from thread " << std::this_thread::get_id() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "Thread " << std::this_thread::get_id() << " finishing..." << std::endl;
    });
    
    std::cout << "Main thread ID: " << std::this_thread::get_id() << std::endl;
    std::cout << "Created thread ID: " << t.get_id() << std::endl;
    
    // Wait for the thread to complete (join)
    t.join();
    std::cout << "Thread joined successfully!" << std::endl;
}

// Function to demonstrate multiple threads
void multipleThreadsExample() {
    std::cout << "\n=== Multiple Threads ===" << std::endl;
    
    const int num_threads = 5;
    std::vector<std::thread> threads;
    
    // Create multiple threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([i]() {
            std::cout << "Thread " << i << " (ID: " << std::this_thread::get_id() 
                      << ") starting..." << std::endl;
            
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * (i + 1)));
            
            std::cout << "Thread " << i << " (ID: " << std::this_thread::get_id() 
                      << ") completed." << std::endl;
        });
    }
    
    // Join all threads
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "All threads joined!" << std::endl;
}

// Function to demonstrate thread arguments
void threadArgumentsExample() {
    std::cout << "\n=== Thread Arguments ===" << std::endl;
    
    std::string message = "Hello from thread!";
    int value = 42;
    
    std::thread t([](std::string msg, int val) {
        std::cout << "Received message: " << msg << std::endl;
        std::cout << "Received value: " << val << std::endl;
        std::cout << "Processing in thread: " << std::this_thread::get_id() << std::endl;
    }, message, value);  // Pass arguments to thread
    
    t.join();
    std::cout << "Thread with arguments completed!" << std::endl;
}

// Function to demonstrate detachable threads
void detachedThreadExample() {
    std::cout << "\n=== Detached Threads ===" << std::endl;
    
    std::thread t([]() {
        std::cout << "Detached thread running with ID: " << std::this_thread::get_id() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "Detached thread " << std::this_thread::get_id() << " finishing..." << std::endl;
    });
    
    // Detach the thread - it will run independently
    t.detach();
    std::cout << "Thread detached. Main thread continuing..." << std::endl;
    
    // Give detached thread time to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
}

int main() {
    std::cout << "Thread Basics - Hands-on Example" << std::endl;
    
    basicThreadExample();
    multipleThreadsExample();
    threadArgumentsExample();
    detachedThreadExample();
    
    std::cout << "\nAll thread examples completed!" << std::endl;
    
    return 0;
}