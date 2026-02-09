// Synchronization Primitives - Hands-on Example
// This example demonstrates mutexes, semaphores, and condition variables

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <semaphore>  // C++20, fallback to counting_semaphore if unavailable
#include <queue>
#include <chrono>
#include <vector>
#include <functional>

// Fallback implementation for semaphore if C++20 semaphore is not available
#include <atomic>
#include <stdexcept>

class counting_semaphore {
private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<int> counter_;

public:
    explicit counting_semaphore(int initial_count) : counter_(initial_count) {}

    void acquire() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return counter_.load() > 0; });
        counter_.fetch_sub(1);
    }

    void release() {
        counter_.fetch_add(1);
        cv_.notify_one();
    }
};

// Shared data structure protected by mutex
struct SharedBuffer {
    std::mutex buffer_mutex;
    std::queue<int> buffer;
    std::condition_variable buffer_cv;
    bool finished = false;
    static const size_t MAX_BUFFER_SIZE = 5;
};

// Producer function using mutex and condition variable
void producer(SharedBuffer& shared, int producer_id, int items_to_produce) {
    for (int i = 0; i < items_to_produce; ++i) {
        {
            std::unique_lock<std::mutex> lock(shared.buffer_mutex);
            // Wait if buffer is full
            shared.buffer_cv.wait(lock, [&shared] { 
                return shared.buffer.size() < SharedBuffer::MAX_BUFFER_SIZE || shared.finished; 
            });
            
            if (shared.finished) break; // Exit if consumer finished
            
            // Produce item
            int item = producer_id * 100 + i;
            shared.buffer.push(item);
            std::cout << "Producer " << producer_id << " produced item: " << item 
                      << " (Buffer size: " << shared.buffer.size() << ")" << std::endl;
        } // Release lock
        
        shared.buffer_cv.notify_all(); // Notify consumers
        
        // Simulate production time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Consumer function using mutex and condition variable
void consumer(SharedBuffer& shared, int consumer_id) {
    while (true) {
        int item;
        {
            std::unique_lock<std::mutex> lock(shared.buffer_mutex);
            // Wait if buffer is empty and producers haven't finished
            shared.buffer_cv.wait(lock, [&shared] { 
                return !shared.buffer.empty() || shared.finished; 
            });
            
            if (shared.buffer.empty() && shared.finished) {
                break; // Exit if no more items and producers finished
            }
            
            // Consume item
            item = shared.buffer.front();
            shared.buffer.pop();
            std::cout << "Consumer " << consumer_id << " consumed item: " << item 
                      << " (Buffer size: " << shared.buffer.size() << ")" << std::endl;
        } // Release lock
        
        shared.buffer_cv.notify_all(); // Notify producers
        
        // Simulate consumption time
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    
    std::cout << "Consumer " << consumer_id << " finished." << std::endl;
}

// Example demonstrating mutex usage
void mutexExample() {
    std::cout << "\n=== Mutex Example ===" << std::endl;
    
    int shared_counter = 0;
    std::mutex counter_mutex;
    
    std::vector<std::thread> threads;
    
    // Create multiple threads that increment the shared counter
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&shared_counter, &counter_mutex, i]() {
            for (int j = 0; j < 100; ++j) {
                {
                    std::lock_guard<std::mutex> lock(counter_mutex);
                    // Critical section - only one thread can access at a time
                    int temp = shared_counter;
                    std::this_thread::sleep_for(std::chrono::microseconds(1)); // Simulate work
                    shared_counter = temp + 1;
                } // Mutex automatically unlocked here
            }
            std::cout << "Thread " << i << " completed increments." << std::endl;
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "Final counter value: " << shared_counter << " (Expected: 500)" << std::endl;
}

// Example demonstrating semaphore usage
void semaphoreExample() {
    std::cout << "\n=== Semaphore Example ===" << std::endl;
    
    counting_semaphore resource_semaphore(3); // Only 3 resources available
    std::mutex output_mutex;
    
    std::vector<std::thread> threads;
    
    // Create more threads than available resources
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&resource_semaphore, &output_mutex, i]() {
            {
                std::lock_guard<std::mutex> output_lock(output_mutex);
                std::cout << "Thread " << i << " requesting resource..." << std::endl;
            }
            
            resource_semaphore.acquire(); // Acquire resource
            
            {
                std::lock_guard<std::mutex> output_lock(output_mutex);
                std::cout << "Thread " << i << " acquired resource." << std::endl;
            }
            
            // Simulate using the resource
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            resource_semaphore.release(); // Release resource
            
            {
                std::lock_guard<std::mutex> output_lock(output_mutex);
                std::cout << "Thread " << i << " released resource." << std::endl;
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
}

// Example demonstrating condition variables
void conditionVariableExample() {
    std::cout << "\n=== Condition Variable Example (Producer-Consumer) ===" << std::endl;
    
    SharedBuffer shared_buffer;
    
    // Create producer and consumer threads
    std::thread prod1(producer, std::ref(shared_buffer), 1, 5);
    std::thread prod2(producer, std::ref(shared_buffer), 2, 5);
    std::thread cons1(consumer, std::ref(shared_buffer), 1);
    std::thread cons2(consumer, std::ref(shared_buffer), 2);
    
    // Wait for producers to finish
    prod1.join();
    prod2.join();
    
    // Signal that production is finished
    {
        std::lock_guard<std::mutex> lock(shared_buffer.buffer_mutex);
        shared_buffer.finished = true;
    }
    shared_buffer.buffer_cv.notify_all(); // Wake up all waiting consumers
    
    // Wait for consumers to finish
    cons1.join();
    cons2.join();
    
    std::cout << "Producer-Consumer example completed." << std::endl;
}

int main() {
    std::cout << "Synchronization Primitives - Hands-on Example" << std::endl;
    
    mutexExample();
    semaphoreExample();
    conditionVariableExample();
    
    std::cout << "\nAll synchronization examples completed!" << std::endl;
    
    return 0;
}