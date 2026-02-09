// Testing Concurrency - Hands-on Example
// This example demonstrates various techniques for testing concurrent code

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>
#include <random>
#include <functional>
#include <cassert>
#include <condition_variable>
#include <future>
#include <algorithm>

// Helper class for measuring performance
class PerformanceTimer {
public:
    PerformanceTimer(const std::string& name) : test_name(name) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    ~PerformanceTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << test_name << " took " << duration.count() << " microseconds" << std::endl;
    }

private:
    std::string test_name;
    std::chrono::high_resolution_clock::time_point start_time;
};

// A simple thread-safe counter for testing
class ThreadSafeCounter {
private:
    mutable std::mutex mtx;
    int count;
    
public:
    ThreadSafeCounter() : count(0) {}
    
    void increment() {
        std::lock_guard<std::mutex> lock(mtx);
        ++count;
    }
    
    int get() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mtx);
        count = 0;
    }
};

// A buggy counter to demonstrate race conditions
class BuggyCounter {
private:
    int count;
    
public:
    BuggyCounter() : count(0) {}
    
    void increment() {
        // Deliberately not thread-safe to demonstrate race conditions
        int temp = count;
        std::this_thread::sleep_for(std::chrono::nanoseconds(1)); // Increase chance of race condition
        count = temp + 1;
    }
    
    int get() const {
        return count;
    }
    
    void reset() {
        count = 0;
    }
};

// Test 1: Basic stress test for thread safety
void testThreadSafety() {
    std::cout << "\n=== Test 1: Thread Safety Stress Test ===" << std::endl;
    
    const int num_threads = 4;
    const int increments_per_thread = 100000;
    const int expected_total = num_threads * increments_per_thread;
    
    ThreadSafeCounter safe_counter;
    
    {
        PerformanceTimer timer("Thread Safe Counter");
        
        std::vector<std::thread> threads;
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&safe_counter, increments_per_thread]() {
                for (int j = 0; j < increments_per_thread; ++j) {
                    safe_counter.increment();
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        int result = safe_counter.get();
        std::cout << "Expected: " << expected_total << ", Got: " << result << std::endl;
        assert(result == expected_total);
        std::cout << "Thread safety test PASSED!" << std::endl;
    }
}

// Test 2: Demonstrate race condition with buggy counter
void testRaceCondition() {
    std::cout << "\n=== Test 2: Race Condition Demonstration ===" << std::endl;
    
    const int num_threads = 4;
    const int increments_per_thread = 10000;
    const int expected_total = num_threads * increments_per_thread;
    
    BuggyCounter buggy_counter;
    
    {
        PerformanceTimer timer("Buggy Counter (with race condition)");
        
        std::vector<std::thread> threads;
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&buggy_counter, increments_per_thread]() {
                for (int j = 0; j < increments_per_thread; ++j) {
                    buggy_counter.increment();
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        int result = buggy_counter.get();
        std::cout << "Expected: " << expected_total << ", Got: " << result << std::endl;
        
        if (result != expected_total) {
            std::cout << "Race condition detected! Result differs from expected." << std::endl;
        } else {
            std::cout << "No race condition detected this time (but it's still there!)." << std::endl;
        }
    }
}

// Test 3: Test with different thread counts to find scaling issues
void testScalability() {
    std::cout << "\n=== Test 3: Scalability Test ===" << std::endl;
    
    const int increments_per_thread = 50000;
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    for (int num_threads : thread_counts) {
        ThreadSafeCounter counter;
        const int expected_total = num_threads * increments_per_thread;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&counter, increments_per_thread]() {
                for (int j = 0; j < increments_per_thread; ++j) {
                    counter.increment();
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        int result = counter.get();
        std::cout << "Threads: " << num_threads 
                  << ", Expected: " << expected_total 
                  << ", Got: " << result
                  << ", Duration: " << duration.count() << "ms" << std::endl;
        
        assert(result == expected_total);
    }
}

// Test 4: Test with randomized delays to expose timing issues
void testWithRandomDelays() {
    std::cout << "\n=== Test 4: Random Delay Test ===" << std::endl;
    
    const int num_threads = 4;
    const int operations_per_thread = 10000;
    const int expected_total = num_threads * operations_per_thread;
    
    ThreadSafeCounter counter;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> delay_dist(0, 1000); // Random delay up to 1000 nanoseconds
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&counter, operations_per_thread, &gen, &delay_dist]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                counter.increment();
                // Add random delay to increase chance of exposing race conditions
                auto delay = std::chrono::nanoseconds(delay_dist(gen));
                std::this_thread::sleep_for(delay);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    int result = counter.get();
    std::cout << "Expected: " << expected_total << ", Got: " << result << std::endl;
    assert(result == expected_total);
    std::cout << "Random delay test PASSED!" << std::endl;
}

// Test 5: Test for deadlocks
class DeadlockTest {
private:
    std::mutex mtx1, mtx2;
    int resource1 = 0;
    int resource2 = 0;

public:
    // Function that could cause deadlock if called from different threads in wrong order
    void transfer1to2(int amount) {
        std::lock_guard<std::mutex> lock1(mtx1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Increase chance of deadlock
        std::lock_guard<std::mutex> lock2(mtx2);
        resource1 -= amount;
        resource2 += amount;
    }
    
    // Function that could cause deadlock if called from different threads in wrong order
    void transfer2to1(int amount) {
        std::lock_guard<std::mutex> lock2(mtx2);  // Different order than transfer1to2!
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Increase chance of deadlock
        std::lock_guard<std::mutex> lock1(mtx1);  // Different order than transfer1to2!
        resource2 -= amount;
        resource1 += amount;
    }
    
    std::pair<int, int> getResources() {
        std::lock_guard<std::mutex> lock1(mtx1);
        std::lock_guard<std::mutex> lock2(mtx2);
        return {resource1, resource2};
    }
};

void testDeadlock() {
    std::cout << "\n=== Test 5: Deadlock Test ===" << std::endl;
    
    DeadlockTest test_obj;
    
    // This test is designed to potentially cause a deadlock
    // In a real testing environment, we'd use timeouts to prevent hanging
    std::cout << "Attempting to trigger deadlock (this might hang if deadlock occurs)..." << std::endl;
    
    std::thread t1([&test_obj]() { test_obj.transfer1to2(10); });
    std::thread t2([&test_obj]() { test_obj.transfer2to1(5); });
    
    // Use futures with timeout to avoid hanging indefinitely
    auto future1 = std::async(std::launch::async, [&t1]() { t1.join(); });
    auto future2 = std::async(std::launch::async, [&t2]() { t2.join(); });
    
    // Wait for a reasonable time, then assume deadlock if not complete
    if (future1.wait_for(std::chrono::seconds(5)) == std::future_status::timeout ||
        future2.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
        std::cout << "Potential deadlock detected (operations timed out)!" << std::endl;
        // In a real system, we'd have better deadlock detection/recovery
    } else {
        std::cout << "Operations completed without deadlock." << std::endl;
    }
    
    auto resources = test_obj.getResources();
    std::cout << "Resources: " << resources.first << ", " << resources.second << std::endl;
}

// Test 6: Property-based testing concept demonstration
void testPropertyBased() {
    std::cout << "\n=== Test 6: Property-Based Testing Concept ===" << std::endl;
    
    // Property: The sum of increments equals the final counter value
    const int num_threads = 3;
    const int increments_per_thread = 50000;
    const int expected_total = num_threads * increments_per_thread;
    
    ThreadSafeCounter counter;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&counter, increments_per_thread]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                counter.increment();
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    int final_value = counter.get();
    
    // Property check: final value equals expected total
    bool property_holds = (final_value == expected_total);
    std::cout << "Property 'final_value == expected_total' holds: " << (property_holds ? "YES" : "NO") << std::endl;
    std::cout << "Expected: " << expected_total << ", Actual: " << final_value << std::endl;
    
    // Additional property: value is non-negative
    bool non_negative = (final_value >= 0);
    std::cout << "Property 'final_value >= 0' holds: " << (non_negative ? "YES" : "NO") << std::endl;
    
    assert(property_holds && non_negative);
    std::cout << "Property-based tests PASSED!" << std::endl;
}

// Test 7: Repeat test multiple times to catch intermittent failures
void testRepeatForFlakiness() {
    std::cout << "\n=== Test 7: Repeat Test for Intermittent Issues ===" << std::endl;
    
    const int num_repeats = 10;
    const int num_threads = 4;
    const int increments_per_thread = 25000;
    const int expected_total = num_threads * increments_per_thread;
    
    bool all_passed = true;
    
    for (int repeat = 0; repeat < num_repeats; ++repeat) {
        ThreadSafeCounter counter;
        
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&counter, increments_per_thread]() {
                for (int j = 0; j < increments_per_thread; ++j) {
                    counter.increment();
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        int result = counter.get();
        bool passed = (result == expected_total);
        
        std::cout << "Repeat " << (repeat + 1) << ": " << (passed ? "PASS" : "FAIL") 
                  << " (Expected: " << expected_total << ", Got: " << result << ")" << std::endl;
        
        if (!passed) {
            all_passed = false;
        }
    }
    
    if (all_passed) {
        std::cout << "All repeats PASSED!" << std::endl;
    } else {
        std::cout << "Some repeats FAILED - potential flakiness detected!" << std::endl;
    }
}

int main() {
    std::cout << "Testing Concurrency - Hands-on Example" << std::endl;
    
    testThreadSafety();
    testRaceCondition();
    testScalability();
    testWithRandomDelays();
    testDeadlock();  // Note: This test might cause a deadlock in some runs
    testPropertyBased();
    testRepeatForFlakiness();
    
    std::cout << "\nAll concurrency testing examples completed!" << std::endl;
    
    return 0;
}