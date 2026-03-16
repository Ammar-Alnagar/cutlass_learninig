// SOLUTION: ex02_atomic_fixes_tearing
// Demonstrates atomic operations that fix tearing

#include <iostream>
#include <thread>
#include <vector>
#include <cstdint>

// ==================== Before: Non-Atomic (Buggy) ====================

uint64_t counter_non_atomic = 0;

void increment_non_atomic(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        counter_non_atomic++;  // Non-atomic — can tear
    }
}

// ==================== After: Atomic (Fixed) ====================

std::atomic<uint64_t> counter_atomic{0};

void increment_atomic(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        counter_atomic++;  // Atomic — no tearing
    }
}

// ==================== Comparison ====================

void compare_implementations() {
    const int num_threads = 4;
    const int iterations = 100000;
    const uint64_t expected = static_cast<uint64_t>(num_threads) * iterations;
    
    std::cout << "=== Non-Atomic Counter (Buggy) ===\n";
    counter_non_atomic = 0;
    
    std::vector<std::thread> threads1;
    for (int i = 0; i < num_threads; ++i) {
        threads1.emplace_back(increment_non_atomic, iterations);
    }
    for (auto& t : threads1) t.join();
    
    std::cout << "Result: " << counter_non_atomic << " (expected: " << expected << ")\n";
    std::cout << "Error: " << static_cast<int64_t>(expected - counter_non_atomic) << "\n\n";
    
    std::cout << "=== Atomic Counter (Fixed) ===\n";
    counter_atomic = 0;
    
    std::vector<std::thread> threads2;
    for (int i = 0; i < num_threads; ++i) {
        threads2.emplace_back(increment_atomic, iterations);
    }
    for (auto& t : threads2) t.join();
    
    std::cout << "Result: " << counter_atomic << " (expected: " << expected << ")\n";
    std::cout << "Error: " << static_cast<int64_t>(expected - counter_atomic) << "\n";
}

// ==================== Atomic Operations Demo ====================

void demonstrate_atomic_operations() {
    std::cout << "\n=== Atomic Operations ===\n";
    
    std::atomic<int> atomic_val{10};
    
    // fetch_add: atomically add and return OLD value
    int old = atomic_val.fetch_add(5);
    std::cout << "fetch_add(5): old=" << old << ", new=" << atomic_val << "\n";
    // atomic_val was 10, now 15. Returns old value (10).
    
    // fetch_sub: atomically subtract and return OLD value
    old = atomic_val.fetch_sub(3);
    std::cout << "fetch_sub(3): old=" << old << ", new=" << atomic_val << "\n";
    // atomic_val was 15, now 12. Returns old value (15).
    
    // exchange: atomically replace and return OLD value
    old = atomic_val.exchange(100);
    std::cout << "exchange(100): old=" << old << ", new=" << atomic_val << "\n";
    // atomic_val was 12, now 100. Returns old value (12).
    
    // compare_exchange_weak: atomically compare and swap
    // If atomic_val == expected, set atomic_val = desired, return true
    // Otherwise, set expected = atomic_val, return false
    int expected = 100;
    bool success = atomic_val.compare_exchange_weak(expected, 200);
    std::cout << "compare_exchange_weak(100->200): success=" << success 
              << ", current=" << atomic_val << "\n";
    // atomic_val was 100, matches expected, so set to 200, return true.
    
    // Try again with wrong expected value
    expected = 999;  // Wrong — current value is 200
    success = atomic_val.compare_exchange_weak(expected, 300);
    std::cout << "compare_exchange_weak(999->300): success=" << success 
              << ", current=" << atomic_val << ", expected updated to=" << expected << "\n";
    // atomic_val is 200, doesn't match expected (999), so set expected=200, return false.
}

// ==================== Lock-Free Queue Pattern ====================

void demonstrate_cas_pattern() {
    std::cout << "\n=== Compare-Exchange Pattern (Lock-Free) ===\n";
    
    std::atomic<int> head{0};
    
    // Simulate lock-free push operation
    for (int i = 0; i < 5; ++i) {
        int old_head = head.load();
        int new_head = old_head + 1;
        
        // Retry loop for compare_exchange_weak
        // Weak version may fail spuriously — retry on failure
        while (!head.compare_exchange_weak(old_head, new_head)) {
            // On failure, old_head is updated to current value
            // Retry with new value
            new_head = old_head + 1;
        }
        
        std::cout << "Pushed item " << i << ", new head = " << new_head << "\n";
    }
    
    std::cout << "Final head: " << head << "\n";
}

int main() {
    compare_implementations();
    demonstrate_atomic_operations();
    demonstrate_cas_pattern();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Layer 1 of atomics: ATOMICITY (prevents tearing)\n";
    std::cout << "  - std::atomic<T> ensures indivisible read/write\n";
    std::cout << "  - fetch_add, fetch_sub, exchange are atomic\n";
    std::cout << "  - compare_exchange_weak for lock-free algorithms\n";
    std::cout << "\nLayer 2 (next exercise): MEMORY ORDER (visibility timing)\n";
    
    return 0;
}

// KEY_INSIGHT:
// Layer 1 (Atomicity): std::atomic<T> prevents tearing.
// - Read/write is indivisible — no partial values
// - fetch_add, fetch_sub, exchange are atomic operations
// - compare_exchange_weak enables lock-free data structures
//
// compare_exchange_weak pattern:
//   while (!atomic.compare_exchange_weak(expected, desired)) {
//       // On failure, expected is updated to current value
//       // Retry with new desired value
//   }
//
// CUDA mapping: Device-side atomics provide same guarantees:
// - atomicAdd, atomicSub, atomicExch, atomicCAS
// - Lock-free queues on GPU use atomicCAS in retry loops
