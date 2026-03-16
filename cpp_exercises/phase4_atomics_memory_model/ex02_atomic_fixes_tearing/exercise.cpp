// CONCEPT: Atomic fixes tearing — scaffold to add atomic
// FORMAT: SCAFFOLD
// TIME_TARGET: 15 min
// WHY_THIS_MATTERS: Layer 1 of the two-layer atomic model — atomicity prevents tearing.
// CUDA_CONNECTION: Converting non-atomic device counter to atomic.

#include <iostream>
#include <thread>
#include <vector>
#include <cstdint>

// ==================== Before: Non-Atomic (Buggy) ====================

// BUG: Non-atomic 64-bit counter
uint64_t counter_non_atomic = 0;

void increment_non_atomic(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        counter_non_atomic++;  // BUG: Can tear
    }
}

// ==================== After: Atomic (Fixed) ====================

// TODO 1: Declare atomic counter
// std::atomic<uint64_t> counter_atomic{0};
std::atomic<uint64_t> counter_atomic{0};

// TODO 2: Implement atomic increment function
// Use counter_atomic++ or counter_atomic.fetch_add(1)
void increment_atomic(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        counter_atomic++;  // Atomic increment — no tearing
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
    
    // TODO 3: Demonstrate different atomic operations
    // fetch_add: atomically add and return OLD value
    int old = atomic_val.fetch_add(5);
    std::cout << "fetch_add(5): old=" << old << ", new=" << atomic_val << "\n";
    
    // fetch_sub: atomically subtract and return OLD value
    old = atomic_val.fetch_sub(3);
    std::cout << "fetch_sub(3): old=" << old << ", new=" << atomic_val << "\n";
    
    // exchange: atomically replace and return OLD value
    old = atomic_val.exchange(100);
    std::cout << "exchange(100): old=" << old << ", new=" << atomic_val << "\n";
    
    // compare_exchange_weak: atomically compare and swap
    int expected = 100;
    bool success = atomic_val.compare_exchange_weak(expected, 200);
    std::cout << "compare_exchange_weak(100->200): success=" << success 
              << ", current=" << atomic_val << "\n";
    
    // Try again with wrong expected value
    expected = 999;  // Wrong — current value is 200
    success = atomic_val.compare_exchange_weak(expected, 300);
    std::cout << "compare_exchange_weak(999->300): success=" << success 
              << ", current=" << atomic_val << ", expected updated to=" << expected << "\n";
}

int main() {
    compare_implementations();
    demonstrate_atomic_operations();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Layer 1 of atomics: ATOMICITY (prevents tearing)\n";
    std::cout << "  - std::atomic<T> ensures indivisible read/write\n";
    std::cout << "  - fetch_add, fetch_sub, exchange are atomic\n";
    std::cout << "  - compare_exchange_weak for lock-free algorithms\n";
    std::cout << "\nLayer 2 (next exercise): MEMORY ORDER (visibility timing)\n";
    
    return 0;
}

// VERIFY: Expected output:
// === Non-Atomic Counter (Buggy) ===
// Result: <varies, less than 400000> (expected: 400000)
// Error: <positive number>
//
// === Atomic Counter (Fixed) ===
// Result: 400000 (expected: 400000)
// Error: 0
//
// === Atomic Operations ===
// fetch_add(5): old=10, new=15
// fetch_sub(3): old=15, new=12
// exchange(100): old=12, new=100
// compare_exchange_weak(100->200): success=1, current=200
// compare_exchange_weak(999->300): success=0, current=200, expected updated to=999
