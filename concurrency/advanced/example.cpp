// Advanced Concurrency Topics - Hands-on Example
// This example demonstrates lock-free programming concepts

#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
#include <chrono>
#include <memory>
#include <random>

// Lock-free stack implementation (Treiber Stack)
template<typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        Node* next;
        
        Node(T const& data_) : data(data_), next(nullptr) {}
    };
    
    std::atomic<Node*> head;

public:
    LockFreeStack() : head(nullptr) {}
    
    void push(T const& data) {
        Node* new_node = new Node(data);
        
        // Atomically update the head
        new_node->next = head.load();
        while (!head.compare_exchange_weak(new_node->next, new_node));
    }
    
    std::shared_ptr<T> pop() {
        Node* old_head = head.load();
        while (old_head && !head.compare_exchange_weak(old_head, old_head->next));
        
        std::shared_ptr<T> res;
        if (old_head) {
            res = std::make_shared<T>(old_head->data);
            delete old_head;
        }
        return res;
    }
    
    bool empty() const {
        return head.load() == nullptr;
    }
};

// Lock-free counter using atomic operations
class AtomicCounter {
private:
    std::atomic<long long> count;
    
public:
    AtomicCounter() : count(0) {}
    
    void increment() {
        count.fetch_add(1, std::memory_order_relaxed);
    }
    
    long long get() const {
        return count.load(std::memory_order_relaxed);
    }
    
    void reset() {
        count.store(0, std::memory_order_relaxed);
    }
};

// Example demonstrating atomic operations and memory ordering
void atomicOperationsExample() {
    std::cout << "\n=== Atomic Operations Example ===" << std::endl;
    
    std::atomic<int> atomic_var(0);
    
    // Different types of atomic operations
    atomic_var.store(10);
    std::cout << "After store(10): " << atomic_var.load() << std::endl;
    
    int old_val = atomic_var.exchange(20);
    std::cout << "exchange(20), returned: " << old_val << ", now: " << atomic_var.load() << std::endl;
    
    // Compare and swap operation
    int expected = 20;
    bool success = atomic_var.compare_exchange_strong(expected, 30);
    std::cout << "compare_exchange_strong(20, 30): " << (success ? "success" : "failed") 
              << ", value now: " << atomic_var.load() << std::endl;
    
    // Atomic arithmetic
    atomic_var.fetch_add(5);
    std::cout << "After fetch_add(5): " << atomic_var.load() << std::endl;
}

// Example demonstrating lock-free stack
void lockFreeStackExample() {
    std::cout << "\n=== Lock-Free Stack Example ===" << std::endl;
    
    LockFreeStack<int> stack;
    
    // Push some values
    for (int i = 1; i <= 5; ++i) {
        stack.push(i * 10);
        std::cout << "Pushed: " << i * 10 << std::endl;
    }
    
    // Pop values
    while (!stack.empty()) {
        auto val = stack.pop();
        if (val) {
            std::cout << "Popped: " << *val << std::endl;
        }
    }
}

// Performance comparison: atomic counter vs mutex-protected counter
#include <mutex>

class MutexCounter {
private:
    long long count;
    mutable std::mutex mtx;
    
public:
    MutexCounter() : count(0) {}
    
    void increment() {
        std::lock_guard<std::mutex> lock(mtx);
        ++count;
    }
    
    long long get() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mtx);
        count = 0;
    }
};

void performanceComparison() {
    std::cout << "\n=== Performance Comparison: Atomic vs Mutex ===" << std::endl;
    
    const int num_threads = 4;
    const int increments_per_thread = 1000000;
    
    // Test atomic counter
    AtomicCounter atomic_counter;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&atomic_counter, increments_per_thread]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                atomic_counter.increment();
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto atomic_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Atomic counter: " << atomic_counter.get() << " in " 
              << atomic_duration.count() << " ms" << std::endl;
    
    // Test mutex counter
    MutexCounter mutex_counter;
    start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&mutex_counter, increments_per_thread]() {
            for (int j = 0; j < increments_per_thread; ++j) {
                mutex_counter.increment();
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    end_time = std::chrono::high_resolution_clock::now();
    auto mutex_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Mutex counter: " << mutex_counter.get() << " in " 
              << mutex_duration.count() << " ms" << std::endl;
    
    std::cout << "Atomic is " << (double)mutex_duration.count() / atomic_duration.count() 
              << "x faster" << std::endl;
}

// Example demonstrating memory orderings
void memoryOrderingExample() {
    std::cout << "\n=== Memory Ordering Example ===" << std::endl;
    
    std::atomic<bool> x(false), y(false);
    std::atomic<int> z(0);
    
    auto write_x = [&x]() { x.store(true, std::memory_order_release); };
    auto write_y = [&y]() { y.store(true, std::memory_order_release); };
    auto read_x_then_y = [&x, &y, &z]() { 
        while (!x.load(std::memory_order_acquire)); 
        if (y.load(std::memory_order_acquire)) { 
            ++z; 
        } 
    };
    auto read_y_then_x = [&x, &y, &z]() { 
        while (!y.load(std::memory_order_acquire)); 
        if (x.load(std::memory_order_acquire)) { 
            ++z; 
        } 
    };
    
    std::thread a(write_x);
    std::thread b(write_y);
    std::thread c(read_x_then_y);
    std::thread d(read_y_then_x);
    
    a.join(); b.join(); c.join(); d.join();
    
    std::cout << "z = " << z.load() << " (can be 0, 1, or 2 depending on execution order)" << std::endl;
}

// Lock-free single-producer, single-consumer queue (simplified)
template<typename T>
class SPSCQueue {
private:
    struct Node {
        T data;
        std::atomic<Node*> next;
        
        Node() : next(nullptr) {}
        Node(const T& data_) : data(data_), next(nullptr) {}
    };
    
    std::atomic<Node*> head;
    std::atomic<Node*> tail;
    
public:
    SPSCQueue() {
        Node* stub = new Node();
        head.store(stub);
        tail.store(stub);
    }
    
    ~SPSCQueue() {
        while (Node* const old_head = head.load()) {
            head.store(old_head->next.load());
            delete old_head;
        }
    }
    
    void enqueue(T const& data) {
        Node* new_node = new Node(data);
        Node* prev = tail.exchange(new_node);
        prev->next.store(new_node);
    }
    
    bool dequeue(T& data) {
        Node* old_head = head.load();
        Node* next = old_head->next.load();
        
        if (next == nullptr) {
            return false;  // Queue is empty
        }
        
        data = next->data;
        head.store(next);
        delete old_head;
        return true;
    }
};

void spscQueueExample() {
    std::cout << "\n=== SPSC Queue Example ===" << std::endl;
    
    SPSCQueue<int> queue;
    
    // Producer thread
    std::thread producer([&queue]() {
        for (int i = 1; i <= 5; ++i) {
            queue.enqueue(i);
            std::cout << "Enqueued: " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    
    // Consumer thread
    std::thread consumer([&queue]() {
        for (int i = 1; i <= 5; ++i) {
            int value;
            while (!queue.dequeue(value)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::cout << "Dequeued: " << value << std::endl;
        }
    });
    
    producer.join();
    consumer.join();
}

int main() {
    std::cout << "Advanced Concurrency Topics - Hands-on Example" << std::endl;
    
    atomicOperationsExample();
    lockFreeStackExample();
    performanceComparison();
    memoryOrderingExample();
    spscQueueExample();
    
    std::cout << "\nAll advanced examples completed!" << std::endl;
    
    return 0;
}