// SOLUTION: ex02_rule_of_five
// Complete implementation with all five special methods

#include <iostream>
#include <cstring>
#include <utility>

class ManagedBuffer {
public:
    size_t size;
    char* data;
    
    // 1. Constructor
    ManagedBuffer(size_t s) : size(s), data(new char[s]) {
        std::memset(data, 0, size);
        std::cout << "[CTOR] Default constructor\n";
    }
    
    // 2. Destructor
    // Always implement this first — it defines how to clean up resources
    ~ManagedBuffer() {
        std::cout << "[DTOR] Destructor\n";
        delete[] data;
    }
    
    // 3. Copy constructor
    // Deep copy: allocate new memory, copy contents
    ManagedBuffer(const ManagedBuffer& other) 
        : size(other.size), data(new char[other.size]) {
        std::cout << "[COPY CTOR] Copy constructor\n";
        std::memcpy(data, other.data, size);
    }
    
    // 4. Copy assignment operator
    // Free current, allocate new, copy contents
    ManagedBuffer& operator=(const ManagedBuffer& other) {
        std::cout << "[COPY ASSIGN] Copy assignment\n";
        if (this == &other) {
            return *this;  // Self-assignment check
        }
        delete[] data;  // Free old resource
        size = other.size;
        data = new char[size];
        std::memcpy(data, other.data, size);
        return *this;
    }
    
    // 5. Move constructor
    // Steal resources, leave source empty
    ManagedBuffer(ManagedBuffer&& other) noexcept 
        : size(other.size), data(other.data) {
        std::cout << "[MOVE CTOR] Move constructor\n";
        other.data = nullptr;
        other.size = 0;
    }
    
    // 6. Move assignment operator
    // Free current, steal, leave source empty
    ManagedBuffer& operator=(ManagedBuffer&& other) noexcept {
        std::cout << "[MOVE ASSIGN] Move assignment\n";
        if (this == &other) {
            return *this;
        }
        delete[] data;  // Free old resource
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
        return *this;
    }
    
    void write(const char* msg) {
        std::strncpy(data, msg, size - 1);
        data[size - 1] = '\0';
    }
    
    void print() const {
        std::cout << "ManagedBuffer: " << (data ? data : "(empty)") << "\n";
    }
};

void test_copy() {
    std::cout << "\n=== Test: Copy construction ===\n";
    ManagedBuffer a(32);
    a.write("Original");
    
    ManagedBuffer b = a;  // Copy constructor
    b.write("Copy");
    
    std::cout << "a: "; a.print();
    std::cout << "b: "; b.print();
}

void test_move() {
    std::cout << "\n=== Test: Move construction ===\n";
    ManagedBuffer a(32);
    a.write("Original");
    
    ManagedBuffer b = std::move(a);  // Move constructor
    std::cout << "a after move: "; a.print();
    std::cout << "b after move: "; b.print();
}

void test_copy_assign() {
    std::cout << "\n=== Test: Copy assignment ===\n";
    ManagedBuffer a(32);
    a.write("Original A");
    
    ManagedBuffer b(64);
    b.write("Original B");
    
    b = a;  // Copy assignment
    std::cout << "a: "; a.print();
    std::cout << "b: "; b.print();
}

void test_move_assign() {
    std::cout << "\n=== Test: Move assignment ===\n";
    ManagedBuffer a(32);
    a.write("Original A");
    
    ManagedBuffer b(64);
    b.write("Original B");
    
    b = std::move(a);  // Move assignment
    std::cout << "a after move: "; a.print();
    std::cout << "b after move: "; b.print();
}

int main() {
    std::cout << "=== Rule of Five Demonstration ===\n";
    
    test_copy();
    test_move();
    test_copy_assign();
    test_move_assign();
    
    std::cout << "\n=== Exiting main (all destructors fire) ===\n";
    return 0;
}

// KEY_INSIGHT:
// Rule of Five: If you define ANY of these, you likely need ALL five:
// 1. Destructor
// 2. Copy constructor
// 3. Copy assignment
// 4. Move constructor
// 5. Move assignment
//
// Why? Because if you manage a resource (like dynamic memory), the default
// operations will do the wrong thing (shallow copy, no cleanup).
//
// Modern C++ alternative: Use unique_ptr or shared_ptr as members.
// Then you can use = default for all five — the smart pointers handle it.
//
// CUDA mapping: A RAII wrapper for cudaMalloc/cudaFree needs all five.
// CUTLASS uses this pattern for device memory management. The destructor
// calls cudaFree, move operations transfer ownership, copy operations
// either deep-copy device memory or are deleted (move-only).
