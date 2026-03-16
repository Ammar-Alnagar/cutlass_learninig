// CONCEPT: Rule of Five — all five special methods
// FORMAT: SCAFFOLD
// TIME_TARGET: 20 min
// WHY_THIS_MATTERS: If you define any of the five, you likely need all five.
// CUDA_CONNECTION: RAII wrappers for cudaMalloc/cudaFree need all five methods.

#include <iostream>
#include <cstring>
#include <utility>

class ManagedBuffer {
public:
    size_t size;
    char* data;
    
    ManagedBuffer(size_t s) : size(s), data(new char[s]) {
        std::memset(data, 0, size);
        std::cout << "[CTOR] Default constructor\n";
    }
    
    // TODO 1: Destructor
    // Print: "[DTOR] Destructor"
    // Free: delete[] data
    ~ManagedBuffer() {
        std::cout << "[DTOR] Destructor\n";
        delete[] data;
    }
    
    // TODO 2: Copy constructor
    // Print: "[COPY CTOR] Copy constructor"
    // Allocate new memory and copy contents (deep copy)
    ManagedBuffer(const ManagedBuffer& other) 
        : size(other.size), data(new char[other.size]) {
        std::cout << "[COPY CTOR] Copy constructor\n";
        std::memcpy(data, other.data, size);
    }
    
    // TODO 3: Copy assignment operator
    // Print: "[COPY ASSIGN] Copy assignment"
    // Check self-assignment, free current, allocate new, copy contents
    ManagedBuffer& operator=(const ManagedBuffer& other) {
        std::cout << "[COPY ASSIGN] Copy assignment\n";
        if (this == &other) {
            return *this;
        }
        delete[] data;
        size = other.size;
        data = new char[size];
        std::memcpy(data, other.data, size);
        return *this;
    }
    
    // TODO 4: Move constructor
    // Print: "[MOVE CTOR] Move constructor"
    // Steal pointer, nullify source
    ManagedBuffer(ManagedBuffer&& other) noexcept 
        : size(other.size), data(other.data) {
        std::cout << "[MOVE CTOR] Move constructor\n";
        other.data = nullptr;
        other.size = 0;
    }
    
    // TODO 5: Move assignment operator
    // Print: "[MOVE ASSIGN] Move assignment"
    // Check self-assignment, free current, steal pointer, nullify source
    ManagedBuffer& operator=(ManagedBuffer&& other) noexcept {
        std::cout << "[MOVE ASSIGN] Move assignment\n";
        if (this == &other) {
            return *this;
        }
        delete[] data;
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

// VERIFY: Expected output shows each method being called:
// [CTOR] Default constructor
// [COPY CTOR] Copy constructor
// [COPY ASSIGN] Copy assignment
// [MOVE CTOR] Move constructor
// [MOVE ASSIGN] Move assignment
// [DTOR] Destructor (multiple times, once per object)
//
// No crashes, no memory leaks, no double-frees.
