// SOLUTION: ex01_rule_of_five_timed
// Complete Rule of Five implementation

#include <iostream>
#include <cstring>

class Buffer {
public:
    size_t size;
    char* data;
    
    // 1. Constructor
    Buffer(size_t s) : size(s), data(new char[s]()) {
        std::cout << "Allocating " << size << " bytes\n";
    }
    
    // 2. Destructor
    ~Buffer() {
        std::cout << "Destroying " << size << "\n";
        delete[] data;
    }
    
    // 3. Copy constructor - deep copy
    Buffer(const Buffer& other) : size(other.size), data(new char[other.size]) {
        std::cout << "Copy ctor\n";
        std::memcpy(data, other.data, size);
    }
    
    // 4. Copy assignment - deep copy with self-assignment check
    Buffer& operator=(const Buffer& other) {
        std::cout << "Copy assign\n";
        if (this != &other) {
            delete[] data;  // Free old resource
            size = other.size;
            data = new char[size];
            std::memcpy(data, other.data, size);
        }
        return *this;
    }
    
    // 5. Move constructor - steal pointer
    Buffer(Buffer&& other) noexcept : size(other.size), data(other.data) {
        std::cout << "Move ctor\n";
        other.data = nullptr;
        other.size = 0;
    }
    
    // 6. Move assignment - steal pointer
    Buffer& operator=(Buffer&& other) noexcept {
        std::cout << "Move assign\n";
        if (this != &other) {
            delete[] data;  // Free old resource
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
    
    void write(const char* msg) {
        std::strncpy(data, msg, size - 1);
        data[size - 1] = '\0';
    }
    
    void print() const {
        std::cout << "Buffer(" << size << "): " << (data ? data : "empty") << "\n";
    }
};

int main() {
    std::cout << "=== Test 1: Copy ===\n";
    Buffer a(32);
    a.write("Hello");
    Buffer b = a;  // Copy ctor
    b.print();
    
    std::cout << "\n=== Test 2: Copy Assign ===\n";
    Buffer c(64);
    c = a;  // Copy assign
    c.print();
    
    std::cout << "\n=== Test 3: Move ===\n";
    Buffer d = std::move(a);  // Move ctor
    d.print();
    a.print();  // Should be empty
    
    std::cout << "\n=== Test 4: Move Assign ===\n";
    Buffer e(16);
    e = std::move(b);  // Move assign
    e.print();
    b.print();  // Should be empty
    
    std::cout << "\n=== Exiting (destructors) ===\n";
    return 0;
}

// INTERVIEW RUBRIC (SOLUTION):
// [✓] Destructor: delete[] data
// [✓] Copy ctor: new char[other.size], memcpy
// [✓] Copy assign: delete[], new, memcpy, self-check
// [✓] Move ctor: steal data, nullify other
// [✓] Move assign: delete[], steal, nullify, self-check
//
// KEY PATTERNS:
// - Copy = deep copy (allocate new, copy contents)
// - Move = steal (copy pointer, nullify source)
// - Assignment = free old first, then copy/steal
// - Always check self-assignment in assignment operators
