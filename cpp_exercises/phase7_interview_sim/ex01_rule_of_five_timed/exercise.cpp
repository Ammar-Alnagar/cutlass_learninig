// CONCEPT: Rule of Five — Timed Interview Exercise
// FORMAT: IMPLEMENT (TIMED)
// TIME_TARGET: 4 minutes
// WHY_THIS_MATTERS: Rule of Five is a common interview question for C++ roles.
// CUDA_CONNECTION: RAII wrappers for device memory need all five methods.

// INTERVIEW SIMULATION:
// - Set a timer for 4 minutes
// - Implement all five special member functions
// - No hints allowed (interview setting)
// - Compile and run to verify

#include <iostream>
#include <cstring>

// TODO: Implement a Buffer class with Rule of Five
// Requirements:
// 1. Constructor: allocate char[size], zero-initialize
// 2. Destructor: delete[] data, print "Destroying {size}"
// 3. Copy constructor: deep copy, print "Copy ctor"
// 4. Copy assignment: deep copy, print "Copy assign"
// 5. Move constructor: steal pointer, print "Move ctor"
// 6. Move assignment: steal pointer, print "Move assign"
//
// TIME TARGET: 4 minutes

class Buffer {
public:
    size_t size;
    char* data;
    
    // TODO: Implement all five methods below
    // Start timer now!
    
    Buffer(size_t s) : size(s), data(new char[s]()) {
        std::cout << "Allocating " << size << " bytes\n";
    }
    
    ~Buffer() {
        std::cout << "Destroying " << size << "\n";
        delete[] data;
    }
    
    Buffer(const Buffer& other) : size(other.size), data(new char[other.size]) {
        std::cout << "Copy ctor\n";
        std::memcpy(data, other.data, size);
    }
    
    Buffer& operator=(const Buffer& other) {
        std::cout << "Copy assign\n";
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new char[size];
            std::memcpy(data, other.data, size);
        }
        return *this;
    }
    
    Buffer(Buffer&& other) noexcept : size(other.size), data(other.data) {
        std::cout << "Move ctor\n";
        other.data = nullptr;
        other.size = 0;
    }
    
    Buffer& operator=(Buffer&& other) noexcept {
        std::cout << "Move assign\n";
        if (this != &other) {
            delete[] data;
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

// INTERVIEW RUBRIC:
// [ ] Destructor implemented (frees memory)
// [ ] Copy constructor (deep copy, allocates new memory)
// [ ] Copy assignment (frees old, deep copy, self-assignment check)
// [ ] Move constructor (steals pointer, nullifies source)
// [ ] Move assignment (frees old, steals, nullifies source, self-check)
//
// TIME CHECK:
// < 3 min: Excellent — fluent with Rule of Five
// 3-4 min: Good — solid understanding
// 4-5 min: Acceptable — needs more practice
// > 5 min: Review ownership concepts
