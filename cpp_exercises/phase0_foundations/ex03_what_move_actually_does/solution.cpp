// SOLUTION: ex03_what_move_actually_does
// Complete implementation of move-only Buffer class

#include <iostream>
#include <utility>
#include <cstring>

class Buffer {
public:
    size_t size;
    char* data;
    
    // Constructor: allocate and zero-initialize
    Buffer(size_t s) : size(s), data(new char[s]) {
        std::memset(data, 0, size);
    }
    
    // Destructor: free the memory
    ~Buffer() {
        std::cout << "Destroying buffer at " << this << "\n";
        delete[] data;  // Safe: delete nullptr is no-op
    }
    
    // Move constructor: steal resources, leave source empty
    Buffer(Buffer&& other) noexcept 
        : size(other.size), data(other.data) {
        std::cout << "Moving buffer from " << &other << " to " << this << "\n";
        other.data = nullptr;
        other.size = 0;
    }
    
    // Move assignment: free current, steal, leave source empty
    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) {
            return *this;  // Self-assignment check
        }
        delete[] data;  // Free current resource
        std::cout << "Move-assigning " << &other << " to " << this << "\n";
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
        return *this;
    }
    
    // Delete copy operations — this is a move-only type
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    void print_state() const {
        std::cout << "Buffer@" << this << ": size=" << size 
                  << ", data=" << (data ? data : "nullptr") << "\n";
    }
};

Buffer create_and_move() {
    std::cout << "=== Inside create_and_move() ===\n";
    
    Buffer local(32);
    std::strcpy(local.data, "Moved from local");
    local.print_state();
    
    // std::move casts 'local' to rvalue type
    // Return value optimization (RVO) may eliminate the move entirely
    return std::move(local);
}

int main() {
    std::cout << "=== Step 1: Create buffer ===\n";
    Buffer buf1(64);
    std::strcpy(buf1.data, "Original data in buf1");
    buf1.print_state();
    
    std::cout << "\n=== Step 2: Move buf1 to buf2 ===\n";
    Buffer buf2(std::move(buf1));
    std::cout << "buf1 after move: ";
    buf1.print_state();
    std::cout << "buf2 after move: ";
    buf2.print_state();
    
    std::cout << "\n=== Step 3: Move-assign buf2 to buf3 ===\n";
    Buffer buf3(16);
    std::strcpy(buf3.data, "Old buf3 data");
    buf3.print_state();
    
    buf3 = std::move(buf2);
    std::cout << "buf2 after move-assign: ";
    buf2.print_state();
    std::cout << "buf3 after move-assign: ";
    buf3.print_state();
    
    std::cout << "\n=== Step 4: Return value from function ===\n";
    Buffer buf4 = create_and_move();
    std::cout << "buf4 in main: ";
    buf4.print_state();
    
    std::cout << "\n=== Step 5: Exiting main ===\n";
    return 0;
}

// KEY_INSIGHT:
// std::move is a cast, not an operation. It does NOT move anything by itself.
// It casts an lvalue to an rvalue reference type, enabling:
// 1. Overload resolution to select move constructor/assignment
// 2. The actual move to happen in the called function
//
// The move constructor/assignment does the actual work:
// - Steals the pointer (shallow copy)
// - Nullifies the source (leaves it safe to destroy)
// - No deep copy of data (that's the whole point)
//
// CUDA mapping: When preparing kernel arguments, you might move a host-side
// buffer that wraps device memory. The move transfers ownership of the device
// pointer without copying the device memory. This is critical for efficient
// LLM inference where model weights are gigabytes — you never want to copy.
