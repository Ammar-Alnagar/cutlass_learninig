// SOLUTION: ex02_named_rvalue_is_lvalue
// The named rvalue rule explained and fixed

#include <iostream>
#include <utility>
#include <cstring>

class Buffer {
public:
    size_t size;
    char* data;
    
    Buffer(size_t s) : size(s), data(new char[s]) {
        std::memset(data, 0, size);
        std::cout << "Buffer(" << size << ") constructed at " << this << "\n";
    }
    
    ~Buffer() {
        std::cout << "Buffer(" << size << ") destroyed at " << this << "\n";
        delete[] data;  // Safe: delete nullptr is a no-op
    }
    
    // FIXED: This is now a proper move constructor
    // Key insight: 'other' is an lvalue inside this function (it has a name),
    // but we're allowed to modify it. We steal its resources and leave it empty.
    Buffer(Buffer&& other) noexcept 
        : size(other.size), data(other.data) {  // Steal the pointer
        std::cout << "  [CORRECT] Moving data (stealing pointer)\n";
        // Leave 'other' in a valid but empty state
        other.data = nullptr;
        other.size = 0;
    }
    
    // Also need copy constructor for completeness (deleted to force move-only)
    // For this exercise, we delete it to ensure move is the only option
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    void print() const {
        if (data) {
            std::cout << "Buffer contents: \"" << data << "\"\n";
        } else {
            std::cout << "Buffer is empty (moved-from)\n";
        }
    }
};

void process(Buffer&& buf) {
    std::cout << "process() received buffer at " << &buf << "\n";
    
    // 'buf' is a named parameter — it is an lvalue here!
    // std::move(buf) casts it back to rvalue so the move constructor is called
    Buffer next(std::move(buf));
    
    std::cout << "process() creating next buffer\n";
    next.print();
}

int main() {
    std::cout << "=== Creating source buffer ===\n";
    Buffer source(64);
    std::strcpy(source.data, "Hello, Move Semantics!");
    
    std::cout << "\n=== Passing to process() as rvalue ===\n";
    process(std::move(source));
    
    std::cout << "\n=== Back in main ===\n";
    std::cout << "Source after move: size = " << source.size 
              << ", data = " << (source.data ? "(valid)" : "nullptr") << "\n";
    
    std::cout << "\n=== Exiting main (destructors fire) ===\n";
    return 0;
}

// KEY_INSIGHT:
// The Named Rvalue Rule: "If it has a name, it is an lvalue — even if its type is T&&"
//
// This single rule explains:
// 1. Why std::move exists — to cast named lvalues to rvalue type
// 2. Why move constructors set the source to nullptr — the source is still an lvalue
//    that will be destroyed normally, so it must be safe to destroy
// 3. Why you need std::move inside functions that take rvalue references — the
//    parameter is named, so it's an lvalue inside the function body
//
// CUDA mapping: In device code, the same rule applies. A kernel parameter declared
// as T&& is an lvalue inside the kernel. If you need to forward it to another
// device function that takes rvalues, you must use std::move (or cuda::std::move).
