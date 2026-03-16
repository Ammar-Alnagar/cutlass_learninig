// CONCEPT: The named rvalue rule — "if it has a name, it is an lvalue"
// FORMAT: DEBUG
// TIME_TARGET: 15 min
// WHY_THIS_MATTERS: This is why std::move exists. Without this rule, move semantics would be automatic.
// CUDA_CONNECTION: Device function parameters — named rvalue references inside kernels.

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
        delete[] data;
    }
    
    // BUG: This is NOT a move constructor — it's a copy constructor!
    // The parameter 'other' has a name, so it is an lvalue, even though
    // its type is Buffer&&. This means the lvalue overload is selected.
    Buffer(Buffer&& other) : size(other.size), data(new char[other.size]) {
        std::cout << "  [WRONG] Copying data (this is a copy, not a move!)\n";
        std::memcpy(data, other.data, size);
    }
    
    // This is the actual move constructor — note the std::move call inside
    // But wait, this won't be called because the above constructor matches first!
    
    void print() const {
        std::cout << "Buffer contents: \"" << data << "\"\n";
    }
};

// SYMPTOMS: 
// 1. Program prints "[WRONG] Copying data" instead of moving
// 2. Double-free crash at exit (both buffers try to delete same memory pattern)
// 3. The "moved-from" buffer still has valid data (it was copied, not moved)

void process(Buffer&& buf) {
    std::cout << "process() received buffer at " << &buf << "\n";
    
    // BUG: 'buf' is a named parameter — it is an lvalue inside this function!
    // You MUST use std::move(buf) to forward it as an rvalue.
    // Without std::move, this calls the copy constructor (if available) or fails.
    Buffer next(std::move(buf));  // FIX: Added std::move — but the constructor is still wrong
    
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
    // SYMPTOM: source should be in a valid but unspecified state after move
    // If it was actually moved, source.data might be nullptr or source.size might be 0
    // If it was copied (the bug), source still has the original data
    
    std::cout << "Source after move: size = " << source.size << "\n";
    // Do NOT access source.data here if it was truly moved — it might be dangling
    
    std::cout << "\n=== Exiting main (destructors fire) ===\n";
    return 0;
}

// VERIFY: Expected behavior after fix:
// 1. No "[WRONG] Copying data" message — should print move constructor message
// 2. No double-free crash
// 3. source.size should be 0 or source.data should be nullptr after move
// 4. Only one destructor should print for the actual data ownership

// FIX INSTRUCTIONS:
// 1. Change the Buffer(Buffer&&) constructor to actually move (steal the pointer)
// 2. Set other.data = nullptr and other.size = 0 after stealing
// 3. This makes the moved-from object safe to destroy (delete nullptr is no-op)
