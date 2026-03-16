// CONCEPT: Deep copy vs shallow copy — double free crash
// FORMAT: DEBUG
// TIME_TARGET: 15 min
// WHY_THIS_MATTERS: Default copy operations do shallow copy. You must implement Rule of Five.
// CUDA_CONNECTION: DeviceMemory wrappers — copying them without deep copy causes double cudaFree.

#include <iostream>
#include <cstring>

class DeviceBuffer {
public:
    size_t size;
    char* data;  // Simulating device memory with host allocation
    
    DeviceBuffer(size_t s) : size(s), data(new char[s]) {
        std::memset(data, 0, size);
        std::cout << "Allocated " << size << " bytes at " << (void*)data << "\n";
    }
    
    ~DeviceBuffer() {
        std::cout << "Freeing " << size << " bytes at " << (void*)data << "\n";
        delete[] data;
    }
    
    void write(const char* msg) {
        std::strncpy(data, msg, size - 1);
        data[size - 1] = '\0';
    }
    
    void print() const {
        std::cout << "DeviceBuffer: " << data << "\n";
    }
    
    // BUG: Default copy constructor does shallow copy (memberwise copy)
    // Both objects end up pointing to the same memory!
    // When both destructors fire, double-free occurs.
    // DeviceBuffer(const DeviceBuffer& other) = default;  // BUG IS HERE
    
    // BUG: Default copy assignment also does shallow copy
    // DeviceBuffer& operator=(const DeviceBuffer& other) = default;  // BUG IS HERE
    
    // SYMPTOMS:
    // 1. Program crashes with "double free or corruption" error
    // 2. Sanitizer reports: heap-use-after-free or double-free
    // 3. Both objects print the same data address
    
    // TODO: Fix by implementing deep copy (copy constructor + copy assignment)
    // OR: Delete copy operations and implement move operations instead
    // For this exercise, implement DEEP COPY to understand the difference
};

void copy_demo() {
    std::cout << "=== Creating original buffer ===\n";
    DeviceBuffer original(64);
    original.write("Original data");
    original.print();
    
    std::cout << "\n=== Copying buffer (shallow vs deep) ===\n";
    DeviceBuffer copy = original;  // Invokes copy constructor
    copy.print();
    
    std::cout << "\n=== Modifying copy ===\n";
    copy.write("Modified copy");
    copy.print();
    std::cout << "Original after modifying copy: ";
    original.print();  // If shallow: also shows "Modified copy" (BUG!)
                       // If deep: still shows "Original data" (CORRECT!)
    
    std::cout << "\n=== Exiting copy_demo (destructors fire) ===\n";
    // If shallow: both destructors try to delete same memory → CRASH
    // If deep: each deletes its own memory → OK
}

int main() {
    copy_demo();
    std::cout << "\n=== Back in main (success!) ===\n";
    return 0;
}

// VERIFY (after fix):
// 1. No crash or sanitizer errors
// 2. Original and copy have DIFFERENT data addresses
// 3. Modifying copy does NOT affect original
// 4. Both destructors fire without error

// FIX INSTRUCTIONS:
// Implement deep copy constructor and copy assignment:
// 1. Allocate NEW memory for the copy
// 2. Copy the CONTENTS (memcpy), not the pointer
// 3. Return *this from assignment operator
