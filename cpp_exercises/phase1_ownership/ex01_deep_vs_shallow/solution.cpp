// SOLUTION: ex01_deep_vs_shallow
// Deep copy implementation to prevent double-free

#include <iostream>
#include <cstring>

class DeviceBuffer {
public:
    size_t size;
    char* data;
    
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
    
    // FIXED: Deep copy constructor
    // Allocates NEW memory and copies the CONTENTS
    DeviceBuffer(const DeviceBuffer& other) 
        : size(other.size), data(new char[other.size]) {
        std::cout << "Deep copying " << other.size << " bytes from " 
                  << (void*)other.data << " to " << (void*)data << "\n";
        std::memcpy(data, other.data, size);
    }
    
    // FIXED: Deep copy assignment operator
    // Frees current memory, allocates new, copies contents
    DeviceBuffer& operator=(const DeviceBuffer& other) {
        if (this == &other) {
            return *this;  // Self-assignment check
        }
        
        delete[] data;  // Free current resource
        
        size = other.size;
        data = new char[size];
        std::memcpy(data, other.data, size);
        
        std::cout << "Deep copy assigned " << size << " bytes\n";
        return *this;
    }
    
    // For completeness, also implement move operations (Rule of Five)
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : size(other.size), data(other.data) {
        other.data = nullptr;
        other.size = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;
        delete[] data;
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
        return *this;
    }
};

void copy_demo() {
    std::cout << "=== Creating original buffer ===\n";
    DeviceBuffer original(64);
    original.write("Original data");
    original.print();
    
    std::cout << "\n=== Copying buffer (shallow vs deep) ===\n";
    DeviceBuffer copy = original;  // Invokes deep copy constructor
    copy.print();
    
    std::cout << "\n=== Modifying copy ===\n";
    copy.write("Modified copy");
    copy.print();
    std::cout << "Original after modifying copy: ";
    original.print();  // Now correctly shows "Original data"
    
    std::cout << "\n=== Exiting copy_demo (destructors fire) ===\n";
    // Both destructors fire safely — each owns its own memory
}

int main() {
    copy_demo();
    std::cout << "\n=== Back in main (success!) ===\n";
    return 0;
}

// KEY_INSIGHT:
// Default copy operations do MEMBERWISE (shallow) copy.
// For pointer members, this copies the POINTER, not the POINTED-TO data.
// Result: two objects own the same memory → double-free crash.
//
// Deep copy: allocate new memory, copy the CONTENTS.
// Each object owns its own memory → no crash.
//
// Alternative: make the type MOVE-ONLY (delete copy, implement move).
// This is often better for resource wrappers (unique_ptr does this).
//
// CUDA mapping: A DeviceMemory wrapper holding a cudaDevicePtr must
// either deep-copy the device memory (expensive!) or be move-only.
// CUTLASS chooses move-only for efficiency — model weights are never copied.
