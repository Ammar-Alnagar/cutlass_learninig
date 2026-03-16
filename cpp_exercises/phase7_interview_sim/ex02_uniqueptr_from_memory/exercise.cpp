// CONCEPT: unique_ptr from memory — Timed Interview Exercise
// FORMAT: IMPLEMENT (TIMED)
// TIME_TARGET: 6 minutes
// WHY_THIS_MATTERS: unique_ptr is fundamental to modern C++ ownership.
// CUDA_CONNECTION: unique_ptr<DeviceMemory, Deleter> for RAII device allocations.

// INTERVIEW SIMULATION:
// - Set a timer for 6 minutes
// - Implement a custom unique_ptr-like class
// - No hints allowed (interview setting)
// - Demonstrate move-only semantics

#include <iostream>
#include <utility>

// TODO: Implement a SimpleUniquePtr class
// Requirements:
// 1. Template: template<typename T> class SimpleUniquePtr
// 2. Constructor: take raw pointer, store it
// 3. Destructor: delete ptr if not null
// 4. Move constructor: steal ptr, nullify source
// 5. Move assignment: free current, steal, nullify source
// 6. Delete copy constructor and copy assignment
// 7. get(), release(), reset() methods
//
// TIME TARGET: 6 minutes

template<typename T>
class SimpleUniquePtr {
private:
    T* ptr;
    
public:
    // Constructor from raw pointer
    explicit SimpleUniquePtr(T* p = nullptr) : ptr(p) {}
    
    // Destructor
    ~SimpleUniquePtr() {
        delete ptr;
    }
    
    // Delete copy operations (move-only)
    SimpleUniquePtr(const SimpleUniquePtr&) = delete;
    SimpleUniquePtr& operator=(const SimpleUniquePtr&) = delete;
    
    // Move constructor
    SimpleUniquePtr(SimpleUniquePtr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    
    // Move assignment
    SimpleUniquePtr& operator=(SimpleUniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    
    // Accessors
    T* get() const { return ptr; }
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
    
    // Release ownership (return ptr, set to null)
    T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }
    
    // Reset with new pointer
    void reset(T* p = nullptr) {
        delete ptr;
        ptr = p;
    }
    
    // Check if valid
    explicit operator bool() const { return ptr != nullptr; }
};

// Helper function (like std::make_unique)
template<typename T, typename... Args>
SimpleUniquePtr<T> make_simple_unique(Args&&... args) {
    return SimpleUniquePtr<T>(new T(std::forward<Args>(args)...));
}

int main() {
    std::cout << "=== Test 1: Basic ownership ===\n";
    SimpleUniquePtr<int> p1(new int(42));
    std::cout << "*p1 = " << *p1 << "\n";
    
    std::cout << "\n=== Test 2: Move constructor ===\n";
    SimpleUniquePtr<int> p2 = std::move(p1);
    std::cout << "p1 after move: " << (p1 ? "valid" : "null") << "\n";
    std::cout << "p2 after move: " << (p2 ? "valid" : "null") << ", *p2 = " << *p2 << "\n";
    
    std::cout << "\n=== Test 3: Move assignment ===\n";
    SimpleUniquePtr<int> p3(new int(100));
    p3 = std::move(p2);
    std::cout << "p2 after move: " << (p2 ? "valid" : "null") << "\n";
    std::cout << "p3 after move: " << (p3 ? "valid" : "null") << ", *p3 = " << *p3 << "\n";
    
    std::cout << "\n=== Test 4: release() ===\n";
    int* raw = p3.release();
    std::cout << "p3 after release: " << (p3 ? "valid" : "null") << "\n";
    std::cout << "raw pointer: " << raw << ", *raw = " << *raw << "\n";
    delete raw;  // Manual cleanup since we released
    
    std::cout << "\n=== Test 5: make_simple_unique ===\n";
    auto p4 = make_simple_unique<int>(999);
    std::cout << "*p4 = " << *p4 << "\n";
    
    std::cout << "\n=== Exiting (destructors) ===\n";
    return 0;
}

// INTERVIEW RUBRIC:
// [ ] Destructor deletes ptr
// [ ] Copy ops deleted (move-only)
// [ ] Move ctor steals and nullifies
// [ ] Move assign frees old, steals, nullifies
// [ ] get() returns raw pointer
// [ ] release() returns ptr and sets to null
// [ ] reset() deletes old, takes new
//
// TIME CHECK:
// < 5 min: Excellent — fluent with ownership
// 5-6 min: Good — solid understanding
// 6-8 min: Acceptable — needs more practice
// > 8 min: Review unique_ptr semantics
