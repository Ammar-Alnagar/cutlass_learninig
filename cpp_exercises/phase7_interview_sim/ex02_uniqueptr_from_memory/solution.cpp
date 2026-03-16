// SOLUTION: ex02_uniqueptr_from_memory
// Complete unique_ptr implementation

#include <iostream>
#include <utility>

template<typename T>
class SimpleUniquePtr {
private:
    T* ptr;
    
public:
    // Constructor from raw pointer
    explicit SimpleUniquePtr(T* p = nullptr) : ptr(p) {}
    
    // Destructor: delete owned pointer
    ~SimpleUniquePtr() {
        delete ptr;
    }
    
    // Delete copy operations — move-only type
    SimpleUniquePtr(const SimpleUniquePtr&) = delete;
    SimpleUniquePtr& operator=(const SimpleUniquePtr&) = delete;
    
    // Move constructor: steal pointer, nullify source
    SimpleUniquePtr(SimpleUniquePtr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    
    // Move assignment: free old, steal, nullify source
    SimpleUniquePtr& operator=(SimpleUniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr;  // Free current resource
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    
    // Accessors
    T* get() const { return ptr; }
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
    
    // Release ownership: return pointer and set to null
    T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }
    
    // Reset: delete old, take new pointer
    void reset(T* p = nullptr) {
        delete ptr;
        ptr = p;
    }
    
    // Bool conversion for if (ptr) checks
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

// INTERVIEW RUBRIC (SOLUTION):
// [✓] Destructor: delete ptr
// [✓] Copy ops deleted (move-only)
// [✓] Move ctor: ptr(other.ptr), other.ptr = nullptr
// [✓] Move assign: delete ptr, steal, nullify, self-check
// [✓] get(): return ptr
// [✓] release(): save ptr, nullify, return saved
// [✓] reset(): delete old, assign new
// [✓] operator bool(): return ptr != nullptr
//
// KEY INSIGHTS:
// - unique_ptr is MOVE-ONLY (copy deleted)
// - Move transfers OWNERSHIP (source becomes null)
// - release() gives up ownership (caller must delete)
// - reset() takes new ownership (deletes old first)
//
// CUDA mapping: unique_ptr<T, Deleter> for device memory:
//   auto deleter = [](void* p) { cudaFree(p); };
//   std::unique_ptr<void, decltype(deleter)> dev(ptr, deleter);
