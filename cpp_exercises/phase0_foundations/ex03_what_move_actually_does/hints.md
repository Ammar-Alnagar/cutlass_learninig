# Hints for ex03_what_move_actually_does

## H1 — Concept Direction
A move constructor steals the pointer and nullifies the source. A move assignment first frees its own current resource, then steals, then nullifies the source. Both leave the moved-from object safe to destroy (delete nullptr is safe).

## H2 — Names the Tool
Use `std::move(x)` to cast `x` to an rvalue. In the return statement, `return std::move(local);` enables the move constructor. For copy deletion: `Buffer(const Buffer&) = delete;` makes the type move-only.

## H3 — Minimal Usage (Unrelated Context)
```cpp
// Move constructor pattern:
MyClass(MyClass&& other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;  // Leave safe
}

// Move assignment pattern:
MyClass& operator=(MyClass&& other) noexcept {
    if (this != &other) {
        delete ptr;       // Free current
        ptr = other.ptr;  // Steal
        other.ptr = nullptr;
    }
    return *this;
}
```
