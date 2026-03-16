# Hints for ex01_deep_vs_shallow

## H1 — Concept Direction
The default copy constructor copies each member variable. For a pointer member, this copies the pointer value (address), not the data it points to. Both objects then point to the same memory. When both destructors call `delete[]`, double-free occurs.

## H2 — Names the Tool
Implement a custom copy constructor that allocates NEW memory: `data(new char[other.size])`, then copies contents: `std::memcpy(data, other.data, size)`. Do the same in copy assignment, but first `delete[] data` to free the old resource.

## H3 — Minimal Usage (Unrelated Context)
```cpp
// Deep copy constructor pattern:
MyClass(const MyClass& other) 
    : ptr(new int[*other.ptr]), size(other.size) {
    std::memcpy(ptr, other.ptr, size * sizeof(int));
}
```
