# Hints for ex02_named_rvalue_is_lvalue

## H1 — Concept Direction
The parameter `other` in `Buffer(Buffer&& other)` has a name. By the named rvalue rule, it is an lvalue inside the constructor body. The issue is not about calling it — it's about what the constructor DOES. A move constructor should steal resources, not copy them.

## H2 — Names the Tool
After stealing the pointer in a move constructor, you MUST leave the source in a valid state for destruction. Set `other.data = nullptr` and `other.size = 0`. The destructor will then safely call `delete[] nullptr` (which is a no-op).

## H3 — Minimal Usage (Unrelated Context)
```cpp
// Correct move constructor pattern:
MyClass(MyClass&& other) noexcept 
    : ptr(other.ptr), size(other.size) {  // Steal
    other.ptr = nullptr;                   // Leave safe
    other.size = 0;
}
```
