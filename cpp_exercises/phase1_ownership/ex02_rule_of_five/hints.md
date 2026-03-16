# Hints for ex02_rule_of_five

## H1 — Concept Direction
The Rule of Five states: if you define any of destructor, copy constructor, copy assignment, move constructor, or move assignment, you likely need all five. This is because resource management (like dynamic memory) requires custom behavior for all lifecycle operations.

## H2 — Names the Tool
Order of implementation: (1) destructor first — it defines cleanup, (2) copy constructor — deep copy, (3) copy assignment — free old, deep copy, (4) move constructor — steal and nullify, (5) move assignment — free old, steal, nullify.

## H3 — Minimal Usage (Unrelated Context)
```cpp
// Minimal Rule of Five skeleton:
~MyClass() { delete ptr; }
MyClass(const MyClass& o) : ptr(new int(*o.ptr)) {}
MyClass& operator=(const MyClass& o) { /* free, alloc, copy */ return *this; }
MyClass(MyClass&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
MyClass& operator=(MyClass&& o) noexcept { /* free, steal, null */ return *this; }
```
