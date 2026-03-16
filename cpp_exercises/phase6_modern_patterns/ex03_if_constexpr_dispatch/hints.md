# Hints for ex03_if_constexpr_dispatch

## H1 — Concept Direction
`if constexpr` evaluates the condition at compile time. The false branch is discarded (not compiled), so code that wouldn't compile for type T is never seen by the compiler. This enables type-specific code in a single template.

## H2 — Names the Tool
Syntax: `if constexpr (std::is_integral_v<T>) { ... }`. Use type traits like `std::is_integral_v`, `std::is_floating_point_v`, `std::is_same_v<T, U>` for conditions.

## H3 — Minimal Usage (Unrelated Context)
```cpp
template<typename T>
void foo(T x) {
    if constexpr (std::is_integral_v<T>) {
        x++;  // Only compiled for integral types
    }
    if constexpr (std::is_floating_point_v<T>) {
        x = std::sin(x);  // Only compiled for float types
    }
}
```
