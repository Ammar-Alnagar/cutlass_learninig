# Hints for ex01_lvalue_rvalue_identity

## H1 — Concept Direction
The key distinction is **addressability**. If you can apply the `&` operator to something, it is an lvalue. If you cannot, it is an rvalue. Named variables have storage — they are lvalues. Temporaries and literals do not — they are rvalues.

## H2 — Names the Tool
Use `std::move()` to cast a named variable to an rvalue reference type. This does NOT move anything by itself — it only changes the type so that overload resolution selects the rvalue overload.

## H3 — Minimal Usage (Unrelated Context)
```cpp
int x = 10;
int& ref1 = x;           // OK: x is lvalue, binds to int&
int&& ref2 = std::move(x); // OK: std::move(x) is rvalue type, binds to int&&
// int&& ref3 = x;       // ERROR: x is lvalue, cannot bind to int&&
```
