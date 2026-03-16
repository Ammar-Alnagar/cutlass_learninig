# Hints for ex02_atomic_fixes_tearing

## H1 — Concept Direction
This exercise scaffolds adding `std::atomic` to fix tearing. The atomic counter uses `std::atomic<uint64_t>` instead of raw `uint64_t`. Atomic operations like `fetch_add` and `compare_exchange_weak` are indivisible.

## H2 — Names the Tool
Declare: `std::atomic<uint64_t> counter{0}`. Increment: `counter++` or `counter.fetch_add(1)`. Compare-and-swap: `counter.compare_exchange_weak(expected, desired)` — returns true if swap succeeded, false otherwise (and updates `expected` to current value).

## H3 — Minimal Usage (Unrelated Context)
```cpp
std::atomic<int> val{10};

val.fetch_add(5);       // val = 15, returns 10 (old)
val.fetch_sub(3);       // val = 12, returns 15 (old)
val.exchange(100);      // val = 100, returns 12 (old)

int exp = 100;
val.compare_exchange_weak(exp, 200);  // If val==100, set val=200, return true
                                      // Else set exp=val, return false
```
