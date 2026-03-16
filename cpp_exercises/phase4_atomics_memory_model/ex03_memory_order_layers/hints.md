# Hints for ex03_memory_order_layers

## H1 — Concept Direction
Memory order controls WHEN writes become visible to other threads. `memory_order_release` on the writer ensures all prior writes are visible. `memory_order_acquire` on the reader ensures all subsequent reads see those writes. Together they create a "synchronizes-with" relationship.

## H2 — Names the Tool
Writer: `ready.store(true, std::memory_order_release)`. Reader: `while(!ready.load(std::memory_order_acquire))`. The release on writer synchronizes with acquire on reader, guaranteeing visibility of prior writes.

## H3 — Minimal Usage (Unrelated Context)
```cpp
std::atomic<int> data{0};
std::atomic<bool> ready{false};

// Writer:
data = 42;
ready.store(true, std::memory_order_release);  // Publish

// Reader:
while (!ready.load(std::memory_order_acquire)) {}  // Wait
use(data);  // Guaranteed to see 42
```
