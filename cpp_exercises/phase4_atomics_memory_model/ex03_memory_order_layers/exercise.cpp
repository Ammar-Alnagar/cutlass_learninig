// CONCEPT: Memory order layers — acquire/release on flag+data
// FORMAT: SCAFFOLD
// TIME_TARGET: 20 min
// WHY_THIS_MATTERS: Layer 2 of atomics — memory_order controls visibility timing.
// CUDA_CONNECTION: Maps to __threadfence() and memory fence semantics.

#include <iostream>
#include <atomic>
#include <thread>
#include <vector>

// ==================== The Two Layers of Atomics ====================

// Layer 1: ATOMICITY (covered in ex01, ex02)
//   - std::atomic<T> prevents tearing
//   - Read/write is indivisible
//   - This is about WHAT operations are atomic

// Layer 2: MEMORY ORDER (this exercise)
//   - memory_order controls WHEN changes become visible
//   - Different orders have different performance/semantics
//   - This is about VISIBILITY TIMING

// ==================== Memory Order Options ====================

// memory_order_relaxed:
//   - Atomicity only, no ordering guarantees
//   - Fastest, but other threads may see operations out of order

// memory_order_acquire (load):
//   - All reads/writes AFTER this cannot be reordered before it
//   - "I'm acquiring — make sure I see all previous writes"

// memory_order_release (store):
//   - All reads/writes BEFORE this cannot be reordered after it
//   - "I'm releasing — make sure all my writes are visible"

// memory_order_acq_rel (read-modify-write):
//   - Both acquire and release

// memory_order_seq_cst (default):
//   - Sequentially consistent — all threads see same order
//   - Slowest, but easiest to reason about

// ==================== Example: Flag + Data Pattern ====================

std::atomic<int> data{0};
std::atomic<bool> ready{false};

// TODO 1: Writer thread with release semantics
// Write data, then set ready with memory_order_release
// This ensures data write is visible before ready becomes true
void writer_thread() {
    data.store(42, std::memory_order_relaxed);  // Write data (no ordering yet)
    // TODO: Add memory fence or use release store for ready
    // ready.store(true, std::memory_order_release);
    ready.store(true, std::memory_order_release);
    std::cout << "Writer: data=42, ready=true (release)\n";
}

// TODO 2: Reader thread with acquire semantics
// Wait for ready, then read data with memory_order_acquire
// This ensures we see the data write that happened before release
void reader_thread() {
    // TODO: Wait for ready with acquire load
    // while (!ready.load(std::memory_order_acquire)) { ... }
    while (!ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::cout << "Reader: ready=true, data=" << data.load(std::memory_order_relaxed) << "\n";
    // Because we used acquire on ready, we're guaranteed to see data=42
}

// ==================== Wrong Pattern: Relaxed on Both ====================

std::atomic<int> data_relaxed{0};
std::atomic<bool> ready_relaxed{false};

void writer_relaxed() {
    data_relaxed.store(42, std::memory_order_relaxed);
    ready_relaxed.store(true, std::memory_order_relaxed);  // BUG: No ordering!
    // With relaxed, the store to ready might become visible BEFORE data
}

void reader_relaxed() {
    while (!ready_relaxed.load(std::memory_order_relaxed)) {
        std::this_thread::yield();
    }
    // BUG: May see ready=true but data=0 (stale value)
    std::cout << "Reader (relaxed): data=" << data_relaxed.load(std::memory_order_relaxed) << "\n";
}

// ==================== Demonstration ====================

void demonstrate_acquire_release() {
    std::cout << "=== Acquire/Release Pattern ===\n";
    
    data = 0;
    ready = false;
    
    std::thread writer(writer_thread);
    std::thread reader(reader_thread);
    
    writer.join();
    reader.join();
    
    std::cout << "(Reader should see data=42)\n\n";
}

void demonstrate_relaxed_problem() {
    std::cout << "=== Relaxed Pattern (Problematic) ===\n";
    
    data_relaxed = 0;
    ready_relaxed = false;
    
    std::thread writer(writer_relaxed);
    std::thread reader(reader_relaxed);
    
    writer.join();
    reader.join();
    
    std::cout << "(Reader MAY see data=0 even though ready=true)\n";
    std::cout << "(This is the visibility problem that acquire/release fixes)\n\n";
}

int main() {
    demonstrate_acquire_release();
    demonstrate_relaxed_problem();
    
    std::cout << "=== KEY LEARNING ===\n";
    std::cout << "Layer 1 (Atomicity): std::atomic prevents tearing\n";
    std::cout << "Layer 2 (Ordering): memory_order controls visibility\n";
    std::cout << "\nAcquire/Release Pattern:\n";
    std::cout << "  Writer: data.store(42); ready.store(true, release);\n";
    std::cout << "  Reader: while(!ready.load(acquire)); read data;\n";
    std::cout << "  Result: Reader sees data=42 (release synchronizes with acquire)\n";
    std::cout << "\nMemory Order Summary:\n";
    std::cout << "  relaxed: atomicity only (fastest)\n";
    std::cout << "  acquire: loads can't move before (reader)\n";
    std::cout << "  release: stores can't move after (writer)\n";
    std::cout << "  acq_rel: both (read-modify-write)\n";
    std::cout << "  seq_cst: all threads see same order (default, slowest)\n";
    
    return 0;
}

// VERIFY: Expected output:
// === Acquire/Release Pattern ===
// Writer: data=42, ready=true (release)
// Reader: ready=true, data=42
// (Reader should see data=42)
//
// === Relaxed Pattern (Problematic) ===
// Reader (relaxed): data=42  (or 0 — non-deterministic!)
// (This is the visibility problem that acquire/release fixes)
