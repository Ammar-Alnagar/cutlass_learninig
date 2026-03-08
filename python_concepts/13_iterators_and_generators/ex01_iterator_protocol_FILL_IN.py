"""
Module 13 — Iterators & Generators
Exercise 01 — Iterator Protocol

WHAT YOU'RE BUILDING:
  The iterator protocol (__iter__, __next__) is how Python loops work.
  Understanding this helps you build custom data loaders, batch iterators,
  and lazy dataset readers — critical for ML pipelines.

OBJECTIVE:
  - Implement __iter__ and __next__ to make a class iterable
  - Understand StopIteration to signal end of iteration
  - Build a batch iterator for kernel benchmarking
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What happens when you call iter() on a list? What does it return?
# Q2: What exception signals the end of iteration in a for loop?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from typing import List, Tuple, Optional, Iterator
import time

# Common benchmark shapes
BENCHMARK_SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Implement a BatchIterator class that yields benchmark shapes.
#              This mimics a data loader yielding batches.
#              - __iter__: return self
#              - __next__: return next shape, raise StopIteration when done
# HINT: Track current index, increment on each __next__ call

class BatchIterator:
    """Iterator over benchmark shapes."""

    def __init__(self, shapes: List[Tuple[int, int, int]]):
        self.shapes = shapes
        self.index = 0

    def __iter__(self):
        # TODO: return self (iterator protocol)
        pass

    def __next__(self) -> Tuple[int, int, int]:
        # TODO: return next shape or raise StopIteration
        pass

# TODO [EASY]: Use the iterator in a for loop.
#              What gets printed? How many iterations?

def test_batch_iterator():
    """Test the BatchIterator."""
    iterator = BatchIterator(BENCHMARK_SHAPES)
    
    # TODO: iterate with for loop and print each shape
    for shape in iterator:
        print(f"Processing shape: {shape}")

# TODO [MEDIUM]: Implement an infinite data stream iterator.
#              This simulates continuous benchmarking or streaming data.
#              It cycles through shapes forever (or until stopped).
# HINT: Use modulo to cycle: self.shapes[self.index % len(self.shapes)]

class InfiniteDataStream:
    """Infinite iterator that cycles through shapes."""

    def __init__(self, shapes: List[Tuple[int, int, int]]):
        self.shapes = shapes
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, int, int]:
        # TODO: cycle through shapes infinitely
        pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How does Python's for loop use __iter__ and __next__ internally?
# C2: When would you use an infinite iterator vs a finite one?

if __name__ == "__main__":
    print("Testing BatchIterator...")
    test_batch_iterator()

    print("\nTesting InfiniteDataStream (first 5 items)...")
    infinite = InfiniteDataStream([(1, 1, 1), (2, 2, 2), (3, 3, 3)])
    for i, shape in enumerate(infinite):
        print(f"  {shape}")
        if i >= 4:
            break

    print("\nDone!")
