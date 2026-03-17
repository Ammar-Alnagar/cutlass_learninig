"""
Scenario: Managing GPU memory and file handles in an inference server.

Your inference server loads model weights into GPU memory, runs inference,
and writes results to a log file.  If any step raises an exception, you
must guarantee that:
  - GPU memory is freed (no memory leak across requests)
  - The log file is closed (no file descriptor leak)
  - Partially-written log entries are flushed or discarded cleanly

Context managers are the standard Python mechanism for this guarantee.
The `with` statement calls __enter__ on entry and __exit__ on exit —
even if an exception is raised inside the block.

This file covers:
  - Section 1: Using built-in context managers (open, tempfile)
  - Section 2: Writing a class-based context manager (__enter__/__exit__)
  - Section 3: Exception handling in __exit__
  - Section 4: Nested context managers and multiple with targets
"""

import os
import tempfile
import time


# ---------------------------------------------------------------------------
# Section 1: Using built-in context managers
# ---------------------------------------------------------------------------
# WHY: `open()` is a context manager — it calls file.close() in __exit__
# even if an exception is raised inside the with block.  Without `with`,
# you'd need a try/finally block to guarantee the file is closed.
#
# The `with` statement desugars to:
#   ctx = open(path)
#   file = ctx.__enter__()
#   try:
#       <body>
#   except:
#       if not ctx.__exit__(*sys.exc_info()): raise
#   else:
#       ctx.__exit__(None, None, None)

# TODO: Use a `with open(...)` block to write the string "hello inference\n"
# to a temporary file, then read it back and assign the content to `content`.
# Use tempfile.mktemp() to get a temp path, and clean up with os.unlink().
# WHY: Always use `with open(...)` — never open() without a context manager
# in production code, because exceptions will leave file handles open.
content = ""  # stub — replace with your implementation

# (cleanup happens inside your with block or after)


def check_1():
    try:
        assert content.strip() == "hello inference", (
            f"Expected 'hello inference', got {content.strip()!r}"
        )
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: Class-based context manager (__enter__ / __exit__)
# ---------------------------------------------------------------------------
# WHY: When you need setup/teardown logic that's more complex than `open()`
# provides, implement __enter__ and __exit__ directly.  This is the pattern
# used by PyTorch's `torch.no_grad()`, `torch.autocast()`, and custom GPU
# memory managers.
#
# __enter__(self)                    → called on `with` entry; return value
#                                      is bound to the `as` variable
# __exit__(self, exc_type, exc_val, tb) → called on exit; return True to
#                                      suppress the exception, False/None to
#                                      re-raise it

# TODO: Implement `FakeGPUMemoryManager`, a context manager that simulates
# allocating and freeing GPU memory.
# - __init__(self, size_mb: int): store size_mb; set self.allocated = False
# - __enter__(self): set self.allocated = True; return self
# - __exit__(self, exc_type, exc_val, tb): set self.allocated = False; return False
# WHY: return False means "don't suppress exceptions" — the caller still sees
# any exception that occurred inside the with block.
class FakeGPUMemoryManager:
    def __init__(self, size_mb: int):
        pass  # TODO: implement

    def __enter__(self):
        pass  # TODO: implement

    def __exit__(self, exc_type, exc_val, tb):
        pass  # TODO: implement


def check_2():
    try:
        mgr = FakeGPUMemoryManager(512)
        assert mgr.allocated is False, "Should not be allocated before with block"

        with mgr as m:
            assert m is mgr, "__enter__ should return self"
            assert mgr.allocated is True, "Should be allocated inside with block"

        assert mgr.allocated is False, "Should be freed after with block"

        # __exit__ must be called even when an exception is raised
        mgr2 = FakeGPUMemoryManager(256)
        try:
            with mgr2:
                raise RuntimeError("simulated OOM")
        except RuntimeError:
            pass  # exception should propagate (return False in __exit__)
        assert mgr2.allocated is False, "Should be freed even after exception"

        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: Exception handling in __exit__
# ---------------------------------------------------------------------------
# WHY: __exit__ receives the exception info (exc_type, exc_val, traceback).
# Returning True suppresses the exception — the with block exits cleanly.
# This is useful for "transient error suppression" patterns, e.g., ignoring
# a FileNotFoundError when cleaning up a temp file that may already be gone.
#
# Be careful: suppressing exceptions silently can hide real bugs.  Only
# suppress specific, expected exception types.

# TODO: Implement `SuppressTransientError`, a context manager that suppresses
# a specific exception type passed to __init__.
# - __init__(self, exc_type): store exc_type
# - __enter__(self): return self
# - __exit__(self, exc_type, exc_val, tb):
#     return True if exc_type is not None and issubclass(exc_type, self.exc_type)
#     else return False
# WHY: This is essentially what contextlib.suppress() does internally.
class SuppressTransientError:
    def __init__(self, exc_type):
        pass  # TODO: implement

    def __enter__(self):
        pass  # TODO: implement

    def __exit__(self, exc_type, exc_val, tb):
        pass  # TODO: implement


def check_3():
    try:
        # Suppressed exception: no error propagates
        with SuppressTransientError(FileNotFoundError):
            raise FileNotFoundError("temp file already gone")
        # If we reach here, the exception was suppressed — correct

        # Non-matching exception: must propagate
        try:
            with SuppressTransientError(FileNotFoundError):
                raise ValueError("wrong type — should not be suppressed")
            assert False, "ValueError should have propagated"
        except ValueError:
            pass  # expected

        # No exception: __exit__ called with (None, None, None) — must not raise
        with SuppressTransientError(OSError):
            x = 1 + 1  # no exception

        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: Nested context managers and multiple with targets
# ---------------------------------------------------------------------------
# WHY: Real inference servers often need multiple resources simultaneously —
# a GPU memory allocation AND an open log file.  Python lets you nest context
# managers in a single `with` statement using commas, which is cleaner than
# nested `with` blocks and avoids the "pyramid of doom".
#
# `with A() as a, B() as b:` is exactly equivalent to:
#   with A() as a:
#       with B() as b:
#           ...
# Both A.__exit__ and B.__exit__ are called on exit, in reverse order.

# TODO: Using a single `with` statement with two targets, open two temp files
# simultaneously: one for writing ("w") and one for reading ("r").
# Write "model_output: 0.95\n" to the write file, then read it back from
# the read file.  Assign the read content to `log_line`.
# Steps:
#   1. Create two temp paths with tempfile.mktemp()
#   2. Write to the first path (outside the with, or use a separate with)
#   3. Open both files in a single `with f1 as w, open(path2) as r:` block
#      (write to w, read from r)
# WHY: Single-line multi-target with is idiomatic Python for acquiring
# multiple resources atomically.
log_line = ""  # stub

# (clean up temp files after your with block)


def check_4():
    try:
        assert "0.95" in log_line, (
            f"Expected '0.95' in log_line, got {log_line!r}"
        )
        assert "model_output" in log_line, (
            f"Expected 'model_output' in log_line, got {log_line!r}"
        )
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
