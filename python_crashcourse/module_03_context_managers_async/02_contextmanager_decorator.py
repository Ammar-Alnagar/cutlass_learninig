"""
Scenario: Timing blocks and suppressing transient errors in a data loader.

Your data loader reads batches from disk.  You need to:
  1. Time each batch-loading step to detect slow reads
  2. Suppress transient I/O errors (e.g., NFS hiccups) without crashing
     the entire training run

The @contextmanager decorator lets you write context managers as generator
functions — much less boilerplate than a full class with __enter__/__exit__.
The pattern is: setup code → yield → teardown code.  The yield is where the
`with` block body runs.  Wrap the yield in try/finally for guaranteed cleanup,
or try/except to suppress specific exceptions.

This file covers:
  - Section 1: Basic @contextmanager — setup/teardown pattern
  - Section 2: @contextmanager with exception suppression
  - Section 3: Yielding a value from @contextmanager
  - Section 4: contextlib.suppress and contextlib.ExitStack
"""

import time
from contextlib import contextmanager, suppress, ExitStack
from typing import Iterator


# ---------------------------------------------------------------------------
# Section 1: Basic @contextmanager — setup/teardown
# ---------------------------------------------------------------------------
# WHY: The @contextmanager decorator turns a generator function into a
# context manager.  The code before `yield` runs as __enter__; the code
# after `yield` (in a finally block) runs as __exit__.
#
# Pattern:
#   @contextmanager
#   def my_cm():
#       # setup
#       try:
#           yield          # <-- with block runs here
#       finally:
#           # teardown (always runs)

# TODO: Implement `timer` as a @contextmanager that:
#   - Records the start time using time.perf_counter() before yield
#   - Records the end time after yield (in a finally block)
#   - Appends the elapsed time (end - start) to the `elapsed_times` list
#     (passed as a parameter)
# WHY: Collecting elapsed times in a list lets callers inspect timing
# results after the with block exits.
@contextmanager
def timer(elapsed_times: list) -> Iterator[None]:
    pass  # TODO: implement (setup → yield → teardown in finally)


def check_1():
    try:
        times: list = []
        with timer(times):
            time.sleep(0.05)  # simulate 50ms work

        assert len(times) == 1, f"Expected 1 elapsed time, got {len(times)}"
        assert times[0] >= 0.04, f"Expected >= 0.04s, got {times[0]:.4f}s"
        assert times[0] < 1.0, f"Expected < 1.0s, got {times[0]:.4f}s"

        # Multiple uses accumulate
        with timer(times):
            time.sleep(0.02)
        assert len(times) == 2

        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: @contextmanager with exception suppression
# ---------------------------------------------------------------------------
# WHY: To suppress a specific exception inside a @contextmanager, wrap the
# yield in try/except.  If you catch the exception and don't re-raise it,
# the with block exits cleanly (no exception propagates to the caller).
#
# This is the generator-based equivalent of returning True from __exit__.
#
# IMPORTANT: You must have exactly ONE yield in a @contextmanager function.
# If an exception is raised inside the with block, it is thrown INTO the
# generator at the yield point — that's how the generator can catch it.

# TODO: Implement `suppress_io_error` as a @contextmanager that:
#   - Suppresses OSError (and its subclasses, e.g., FileNotFoundError)
#   - Lets all other exceptions propagate normally
#   - Appends True to `suppressed_log` if an OSError was suppressed,
#     False if the block exited cleanly
# WHY: Logging whether an error was suppressed helps with debugging —
# you know a transient error occurred even though the run continued.
@contextmanager
def suppress_io_error(suppressed_log: list) -> Iterator[None]:
    pass  # TODO: implement


def check_2():
    try:
        log: list = []

        # OSError suppressed — no exception propagates
        with suppress_io_error(log):
            raise FileNotFoundError("NFS hiccup")
        assert log == [True], f"Expected [True], got {log}"

        # Clean exit — no exception
        with suppress_io_error(log):
            x = 1 + 1
        assert log == [True, False], f"Expected [True, False], got {log}"

        # Non-OSError must propagate
        try:
            with suppress_io_error(log):
                raise ValueError("not an IO error")
            assert False, "ValueError should have propagated"
        except ValueError:
            pass

        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: Yielding a value from @contextmanager
# ---------------------------------------------------------------------------
# WHY: `yield value` in a @contextmanager makes `value` available as the
# `as` target in the with statement.  This is how you expose a resource
# handle (e.g., a file, a connection, a stats dict) to the with block body.

# TODO: Implement `batch_loader` as a @contextmanager that:
#   - Creates a stats dict: {"batches_loaded": 0, "errors": 0}
#   - Yields the stats dict (so the caller can update it inside the with block)
#   - After the with block, prints a summary line:
#     f"Loaded {stats['batches_loaded']} batches, {stats['errors']} errors"
#   - Uses try/finally so the summary always prints even on exception
# WHY: Yielding a mutable dict lets the with block accumulate stats that
# the context manager can then summarise on exit.
@contextmanager
def batch_loader() -> Iterator[dict]:
    pass  # TODO: implement


def check_3():
    try:
        with batch_loader() as stats:
            assert isinstance(stats, dict), "stats should be a dict"
            assert stats["batches_loaded"] == 0
            stats["batches_loaded"] += 5
            stats["errors"] += 1

        # After the with block, stats should still be accessible
        assert stats["batches_loaded"] == 5
        assert stats["errors"] == 1

        # Summary should print even when an exception occurs
        try:
            with batch_loader() as stats2:
                stats2["batches_loaded"] = 3
                raise RuntimeError("disk full")
        except RuntimeError:
            pass
        assert stats2["batches_loaded"] == 3

        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: contextlib.suppress and contextlib.ExitStack
# ---------------------------------------------------------------------------
# WHY:
#   contextlib.suppress(ExcType) — stdlib equivalent of SuppressTransientError
#     from the previous file.  Use it instead of rolling your own.
#
#   contextlib.ExitStack — a dynamic context manager that lets you push an
#     arbitrary number of context managers onto a stack at runtime.  Useful
#     when you don't know at write-time how many resources you'll need
#     (e.g., opening N files where N is determined at runtime).

# TODO: Use contextlib.suppress to suppress a ZeroDivisionError.
# Inside the suppress block, compute 1 // 0 and assign the result to
# `result` if no exception, or leave `result` as the sentinel value -1.
# WHY: contextlib.suppress is cleaner than try/except for one-liners.
result = -1  # sentinel — replace with your implementation using suppress


# TODO: Use contextlib.ExitStack to open 3 temp files simultaneously.
# Create 3 temp paths, push each open() call onto the stack, write
# "file_N\n" to each, then read them back and collect the contents in
# `file_contents` (a list of 3 strings, stripped).
# WHY: ExitStack is the right tool when the number of resources is dynamic.
import tempfile, os
file_contents: list = []  # stub — replace with your ExitStack implementation


def check_4():
    try:
        assert result == -1, (
            f"ZeroDivisionError should have been suppressed; result should stay -1, got {result}"
        )

        assert len(file_contents) == 3, f"Expected 3 file contents, got {len(file_contents)}"
        for i, content in enumerate(file_contents):
            assert content == f"file_{i}", (
                f"Expected 'file_{i}', got {content!r}"
            )

        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
