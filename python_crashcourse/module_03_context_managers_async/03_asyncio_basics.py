"""
Scenario: Concurrent model health checks for an inference server fleet.

Your inference platform runs dozens of model replicas.  Before routing
traffic, you need to health-check all of them concurrently.  Doing this
sequentially would take N * timeout seconds; doing it concurrently with
asyncio takes only max(timeout) seconds.

asyncio is Python's built-in cooperative multitasking framework.  A single
thread runs an event loop that switches between coroutines whenever one
awaits an I/O operation.  No threads, no GIL contention — just cooperative
yielding.

This file covers:
  - Section 1: async def / await — writing and calling coroutines
  - Section 2: asyncio.gather — running multiple coroutines concurrently
  - Section 3: asyncio.create_task — fire-and-forget tasks
  - Section 4: Timeouts with asyncio.wait_for
"""

import asyncio
import time


# ---------------------------------------------------------------------------
# Section 1: async def / await
# ---------------------------------------------------------------------------
# WHY: `async def` defines a coroutine function.  Calling it returns a
# coroutine object — it does NOT run the function.  You must `await` it
# (inside another coroutine) or pass it to asyncio.run() to execute it.
#
# `await` suspends the current coroutine and yields control back to the
# event loop until the awaited coroutine completes.  During that suspension,
# the event loop can run other coroutines — that's the concurrency.
#
# asyncio.sleep(n) is the async equivalent of time.sleep(n).  It suspends
# the coroutine for n seconds WITHOUT blocking the event loop thread.

# TODO: Implement `fake_health_check(host: str, latency: float) -> str`
# as an async function that:
#   - Awaits asyncio.sleep(latency) to simulate network I/O
#   - Returns f"{host}: OK"
# WHY: asyncio.sleep yields control so other health checks can run
# concurrently during the simulated network wait.
async def fake_health_check(host: str, latency: float) -> str:
    pass  # TODO: implement


# TODO: Implement `run_single_check(host: str) -> str` as an async function
# that calls fake_health_check(host, 0.05) and returns the result.
async def run_single_check(host: str) -> str:
    pass  # TODO: implement


def check_1():
    try:
        result = asyncio.run(run_single_check("model-01"))
        assert result == "model-01: OK", f"Expected 'model-01: OK', got {result!r}"
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: asyncio.gather — concurrent execution
# ---------------------------------------------------------------------------
# WHY: asyncio.gather(*coroutines) runs all coroutines concurrently and
# returns a list of their results in the same order as the input.
#
# Key difference from sequential await:
#   Sequential: total time = sum of all latencies
#   gather:     total time = max of all latencies
#
# This is the primary tool for "fan-out" patterns — sending the same
# request to multiple backends simultaneously.

HOSTS = ["model-01", "model-02", "model-03", "model-04"]
LATENCIES = [0.05, 0.03, 0.07, 0.02]  # seconds

# TODO: Implement `check_all_hosts() -> list` as an async function that:
#   - Uses asyncio.gather to run fake_health_check for all HOSTS/LATENCIES
#   - Returns the list of results
# WHY: gather runs all 4 checks concurrently; total time ≈ 0.07s (max),
# not 0.17s (sum).
async def check_all_hosts() -> list:
    pass  # TODO: implement


def check_2():
    try:
        start = time.perf_counter()
        results = asyncio.run(check_all_hosts())
        elapsed = time.perf_counter() - start

        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        for host, result in zip(HOSTS, results):
            assert result == f"{host}: OK", f"Expected '{host}: OK', got {result!r}"

        # Concurrent execution: should finish in ~max(latencies) not sum(latencies)
        assert elapsed < sum(LATENCIES), (
            f"Expected concurrent execution (< {sum(LATENCIES):.2f}s), "
            f"but took {elapsed:.2f}s — did you use gather?"
        )
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: asyncio.create_task — fire-and-forget tasks
# ---------------------------------------------------------------------------
# WHY: asyncio.gather waits for ALL coroutines to finish before returning.
# asyncio.create_task schedules a coroutine to run "in the background" and
# returns a Task object immediately.  You can await the task later, or let
# it run independently.
#
# Use create_task when:
#   - You want to start work now but collect results later
#   - You want to run background work while doing something else
#   - You need to cancel a task independently

# TODO: Implement `run_with_tasks() -> list` as an async function that:
#   - Creates tasks for all HOSTS/LATENCIES using asyncio.create_task
#   - Stores them in a list `tasks`
#   - Awaits all tasks using `await asyncio.gather(*tasks)` and returns results
# WHY: create_task starts the coroutines immediately (before the first await),
# while gather(*coros) only starts them when gather itself is awaited.
async def run_with_tasks() -> list:
    pass  # TODO: implement


def check_3():
    try:
        results = asyncio.run(run_with_tasks())
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        for host, result in zip(HOSTS, results):
            assert result == f"{host}: OK", f"Expected '{host}: OK', got {result!r}"
        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: Timeouts with asyncio.wait_for
# ---------------------------------------------------------------------------
# WHY: In production, you can't wait forever for a health check.  If a
# replica is down, it might never respond.  asyncio.wait_for(coro, timeout)
# cancels the coroutine and raises asyncio.TimeoutError if it doesn't
# complete within `timeout` seconds.
#
# This is the async equivalent of socket.settimeout() — but it works at
# the coroutine level, not the OS level.

# TODO: Implement `check_with_timeout(host: str, latency: float, timeout: float) -> str`
# as an async function that:
#   - Wraps fake_health_check(host, latency) in asyncio.wait_for(..., timeout=timeout)
#   - Returns the result if it completes in time
#   - Catches asyncio.TimeoutError and returns f"{host}: TIMEOUT"
# WHY: Returning a sentinel string instead of raising lets the caller
# distinguish between healthy, slow, and timed-out replicas.
async def check_with_timeout(host: str, latency: float, timeout: float) -> str:
    pass  # TODO: implement


# TODO: Implement `check_fleet_with_timeout() -> list` as an async function
# that uses asyncio.gather to run check_with_timeout for all HOSTS/LATENCIES
# with a timeout of 0.04 seconds.
# WHY: With timeout=0.04, hosts with latency > 0.04 should return "TIMEOUT".
async def check_fleet_with_timeout() -> list:
    pass  # TODO: implement


def check_4():
    try:
        results = asyncio.run(check_fleet_with_timeout())
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"

        # latency=0.05 > timeout=0.04 → TIMEOUT
        assert results[0] == "model-01: TIMEOUT", (
            f"model-01 (latency=0.05 > timeout=0.04) should be TIMEOUT, got {results[0]!r}"
        )
        # latency=0.03 < timeout=0.04 → OK
        assert results[1] == "model-02: OK", (
            f"model-02 (latency=0.03 < timeout=0.04) should be OK, got {results[1]!r}"
        )
        # latency=0.07 > timeout=0.04 → TIMEOUT
        assert results[2] == "model-03: TIMEOUT", (
            f"model-03 (latency=0.07 > timeout=0.04) should be TIMEOUT, got {results[2]!r}"
        )
        # latency=0.02 < timeout=0.04 → OK
        assert results[3] == "model-04: OK", (
            f"model-04 (latency=0.02 < timeout=0.04) should be OK, got {results[3]!r}"
        )
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


if __name__ == "__main__":
    check_4()
