"""
Challenge: Async Inference Client with Connection Pooling and Streaming

Scenario: You are building an async client for a remote inference service.
The client must:
  1. Manage a pool of connections using an async context manager
  2. Send requests with a timeout and handle cancellation gracefully
  3. Stream results token-by-token using an async generator
  4. Compose all three patterns in a single end-to-end pipeline

This challenge integrates all Module 03 topics:
  - Class-based context managers (__enter__/__exit__)
  - @contextmanager decorator
  - asyncio.gather, create_task, wait_for
  - Async context managers (__aenter__/__aexit__)
  - Async generators (async def + yield)
  - Cancellation (task.cancel(), asyncio.CancelledError)

This file covers:
  - Section 1: Async context manager — connection pool lifecycle
  - Section 2: Async generator — streaming token-by-token results
  - Section 3: Timeout + cancellation — robust request handling
  - Section 4: End-to-end pipeline — pool + streaming + timeout
"""

import asyncio
from typing import AsyncIterator


# ---------------------------------------------------------------------------
# Section 1: Async context manager — connection pool lifecycle
# ---------------------------------------------------------------------------
# WHY: An async context manager is like a regular context manager but its
# __aenter__ and __aexit__ methods are coroutines (async def).  Use it when
# setup/teardown involves async I/O — e.g., opening a network connection,
# acquiring a semaphore, or initialising a database pool.
#
# Syntax:
#   async with MyAsyncCM() as resource:
#       ...
#
# This calls `await cm.__aenter__()` on entry and `await cm.__aexit__(...)` on exit.

# TODO: Implement `ConnectionPool` as an async context manager class with:
#   - __init__(self, max_connections: int): store max_connections;
#     set self.active = 0; set self.open = False
#   - async __aenter__(self): set self.open = True; return self
#   - async __aexit__(self, *args): set self.open = False; set self.active = 0
#   - async def acquire(self) -> int:
#       if self.active >= self.max_connections: raise RuntimeError("pool exhausted")
#       self.active += 1; return self.active
#   - async def release(self): self.active = max(0, self.active - 1)
# WHY: The async context manager guarantees the pool is closed (open=False)
# even if an exception occurs inside the async with block.
class ConnectionPool:
    def __init__(self, max_connections: int):
        pass  # TODO: implement

    async def __aenter__(self):
        pass  # TODO: implement

    async def __aexit__(self, *args):
        pass  # TODO: implement

    async def acquire(self) -> int:
        pass  # TODO: implement

    async def release(self):
        pass  # TODO: implement


def check_1():
    async def _test():
        pool = ConnectionPool(max_connections=3)
        assert pool.open is False

        async with pool as p:
            assert p is pool
            assert pool.open is True

            conn_id = await pool.acquire()
            assert conn_id == 1
            assert pool.active == 1

            conn_id2 = await pool.acquire()
            assert conn_id2 == 2

            await pool.release()
            assert pool.active == 1

        assert pool.open is False
        assert pool.active == 0

        # Pool exhaustion
        async with ConnectionPool(max_connections=1) as small_pool:
            await small_pool.acquire()
            try:
                await small_pool.acquire()
                assert False, "Should raise RuntimeError when pool exhausted"
            except RuntimeError:
                pass

    try:
        asyncio.run(_test())
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: Async generator — streaming token-by-token
# ---------------------------------------------------------------------------
# WHY: An async generator is an `async def` function that contains `yield`.
# It produces values asynchronously — each `yield` suspends the generator
# and returns a value to the consumer.  Consume with `async for`.
#
# This is the pattern used by streaming LLM APIs (OpenAI, Anthropic) where
# tokens arrive one at a time over a long-lived HTTP connection.

# TODO: Implement `stream_tokens(prompt: str, num_tokens: int)` as an async
# generator that:
#   - For each token index i in range(num_tokens):
#       - Awaits asyncio.sleep(0.01) to simulate network latency per token
#       - Yields f"token_{i}"
# WHY: The await inside the generator lets other coroutines run between
# tokens — the event loop isn't blocked while waiting for each token.
async def stream_tokens(prompt: str, num_tokens: int) -> AsyncIterator[str]:
    pass  # TODO: implement (use `yield` to make this an async generator)


# TODO: Implement `collect_stream(prompt: str, num_tokens: int) -> list` as
# an async function that:
#   - Uses `async for token in stream_tokens(...)` to collect all tokens
#   - Returns the list of tokens
async def collect_stream(prompt: str, num_tokens: int) -> list:
    pass  # TODO: implement


def check_2():
    async def _test():
        tokens = await collect_stream("hello", 5)
        assert tokens == ["token_0", "token_1", "token_2", "token_3", "token_4"], (
            f"Expected 5 tokens, got {tokens}"
        )

        # Empty stream
        empty = await collect_stream("hi", 0)
        assert empty == [], f"Expected [], got {empty}"

    try:
        asyncio.run(_test())
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: Timeout + cancellation
# ---------------------------------------------------------------------------
# WHY: In production, a streaming response might stall mid-stream.  You need
# to cancel the stream after a deadline and handle asyncio.CancelledError
# gracefully — logging the partial result rather than crashing.
#
# asyncio.CancelledError is raised at the next `await` point when a task is
# cancelled.  You can catch it to do cleanup, but you MUST re-raise it (or
# let it propagate) so the event loop knows the task is done.

# TODO: Implement `stream_with_timeout(prompt: str, num_tokens: int, timeout: float) -> list`
# as an async function that:
#   - Collects tokens from stream_tokens into a list, one at a time
#   - Wraps the entire collection in asyncio.wait_for(..., timeout=timeout)
#   - If asyncio.TimeoutError is raised, returns whatever tokens were collected so far
#     (hint: collect into a list outside the wait_for, or catch TimeoutError)
# WHY: Returning partial results is better than returning nothing — the caller
# can decide whether partial output is useful.
async def stream_with_timeout(prompt: str, num_tokens: int, timeout: float) -> list:
    pass  # TODO: implement


def check_3():
    async def _test():
        # 5 tokens * 0.01s each = 0.05s total; timeout=1.0 → all tokens
        tokens = await stream_with_timeout("hi", 5, timeout=1.0)
        assert tokens == ["token_0", "token_1", "token_2", "token_3", "token_4"], (
            f"Expected all 5 tokens, got {tokens}"
        )

        # timeout=0.025s → only ~2 tokens before timeout
        partial = await stream_with_timeout("hi", 10, timeout=0.025)
        assert len(partial) < 10, (
            f"Expected partial result (< 10 tokens), got {len(partial)}"
        )
        assert len(partial) >= 1, (
            f"Expected at least 1 token before timeout, got {len(partial)}"
        )

    try:
        asyncio.run(_test())
        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: End-to-end pipeline
# ---------------------------------------------------------------------------
# WHY: Real inference clients combine all three patterns:
#   1. Acquire a connection from the pool (async context manager)
#   2. Stream tokens from the remote model (async generator)
#   3. Apply a per-request timeout (asyncio.wait_for)
#
# This section wires them together into a single `inference_request` function.

# TODO: Implement `inference_request(pool: ConnectionPool, prompt: str,
#                                    num_tokens: int, timeout: float) -> dict`
# as an async function that:
#   - Acquires a connection from the pool using `await pool.acquire()`
#   - Collects tokens using stream_with_timeout(prompt, num_tokens, timeout)
#   - Releases the connection using `await pool.release()` in a finally block
#   - Returns {"conn_id": conn_id, "tokens": tokens, "prompt": prompt}
# WHY: The finally block guarantees the connection is released even if
# stream_with_timeout raises an exception.
async def inference_request(
    pool: ConnectionPool, prompt: str, num_tokens: int, timeout: float
) -> dict:
    pass  # TODO: implement


def check_4():
    async def _test():
        async with ConnectionPool(max_connections=2) as pool:
            # Single request
            result = await inference_request(pool, "hello", 3, timeout=1.0)
            assert result["prompt"] == "hello"
            assert result["tokens"] == ["token_0", "token_1", "token_2"]
            assert result["conn_id"] == 1
            assert pool.active == 0, "Connection should be released after request"

            # Two concurrent requests
            r1, r2 = await asyncio.gather(
                inference_request(pool, "a", 2, timeout=1.0),
                inference_request(pool, "b", 2, timeout=1.0),
            )
            assert r1["tokens"] == ["token_0", "token_1"]
            assert r2["tokens"] == ["token_0", "token_1"]
            assert pool.active == 0, "Both connections should be released"

    try:
        asyncio.run(_test())
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


if __name__ == "__main__":
    check_4()
