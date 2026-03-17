# Module 03 — Context Managers and Async Basics

## Motivation

ML serving systems have two recurring resource management problems: they hold expensive resources (GPU memory, file handles, network connections) that must be released even when exceptions occur, and they need to handle many concurrent requests without blocking on I/O. Context managers solve the first problem — they guarantee cleanup code runs regardless of how a block exits. Async I/O solves the second — `asyncio` lets a single thread handle thousands of concurrent health checks, model downloads, or streaming responses by yielding control during I/O waits instead of blocking. Together, these two patterns are the foundation of every production Python inference server.

---

## Concept Map

```
context managers
  (resource acquisition + guaranteed cleanup)
        │
        ├── with statement
        │     (calls __enter__ on entry, __exit__ on exit)
        │
        ├── __enter__ / __exit__
        │     (__exit__ receives exc_type, exc_val, tb;
        │      return True to suppress the exception)
        │
        └── @contextmanager decorator
              (generator-based: yield = "with block runs here";
               exception suppression via try/except around yield)
                    │
                    ▼
            asyncio event loop
              (single-threaded cooperative multitasking)
                    │
                    ├── async def / await
                    │     (coroutine functions; await suspends until ready)
                    │
                    ├── asyncio.gather / asyncio.create_task
                    │     (run multiple coroutines concurrently)
                    │
                    ├── async context managers
                    │     (__aenter__ / __aexit__, async with)
                    │
                    └── async generators
                          (async def + yield; async for to consume)
```

---

## How to Run

```bash
# Exercise 1 — with statement, __enter__/__exit__
python python_crashcourse/module_03_context_managers_async/01_context_managers.py

# Exercise 2 — @contextmanager decorator, exception suppression
python python_crashcourse/module_03_context_managers_async/02_contextmanager_decorator.py

# Exercise 3 — asyncio basics: async def, await, gather, create_task
python python_crashcourse/module_03_context_managers_async/03_asyncio_basics.py

# Exercise 4 — challenge: async context managers, generators, cancellation
python python_crashcourse/module_03_context_managers_async/04_challenge.py
```

---

## What to Do If Stuck

| Exercise file                    | Solution file                                       |
| -------------------------------- | --------------------------------------------------- |
| `01_context_managers.py`         | `solutions/01_context_managers_solution.py`         |
| `02_contextmanager_decorator.py` | `solutions/02_contextmanager_decorator_solution.py` |
| `03_asyncio_basics.py`           | `solutions/03_asyncio_basics_solution.py`           |
| `04_challenge.py`                | `solutions/04_challenge_solution.py`                |

### Official documentation

- **contextlib**: <https://docs.python.org/3/library/contextlib.html>
- **asyncio**: <https://docs.python.org/3/library/asyncio.html>
- **asyncio tasks and coroutines**: <https://docs.python.org/3/library/asyncio-task.html>
- **async context managers**: <https://docs.python.org/3/reference/datamodel.html#asynchronous-context-managers>
- **async generators**: <https://docs.python.org/3/reference/expressions.html#asynchronous-generator-functions>
