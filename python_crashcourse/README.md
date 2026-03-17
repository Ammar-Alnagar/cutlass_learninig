# Python Crash Course

A hands-on, self-contained Python learning directory for engineers who need to get up to speed fast on Python topics relevant to ML systems, infrastructure, and production engineering.

## Who This Is For

This course targets engineers with general programming experience (in any language) who need practical Python fluency in areas like tensor operations, async I/O, inter-process communication, and observability. If you already write code professionally but haven't worked deeply with Python's ML/systems ecosystem, this is for you.

No prior Python expertise required — but you should be comfortable with basic programming concepts (functions, loops, data structures).

---

## Install All Dependencies

Run this single command to install every library used across all seven modules:

```bash
pip install torch numpy pydantic fastapi uvicorn httpx structlog pyzmq opentelemetry-sdk prometheus_client
```

> **Note:** `asyncio`, `itertools`, `multiprocessing`, `time`, `timeit`, `cProfile`, and `logging` are part of the Python standard library — no installation needed.

---

## Modules

| Module | Name                                          | Description                                                                                                               |
| ------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 01     | `module_01_tensor_ops`                        | Tensor creation, reshaping, slicing, fancy indexing, broadcasting, einsum, and strides using PyTorch and NumPy            |
| 02     | `module_02_dataclasses_pydantic`              | Python type hints, `@dataclass`, Pydantic `BaseModel`, `TypedDict`, and `Protocol` for safe, self-documenting data models |
| 03     | `module_03_context_managers_async`            | Resource management with context managers and non-blocking async I/O with `asyncio`                                       |
| 04     | `module_04_generators_comprehensions`         | Lazy evaluation, generator pipelines, comprehensions, and streaming large datasets with `itertools`                       |
| 05     | `module_05_zeromq_multiprocessing`            | Inter-process communication with ZeroMQ socket patterns, `multiprocessing` primitives, and shared memory                  |
| 06     | `module_06_fastapi_error_logging`             | Building robust HTTP APIs with FastAPI, structured error handling, and production-grade logging with `structlog`          |
| 07     | `module_07_timing_benchmarking_observability` | Profiling, statistical benchmarking, OpenTelemetry tracing, and Prometheus metrics for ML workloads                       |

---

## Recommended Completion Order

Work through the modules in order — earlier modules introduce vocabulary (type hints, generators) that later modules build on (FastAPI, observability). That said, each module is self-contained enough to study independently if you have a specific gap to fill.

| Order | Module                                        | Estimated Time | Notes                                                                             |
| ----- | --------------------------------------------- | -------------- | --------------------------------------------------------------------------------- |
| 1     | `module_01_tensor_ops`                        | 3–4 hours      | Start here if you work with ML models; foundational for understanding data shapes |
| 2     | `module_02_dataclasses_pydantic`              | 2–3 hours      | Type safety patterns used in every subsequent module                              |
| 3     | `module_03_context_managers_async`            | 3–4 hours      | Async patterns are essential for serving and data ingestion                       |
| 4     | `module_04_generators_comprehensions`         | 2–3 hours      | Lazy evaluation is critical for large-scale data pipelines                        |
| 5     | `module_05_zeromq_multiprocessing`            | 4–5 hours      | Heaviest module; covers IPC patterns for high-throughput inference                |
| 6     | `module_06_fastapi_error_logging`             | 3–4 hours      | Builds on modules 02 and 03; covers production API patterns                       |
| 7     | `module_07_timing_benchmarking_observability` | 3–4 hours      | Capstone module; ties everything together with observability tooling              |

**Total estimated time: 20–27 hours** (spread across multiple sessions for best retention)

---

## How to Use This Course

Each module contains:

- A `README.md` with motivation, concept map, run commands, and help pointers
- Four exercise files (`01_basics.py` through `04_challenge.py`) with `# TODO:` markers and embedded self-checks
- A `solutions/` directory with complete, commented reference implementations

Run any exercise file directly:

```bash
python module_01_tensor_ops/01_basics.py
```

Self-checks run automatically and print `✓ Section N passed` or `✗ Section N failed` — no test runner needed.
