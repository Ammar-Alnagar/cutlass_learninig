# Module 04 — Generators, Comprehensions, and Streaming

## Motivation

ML data pipelines routinely process datasets that are too large to fit in memory — billions of tokens, terabytes of images, or continuous streams from a message queue. Loading everything into a list first is not an option. Generators solve this by producing values one at a time, on demand, without materialising the entire sequence. Comprehensions give you a concise, readable syntax for building lists, dicts, and sets from iterables. Together with `itertools` and custom `__iter__`/`__next__` classes, these tools let you build lazy, composable data pipelines that process arbitrarily large datasets with constant memory overhead — the same pattern used by PyTorch's `DataLoader`, HuggingFace's `datasets`, and every streaming inference server.

---

## Concept Map

```
yield mechanics
  (generator function: calling it returns a generator object;
   each next() call runs until the next yield)
        │
        ├── generator state
        │     (local variables persist between next() calls)
        │
        ├── send()
        │     (send a value INTO the generator at the yield point)
        │
        └── yield from
              (delegate to a sub-generator; propagates send/throw/close)
                    │
                    ▼
            generator pipelines
              (chain generators: source → transform → sink)
                    │
                    ▼
            comprehensions
              (list / dict / set — eager, returns a container)
                    │
                    ├── nested comprehensions
                    │     ([[...] for ...] for ...])
                    │
                    └── generator expressions
                          ((...) — lazy, returns a generator object)
                                │
                                ▼
                          itertools toolkit
                            (chain, islice, groupby, product,
                             accumulate, takewhile, dropwhile)
                                │
                                ▼
                          custom __iter__ / __next__
                            (dataset class that is its own iterator)
```

---

## How to Run

```bash
# Exercise 1 — yield mechanics, generator state, send()
python python_crashcourse/module_04_generators_comprehensions/01_generators.py

# Exercise 2 — yield from, generator delegation, pipelines
python python_crashcourse/module_04_generators_comprehensions/02_yield_from.py

# Exercise 3 — comprehensions, generator expressions, itertools
python python_crashcourse/module_04_generators_comprehensions/03_comprehensions.py

# Exercise 4 — challenge: lazy streaming data pipeline
python python_crashcourse/module_04_generators_comprehensions/04_challenge.py
```

---

## What to Do If Stuck

| Exercise file          | Solution file                             |
| ---------------------- | ----------------------------------------- |
| `01_generators.py`     | `solutions/01_generators_solution.py`     |
| `02_yield_from.py`     | `solutions/02_yield_from_solution.py`     |
| `03_comprehensions.py` | `solutions/03_comprehensions_solution.py` |
| `04_challenge.py`      | `solutions/04_challenge_solution.py`      |

### Official documentation

- **Generator functions**: <https://docs.python.org/3/reference/expressions.html#yield-expressions>
- **itertools**: <https://docs.python.org/3/library/itertools.html>
- **List comprehensions**: <https://docs.python.org/3/tutorial/datastructures.html#list-
