# Module 02 — Dataclasses, Pydantic, and Type Hints

## Motivation

ML systems are configuration-heavy: training runs have dozens of hyperparameters, inference servers accept structured request payloads, and data pipelines pass typed records between stages. Without a disciplined approach to data modeling, these configs become stringly-typed dicts that silently accept wrong values and explode at runtime — often hours into a training job. Python's type annotation system, `@dataclass`, and Pydantic give you three complementary tools: type hints document intent and enable static analysis; dataclasses add lightweight structure with zero boilerplate; Pydantic adds runtime validation so bad data is rejected at the boundary, not deep inside your model code. Mastering all three — plus `TypedDict` and `Protocol` for structural typing — lets you write ML configs and API schemas that are self-documenting, IDE-friendly, and safe.

---

## Concept Map

```
type annotations
  (int, str, float, bool — basic types)
        │
        ├── generics
        │     (List[int], Dict[str, float], Tuple[int, ...])
        │
        ├── Optional / Union
        │     (Optional[str] == Union[str, None])
        │
        ├── Literal
        │     (Literal["train", "eval"] — constrained string values)
        │
        └── TypeVar
              (T = TypeVar("T") — generic functions/classes)
                    │
                    ▼
            @dataclass
              (fields, defaults, __post_init__, frozen=True)
                    │
                    ▼
            Pydantic BaseModel
              (Field, validators, model_config,
               model_dump / model_validate)
                    │
                    ▼
            TypedDict + Protocol
              (structural subtyping — "duck typing with types")
```

---

## How to Run

```bash
# Exercise 1 — type annotation syntax
python python_crashcourse/module_02_dataclasses_pydantic/01_type_hints.py

# Exercise 2 — @dataclass
python python_crashcourse/module_02_dataclasses_pydantic/02_dataclasses.py

# Exercise 3 — Pydantic BaseModel
python python_crashcourse/module_02_dataclasses_pydantic/03_pydantic.py

# Exercise 4 — challenge: TypedDict + Protocol + full config schema
python python_crashcourse/module_02_dataclasses_pydantic/04_challenge.py
```

Each file prints `✓ Section N passed` or `✗ Section N failed: <reason>` — no test runner needed.

---

## What to Do If Stuck

### Solution files

| Exercise file       | Solution file                          |
| ------------------- | -------------------------------------- |
| `01_type_hints.py`  | `solutions/01_type_hints_solution.py`  |
| `02_dataclasses.py` | `solutions/02_dataclasses_solution.py` |
| `03_pydantic.py`    | `solutions/03_pydantic_solution.py`    |
| `04_challenge.py`   | `solutions/04_challenge_solution.py`   |

### Official documentation

- **Python type hints (typing module)**: <https://docs.python.org/3/library/typing.html>
- **dataclasses**: <https://docs.python.org/3/library/dataclasses.html>
- **Pydantic v2 docs**: <https://docs.pydantic.dev/latest/>
- **Pydantic Field**: <https://docs.pydantic.dev/latest/concepts/fields/>
- **Pydantic validators**: <https://docs.pydantic.dev/latest/concepts/validators/>
- **TypedDict**: <https://docs.python.org/3/library/typing.html#typing.TypedDict>
- **Protocol**: <https://docs.python.org/3/library/typing.html#typing.Protocol>
