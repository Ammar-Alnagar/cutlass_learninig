"""
Scenario: Annotating an ML config loader.

Your team's training framework reads hyperparameters from a YAML file and
passes them around as Python dicts.  You're tasked with adding type
annotations so that IDEs can autocomplete, mypy can catch mistakes, and
new engineers can understand the expected types at a glance — without
changing any runtime behaviour.

This file covers:
  - Section 1: Basic types — int, float, str, bool, None
  - Section 2: Generics — List, Dict, Tuple, Set
  - Section 3: Optional and Union — nullable fields and multi-type values
  - Section 4: Literal and TypeVar — constrained values and generic functions
"""

from typing import Dict, List, Literal, Optional, Set, Tuple, TypeVar, Union


# ---------------------------------------------------------------------------
# Section 1: Basic type annotations
# ---------------------------------------------------------------------------
# WHY: Type annotations are purely informational at runtime — Python does NOT
# enforce them.  Their value is for static analysis tools (mypy, pyright) and
# for human readers.  Annotating function signatures is the highest-leverage
# place to start because it documents the contract at the call boundary.

# TODO: Annotate the function `parse_learning_rate` so that:
#   - `value` is annotated as str (raw string from YAML)
#   - the return type is float
# Then implement it: convert value to float and return it.
# WHY: Explicit return type annotations prevent callers from accidentally
# treating the result as a string.
def parse_learning_rate(value) -> None:  # stub — fix annotation and implement
    pass


# TODO: Annotate `is_valid_batch_size` so that:
#   - `n` is annotated as int
#   - return type is bool
# Implement: return True if n > 0 and n is a power of 2 (n & (n-1) == 0).
# WHY: Batch sizes must be positive powers of 2 for most GPU kernels.
def is_valid_batch_size(n) -> None:  # stub
    pass


def check_1():
    try:
        lr = parse_learning_rate("3e-4")
        assert isinstance(lr, float), f"Expected float, got {type(lr)}"
        assert abs(lr - 3e-4) < 1e-10, f"Expected 3e-4, got {lr}"

        assert is_valid_batch_size(32) is True, "32 is a valid batch size"
        assert is_valid_batch_size(0) is False, "0 is not a valid batch size"
        assert is_valid_batch_size(33) is False, "33 is not a power of 2"
        assert is_valid_batch_size(128) is True, "128 is a valid batch size"
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: Generics — List, Dict, Tuple, Set
# ---------------------------------------------------------------------------
# WHY: Generic types let you annotate the *contents* of a container, not just
# the container itself.  `List[float]` is far more informative than `list`
# because it tells the reader (and mypy) what's inside.
#
# Key generics:
#   List[T]           — ordered, mutable sequence of T
#   Dict[K, V]        — mapping from K to V
#   Tuple[T1, T2]     — fixed-length heterogeneous sequence
#   Tuple[T, ...]     — variable-length homogeneous sequence
#   Set[T]            — unordered collection of unique T

# TODO: Annotate `summarise_losses` so that:
#   - `losses` is List[float]
#   - return type is Dict[str, float]
# Implement: return {"min": min(losses), "max": max(losses), "mean": sum(losses)/len(losses)}.
# WHY: Returning a typed dict makes it clear what keys callers can expect.
def summarise_losses(losses) -> None:  # stub
    pass


# TODO: Annotate `layer_config` so that:
#   - `name` is str
#   - `in_out` is Tuple[int, int]  (input_dim, output_dim)
#   - `tags` is Set[str]
#   - return type is Dict[str, object]
# Implement: return {"name": name, "in": in_out[0], "out": in_out[1], "tags": tags}.
def layer_config(name, in_out, tags) -> None:  # stub
    pass


def check_2():
    try:
        summary = summarise_losses([1.0, 2.0, 3.0, 4.0])
        assert isinstance(summary, dict), f"Expected dict, got {type(summary)}"
        assert abs(summary["min"] - 1.0) < 1e-9
        assert abs(summary["max"] - 4.0) < 1e-9
        assert abs(summary["mean"] - 2.5) < 1e-9

        cfg = layer_config("linear", (512, 256), {"projection", "frozen"})
        assert cfg["name"] == "linear"
        assert cfg["in"] == 512
        assert cfg["out"] == 256
        assert "projection" in cfg["tags"]
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: Optional and Union
# ---------------------------------------------------------------------------
# WHY: Many config fields are optional — a learning rate schedule might not
# be specified, or a checkpoint path might be None.  `Optional[T]` is
# shorthand for `Union[T, None]` and signals "this might be absent".
#
# `Union[A, B]` means "either A or B" — useful when a field can hold
# different types depending on context (e.g., a seed that can be an int
# or the string "random").

# TODO: Annotate `get_checkpoint_path` so that:
#   - `base_dir` is str
#   - `step` is Optional[int]  (None means "use latest")
#   - return type is Optional[str]
# Implement: if step is None return None, else return f"{base_dir}/ckpt_{step}.pt".
def get_checkpoint_path(base_dir, step) -> None:  # stub
    pass


# TODO: Annotate `parse_seed` so that:
#   - `value` is Union[int, str]
#   - return type is int
# Implement: if value is already an int return it; if it's the string "random"
# return 42; otherwise raise ValueError(f"Invalid seed: {value}").
def parse_seed(value) -> None:  # stub
    pass


def check_3():
    try:
        assert get_checkpoint_path("/models", 1000) == "/models/ckpt_1000.pt"
        assert get_checkpoint_path("/models", None) is None

        assert parse_seed(7) == 7
        assert parse_seed("random") == 42
        try:
            parse_seed("bad")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: Literal and TypeVar
# ---------------------------------------------------------------------------
# WHY: `Literal` constrains a type to a fixed set of values — ideal for
# config fields like "optimizer" that must be one of a known list.  Static
# analysers will flag any call that passes a value outside the Literal set.
#
# `TypeVar` enables generic functions that preserve the type of their input.
# Without TypeVar, a function that "returns its input unchanged" would have
# to be annotated as returning `object`, losing all type information.

# TODO: Annotate `get_optimizer_lr` so that:
#   - `optimizer` is Literal["adam", "sgd", "adamw"]
#   - return type is float
# Implement: return 1e-3 for "adam"/"adamw", 1e-2 for "sgd".
def get_optimizer_lr(optimizer) -> None:  # stub
    pass


# TODO: Define a TypeVar T and annotate `first_or_default` so that:
#   - T is a TypeVar (define it above the function)
#   - `items` is List[T]
#   - `default` is T
#   - return type is T
# Implement: return items[0] if items else default.
# WHY: With TypeVar, mypy knows that first_or_default([1,2,3], 0) returns int,
# not object — the return type matches the element type of the list.
T = TypeVar("T")  # already defined for you — use it in your annotation


def first_or_default(items, default):  # TODO: add annotations
    pass  # stub


def check_4():
    try:
        assert abs(get_optimizer_lr("adam") - 1e-3) < 1e-10
        assert abs(get_optimizer_lr("sgd") - 1e-2) < 1e-10
        assert abs(get_optimizer_lr("adamw") - 1e-3) < 1e-10

        assert first_or_default([10, 20, 30], 0) == 10
        assert first_or_default([], 99) == 99
        assert first_or_default(["a", "b"], "z") == "a"
        assert first_or_default([], "default") == "default"
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
