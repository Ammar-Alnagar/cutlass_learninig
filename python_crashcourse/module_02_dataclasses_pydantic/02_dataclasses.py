"""
Scenario: Modeling a training run config with @dataclass.

Your ML platform stores training configurations as Python objects that are
passed between the experiment scheduler, the trainer, and the checkpoint
manager.  You need a lightweight, self-documenting config class that:
  - Has sensible defaults so researchers can override only what they need
  - Validates invariants at construction time (not silently at step 10000)
  - Is immutable once created so no component can accidentally mutate it

This file covers:
  - Section 1: Basic @dataclass — fields and defaults
  - Section 2: __post_init__ — validation at construction time
  - Section 3: frozen=True — immutable dataclasses
  - Section 4: Nested dataclasses and field(default_factory=...)
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Section 1: Basic @dataclass — fields and defaults
# ---------------------------------------------------------------------------
# WHY: @dataclass auto-generates __init__, __repr__, and __eq__ from the
# field annotations.  This eliminates ~10 lines of boilerplate per class
# and keeps the field definitions as the single source of truth.
#
# Fields without defaults MUST come before fields with defaults (same rule
# as regular function parameters).

# TODO: Define a dataclass `OptimizerConfig` with these fields:
#   - name: str                    (no default — required)
#   - learning_rate: float = 1e-3
#   - weight_decay: float = 0.0
#   - momentum: float = 0.9
# WHY: Separating optimizer config from model config makes it easy to swap
# optimizers without touching the model definition.
# (Replace the `pass` below with the full class definition.)
@dataclass
class OptimizerConfig:
    pass  # TODO: add fields


# TODO: Define a dataclass `DataConfig` with these fields:
#   - dataset_path: str            (required)
#   - batch_size: int = 32
#   - num_workers: int = 4
#   - shuffle: bool = True
@dataclass
class DataConfig:
    pass  # TODO: add fields


def check_1():
    try:
        opt = OptimizerConfig(name="adam")
        assert opt.name == "adam"
        assert abs(opt.learning_rate - 1e-3) < 1e-10, f"Expected lr=1e-3, got {opt.learning_rate}"
        assert opt.weight_decay == 0.0
        assert opt.momentum == 0.9

        opt2 = OptimizerConfig(name="sgd", learning_rate=1e-2, momentum=0.95)
        assert opt2.learning_rate == 1e-2
        assert opt2.momentum == 0.95

        data = DataConfig(dataset_path="/data/imagenet")
        assert data.dataset_path == "/data/imagenet"
        assert data.batch_size == 32
        assert data.shuffle is True

        # __repr__ should include field values
        assert "adam" in repr(opt)
        # __eq__ should compare field values
        assert OptimizerConfig(name="adam") == OptimizerConfig(name="adam")
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: __post_init__ — validation at construction time
# ---------------------------------------------------------------------------
# WHY: @dataclass calls __post_init__ immediately after __init__ completes.
# This is the right place to validate invariants — if the config is invalid,
# you want to fail fast at construction, not silently produce wrong results
# hours into a training run.

# TODO: Define a dataclass `TrainingConfig` with these fields:
#   - max_steps: int               (required)
#   - warmup_steps: int = 0
#   - gradient_clip: float = 1.0
#   - log_every: int = 100
# Add a __post_init__ method that raises ValueError if:
#   - max_steps <= 0
#   - warmup_steps < 0
#   - warmup_steps >= max_steps
#   - gradient_clip <= 0
# WHY: These are all invariants that would cause silent failures or crashes
# deep in the training loop if not caught early.
@dataclass
class TrainingConfig:
    pass  # TODO: add fields and __post_init__


def check_2():
    try:
        cfg = TrainingConfig(max_steps=10000, warmup_steps=500)
        assert cfg.max_steps == 10000
        assert cfg.warmup_steps == 500
        assert cfg.gradient_clip == 1.0

        # Invalid configs must raise ValueError
        for bad_kwargs, desc in [
            ({"max_steps": 0}, "max_steps=0 should raise"),
            ({"max_steps": -1}, "max_steps=-1 should raise"),
            ({"max_steps": 100, "warmup_steps": -1}, "warmup_steps=-1 should raise"),
            ({"max_steps": 100, "warmup_steps": 100}, "warmup_steps >= max_steps should raise"),
            ({"max_steps": 100, "gradient_clip": 0.0}, "gradient_clip=0 should raise"),
        ]:
            try:
                TrainingConfig(**bad_kwargs)
                assert False, f"{desc} but no error was raised"
            except ValueError:
                pass  # expected
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: frozen=True — immutable dataclasses
# ---------------------------------------------------------------------------
# WHY: A frozen dataclass raises FrozenInstanceError if you try to assign
# to any field after construction.  This is valuable for configs that are
# shared across threads or passed to multiple components — you want a
# guarantee that no component can silently mutate the config.
#
# Frozen dataclasses are also hashable (if all fields are hashable), so
# they can be used as dict keys or set members.

# TODO: Define a frozen dataclass `ModelConfig` with these fields:
#   - architecture: str            (required)
#   - hidden_dim: int = 768
#   - num_layers: int = 12
#   - num_heads: int = 12
#   - dropout: float = 0.1
# Use frozen=True in the @dataclass decorator.
@dataclass(frozen=True)
class ModelConfig:
    pass  # TODO: add fields


def check_3():
    try:
        cfg = ModelConfig(architecture="bert-base")
        assert cfg.architecture == "bert-base"
        assert cfg.hidden_dim == 768
        assert cfg.num_layers == 12

        # Attempting to mutate a frozen dataclass must raise
        try:
            cfg.hidden_dim = 512  # type: ignore
            assert False, "Should have raised FrozenInstanceError"
        except Exception as exc:
            # FrozenInstanceError is a subclass of AttributeError
            assert "frozen" in type(exc).__name__.lower() or isinstance(exc, AttributeError), (
                f"Expected FrozenInstanceError, got {type(exc)}"
            )

        # Frozen dataclasses are hashable
        cfg_set = {ModelConfig(architecture="bert"), ModelConfig(architecture="gpt")}
        assert len(cfg_set) == 2
        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: Nested dataclasses and field(default_factory=...)
# ---------------------------------------------------------------------------
# WHY: Real configs are hierarchical — a top-level ExperimentConfig contains
# a ModelConfig, a TrainingConfig, etc.  Nesting dataclasses keeps each
# concern separate and makes the config tree easy to serialise.
#
# `field(default_factory=list)` is required for mutable defaults like lists
# and dicts.  You CANNOT write `tags: List[str] = []` in a dataclass because
# Python would share the same list object across all instances — a classic
# mutable-default-argument bug.  `default_factory` creates a fresh list for
# each instance.

# TODO: Define a dataclass `ExperimentConfig` with these fields:
#   - name: str                                    (required)
#   - model: ModelConfig                           (required)
#   - optimizer: OptimizerConfig                   (required)
#   - tags: List[str] = field(default_factory=list)
#   - notes: Optional[str] = None
# WHY: Nesting the sub-configs keeps each concern isolated and makes it
# easy to swap out just the optimizer without touching the model config.
@dataclass
class ExperimentConfig:
    pass  # TODO: add fields


def check_4():
    try:
        model_cfg = ModelConfig(architecture="gpt2")
        opt_cfg = OptimizerConfig(name="adamw", learning_rate=3e-4)
        exp = ExperimentConfig(name="run_001", model=model_cfg, optimizer=opt_cfg)

        assert exp.name == "run_001"
        assert exp.model.architecture == "gpt2"
        assert exp.optimizer.name == "adamw"
        assert exp.tags == []
        assert exp.notes is None

        # Each instance must get its OWN list (not a shared default)
        exp2 = ExperimentConfig(name="run_002", model=model_cfg, optimizer=opt_cfg)
        exp.tags.append("baseline")
        assert exp2.tags == [], (
            "exp2.tags should be independent of exp.tags (default_factory)"
        )

        exp3 = ExperimentConfig(
            name="run_003",
            model=model_cfg,
            optimizer=opt_cfg,
            tags=["ablation", "fp16"],
            notes="Testing mixed precision",
        )
        assert "ablation" in exp3.tags
        assert exp3.notes == "Testing mixed precision"
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
