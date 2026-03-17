"""
Challenge: Full ML Experiment Config Schema

Scenario: You are building the config layer for an ML experiment platform.
Configs arrive as JSON from a web UI, a CLI, or a YAML file.  They must be:
  - Structurally typed so IDEs and mypy can catch mistakes
  - Validated at the boundary so bad configs fail fast
  - Composable — sub-configs can be reused across experiment types
  - Protocol-based so the trainer can accept any config that has the right shape

This challenge integrates all Module 02 topics:
  - TypedDict for lightweight structural typing of raw dicts
  - Protocol for structural subtyping (duck typing with types)
  - @dataclass for immutable sub-configs
  - Pydantic BaseModel for boundary validation with Field and validators

This file covers:
  - Section 1: TypedDict — typed raw dicts for intermediate representations
  - Section 2: Protocol — structural subtyping for the trainer interface
  - Section 3: Full Pydantic schema — composing all sub-configs with validation
  - Section 4: Round-trip — JSON → model_validate → model_dump → JSON
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
    from pydantic import ConfigDict
except ImportError:
    raise SystemExit("Install pydantic v2: pip install 'pydantic>=2.0'")


# ---------------------------------------------------------------------------
# Section 1: TypedDict — typed raw dicts
# ---------------------------------------------------------------------------
# WHY: TypedDict lets you annotate a plain Python dict with specific key types.
# It's useful for intermediate representations — e.g., a raw dict parsed from
# YAML before it's validated into a Pydantic model.  Unlike BaseModel, a
# TypedDict is just a dict at runtime; the type info is only for static tools.
#
# Use TypedDict when:
#   - You need to pass data to code that expects a plain dict (e.g., json.dumps)
#   - You want lightweight typing without Pydantic's validation overhead
#   - You're annotating existing dict-based APIs you can't change

from typing import TypedDict

# TODO: Define a TypedDict `RawCheckpointDict` with these keys:
#   - path: str
#   - step: int
#   - metric: float
# WHY: This represents a raw checkpoint record as it comes out of a database
# query — a plain dict, but with known key types.
class RawCheckpointDict(TypedDict):
    pass  # TODO: add keys


# TODO: Write a function `parse_checkpoint` that:
#   - Takes a `RawCheckpointDict` and returns a tuple (str, int, float)
#     representing (path, step, metric)
# WHY: Extracting into a tuple makes it easy to unpack in calling code.
def parse_checkpoint(record: "RawCheckpointDict"):  # TODO: add return annotation
    pass  # stub


def check_1():
    try:
        record: RawCheckpointDict = {"path": "/ckpts/step_1000.pt", "step": 1000, "metric": 0.92}
        path, step, metric = parse_checkpoint(record)
        assert path == "/ckpts/step_1000.pt"
        assert step == 1000
        assert abs(metric - 0.92) < 1e-9
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: Protocol — structural subtyping
# ---------------------------------------------------------------------------
# WHY: A Protocol defines an interface by the methods/attributes an object
# must have — not by inheritance.  This is "duck typing with types": any
# class that has the right shape satisfies the Protocol, even if it doesn't
# explicitly inherit from it.
#
# Use @runtime_checkable to enable isinstance() checks at runtime.
# Without it, Protocol is only useful for static analysis.
#
# This is valuable for the trainer: it should accept any config object that
# has `learning_rate`, `max_steps`, and `batch_size` — regardless of whether
# it's a dataclass, a Pydantic model, or a plain object.

# TODO: Define a @runtime_checkable Protocol `TrainerConfigProtocol` with:
#   - learning_rate: float  (attribute)
#   - max_steps: int        (attribute)
#   - batch_size: int       (attribute)
# WHY: The trainer only needs these three values; it shouldn't care about
# the rest of the config hierarchy.
@runtime_checkable
class TrainerConfigProtocol(Protocol):
    pass  # TODO: add attribute annotations


# TODO: Define a plain class `MinimalConfig` (NOT a dataclass, NOT Pydantic)
# with these attributes set in __init__:
#   - learning_rate: float
#   - max_steps: int
#   - batch_size: int
# WHY: This demonstrates that Protocol works with any class — no inheritance needed.
class MinimalConfig:
    pass  # TODO: implement __init__


def check_2():
    try:
        cfg = MinimalConfig()
        # MinimalConfig must satisfy TrainerConfigProtocol without inheriting from it
        assert isinstance(cfg, TrainerConfigProtocol), (
            "MinimalConfig should satisfy TrainerConfigProtocol"
        )
        assert cfg.learning_rate == 1e-3
        assert cfg.max_steps == 1000
        assert cfg.batch_size == 32

        # A dict does NOT satisfy the protocol (no .learning_rate attribute)
        assert not isinstance({"learning_rate": 1e-3}, TrainerConfigProtocol), (
            "A plain dict should NOT satisfy TrainerConfigProtocol"
        )
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: Full Pydantic schema
# ---------------------------------------------------------------------------
# WHY: Now we compose everything into a production-grade config schema.
# The top-level ExperimentSchema uses nested Pydantic models for each
# sub-config, Field() for constraints, and validators for cross-field rules.

# TODO: Define `OptimizerSchema` (Pydantic BaseModel) with:
#   - name: Literal["adam", "sgd", "adamw"] = "adamw"
#   - lr: float = Field(1e-3, gt=0, description="Learning rate")
#   - weight_decay: float = Field(0.0, ge=0.0)
class OptimizerSchema(BaseModel):
    pass  # TODO: add fields


# TODO: Define `TrainSchema` (Pydantic BaseModel) with:
#   - max_steps: int = Field(..., gt=0)
#   - warmup_steps: int = Field(0, ge=0)
#   - batch_size: int = Field(32, ge=1, le=4096)
# Add a @model_validator(mode="after") that raises ValueError if
# warmup_steps >= max_steps.
class TrainSchema(BaseModel):
    pass  # TODO: add fields and validator


# TODO: Define `ExperimentSchema` (Pydantic BaseModel) with:
#   - model_config = ConfigDict(extra="forbid")
#   - name: str = Field(..., min_length=1)
#   - optimizer: OptimizerSchema = Field(default_factory=OptimizerSchema)
#   - training: TrainSchema
#   - tags: List[str] = Field(default_factory=list)
#   - description: Optional[str] = None
class ExperimentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pass  # TODO: add fields


def check_3():
    try:
        exp = ExperimentSchema(
            name="baseline",
            training=TrainSchema(max_steps=10000, warmup_steps=500),
        )
        assert exp.name == "baseline"
        assert exp.optimizer.name == "adamw"
        assert exp.training.max_steps == 10000
        assert exp.tags == []

        # Cross-field validator: warmup_steps >= max_steps must raise
        try:
            ExperimentSchema(
                name="bad",
                training=TrainSchema(max_steps=100, warmup_steps=100),
            )
            assert False, "warmup_steps >= max_steps should raise"
        except ValidationError:
            pass

        # extra="forbid": unknown top-level keys must raise
        try:
            ExperimentSchema(
                name="bad",
                training=TrainSchema(max_steps=100),
                unknown_field="oops",  # type: ignore
            )
            assert False, "Extra field should raise"
        except ValidationError:
            pass

        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: Round-trip — JSON → model_validate → model_dump → JSON
# ---------------------------------------------------------------------------
# WHY: In production, configs arrive as JSON strings (from HTTP requests,
# config files, or message queues) and must be serialised back to JSON for
# logging, storage, or forwarding.  Pydantic's model_validate / model_dump
# handle this round-trip cleanly.

RAW_JSON = """{
  "name": "experiment_42",
  "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.01},
  "training": {"max_steps": 5000, "warmup_steps": 200, "batch_size": 64},
  "tags": ["fp16", "large-batch"],
  "description": "Baseline run with Adam"
}"""

# TODO: Parse RAW_JSON into a Python dict using json.loads, then validate it
# into an ExperimentSchema using model_validate.
# Assign to `experiment`.
experiment = None  # stub

# TODO: Serialise `experiment` back to a dict using model_dump().
# Assign to `dumped`.
dumped = {}  # stub

# TODO: Re-serialise `dumped` to a JSON string using json.dumps.
# Assign to `round_tripped_json`.
round_tripped_json = ""  # stub


def check_4():
    try:
        assert experiment is not None, "experiment should not be None"
        assert isinstance(experiment, ExperimentSchema), (
            f"Expected ExperimentSchema, got {type(experiment)}"
        )
        assert experiment.name == "experiment_42"
        assert experiment.optimizer.name == "adam"
        assert experiment.training.batch_size == 64
        assert "fp16" in experiment.tags
        assert experiment.description == "Baseline run with Adam"

        assert isinstance(dumped, dict), f"Expected dict, got {type(dumped)}"
        assert dumped["name"] == "experiment_42"
        assert dumped["training"]["max_steps"] == 5000

        # Round-tripped JSON must be valid JSON
        parsed_back = json.loads(round_tripped_json)
        assert parsed_back["name"] == "experiment_42"
        assert parsed_back["optimizer"]["name"] == "adam"
        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
