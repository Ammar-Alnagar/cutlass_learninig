"""
Scenario: Validating an inference request payload.

Your inference server receives JSON payloads over HTTP.  Clients can send
malformed data — wrong types, out-of-range values, missing required fields.
You need a validation layer that:
  - Rejects bad requests with a clear error message before they reach the model
  - Coerces compatible types (e.g., "32" → 32) where safe
  - Documents the expected schema so API consumers know what to send

Pydantic v2 BaseModel is the standard tool for this in Python ML serving.

This file covers:
  - Section 1: Basic BaseModel — fields, types, and automatic validation
  - Section 2: Field — constraints, aliases, and descriptions
  - Section 3: Validators — custom field and model-level validation
  - Section 4: model_config, model_dump, and model_validate
"""

from typing import List, Optional

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
    from pydantic import ConfigDict
except ImportError:
    raise SystemExit("Install pydantic v2: pip install 'pydantic>=2.0'")


# ---------------------------------------------------------------------------
# Section 1: Basic BaseModel
# ---------------------------------------------------------------------------
# WHY: Pydantic BaseModel auto-generates __init__ with runtime type checking.
# Unlike @dataclass, Pydantic actually ENFORCES types at runtime — passing
# a string where an int is expected raises ValidationError, not a silent bug.
# Pydantic also coerces compatible types: "42" → 42 for an int field.

# TODO: Define a Pydantic model `InferenceRequest` with these fields:
#   - model_name: str              (required)
#   - inputs: List[float]          (required)
#   - max_tokens: int = 128
#   - temperature: float = 1.0
# WHY: This mirrors a typical LLM inference API request body.
class InferenceRequest(BaseModel):
    pass  # TODO: add fields


# TODO: Define a Pydantic model `InferenceResponse` with these fields:
#   - model_name: str
#   - outputs: List[float]
#   - tokens_generated: int
#   - latency_ms: float
class InferenceResponse(BaseModel):
    pass  # TODO: add fields


def check_1():
    try:
        req = InferenceRequest(model_name="gpt2", inputs=[1.0, 2.0, 3.0])
        assert req.model_name == "gpt2"
        assert req.inputs == [1.0, 2.0, 3.0]
        assert req.max_tokens == 128
        assert req.temperature == 1.0

        # Pydantic coerces "64" → 64 for int fields
        req2 = InferenceRequest(model_name="bert", inputs=[0.5], max_tokens="64")  # type: ignore
        assert req2.max_tokens == 64, f"Expected 64, got {req2.max_tokens}"

        # Missing required field must raise ValidationError
        try:
            InferenceRequest(inputs=[1.0])  # type: ignore
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass

        resp = InferenceResponse(
            model_name="gpt2", outputs=[0.1, 0.9], tokens_generated=5, latency_ms=12.3
        )
        assert resp.latency_ms == 12.3
        print("✓ Section 1 passed")
    except AssertionError as e:
        print(f"✗ Section 1 failed: {e}")


check_1()


# ---------------------------------------------------------------------------
# Section 2: Field — constraints, aliases, and descriptions
# ---------------------------------------------------------------------------
# WHY: `Field(...)` lets you attach metadata to a field:
#   - gt/ge/lt/le: numeric constraints (greater-than, less-than, etc.)
#   - min_length/max_length: string/list length constraints
#   - alias: accept a different key name in the input JSON
#   - description: documents the field for OpenAPI schema generation
#
# Pydantic enforces gt/ge/lt/le at validation time — no need for manual
# if-statements in __post_init__.

# TODO: Define `ConstrainedRequest` with these fields using Field():
#   - model_name: str = Field(..., min_length=1, description="Model identifier")
#   - batch_size: int = Field(1, ge=1, le=512, description="Batch size (1-512)")
#   - top_p: float = Field(0.9, gt=0.0, le=1.0, description="Nucleus sampling p")
#   - prompt: str = Field(..., alias="input_text", min_length=1)
#     (alias means the JSON key is "input_text" but the Python attr is "prompt")
# WHY: Field constraints replace manual validation code and are automatically
# reflected in the OpenAPI schema that FastAPI generates.
class ConstrainedRequest(BaseModel):
    pass  # TODO: add fields with Field()


def check_2():
    try:
        # alias: pass "input_text" in the dict, access as .prompt
        req = ConstrainedRequest(model_name="llama", input_text="Hello world")  # type: ignore
        assert req.prompt == "Hello world"
        assert req.batch_size == 1
        assert req.top_p == 0.9

        # Constraint violations must raise ValidationError
        for bad_kwargs, desc in [
            ({"model_name": "", "input_text": "hi"}, "empty model_name"),
            ({"model_name": "m", "input_text": "hi", "batch_size": 0}, "batch_size < 1"),
            ({"model_name": "m", "input_text": "hi", "batch_size": 513}, "batch_size > 512"),
            ({"model_name": "m", "input_text": "hi", "top_p": 0.0}, "top_p == 0"),
            ({"model_name": "m", "input_text": "hi", "top_p": 1.1}, "top_p > 1"),
            ({"model_name": "m", "input_text": ""}, "empty input_text"),
        ]:
            try:
                ConstrainedRequest(**bad_kwargs)  # type: ignore
                assert False, f"{desc} should have raised ValidationError"
            except ValidationError:
                pass
        print("✓ Section 2 passed")
    except AssertionError as e:
        print(f"✗ Section 2 failed: {e}")


check_2()


# ---------------------------------------------------------------------------
# Section 3: Validators — custom field and model-level validation
# ---------------------------------------------------------------------------
# WHY: Field constraints handle simple numeric/length checks, but sometimes
# you need custom logic — e.g., "temperature must be 0 if do_sample is False",
# or "model_name must be in the registry".
#
# Pydantic v2 provides:
#   @field_validator("field_name")  — validates a single field
#   @model_validator(mode="after")  — validates the whole model after init
#     (use this for cross-field constraints)

# TODO: Define `SamplingConfig` with:
#   - temperature: float = 1.0
#   - do_sample: bool = True
#   - top_k: Optional[int] = None
# Add a @field_validator for "temperature" that raises ValueError if
# temperature <= 0 or temperature > 2.0.
# Add a @model_validator(mode="after") that raises ValueError if
# do_sample is False but temperature != 1.0
# (greedy decoding ignores temperature, so setting it is misleading).
class SamplingConfig(BaseModel):
    pass  # TODO: add fields and validators


def check_3():
    try:
        cfg = SamplingConfig()
        assert cfg.temperature == 1.0
        assert cfg.do_sample is True

        # Field validator: temperature out of range
        for bad_temp in [0.0, -0.5, 2.1]:
            try:
                SamplingConfig(temperature=bad_temp)
                assert False, f"temperature={bad_temp} should raise"
            except ValidationError:
                pass

        # Model validator: do_sample=False with non-default temperature
        try:
            SamplingConfig(do_sample=False, temperature=0.7)
            assert False, "do_sample=False + temperature!=1.0 should raise"
        except ValidationError:
            pass

        # Valid: do_sample=False with temperature=1.0 (default) is fine
        cfg2 = SamplingConfig(do_sample=False, temperature=1.0)
        assert cfg2.do_sample is False
        print("✓ Section 3 passed")
    except AssertionError as e:
        print(f"✗ Section 3 failed: {e}")


check_3()


# ---------------------------------------------------------------------------
# Section 4: model_config, model_dump, and model_validate
# ---------------------------------------------------------------------------
# WHY:
#   model_config = ConfigDict(...)  — class-level settings (e.g., forbid
#     extra fields, use enum values, populate by alias)
#   model_dump()  — serialize to a plain dict (replaces .dict() in v1)
#   model_validate()  — deserialize from a dict/JSON (replaces .parse_obj())
#
# `extra="forbid"` is especially useful for configs: it raises ValidationError
# if the input contains any key not declared in the model, catching typos like
# "learing_rate" before they silently use the default.

# TODO: Define `StrictConfig` with model_config = ConfigDict(extra="forbid") and:
#   - lr: float = 1e-3
#   - epochs: int = 10
# WHY: extra="forbid" turns typos in config files into hard errors.
class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pass  # TODO: add fields


def check_4():
    try:
        cfg = StrictConfig(lr=3e-4, epochs=5)
        assert abs(cfg.lr - 3e-4) < 1e-10
        assert cfg.epochs == 5

        # extra="forbid": unknown keys must raise ValidationError
        try:
            StrictConfig(lr=1e-3, epochs=10, learing_rate=5e-4)  # type: ignore
            assert False, "Extra field should raise ValidationError"
        except ValidationError:
            pass

        # model_dump() returns a plain dict
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert "lr" in d and "epochs" in d

        # model_validate() round-trips through a dict
        cfg2 = StrictConfig.model_validate({"lr": 1e-4, "epochs": 20})
        assert cfg2.epochs == 20

        # model_validate() also enforces extra="forbid"
        try:
            StrictConfig.model_validate({"lr": 1e-3, "epochs": 10, "extra_key": 99})
            assert False, "model_validate with extra key should raise"
        except ValidationError:
            pass

        print("✓ Section 4 passed")
    except AssertionError as e:
        print(f"✗ Section 4 failed: {e}")


check_4()
