"""
Module 06 - Exercise 02: Error Handling in FastAPI

Scenario: Your inference API needs to handle various error conditions
gracefully - invalid inputs, model loading failures, rate limiting.

Topics covered:
- HTTPException for standard HTTP errors
- Custom exception classes
- Exception handlers with @app.exception_handler
- Error response schemas
- Middleware for global error catching

Run with: uvicorn 02_error_handling:app --reload --port 8001
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Any, Dict
import time


# =============================================================================
# Part 1: Error Response Schema
# =============================================================================

class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path that caused error")


# TODO: Create FastAPI app
app = FastAPI(title="Error Handling Demo")


# =============================================================================
# Part 2: HTTPException for Standard Errors
# =============================================================================

class ModelNotLoadedError(Exception):
    """Custom exception for model not loaded."""
    pass


class RateLimitExceededError(Exception):
    """Custom exception for rate limiting."""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after


# Simulated global state
model_loaded = False
request_counts: Dict[str, int] = {}


@app.get("/predict/{text}")
def predict(text: str, confidence_threshold: float = 0.5):
    """Predict endpoint with various error conditions."""
    # TODO: Validate confidence_threshold (must be 0-1)
    if not (0 <= confidence_threshold <= 1):
        raise None  # Replace with HTTPException
    
    # TODO: Check if model is loaded
    if not model_loaded:
        raise None  # Replace with HTTPException
    
    # TODO: Simple rate limiting
    client_id = "demo_client"
    request_counts[client_id] = request_counts.get(client_id, 0) + 1
    
    if request_counts[client_id] > 10:
        raise None  # Replace with HTTPException for rate limit
    
    return {
        "text": text,
        "prediction": "positive",
        "confidence": 0.95
    }


# =============================================================================
# Part 3: Custom Exception Handlers
# =============================================================================

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    """Handle ModelNotLoadedError with a proper error response."""
    # TODO: Return JSONResponse with ErrorResponse schema, status_code=503
    return JSONResponse(
        status_code=503,
        content=None
    )


@app.exception_handler(RateLimitExceededError)
async def rate_limit_handler(request: Request, exc: RateLimitExceededError):
    """Handle rate limit errors with Retry-After header."""
    # TODO: Return JSONResponse with status_code=429 and Retry-After header
    return JSONResponse(
        status_code=None,
        content=None,
        headers=None
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unhandled errors."""
    print(f"Unhandled error: {type(exc).__name__}: {exc}")
    
    # TODO: Return JSONResponse with status_code=500
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "An internal error occurred",
            "path": str(request.url.path)
        }
    )


# =============================================================================
# Part 4: Error Handling Middleware
# =============================================================================

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs all errors and ensures consistent error format."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            # TODO: Call the next handler and return response
            response = await call_next(request)
            return response
        except Exception as exc:
            print(f"Middleware caught error: {type(exc).__name__} on {request.url.path}")
            
            # TODO: Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "MIDDLEWARE_ERROR",
                    "message": "Request processing failed"
                }
            )


# TODO: Add middleware to app
# app.add_middleware(ErrorLoggingMiddleware)


# =============================================================================
# Part 5: Practical Error Patterns
# =============================================================================

class InferenceRequest(BaseModel):
    """Request with validation requirements."""
    text: str = Field(..., min_length=1, max_length=5000)
    model_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    temperature: float = Field(1.0, ge=0, le=2)


@app.post("/infer", response_model=dict)
def infer(request: InferenceRequest):
    """Inference endpoint with Pydantic validation."""
    valid_models = ["model_v1", "model_v2", "model_v3"]
    if request.model_id not in valid_models:
        # TODO: Raise HTTPException for invalid model
        pass
    
    return {
        "model_id": request.model_id,
        "result": f"Processed: {request.text[:20]}...",
        "temperature": request.temperature
    }


@app.get("/model/{model_id}/status")
def model_status(model_id: str):
    """Get model status, demonstrating path parameter validation."""
    known_models = {"model_v1": "loaded", "model_v2": "loading", "model_v3": "offline"}
    
    if model_id not in known_models:
        # TODO: Raise HTTPException 404
        pass
    
    return {
        "model_id": model_id,
        "status": known_models[model_id]
    }


# =============================================================================
# Self-Check Functions
# =============================================================================

def check_local():
    """Run local checks without starting the server."""
    print("=" * 60)
    print("Module 06 - Exercise 02: Local Self-Check")
    print("=" * 60)
    
    # Check 1: ErrorResponse schema
    error = ErrorResponse(
        error_code="TEST_ERROR",
        message="Test error message",
        path="/test"
    )
    assert error.error_code == "TEST_ERROR", "ErrorResponse should have error_code"
    print("[PASS] ErrorResponse schema")
    
    # Check 2: Custom exceptions
    try:
        raise ModelNotLoadedError("Model not ready")
    except ModelNotLoadedError as e:
        assert str(e) == "Model not ready"
    print("[PASS] ModelNotLoadedError")
    
    # Check 3: RateLimitExceededError with retry_after
    try:
        raise RateLimitExceededError(retry_after=60)
    except RateLimitExceededError as e:
        assert e.retry_after == 60
    print("[PASS] RateLimitExceededError")
    
    # Check 4: InferenceRequest validation
    valid_request = InferenceRequest(text="Hello", model_id="model_v1", temperature=0.8)
    assert valid_request.text == "Hello"
    print("[PASS] InferenceRequest valid input")
    
    # Check 5: InferenceRequest validation fails
    try:
        InferenceRequest(text="", model_id="model_v1")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    print("[PASS] InferenceRequest validation rejects invalid input")
    
    print("=" * 60)
    print("All local checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check_local()
