"""
Module 06 - Exercise 04: Complete Inference API Challenge

Scenario: Build a production-ready inference API for an image classification
model with error handling, structured logging, and multiple endpoints.

This challenge integrates FastAPI, error handling, and structured logging.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import uuid
import structlog
from enum import Enum


# =============================================================================
# Part 1: Models and Schemas
# =============================================================================

class ModelStatus(str, Enum):
    LOADING = "loading"
    READY = "ready"
    OFFLINE = "offline"
    ERROR = "error"


class ModelInfo(BaseModel):
    model_id: str
    name: str
    version: str
    status: ModelStatus
    input_shape: List[int]
    classes: List[str]


class PredictionResult(BaseModel):
    label: str
    confidence: float = Field(..., ge=0, le=1)
    class_index: Optional[int] = None


class InferenceResponse(BaseModel):
    request_id: str
    model_id: str
    predictions: List[PredictionResult]
    processing_time_ms: float
    timestamp: float


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


# =============================================================================
# Part 2: Model Registry
# =============================================================================

class ModelRegistry:
    """Registry for managing multiple models."""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
    
    def register(self, model: ModelInfo):
        """Register a model."""
        # TODO: Add model to registry
        pass
    
    def get(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        # TODO: Return model or None if not found
        pass
    
    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        # TODO: Return list of all models
        pass
    
    def is_ready(self, model_id: str) -> bool:
        """Check if model is ready for inference."""
        # TODO: Return True if model status is READY
        pass


model_registry = ModelRegistry()


# =============================================================================
# Part 3: Structured Logging Setup
# =============================================================================

def setup_logging():
    """Configure logging for the application."""
    # TODO: Configure structlog with JSON output
    structlog.configure(
        processors=[
            # TODO: Add necessary processors
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_request_logger(request_id: str, model_id: str):
    """Get a logger bound with request context."""
    # TODO: Return logger with bound request_id and model_id
    pass


# =============================================================================
# Part 4: Request ID Middleware
# =============================================================================

async def add_request_id_header(request: Request, call_next):
    """Middleware to add request ID to all requests."""
    # TODO: Get or generate request ID
    request_id = None
    
    # TODO: Store request_id in request state
    request.state.request_id = request_id
    
    # TODO: Call next handler
    response = await call_next(request)
    
    # TODO: Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response


# =============================================================================
# Part 5: FastAPI Application
# =============================================================================

# TODO: Create FastAPI app with metadata
app = FastAPI(
    title="Image Classification API",
    description="Production-ready inference API for image classification",
    version="1.0.0",
)

# TODO: Add CORS middleware
# app.add_middleware(CORSMiddleware, ...)

# TODO: Add request ID middleware
# app.middleware("http")(add_request_id_header)


# =============================================================================
# Part 6: API Endpoints
# =============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint for load balancers."""
    # TODO: Return health status
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/models")
def list_models():
    """List all registered models."""
    # TODO: Return list of models from registry
    return {"models": []}


@app.get("/models/{model_id}")
def get_model(model_id: str):
    """Get details for a specific model."""
    # TODO: Get model from registry, return 404 if not found
    model = None
    if model is None:
        raise None  # HTTPException 404
    return model


@app.post("/predict/{model_id}", response_model=InferenceResponse)
async def predict(
    model_id: str,
    file: UploadFile = File(...),
    top_k: int = 3,
    x_request_id: Optional[str] = Header(None)
):
    """Run inference on an uploaded image."""
    request_id = getattr(app.state, "request_id", str(uuid.uuid4()))
    start_time = time.perf_counter()
    
    # TODO: Validate model exists and is ready
    model = model_registry.get(model_id)
    if model is None:
        raise None  # HTTPException
    if not model_registry.is_ready(model_id):
        raise None  # HTTPException 503
    
    # TODO: Validate file type (must be image)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise None  # HTTPException 400
    
    # TODO: Simulate inference
    predictions = []
    for i in range(min(top_k, len(model.classes))):
        predictions.append(PredictionResult(
            label=model.classes[i],
            confidence=round(0.9 - i * 0.15, 3),
            class_index=i
        ))
    
    processing_time = (time.perf_counter() - start_time) * 1000
    
    return InferenceResponse(
        request_id=request_id,
        model_id=model_id,
        predictions=predictions,
        processing_time_ms=processing_time,
        timestamp=time.time()
    )


@app.get("/metrics")
def get_metrics():
    """Get API metrics."""
    # TODO: Return fake metrics
    return {
        "requests_total": 10000,
        "requests_in_flight": 5,
        "avg_latency_ms": 45.2,
        "p99_latency_ms": 150.0,
    }


# =============================================================================
# Part 7: Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error responses."""
    request_id = getattr(request.state, "request_id", None)
    
    # TODO: Return JSONResponse with ErrorResponse schema
    return JSONResponse(
        status_code=exc.status_code,
        content=None
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    # TODO: Log the exception and return 500 error response
    return JSONResponse(
        status_code=500,
        content=None
    )


# =============================================================================
# Self-Check Functions
# =============================================================================

def check_local():
    """Run local checks without starting the server."""
    print("=" * 60)
    print("Module 06 - Exercise 04: Local Self-Check")
    print("=" * 60)
    
    # Check 1: ModelInfo schema
    model = ModelInfo(
        model_id="test_model",
        name="Test Model",
        version="1.0",
        status=ModelStatus.READY,
        input_shape=[224, 224, 3],
        classes=["cat", "dog"]
    )
    assert model.model_id == "test_model"
    print("[PASS] ModelInfo schema")
    
    # Check 2: PredictionResult schema
    pred = PredictionResult(label="cat", confidence=0.95, class_index=0)
    assert pred.label == "cat"
    assert 0 <= pred.confidence <= 1
    print("[PASS] PredictionResult schema")
    
    # Check 3: InferenceResponse schema
    response = InferenceResponse(
        request_id="req-123",
        model_id="test_model",
        predictions=[pred],
        processing_time_ms=10.5,
        timestamp=time.time()
    )
    assert response.request_id == "req-123"
    print("[PASS] InferenceResponse schema")
    
    # Check 4: ModelRegistry
    registry = ModelRegistry()
    registry.register(model)
    assert registry.get("test_model") is not None
    assert registry.get("unknown") is None
    assert registry.is_ready("test_model") is True
    print("[PASS] ModelRegistry")
    
    # Check 5: ErrorResponse schema
    error = ErrorResponse(
        error_code="MODEL_NOT_FOUND",
        message="Model 'unknown' not found",
        request_id="req-123"
    )
    assert error.error_code == "MODEL_NOT_FOUND"
    print("[PASS] ErrorResponse schema")
    
    # Check 6: App exists
    assert app is not None, "FastAPI app should be created"
    print("[PASS] FastAPI app created")
    
    print("\n" + "=" * 60)
    print("All local checks passed!")
    print("=" * 60)
    print("\nTo test the full API:")
    print("  uvicorn 04_api_challenge:app --reload --port 8002")


if __name__ == "__main__":
    check_local()
