"""
Module 06 - Exercise 01: FastAPI Basics

Scenario: You're building an inference API for a text classification model.
Clients send text via HTTP POST, and your API returns predictions with
confidence scores.

Topics covered:
- Creating a FastAPI app
- Path, query, and body parameters
- Pydantic models for request/response validation
- APIRouter for modular endpoint organization
- Background tasks for async work

Run with: uvicorn 01_fastapi_basics:app --reload --port 8000
Test at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import time


# =============================================================================
# Part 1: Basic App and Simple Endpoints
# =============================================================================

# TODO: Create a FastAPI app instance
app = FastAPI(
    title="Text Classification API",
    description="API for classifying text into categories",
    version="1.0.0"
)


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    timestamp: float = Field(..., description="Unix timestamp of check")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    # TODO: Return HealthResponse with status='healthy' and current timestamp
    return None


# =============================================================================
# Part 2: Path and Query Parameters
# =============================================================================

@app.get("/model/{model_id}")
def get_model_info(
    model_id: str,
    include_stats: bool = Query(False, description="Include usage statistics")
):
    """Get information about a specific model."""
    info = {
        "model_id": model_id,
        "name": f"Text Classifier v1 ({model_id})",
        "type": "classification",
    }
    # TODO: Add stats if requested
    return info


@app.get("/models")
def list_models(
    skip: int = Query(0, ge=0, description="Number to skip"),
    limit: int = Query(10, ge=1, le=100, description="Max to return")
):
    """List available models with pagination."""
    all_models = [{"id": f"model_{i}", "name": f"Model {i}"} for i in range(100)]
    # TODO: Slice the list based on skip and limit
    return None


# =============================================================================
# Part 3: Request/Response Models (Pydantic)
# =============================================================================

class ClassificationRequest(BaseModel):
    """Request model for text classification."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    top_k: int = Field(3, ge=1, le=10, description="Number of top predictions to return")


class Prediction(BaseModel):
    """Single prediction with confidence."""
    label: str
    confidence: float = Field(..., ge=0, le=1)


class ClassificationResponse(BaseModel):
    """Response model for text classification."""
    text: str = Field(..., description="Original input text")
    predictions: List[Prediction] = Field(..., description="Top predictions")
    processing_time_ms: float = Field(..., description="Time taken in milliseconds")


def simulate_classification(text: str, top_k: int) -> List[Prediction]:
    """Simulate model classification (replace with real model in production)."""
    labels = ["positive", "negative", "neutral", "question", "statement"]
    predictions = []
    base_conf = 0.9
    for i in range(top_k):
        predictions.append(Prediction(
            label=labels[i % len(labels)],
            confidence=round(base_conf - i * 0.15, 3)
        ))
    return predictions


@app.post("/classify", response_model=ClassificationResponse)
def classify_text(request: ClassificationRequest):
    """Classify input text into categories."""
    start_time = time.perf_counter()
    
    # TODO: Run classification using simulate_classification()
    predictions = None
    
    processing_time = (time.perf_counter() - start_time) * 1000
    
    # TODO: Return ClassificationResponse
    return None


# =============================================================================
# Part 4: APIRouter for Modular Endpoints
# =============================================================================

# TODO: Create an APIRouter for admin endpoints
admin_router = None  # Create APIRouter with prefix="/admin"


@admin_router.get("/stats")
def get_admin_stats():
    """Get API usage statistics (admin only)."""
    return {
        "total_requests": 10000,
        "avg_latency_ms": 45.2,
        "error_rate": 0.02,
    }


@admin_router.get("/models/{model_id}/metrics")
def get_model_metrics(model_id: str):
    """Get metrics for a specific model."""
    return {
        "model_id": model_id,
        "requests_last_hour": 500,
        "p50_latency_ms": 30,
        "p99_latency_ms": 150,
    }


# TODO: Include the admin router in the main app
# app.include_router(admin_router)


# =============================================================================
# Part 5: Background Tasks
# =============================================================================

def log_prediction(text: str, predictions: List[Prediction], client_id: str):
    """Log prediction for analytics (runs in background)."""
    time.sleep(0.1)
    print(f"[Background] Logged prediction for client {client_id}: {predictions[0].label}")


@app.post("/classify_with_logging", response_model=ClassificationResponse)
def classify_and_log(
    request: ClassificationRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Query(..., description="Client identifier for logging")
):
    """Classify text and log the prediction in the background."""
    start_time = time.perf_counter()
    
    # TODO: Run classification
    predictions = simulate_classification(request.text, request.top_k)
    
    processing_time = (time.perf_counter() - start_time) * 1000
    
    # TODO: Add the logging as a background task
    
    response = ClassificationResponse(
        text=request.text,
        predictions=predictions,
        processing_time_ms=processing_time
    )
    
    return response


# =============================================================================
# Self-Check Functions
# =============================================================================

def check_local():
    """Run local checks without starting the server."""
    print("=" * 60)
    print("Module 06 - Exercise 01: Local Self-Check")
    print("=" * 60)
    
    # Check 1: App exists
    assert app is not None, "App should be created"
    print("[PASS] FastAPI app created")
    
    # Check 2: Pydantic models work
    req = ClassificationRequest(text="This is great!", top_k=2)
    assert req.text == "This is great!", "ClassificationRequest should store text"
    print("[PASS] ClassificationRequest model")
    
    # Check 3: Prediction model
    pred = Prediction(label="positive", confidence=0.95)
    assert pred.label == "positive", "Prediction should have label"
    print("[PASS] Prediction model")
    
    # Check 4: Classification response
    response = ClassificationResponse(
        text="Test",
        predictions=[pred],
        processing_time_ms=10.5
    )
    assert len(response.predictions) == 1, "Response should have predictions"
    print("[PASS] ClassificationResponse model")
    
    # Check 5: simulate_classification
    preds = simulate_classification("test text", 3)
    assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"
    print("[PASS] simulate_classification")
    
    # Check 6: Health response structure
    health = health_check()
    assert health is not None, "health_check should return response"
    print("[PASS] health_check")
    
    print("=" * 60)
    print("All local checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check_local()
