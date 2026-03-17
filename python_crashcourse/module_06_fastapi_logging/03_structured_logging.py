"""
Module 06 - Exercise 03: Structured Logging with structlog

Scenario: Your inference API serves thousands of requests daily. When
something goes wrong, you need to quickly find all logs for a specific
request, user, or model. Structured logging with context binding makes
this possible.

Topics covered:
- Python logging module basics
- structlog for structured JSON logs
- Context binding (request IDs, user IDs)
- Log levels and handlers
- Request logging middleware

Prerequisites:
    pip install structlog
"""

import logging
import structlog
from typing import Optional
import time


# =============================================================================
# Part 1: Python Logging Basics
# =============================================================================

def setup_basic_logging():
    """Configure Python's built-in logging module."""
    # TODO: Configure logging with format and level DEBUG
    logging.basicConfig(
        format=None,
        level=None,
    )


def demonstrate_logging_levels():
    """Demonstrate different logging levels."""
    logger = logging.getLogger(__name__)
    
    # TODO: Log at each level: debug, info, warning, error, critical
    logger.debug(None)
    logger.info(None)
    logger.warning(None)
    logger.error(None)
    logger.critical(None)


# =============================================================================
# Part 2: structlog Setup
# =============================================================================

def setup_structlog():
    """Configure structlog for structured JSON logging."""
    structlog.configure(
        processors=[
            # TODO: Add processors for log level, logger name, timestamp, JSON renderer
        ],
        wrapper_class=None,
        context_class=None,
        logger_factory=None,
        cache_logger_on_first_use=True,
    )


def get_structured_logger(name: str):
    """Get a structured logger instance."""
    # TODO: Return structlog.get_logger(name)
    return None


# =============================================================================
# Part 3: Context Binding
# =============================================================================

def log_with_context(request_id: str, user_id: Optional[str] = None):
    """Create a logger with bound context."""
    logger = get_structured_logger("request_handler")
    
    # TODO: Bind request_id and user_id to context
    # logger = logger.bind(request_id=request_id, user_id=user_id)
    
    return logger


def demonstrate_context_binding():
    """Show how context appears in all log entries."""
    logger = log_with_context(request_id="req-12345", user_id="user-789")
    
    # TODO: Log several messages - all should include request_id and user_id
    logger.info("Request started")
    logger.debug("Processing input data")
    logger.info("Classification complete", prediction="positive", confidence=0.95)
    logger.info("Request finished")


# =============================================================================
# Part 4: Request Logging Middleware Pattern
# =============================================================================

class RequestLogger:
    """Helper class for consistent request logging."""
    
    def __init__(self, request_id: str, method: str, path: str):
        self.request_id = None
        self.method = None
        self.path = None
        self.start_time = None
        self.logger = None
    
    def start(self):
        """Log request start."""
        self.start_time = time.perf_counter()
        # TODO: Log request start with method and path
        self.logger.info("Request started")
    
    def finish(self, status_code: int, **extra_context):
        """Log request completion."""
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        # TODO: Log request finish with status_code, duration_ms, and extra_context
        self.logger.info("Request finished")
    
    def error(self, exc: Exception, **extra_context):
        """Log request error."""
        # TODO: Log error with exception type and message
        self.logger.error("Request failed", exc_info=True)


# =============================================================================
# Part 5: Practical Logging Patterns for ML APIs
# =============================================================================

class ModelInferenceLogger:
    """Specialized logger for ML inference operations."""
    
    def __init__(self, model_id: str):
        self.model_id = None
        self.logger = None
    
    def log_prediction(self, input_hash: str, prediction: str, confidence: float, latency_ms: float):
        """Log a single prediction."""
        # TODO: Log prediction with all metadata
        pass
    
    def log_batch_stats(self, batch_size: int, avg_latency_ms: float, predictions_distribution: dict):
        """Log batch processing statistics."""
        # TODO: Log batch stats
        pass
    
    def log_model_load(self, load_time_ms: float, model_size_mb: float):
        """Log model loading event."""
        # TODO: Log model load event
        pass


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 06 - Exercise 03: Self-Check")
    print("=" * 60)
    
    # Check 1: Basic logging setup
    print("\nSetting up basic logging...")
    setup_basic_logging()
    print("[PASS] setup_basic_logging")
    
    # Check 2: Logging levels demonstration
    print("\nDemonstrating logging levels:")
    demonstrate_logging_levels()
    print("[PASS] demonstrate_logging_levels")
    
    # Check 3: structlog setup
    print("\nSetting up structlog...")
    setup_structlog()
    print("[PASS] setup_structlog")
    
    # Check 4: Get structured logger
    logger = get_structured_logger("test")
    assert logger is not None, "get_structured_logger should return a logger"
    print("[PASS] get_structured_logger")
    
    # Check 5: Context binding demonstration
    print("\nDemonstrating context binding:")
    demonstrate_context_binding()
    print("[PASS] demonstrate_context_binding")
    
    # Check 6: RequestLogger
    print("\nTesting RequestLogger...")
    req_logger = RequestLogger("test-req-001", "POST", "/predict")
    assert req_logger is not None, "RequestLogger should be created"
    print("[PASS] RequestLogger creation")
    
    # Check 7: ModelInferenceLogger
    print("\nTesting ModelInferenceLogger...")
    model_logger = ModelInferenceLogger("model_v1")
    assert model_logger.model_id == "model_v1", "ModelInferenceLogger should store model_id"
    print("[PASS] ModelInferenceLogger creation")
    
    print("\n" + "=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
