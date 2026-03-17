"""
Module 07 - Exercise 04: Observability Challenge

Scenario: Build a complete observability stack for an ML inference service.
The service needs distributed tracing, metrics collection, and structured logging.

This challenge integrates timing, benchmarking, and observability.
"""

import time
import random
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager
import json


# =============================================================================
# Part 1: OpenTelemetry Tracing Setup
# =============================================================================

class TracingSpan:
    """Simplified span implementation for learning distributed tracing."""
    
    def __init__(self, name: str, parent: Optional['TracingSpan'] = None):
        self.name = name
        self.parent = parent
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, str] = {}
        self.children: List['TracingSpan'] = []
        
        if parent:
            parent.children.append(self)
    
    def start(self):
        """Start the span."""
        self.start_time = time.perf_counter()
    
    def end(self):
        """End the span."""
        self.end_time = time.perf_counter()
    
    def set_attribute(self, key: str, value: str):
        """Set a span attribute."""
        self.attributes[key] = value
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def to_dict(self) -> dict:
        """Convert span to dictionary for export."""
        return {
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'attributes': self.attributes,
            'children': [c.to_dict() for c in self.children],
        }


class Tracer:
    """Simplified tracer for managing spans."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.active_spans: Dict[int, TracingSpan] = {}
        self.completed_spans: List[TracingSpan] = []
    
    def start_span(self, name: str, parent: Optional[TracingSpan] = None) -> TracingSpan:
        """Start a new span."""
        span = TracingSpan(name, parent)
        self.active_spans[threading.get_ident()] = span
        span.start()
        return span
    
    def end_span(self, span: TracingSpan):
        """End a span and store it."""
        span.end()
        self.active_spans.pop(threading.get_ident(), None)
        self.completed_spans.append(span)
    
    @contextmanager
    def trace(self, name: str, **attributes):
        """Context manager for tracing a code block."""
        parent = self.active_spans.get(threading.get_ident())
        span = self.start_span(name, parent)
        
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
        
        try:
            yield span
        finally:
            self.end_span(span)
    
    def get_trace_summary(self) -> dict:
        """Get summary of all completed traces."""
        return {
            'service_name': self.service_name,
            'total_spans': len(self.completed_spans),
            'spans': [s.to_dict() for s in self.completed_spans],
        }


# =============================================================================
# Part 2: Prometheus Metrics
# =============================================================================

class Counter:
    """Simplified Counter metric (like Prometheus Counter)."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()
    
    def inc(self, amount: float = 1):
        """Increment the counter."""
        with self._lock:
            self._value += amount
    
    @property
    def value(self) -> float:
        return self._value


class Gauge:
    """Simplified Gauge metric (like Prometheus Gauge)."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: float):
        with self._lock:
            self._value = value
    
    def inc(self, amount: float = 1):
        with self._lock:
            self._value += amount
    
    def dec(self, amount: float = 1):
        with self._lock:
            self._value -= amount
    
    @property
    def value(self) -> float:
        return self._value


class Histogram:
    """Simplified Histogram metric (like Prometheus Histogram)."""
    
    def __init__(self, name: str, description: str, buckets: List[float] = None):
        self.name = name
        self.description = description
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._bucket_counts[float('inf')] = 0
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()
    
    def observe(self, value: float):
        """Record an observation."""
        with self._lock:
            self._sum += value
            self._count += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
            self._bucket_counts[float('inf')] += 1
    
    def get_percentile(self, percentile: float) -> float:
        """Estimate percentile from histogram."""
        if self._count == 0:
            return 0.0
        
        target_count = self._count * percentile
        cumulative = 0
        for bucket in sorted(self._bucket_counts.keys()):
            cumulative += self._bucket_counts[bucket]
            if cumulative >= target_count:
                return bucket
        return max(self.buckets)
    
    @property
    def summary(self) -> dict:
        """Get histogram summary."""
        return {
            'count': self._count,
            'sum': self._sum,
            'mean': self._sum / self._count if self._count > 0 else 0,
            'p50': self.get_percentile(0.5),
            'p95': self.get_percentile(0.95),
            'p99': self.get_percentile(0.99),
        }


class MetricsRegistry:
    """Registry for all metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, object] = {}
    
    def counter(self, name: str, description: str) -> Counter:
        if name not in self.metrics:
            self.metrics[name] = Counter(name, description)
        return self.metrics[name]
    
    def gauge(self, name: str, description: str) -> Gauge:
        if name not in self.metrics:
            self.metrics[name] = Gauge(name, description)
        return self.metrics[name]
    
    def histogram(self, name: str, description: str, buckets: List[float] = None) -> Histogram:
        if name not in self.metrics:
            self.metrics[name] = Histogram(name, description, buckets)
        return self.metrics[name]
    
    def get_all_metrics(self) -> dict:
        """Get all metrics as dictionary."""
        result = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, Counter):
                result[name] = {'type': 'counter', 'value': metric.value}
            elif isinstance(metric, Gauge):
                result[name] = {'type': 'gauge', 'value': metric.value}
            elif isinstance(metric, Histogram):
                result[name] = {'type': 'histogram', **metric.summary}
        return result


# =============================================================================
# Part 3: ML Inference Service with Observability
# =============================================================================

@dataclass
class InferenceRequest:
    request_id: str
    model_id: str
    input_data: List[float]


@dataclass
class InferenceResponse:
    request_id: str
    predictions: List[float]
    latency_ms: float


class ObservableInferenceService:
    """ML inference service with full observability."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tracer = Tracer(f"inference-service-{model_id}")
        self.metrics = MetricsRegistry()
        
        self.request_counter = self.metrics.counter('inference_requests_total', 'Total requests')
        self.error_counter = self.metrics.counter('inference_errors_total', 'Total errors')
        self.active_requests = self.metrics.gauge('active_inference_requests', 'Active requests')
        self.latency_histogram = self.metrics.histogram(
            'inference_latency_seconds',
            'Inference latency distribution',
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        self.model_loaded = True
    
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference with full observability."""
        self.request_counter.inc()
        self.active_requests.inc()
        
        start_time = time.perf_counter()
        
        with self.tracer.trace("inference", model_id=self.model_id, request_id=request.request_id):
            try:
                with self.tracer.trace("preprocessing"):
                    time.sleep(random.uniform(0.01, 0.03))
                    processed_data = [x / 255.0 for x in request.input_data]
                
                with self.tracer.trace("model_forward"):
                    time.sleep(random.uniform(0.05, 0.15))
                    predictions = [sum(processed_data) / len(processed_data) * random.uniform(0.8, 1.2)]
                
                with self.tracer.trace("postprocessing"):
                    time.sleep(random.uniform(0.005, 0.015))
                    predictions = [round(p, 4) for p in predictions]
                
            except Exception as e:
                self.error_counter.inc()
                raise
            finally:
                latency = time.perf_counter() - start_time
                self.latency_histogram.observe(latency)
                self.active_requests.dec()
        
        return InferenceResponse(
            request_id=request.request_id,
            predictions=predictions,
            latency_ms=latency * 1000
        )
    
    def get_observability_report(self) -> dict:
        """Get complete observability report."""
        return {
            'traces': self.tracer.get_trace_summary(),
            'metrics': self.metrics.get_all_metrics(),
        }


# =============================================================================
# Part 4: Dashboard Data Generation
# =============================================================================

def generate_dashboard_data(service: ObservableInferenceService, num_requests: int = 100):
    """Simulate traffic and generate dashboard data."""
    latencies = []
    errors = 0
    
    for i in range(num_requests):
        request = InferenceRequest(
            request_id=f"req-{i}",
            model_id=service.model_id,
            input_data=[random.random() * 255 for _ in range(random.randint(100, 1000))]
        )
        
        try:
            response = service.infer(request)
            latencies.append(response.latency_ms)
        except Exception:
            errors += 1
    
    dashboard = {
        'total_requests': num_requests,
        'successful_requests': num_requests - errors,
        'error_count': errors,
        'error_rate': errors / num_requests if num_requests > 0 else 0,
        'latency': {
            'min_ms': min(latencies) if latencies else 0,
            'max_ms': max(latencies) if latencies else 0,
            'avg_ms': sum(latencies) / len(latencies) if latencies else 0,
        },
        'current_metrics': service.get_observability_report()['metrics'],
    }
    
    return dashboard


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 07 - Exercise 04: Self-Check")
    print("=" * 60)
    
    # Check 1: TracingSpan
    print("\nTesting TracingSpan...")
    span = TracingSpan("test_span")
    span.start()
    time.sleep(0.01)
    span.end()
    assert span.duration_ms is not None
    assert span.duration_ms >= 10
    print(f"[PASS] TracingSpan (duration: {span.duration_ms:.2f}ms)")
    
    # Check 2: Tracer
    print("\nTesting Tracer...")
    tracer = Tracer("test-service")
    with tracer.trace("outer_operation", key1="value1"):
        with tracer.trace("inner_operation", key2="value2"):
            time.sleep(0.01)
    
    trace_summary = tracer.get_trace_summary()
    assert trace_summary['total_spans'] == 2
    print(f"[PASS] Tracer ({trace_summary['total_spans']} spans recorded)")
    
    # Check 3: Counter
    print("\nTesting Counter...")
    counter = Counter("test_counter", "Test counter")
    counter.inc()
    counter.inc(5)
    assert counter.value == 6
    print(f"[PASS] Counter (value: {counter.value})")
    
    # Check 4: Gauge
    print("\nTesting Gauge...")
    gauge = Gauge("test_gauge", "Test gauge")
    gauge.set(10)
    gauge.inc(5)
    gauge.dec(3)
    assert gauge.value == 12
    print(f"[PASS] Gauge (value: {gauge.value})")
    
    # Check 5: Histogram
    print("\nTesting Histogram...")
    histogram = Histogram("test_histogram", "Test histogram")
    for val in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        histogram.observe(val)
    
    summary = histogram.summary
    assert summary['count'] == 7
    print(f"[PASS] Histogram (count: {summary['count']}, p50: {summary['p50']:.3f}s)")
    
    # Check 6: MetricsRegistry
    print("\nTesting MetricsRegistry...")
    registry = MetricsRegistry()
    registry.counter('requests', 'Request count')
    registry.gauge('active', 'Active connections')
    registry.histogram('latency', 'Latency distribution')
    
    all_metrics = registry.get_all_metrics()
    assert len(all_metrics) == 3
    print(f"[PASS] MetricsRegistry ({len(all_metrics)} metrics)")
    
    # Check 7: ObservableInferenceService
    print("\nTesting ObservableInferenceService...")
    service = ObservableInferenceService("test-model")
    
    request = InferenceRequest(
        request_id="test-req-1",
        model_id="test-model",
        input_data=[random.random() * 255 for _ in range(100)]
    )
    
    response = service.infer(request)
    assert response.request_id == "test-req-1"
    assert response.latency_ms > 0
    print(f"[PASS] ObservableInferenceService (latency: {response.latency_ms:.2f}ms)")
    
    # Check 8: Dashboard data
    print("\nGenerating dashboard data...")
    dashboard = generate_dashboard_data(service, num_requests=20)
    assert dashboard['total_requests'] == 20
    assert dashboard['latency']['avg_ms'] > 0
    print(f"[PASS] Dashboard data")
    print(f"  Total requests: {dashboard['total_requests']}")
    print(f"  Error rate: {dashboard['error_rate']*100:.1f}%")
    print(f"  Avg latency: {dashboard['latency']['avg_ms']:.2f}ms")
    
    print("\n" + "=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
