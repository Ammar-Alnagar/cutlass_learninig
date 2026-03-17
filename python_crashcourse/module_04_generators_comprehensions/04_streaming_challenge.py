"""
Module 04 - Exercise 04: Streaming Challenge

Scenario: You're building a log file analyzer for an ML inference service.
Log files can be gigabytes in size, so you cannot load them entirely into
memory. You need to:
1. Stream the file line by line
2. Parse and filter relevant entries
3. Compute statistics in a single pass
4. Generate reports using memory-efficient patterns

This challenge combines generators, comprehensions, and itertools.
"""

import itertools
import re
from collections import defaultdict


# =============================================================================
# Part 1: Streaming File Reader
# =============================================================================

def stream_log_file(filepath):
    """
    Stream a log file line by line, yielding parsed entries.
    
    Log format: "TIMESTAMP LEVEL MESSAGE"
    Example: "2024-01-15T10:30:00 INFO Request processed in 45ms"
    
    Args:
        filepath: Path to log file
        
    Yields:
        dict: {'timestamp': str, 'level': str, 'message': str}
    """
    pattern = re.compile(r'^(\S+)\s+(\w+)\s+(.*)$')
    
    with open(filepath, 'r') as f:
        for line in f:
            # Strip whitespace and parse
            line = line.strip()
            match = pattern.match(line)
            if match:
                # TODO: Yield parsed components as dict
                pass


def filter_by_level(entries, level):
    """
    Filter log entries by log level.
    
    Args:
        entries: Iterator of log entry dicts
        level: Log level to filter (e.g., 'ERROR', 'WARNING')
        
    Yields:
        dict: Only entries matching the level
    """
    # TODO: Yield only entries where entry['level'] == level
    for entry in entries:
        pass  # Add filter condition


def filter_by_keyword(entries, keyword):
    """
    Filter log entries containing a keyword in the message.
    
    Args:
        entries: Iterator of log entry dicts
        keyword: Keyword to search for (case-insensitive)
        
    Yields:
        dict: Only entries containing the keyword
    """
    # TODO: Yield entries where keyword is in entry['message'] (case-insensitive)
    for entry in entries:
        pass  # Add filter condition


# =============================================================================
# Part 2: Streaming Statistics
# =============================================================================

def count_by_level(entries):
    """
    Count entries per log level in a single pass.
    
    Args:
        entries: Iterator of log entry dicts
        
    Returns:
        dict: {level: count}
    """
    counts = defaultdict(int)
    for entry in entries:
        # TODO: Increment count for entry['level']
        pass
    return dict(counts)


def extract_latency_values(entries):
    """
    Extract latency values from log messages.
    
    Assumes messages contain "XXXms" pattern for latency.
    
    Args:
        entries: Iterator of log entry dicts
        
    Yields:
        float: Latency values in milliseconds
    """
    pattern = re.compile(r'(\d+(?:\.\d+)?)ms')
    for entry in entries:
        # TODO: Find and yield latency values
        pass


def compute_latency_stats(entries):
    """
    Compute min, max, sum, count for latencies in a single pass.
    
    Args:
        entries: Iterator of log entry dicts
        
    Returns:
        dict: {'min': float, 'max': float, 'sum': float, 'count': int, 'avg': float}
    """
    min_lat = float('inf')
    max_lat = float('-inf')
    total = 0
    count = 0
    
    for entry in entries:
        # TODO: Extract latency and update stats
        pass
    
    # TODO: Compute average and return stats dict
    return None


# =============================================================================
# Part 3: Chunked Processing
# =============================================================================

def chunked_iterator(items, chunk_size):
    """
    Split an iterator into chunks without loading all into memory.
    
    Args:
        items: Iterator of items
        chunk_size: Number of items per chunk
        
    Yields:
        list: Chunks of items (last chunk may be smaller)
    """
    it = iter(items)
    while True:
        # TODO: Get next chunk using islice
        chunk = None
        if not chunk:
            break
        yield list(chunk)


def process_in_chunks(filepath, chunk_size=100):
    """
    Process log file in chunks, computing per-chunk statistics.
    
    Args:
        filepath: Path to log file
        chunk_size: Number of entries per chunk
        
    Yields:
        dict: Per-chunk statistics
    """
    # TODO: Create a generator that parses log entries
    # Use chunked_iterator to process in batches
    # Yield stats for each chunk
    pass


# =============================================================================
# Part 4: Memory-Efficient Aggregation
# =============================================================================

def top_k_latencies(entries, k=10):
    """
    Find top K highest latencies without storing all values.
    
    Use case: Finding outliers in a massive dataset.
    
    Args:
        entries: Iterator of log entry dicts
        k: Number of top values to keep
        
    Returns:
        list: Top K latency values, sorted descending
    """
    top_k = []
    
    for entry in entries:
        # TODO: Extract latency and update top_k list
        pass
    
    return top_k


def running_percentile(entries, percentile=90):
    """
    Estimate percentile using reservoir sampling for massive streams.
    
    For truly massive streams, use reservoir sampling to maintain
    a representative sample of fixed size.
    
    Args:
        entries: Iterator of log entry dicts
        percentile: Percentile to compute (0-100)
        
    Returns:
        float: Estimated percentile value
    """
    import random
    
    reservoir = []
    n = 0
    
    for entry in entries:
        # TODO: Extract latency and apply reservoir sampling
        n += 1
        pass
    
    # TODO: Sort reservoir and return value at percentile index
    return None


# =============================================================================
# Part 5: Pipeline Composition
# =============================================================================

def create_analysis_pipeline(filepath):
    """
    Compose a full analysis pipeline using generator composition.
    
    This demonstrates the power of generators: you can chain operations
    that each process one item at a time, with minimal memory overhead.
    
    Args:
        filepath: Path to log file
        
    Returns:
        dict: Complete analysis results
    """
    # TODO: Build a pipeline that:
    # 1. Streams the log file
    # 2. Filters to only ERROR and WARNING entries
    # 3. Counts by level
    # 4. Extracts latencies from INFO entries
    # 5. Computes latency statistics
    # 6. Finds top 10 latencies
    
    results = {
        'level_counts': None,
        'latency_stats': None,
        'top_10_latencies': None,
    }
    
    return results


# =============================================================================
# Self-Check with Synthetic Data
# =============================================================================

def create_test_log_file(filepath):
    """Create a small test log file for verification."""
    content = """2024-01-15T10:00:00 INFO Request processed in 45ms
2024-01-15T10:00:01 INFO Request processed in 32ms
2024-01-15T10:00:02 WARNING High memory usage detected
2024-01-15T10:00:03 INFO Request processed in 78ms
2024-01-15T10:00:04 ERROR Connection timeout after 5000ms
2024-01-15T10:00:05 INFO Request processed in 23ms
2024-01-15T10:00:06 INFO Request processed in 156ms
2024-01-15T10:00:07 WARNING Slow response time
2024-01-15T10:00:08 INFO Request processed in 41ms
2024-01-15T10:00:09 ERROR Database query failed
"""
    with open(filepath, 'w') as f:
        f.write(content)


def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 04 - Exercise 04: Self-Check")
    print("=" * 60)
    
    # Create test file
    test_file = '/tmp/test_logs.txt'
    create_test_log_file(test_file)
    
    # Check 1: stream_log_file
    entries = list(stream_log_file(test_file))
    assert len(entries) == 10, f"stream_log_file: expected 10 entries, got {len(entries)}"
    assert entries[0]['level'] == 'INFO', f"stream_log_file: first entry level should be INFO"
    print("[PASS] stream_log_file")
    
    # Check 2: filter_by_level
    entries = list(stream_log_file(test_file))
    errors = list(filter_by_level(iter(entries), 'ERROR'))
    assert len(errors) == 2, f"filter_by_level: expected 2 ERRORs, got {len(errors)}"
    print("[PASS] filter_by_level")
    
    # Check 3: filter_by_keyword
    entries = list(stream_log_file(test_file))
    requests = list(filter_by_keyword(iter(entries), 'Request'))
    assert len(requests) == 5, f"filter_by_keyword: expected 5 Request entries, got {len(requests)}"
    print("[PASS] filter_by_keyword")
    
    # Check 4: count_by_level
    entries = list(stream_log_file(test_file))
    counts = count_by_level(iter(entries))
    assert counts.get('INFO', 0) == 6, f"count_by_level: expected 6 INFOs, got {counts.get('INFO', 0)}"
    assert counts.get('ERROR', 0) == 2, f"count_by_level: expected 2 ERRORs, got {counts.get('ERROR', 0)}"
    assert counts.get('WARNING', 0) == 2, f"count_by_level: expected 2 WARNINGs, got {counts.get('WARNING', 0)}"
    print("[PASS] count_by_level")
    
    # Check 5: extract_latency_values
    entries = list(stream_log_file(test_file))
    latencies = list(extract_latency_values(iter(entries)))
    assert len(latencies) == 6, f"extract_latency_values: expected 6 latencies, got {len(latencies)}"
    assert 45.0 in latencies, "extract_latency_values: should contain 45.0"
    print("[PASS] extract_latency_values")
    
    # Check 6: compute_latency_stats
    entries = list(stream_log_file(test_file))
    stats = compute_latency_stats(iter(entries))
    assert stats is not None, "compute_latency_stats: returned None"
    assert stats['count'] == 6, f"compute_latency_stats: expected count 6, got {stats['count']}"
    assert stats['min'] == 23.0, f"compute_latency_stats: expected min 23.0, got {stats['min']}"
    assert stats['max'] == 156.0, f"compute_latency_stats: expected max 156.0, got {stats['max']}"
    print("[PASS] compute_latency_stats")
    
    # Check 7: chunked_iterator
    result = list(chunked_iterator([1, 2, 3, 4, 5, 6, 7], 3))
    assert result == [[1, 2, 3], [4, 5, 6], [7]], f"chunked_iterator: got {result}"
    print("[PASS] chunked_iterator")
    
    # Check 8: top_k_latencies
    entries = list(stream_log_file(test_file))
    top = top_k_latencies(iter(entries), k=3)
    assert top == [156.0, 78.0, 45.0], f"top_k_latencies: expected [156,78,45], got {top}"
    print("[PASS] top_k_latencies")
    
    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)
    print("\nNote: running_percentile uses random sampling, so results vary.")
    print("Test it manually with: running_percentile(stream_log_file('test.log'), 90)")


if __name__ == "__main__":
    check()
