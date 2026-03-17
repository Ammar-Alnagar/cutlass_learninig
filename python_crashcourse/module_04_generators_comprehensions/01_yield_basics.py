"""
Module 04 - Exercise 01: Yield Basics

Scenario: You're building a data loader that reads training samples from
disk. Loading all samples into memory at once would cause OOM errors.
Instead, you'll use generators to yield one sample at a time.

Topics covered:
- yield mechanics (pauses execution, preserves local state)
- generator objects and iteration
- send() for two-way communication with generators
- yield from for delegating to sub-generators
"""


# =============================================================================
# Part 1: Basic Generator Functions
# =============================================================================

def number_generator(n):
    """
    Generate numbers from 0 to n-1, one at a time.
    
    Unlike a function that returns a list, this generator yields values
    one at a time. The function's local state (including the value of i)
    is preserved between calls.
    
    Args:
        n: Upper bound (exclusive)
        
    Yields:
        int: Numbers from 0 to n-1
    """
    # TODO: Use a for loop and yield to generate numbers 0 to n-1
    # The function pauses at each yield, returning control to the caller
    # When iteration resumes, execution continues from the yield point
    pass  # Replace with your implementation


def finite_sequence():
    """
    A generator that yields a fixed sequence of values.
    
    This demonstrates that generators can yield any values, not just
    numbers from a loop. Useful for yielding pre-computed constants
    or configuration values lazily.
    
    Yields:
        str: "start", "process", "end" in order
    """
    # TODO: Yield the three strings in order
    pass  # Replace with your implementation


def manual_iteration():
    """
    Demonstrate manual generator iteration using next().
    
    Generators are iterators — they implement __iter__ and __next__.
    You can call next() directly instead of using a for loop.
    
    Returns:
        tuple: (first_value, second_value, third_value) from the generator
    """
    gen = number_generator(5)
    
    # TODO: Use next(gen) three times to get the first three values
    first = None
    second = None
    third = None
    
    return first, second, third


# =============================================================================
# Part 2: Generator State Preservation
# =============================================================================

def running_sum():
    """
    A generator that maintains running state.
    
    Each time you send a value to this generator, it adds it to a
    running total and yields the new sum. This demonstrates how
    generators preserve local variables across yields.
    
    Yields:
        int: The running sum after each input
        
    Receives:
        int: Values to add to the running sum (via send())
    """
    total = 0
    while True:
        # TODO: Receive a value via send(), add to total, yield the new sum
        # Use: value = yield total  (this yields current total, receives next input)
        # Then add the received value to total
        pass  # Replace with your implementation


def test_running_sum():
    """
    Test the running_sum generator with send().
    
    send() does two things:
    1. Sends a value into the generator (becomes the yield expression result)
    2. Advances the generator to the next yield
    
    Returns:
        list: Running sums after sending [1, 2, 3, 4, 5]
    """
    gen = running_sum()
    results = []
    
    # Prime the generator (advance to first yield)
    next(gen)
    
    # TODO: Send values 1 through 5, collecting the yielded sums
    # Each send() returns the yielded value from the generator
    # Expected results: [1, 3, 6, 10, 15]
    
    return results


# =============================================================================
# Part 3: Yield From (Delegation)
# =============================================================================

def sub_generator():
    """A simple sub-generator that yields letters."""
    yield 'a'
    yield 'b'
    yield 'c'


def delegating_generator():
    """
    Delegate to sub_generator using yield from.
    
    yield from is syntactic sugar for iterating over another
    generator. It automatically yields all values from the
    sub-generator, then resumes with subsequent code.
    
    Yields:
        str: Numbers 1-3, then letters a-c, then numbers 4-5
    """
    # TODO: Yield numbers 1, 2, 3
    # TODO: Use yield from to delegate to sub_generator()
    # TODO: Yield numbers 4, 5
    pass  # Replace with your implementation


def flatten_nested(nested_list):
    """
    Flatten a nested list using yield from.
    
    This is a common pattern: recursively yield from sub-generators
    to flatten arbitrarily nested structures.
    
    Args:
        nested_list: A list that may contain nested lists (e.g., [1, [2, 3], 4])
        
    Yields:
        int: Each non-list element in depth-first order
    """
    for item in nested_list:
        if isinstance(item, list):
            # TODO: Recursively yield from flatten_nested(item)
            pass  # Replace with yield from call
        else:
            # TODO: Yield the item directly
            pass  # Replace with yield statement


# =============================================================================
# Part 4: Generator Lifecycle
# =============================================================================

def countdown(n):
    """
    A generator that counts down from n to 1.
    
    Demonstrates that generators naturally terminate when they
    run out of yield statements, raising StopIteration.
    
    Args:
        n: Starting number
        
    Yields:
        int: Numbers from n down to 1
    """
    # TODO: Yield numbers from n down to 1
    pass  # Replace with your implementation


def consume_generator():
    """
    Consume a generator in different ways.
    
    Generators can be consumed by:
    - for loops
    - list() to materialize all values
    - sum(), max(), min() for aggregations
    - next() for single values
    
    Returns:
        tuple: (as_list, total, max_value) from countdown(5)
    """
    # TODO: Create a generator for countdown(5)
    # Convert to list
    as_list = None
    
    # TODO: Create a fresh generator and compute sum
    total = None
    
    # TODO: Create a fresh generator and find max value
    max_value = None
    
    return as_list, total, max_value


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 04 - Exercise 01: Self-Check")
    print("=" * 60)
    
    # Check 1: number_generator
    result = list(number_generator(5))
    assert result == [0, 1, 2, 3, 4], f"number_generator: expected [0,1,2,3,4], got {result}"
    print("[PASS] number_generator")
    
    # Check 2: finite_sequence
    result = list(finite_sequence())
    assert result == ['start', 'process', 'end'], f"finite_sequence: expected ['start','process','end'], got {result}"
    print("[PASS] finite_sequence")
    
    # Check 3: manual_iteration
    first, second, third = manual_iteration()
    assert (first, second, third) == (0, 1, 2), f"manual_iteration: expected (0,1,2), got {(first, second, third)}"
    print("[PASS] manual_iteration")
    
    # Check 4: running_sum
    results = test_running_sum()
    assert results == [1, 3, 6, 10, 15], f"running_sum: expected [1,3,6,10,15], got {results}"
    print("[PASS] running_sum")
    
    # Check 5: delegating_generator
    result = list(delegating_generator())
    assert result == [1, 2, 3, 'a', 'b', 'c', 4, 5], f"delegating_generator: expected [1,2,3,'a','b','c',4,5], got {result}"
    print("[PASS] delegating_generator")
    
    # Check 6: flatten_nested
    nested = [1, [2, 3], [4, [5, 6]], 7]
    result = list(flatten_nested(nested))
    assert result == [1, 2, 3, 4, 5, 6, 7], f"flatten_nested: expected [1,2,3,4,5,6,7], got {result}"
    print("[PASS] flatten_nested")
    
    # Check 7: consume_generator
    as_list, total, max_val = consume_generator()
    assert as_list == [5, 4, 3, 2, 1], f"consume_generator list: expected [5,4,3,2,1], got {as_list}"
    assert total == 15, f"consume_generator sum: expected 15, got {total}"
    assert max_val == 5, f"consume_generator max: expected 5, got {max_val}"
    print("[PASS] consume_generator")
    
    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
