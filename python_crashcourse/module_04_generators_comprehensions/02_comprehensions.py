"""
Module 04 - Exercise 02: Comprehensions

Scenario: You're preprocessing a dataset of model predictions. You need to
filter invalid predictions, extract confidence scores, and build lookup
tables — all common data transformation tasks in ML evaluation pipelines.

Topics covered:
- List comprehensions (eager evaluation)
- Dict and set comprehensions
- Nested comprehensions
- Generator expressions (lazy evaluation)
- Conditional filtering in comprehensions
"""


# =============================================================================
# Part 1: List Comprehensions
# =============================================================================

def square_numbers(numbers):
    """
    Square each number in a list using list comprehension.
    
    List comprehensions are more concise and often faster than
    equivalent for loops with append().
    
    Args:
        numbers: List of numbers
        
    Returns:
        list: Squared values
    """
    # TODO: Use list comprehension to square each number
    # Syntax: [expression for item in iterable]
    result = None
    return result


def filter_positive(numbers):
    """
    Keep only positive numbers using list comprehension with condition.
    
    The if clause filters elements before they're included in the result.
    
    Args:
        numbers: List of numbers (may include negatives and zero)
        
    Returns:
        list: Only positive numbers (> 0)
    """
    # TODO: Use list comprehension with if condition to filter positives
    # Syntax: [expression for item in iterable if condition]
    result = None
    return result


def transform_and_filter(predictions):
    """
    Apply transformation and filter in one comprehension.
    
    Scenario: Extract confidence scores above a threshold from prediction tuples.
    
    Args:
        predictions: List of (label, confidence) tuples
        
    Returns:
        list: Confidence values > 0.5
    """
    # TODO: Extract confidence values where confidence > 0.5
    # Input example: [('cat', 0.9), ('dog', 0.3), ('bird', 0.7)]
    # Expected output: [0.9, 0.7]
    result = None
    return result


def flatten_matrix(matrix):
    """
    Flatten a 2D matrix using nested list comprehension.
    
    Nested comprehensions read left-to-right: first the outer loop,
    then the inner loop.
    
    Args:
        matrix: 2D list (list of lists)
        
    Returns:
        list: All elements in a single flat list
    """
    # TODO: Flatten using nested comprehension
    # Syntax: [item for row in matrix for item in row]
    result = None
    return result


# =============================================================================
# Part 2: Dict and Set Comprehensions
# =============================================================================

def create_label_mapping(labels):
    """
    Create a label-to-index mapping using dict comprehension.
    
    Common pattern in ML: mapping string labels to integer indices
    for model input/output.
    
    Args:
        labels: List of label strings
        
    Returns:
        dict: {label: index} for each unique label
    """
    # TODO: Create dict mapping each label to its index
    # Use enumerate(labels) to get (index, label) pairs
    # Syntax: {key: value for item in iterable}
    result = None
    return result


def invert_mapping(label_to_idx):
    """
    Invert a dict (swap keys and values) using dict comprehension.
    
    Args:
        label_to_idx: Dict mapping labels to indices
        
    Returns:
        dict: {index: label} inverse mapping
    """
    # TODO: Invert the mapping using dict comprehension
    # Syntax: {value: key for key, value in original_dict.items()}
    result = None
    return result


def unique_labels(predictions):
    """
    Extract unique labels using set comprehension.
    
    Sets automatically deduplicate. Useful for finding unique
    classes in a dataset.
    
    Args:
        predictions: List of (label, confidence) tuples
        
    Returns:
        set: Unique label strings
    """
    # TODO: Extract unique labels using set comprehension
    # Syntax: {expression for item in iterable}
    result = None
    return result


# =============================================================================
# Part 3: Generator Expressions
# =============================================================================

def sum_of_squares(numbers):
    """
    Compute sum of squares using a generator expression.
    
    Generator expressions use () instead of []. They compute values
    lazily, one at a time, which is memory-efficient for large inputs.
    
    Key difference:
    - [x*x for x in numbers] creates a full list in memory
    - (x*x for x in numbers) yields values one at a time
    
    Args:
        numbers: List of numbers
        
    Returns:
        int/float: Sum of squared values
    """
    # TODO: Use generator expression with sum() to compute sum of squares
    # Syntax: sum(expression for item in iterable)
    result = None
    return result


def first_high_confidence(predictions, threshold=0.8):
    """
    Find first prediction above threshold using generator expression.
    
    Generator expressions are ideal when you need only the first
    matching element — they stop computing once found.
    
    Args:
        predictions: List of (label, confidence) tuples
        threshold: Minimum confidence to match
        
    Returns:
        tuple or None: First (label, confidence) above threshold, or None
    """
    # TODO: Use next() with a generator expression to find first match
    # Syntax: next((item for item in iterable if condition), default)
    result = None
    return result


def memory_comparison():
    """
    Demonstrate memory difference between list comprehension and generator.
    
    Returns:
        tuple: (list_size_bytes, generator_size_bytes)
    """
    import sys
    
    # List comprehension: materializes all 1 million values
    list_comp = [x * 2 for x in range(1000000)]
    
    # Generator expression: stores only the generator object
    gen_expr = (x * 2 for x in range(1000000))
    
    # TODO: Get sizes using sys.getsizeof()
    list_size = None
    gen_size = None
    
    return list_size, gen_size


# =============================================================================
# Part 4: Advanced Comprehensions
# =============================================================================

def transpose_matrix(matrix):
    """
    Transpose a matrix using nested list comprehension.
    
    Args:
        matrix: 2D list (m x n)
        
    Returns:
        list: Transposed matrix (n x m)
    """
    # TODO: Transpose using nested comprehension
    # For each column index j, create a row of matrix[i][j] for all i
    # Syntax: [[matrix[i][j] for i in range(rows)] for j in range(cols)]
    result = None
    return result


def conditional_transform(values):
    """
    Apply different transformations based on conditions.
    
    Scenario: Normalize values differently based on their range.
    
    Args:
        values: List of numbers
        
    Returns:
        list: Squared if < 5, unchanged if 5-10, halved if > 10
    """
    # TODO: Use list comprehension with conditional expression
    # Syntax: [transform1 if cond1 else transform2 if cond2 else transform3 for x in values]
    result = None
    return result


def zip_comprehension(keys, values):
    """
    Combine two lists using zip in a comprehension.
    
    Args:
        keys: List of keys
        values: List of values
        
    Returns:
        dict: {key: value} pairs from zipped lists
    """
    # TODO: Create dict using zip and dict comprehension
    result = None
    return result


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 04 - Exercise 02: Self-Check")
    print("=" * 60)
    
    # Check 1: square_numbers
    result = square_numbers([1, 2, 3, 4])
    assert result == [1, 4, 9, 16], f"square_numbers: expected [1,4,9,16], got {result}"
    print("[PASS] square_numbers")
    
    # Check 2: filter_positive
    result = filter_positive([-2, -1, 0, 1, 2, 3])
    assert result == [1, 2, 3], f"filter_positive: expected [1,2,3], got {result}"
    print("[PASS] filter_positive")
    
    # Check 3: transform_and_filter
    preds = [('cat', 0.9), ('dog', 0.3), ('bird', 0.7), ('fish', 0.4)]
    result = transform_and_filter(preds)
    assert result == [0.9, 0.7], f"transform_and_filter: expected [0.9,0.7], got {result}"
    print("[PASS] transform_and_filter")
    
    # Check 4: flatten_matrix
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = flatten_matrix(matrix)
    assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9], f"flatten_matrix: expected [1..9], got {result}"
    print("[PASS] flatten_matrix")
    
    # Check 5: create_label_mapping
    labels = ['cat', 'dog', 'bird']
    result = create_label_mapping(labels)
    assert result == {'cat': 0, 'dog': 1, 'bird': 2}, f"create_label_mapping: got {result}"
    print("[PASS] create_label_mapping")
    
    # Check 6: invert_mapping
    mapping = {'cat': 0, 'dog': 1, 'bird': 2}
    result = invert_mapping(mapping)
    assert result == {0: 'cat', 1: 'dog', 2: 'bird'}, f"invert_mapping: got {result}"
    print("[PASS] invert_mapping")
    
    # Check 7: unique_labels
    preds = [('cat', 0.9), ('dog', 0.3), ('cat', 0.8), ('bird', 0.7)]
    result = unique_labels(preds)
    assert result == {'cat', 'dog', 'bird'}, f"unique_labels: got {result}"
    print("[PASS] unique_labels")
    
    # Check 8: sum_of_squares
    result = sum_of_squares([1, 2, 3, 4])
    assert result == 30, f"sum_of_squares: expected 30, got {result}"
    print("[PASS] sum_of_squares")
    
    # Check 9: first_high_confidence
    preds = [('cat', 0.5), ('dog', 0.7), ('bird', 0.9), ('fish', 0.85)]
    result = first_high_confidence(preds, threshold=0.8)
    assert result == ('bird', 0.9), f"first_high_confidence: got {result}"
    print("[PASS] first_high_confidence")
    
    # Check 10: memory_comparison
    list_size, gen_size = memory_comparison()
    assert list_size > gen_size * 100, f"memory: list ({list_size}) should be >> generator ({gen_size})"
    print(f"[PASS] memory_comparison (list: {list_size} bytes, generator: {gen_size} bytes)")
    
    # Check 11: transpose_matrix
    matrix = [[1, 2, 3], [4, 5, 6]]
    result = transpose_matrix(matrix)
    assert result == [[1, 4], [2, 5], [3, 6]], f"transpose_matrix: got {result}"
    print("[PASS] transpose_matrix")
    
    # Check 12: conditional_transform
    result = conditional_transform([1, 5, 10, 20])
    assert result == [1, 5, 10, 10], f"conditional_transform: expected [1,5,10,10], got {result}"
    print("[PASS] conditional_transform")
    
    # Check 13: zip_comprehension
    keys = ['a', 'b', 'c']
    values = [1, 2, 3]
    result = zip_comprehension(keys, values)
    assert result == {'a': 1, 'b': 2, 'c': 3}, f"zip_comprehension: got {result}"
    print("[PASS] zip_comprehension")
    
    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
