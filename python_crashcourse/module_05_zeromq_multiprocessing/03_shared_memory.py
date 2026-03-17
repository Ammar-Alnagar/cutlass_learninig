"""
Module 05 - Exercise 03: Shared Memory for NumPy Arrays

Scenario: You're processing large image batches or feature matrices that
are too expensive to pickle and copy between processes. Using shared
memory, multiple processes can access the same data without serialization.

Topics covered:
- multiprocessing.shared_memory (Python 3.8+)
- Creating numpy arrays backed by shared memory
- Passing shared memory names between processes
- Cleanup and resource management

Note: Requires Python 3.8+
"""

import numpy as np
from multiprocessing import Process, shared_memory
import time


# =============================================================================
# Part 1: Creating Shared Memory
# =============================================================================

def create_shared_array(shape, dtype=np.float32):
    """
    Create a numpy array backed by shared memory.
    
    This array can be accessed by multiple processes without copying.
    
    Args:
        shape: Tuple shape of the array
        dtype: NumPy dtype
        
    Returns:
        tuple: (numpy_array, SharedMemory object)
    """
    # TODO: Calculate size in bytes: np.prod(shape) * np.dtype(dtype).itemsize
    size = None
    
    # TODO: Create SharedMemory with size
    shm = None
    
    # TODO: Create numpy array backed by shared memory buffer
    arr = None
    
    return arr, shm


def fill_shared_array(arr, value):
    """
    Fill a shared array with a value.
    
    Args:
        arr: Shared numpy array
        value: Value to fill
    """
    # TODO: Fill array with value using arr.fill() or arr[:] = value
    pass


# =============================================================================
# Part 2: Accessing Shared Memory from Another Process
# =============================================================================

def worker_process_array(shm_name, shape, dtype, offset=0):
    """
    Worker that attaches to existing shared memory and modifies it.
    
    Args:
        shm_name: Name of the shared memory block
        shape: Shape of the array
        dtype: Data type
        offset: Value to add to each element
    """
    # TODO: Attach to existing shared memory by name
    shm = None
    
    # TODO: Create numpy array viewing the shared memory
    arr = None
    
    # TODO: Modify the array (e.g., add offset to each element)
    
    # TODO: Close the shared memory (but don't unlink - parent owns it)
    shm.close()


def run_shared_memory_demo():
    """
    Demonstrate sharing a large array between processes.
    
    Returns:
        numpy.ndarray: Final state of the shared array
    """
    shape = (1000, 100)
    
    # TODO: Create shared array in parent process
    arr, shm = create_shared_array(shape)
    
    # TODO: Initialize array with values
    arr[:] = np.arange(np.prod(shape)).reshape(shape)
    
    # TODO: Create worker process that will modify the array
    p = None  # Create Process
    
    # TODO: Start and join the process
    p.start()
    p.join()
    
    # TODO: Cleanup - close and unlink shared memory
    shm.close()
    shm.unlink()
    
    return arr


# =============================================================================
# Part 3: Multiple Arrays with Named Shared Memory
# =============================================================================

class SharedMemoryManager:
    """
    Manage multiple shared memory blocks for numpy arrays.
    """
    
    def __init__(self):
        self.shm_blocks = {}
        self.arrays = {}
    
    def create_array(self, name, shape, dtype=np.float32):
        """
        Create a named shared array.
        
        Args:
            name: Unique name for this array
            shape: Array shape
            dtype: Array dtype
        """
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        
        # TODO: Store references
        self.shm_blocks[name] = None
        self.arrays[name] = None
    
    def get_array(self, name, shape, dtype=np.float32):
        """
        Attach to an existing named shared array.
        
        Args:
            name: Name of the shared array
            shape: Array shape (must match original)
            dtype: Data type (must match original)
            
        Returns:
            numpy.ndarray: Array backed by shared memory
        """
        # TODO: Attach to existing shared memory by name
        shm = None
        arr = None
        return arr
    
    def close_all(self):
        """Close and unlink all shared memory blocks."""
        for name, shm in self.shm_blocks.items():
            shm.close()
            shm.unlink()
        self.shm_blocks.clear()
        self.arrays.clear()


# =============================================================================
# Part 4: Shared Memory for Model Weights
# =============================================================================

def worker_update_weights(shm_name, shape, update_factor):
    """
    Worker that updates shared model weights.
    
    Args:
        shm_name: Name of shared memory with weights
        shape: Shape of weight matrix
        update_factor: Factor to multiply updates by
    """
    shm = None
    weights = None
    # TODO: Apply update
    pass
    shm.close()


def parallel_weight_update():
    """
    Demonstrate multiple workers updating shared weights.
    
    Returns:
        numpy.ndarray: Final weights after all updates
    """
    shape = (100, 50)
    initial_weights = np.random.randn(*shape).astype(np.float32)
    
    # TODO: Create shared memory for weights
    size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=size, name="model_weights")
    weights = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    weights[:] = initial_weights
    
    # TODO: Create multiple worker processes
    processes = []
    factors = [0.9, 0.8, 0.7]
    
    for i, factor in enumerate(factors):
        p = None  # Create Process
        processes.append(p)
    
    # TODO: Start and join all processes
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    final_weights = weights.copy()
    
    # Cleanup
    shm.close()
    shm.unlink()
    
    return final_weights


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 05 - Exercise 03: Self-Check")
    print("=" * 60)
    
    # Check 1: create_shared_array
    arr, shm = create_shared_array((10, 10))
    assert arr.shape == (10, 10), f"create_shared_array: wrong shape {arr.shape}"
    assert arr.dtype == np.float32, f"create_shared_array: wrong dtype {arr.dtype}"
    shm.close()
    shm.unlink()
    print("[PASS] create_shared_array")
    
    # Check 2: fill_shared_array
    arr, shm = create_shared_array((5, 5))
    fill_shared_array(arr, 42.0)
    assert np.all(arr == 42.0), f"fill_shared_array: expected all 42.0, got {arr[0,0]}"
    shm.close()
    shm.unlink()
    print("[PASS] fill_shared_array")
    
    # Check 3: run_shared_memory_demo
    final_arr = run_shared_memory_demo()
    assert final_arr is not None, "run_shared_memory_demo: returned None"
    print("[PASS] run_shared_memory_demo")
    
    # Check 4: SharedMemoryManager
    manager = SharedMemoryManager()
    manager.create_array("test_arr", (20, 20))
    assert "test_arr" in manager.arrays, "SharedMemoryManager: array not stored"
    assert manager.arrays["test_arr"].shape == (20, 20), "SharedMemoryManager: wrong shape"
    manager.close_all()
    print("[PASS] SharedMemoryManager")
    
    # Check 5: parallel_weight_update
    final_weights = parallel_weight_update()
    assert final_weights.shape == (100, 50), f"parallel_weight_update: wrong shape {final_weights.shape}"
    print("[PASS] parallel_weight_update")
    
    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
