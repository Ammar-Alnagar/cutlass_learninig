"""
Module 05 - Exercise 04: Distributed Inference Pipeline Challenge

Scenario: You're building a distributed inference system with:
1. A coordinator that receives inference requests
2. Multiple worker processes that run model inference
3. Shared memory for large input/output data
4. ZeroMQ for task distribution and result collection

This challenge integrates all concepts from Module 05.
"""

import zmq
import numpy as np
from multiprocessing import Process, shared_memory, Queue, Event
import time
import os


# =============================================================================
# Part 1: System Architecture
# =============================================================================

class InferenceCoordinator:
    """
    Central coordinator that distributes inference tasks to workers.
    """
    
    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        self.context = zmq.Context()
        self.task_queue = None
        self.request_socket = None
        self.shm_blocks = {}
        self.stop_event = Event()
        
    def setup_sockets(self):
        """Create and bind ZeroMQ sockets."""
        # TODO: Create REP socket for client requests (bind to tcp://*:5555")
        # TODO: Create PUSH socket for worker tasks (bind to tcp://*:5556")
        self.request_socket = None
        self.task_queue = None
    
    def allocate_shared_input(self, client_id, shape):
        """
        Allocate shared memory for input data.
        
        Args:
            client_id: Unique client identifier
            shape: Shape of input tensor
            
        Returns:
            tuple: (shm_name, numpy_array)
        """
        shm_name = f"inference_input_{client_id}"
        size = int(np.prod(shape) * np.float32().itemsize)
        shm = None
        arr = None
        return shm_name, arr
    
    def send_task_to_worker(self, worker_id, task_info):
        """
        Send a task description to a worker.
        
        Args:
            worker_id: Target worker
            task_info: Dict with shm_name, shape, client_id
        """
        import json
        message = json.dumps(task_info).encode('utf-8')
        # TODO: Send message
        pass
    
    def handle_client_request(self, request):
        """
        Process a client inference request.
        
        Args:
            request: Dict with 'input_data' and 'client_id'
            
        Returns:
            dict: Response with results
        """
        input_data = request.get('input_data')
        client_id = request.get('client_id')
        shape = input_data.shape if hasattr(input_data, 'shape') else np.array(input_data).shape
        
        # TODO: Implement the flow
        return {'status': 'not_implemented'}
    
    def cleanup(self):
        """Clean up all resources."""
        for shm in self.shm_blocks.values():
            shm.close()
            shm.unlink()
        if self.task_queue:
            self.task_queue.close()
        if self.request_socket:
            self.request_socket.close()
        self.context.term()


class InferenceWorker:
    """
    Worker process that performs inference on shared memory data.
    """
    
    def __init__(self, worker_id, task_address="tcp://localhost:5556"):
        self.worker_id = worker_id
        self.context = zmq.Context()
        self.task_socket = None
        self.task_address = task_address
        
    def setup_socket(self):
        """Create and connect PULL socket for tasks."""
        # TODO: Create PULL socket and connect to task_address
        self.task_socket = None
    
    def receive_task(self, timeout_ms=1000):
        """
        Wait for a task from the coordinator.
        
        Returns:
            dict: Task info, or None on timeout
        """
        try:
            # TODO: Receive and parse message
            return None
        except zmq.Again:
            return None
    
    def attach_to_shared_input(self, shm_name, shape):
        """
        Attach to existing shared memory with input data.
        
        Args:
            shm_name: Name of shared memory block
            shape: Shape of input array
            
        Returns:
            numpy.ndarray: Array view of shared memory
        """
        shm = None
        arr = None
        return arr
    
    def run_inference(self, input_data):
        """
        Perform simulated inference on input data.
        
        Args:
            input_data: Input numpy array
            
        Returns:
            numpy.ndarray: Inference results
        """
        # TODO: Implement a simple "inference" computation
        result = None
        return result
    
    def run(self):
        """Main worker loop."""
        self.setup_socket()
        print(f"Worker {self.worker_id} started, waiting for tasks...")
        
        while True:
            task = self.receive_task()
            if task is None:
                continue
                
            print(f"Worker {self.worker_id}: Processing task {task}")
            # TODO: Implement task processing


# =============================================================================
# Part 2: Worker Pool Manager
# =============================================================================

class WorkerPoolManager:
    """
    Manage a pool of inference worker processes.
    """
    
    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        self.workers = []
        self.worker_processes = []
        
    def spawn_workers(self):
        """Spawn all worker processes."""
        for i in range(self.num_workers):
            # TODO: Create Process targeting worker main function
            pass
    
    def check_worker_health(self):
        """
        Check if all workers are still alive.
        
        Returns:
            list: Indices of dead workers
        """
        dead_workers = []
        for i, proc in enumerate(self.worker_processes):
            if not proc.is_alive():
                dead_workers.append(i)
        return dead_workers
    
    def restart_worker(self, worker_index):
        """
        Restart a dead worker process.
        
        Args:
            worker_index: Index of worker to restart
        """
        # TODO: Terminate old process and create new one
        pass
    
    def shutdown(self):
        """Gracefully shutdown all workers."""
        for proc in self.worker_processes:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()


# =============================================================================
# Part 3: End-to-End Demo
# =============================================================================

def run_distributed_inference_demo():
    """
    Run a complete distributed inference demonstration.
    
    Returns:
        dict: Demo results and statistics
    """
    print("Starting distributed inference demo...")
    
    # TODO: Set up the complete system and run demo
    
    stats = {
        'requests_processed': 0,
        'avg_latency_ms': 0,
        'workers_used': 0,
    }
    
    return stats


# =============================================================================
# Part 4: Performance Comparison
# =============================================================================

def benchmark_sequential_vs_parallel(data_size=1000, num_workers=4):
    """
    Compare sequential processing vs parallel multiprocessing.
    
    Args:
        data_size: Number of items to process
        num_workers: Number of parallel workers
        
    Returns:
        dict: Timing comparison results
    """
    import time
    
    # Simulated computation (CPU-bound)
    def compute_item(x):
        return sum(i * i for i in range(1000))
    
    # TODO: Benchmark sequential execution
    start = time.perf_counter()
    sequential_results = [compute_item(x) for x in range(data_size)]
    sequential_time = time.perf_counter() - start
    
    # TODO: Benchmark parallel execution using ProcessPool
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        start = time.perf_counter()
        parallel_results = list(executor.map(compute_item, range(data_size)))
        parallel_time = time.perf_counter() - start
    
    # TODO: Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    
    return {
        'sequential_time_ms': sequential_time * 1000,
        'parallel_time_ms': parallel_time * 1000,
        'speedup': speedup,
        'num_workers': num_workers,
    }


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 05 - Exercise 04: Self-Check")
    print("=" * 60)
    
    # Check 1: Benchmark comparison
    print("\nRunning performance benchmark...")
    results = benchmark_sequential_vs_parallel(data_size=100, num_workers=4)
    
    print(f"Sequential: {results['sequential_time_ms']:.2f} ms")
    print(f"Parallel:   {results['parallel_time_ms']:.2f} ms")
    print(f"Speedup:    {results['speedup']:.2f}x")
    
    assert results['sequential_time_ms'] > 0, "Sequential time should be > 0"
    assert results['parallel_time_ms'] > 0, "Parallel time should be > 0"
    print("[PASS] benchmark_sequential_vs_parallel")
    
    # Check 2: WorkerPoolManager basic functionality
    manager = WorkerPoolManager(num_workers=2)
    assert manager.num_workers == 2, "WorkerPoolManager: num_workers not set"
    print("[PASS] WorkerPoolManager initialization")
    
    # Check 3: InferenceCoordinator basic functionality
    coordinator = InferenceCoordinator(num_workers=2)
    assert coordinator.num_workers == 2, "InferenceCoordinator: num_workers not set"
    print("[PASS] InferenceCoordinator initialization")
    
    print("\n" + "=" * 60)
    print("Basic checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
