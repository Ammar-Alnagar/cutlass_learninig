"""
Module 05 - Exercise 01: ZeroMQ Basics

Scenario: You're building a distributed inference system where a central
coordinator sends requests to multiple worker nodes. ZeroMQ provides
lightweight, high-performance messaging without the overhead of HTTP.

Topics covered:
- ZeroMQ socket types (PUSH/PULL, REQ/REP)
- Bind vs connect patterns
- Sending and receiving messages
- Multipart messages for request routing

Prerequisites:
    pip install pyzmq
"""

import zmq
import time
import threading


# =============================================================================
# Part 1: PUSH/PULL Pattern (Pipeline)
# =============================================================================

def create_push_socket(context, address):
    """
    Create a PUSH socket that sends messages downstream.
    
    PUSH sockets distribute messages round-robin among connected PULL sockets.
    This is ideal for load-balancing work among workers.
    
    Args:
        context: ZeroMQ context
        address: Address to bind (e.g., "tcp://*:5555")
        
    Returns:
        zmq.Socket: Configured PUSH socket
    """
    # TODO: Create a PUSH socket using context.socket(zmq.PUSH)
    # Bind to the specified address
    socket = None
    return socket


def create_pull_socket(context, address):
    """
    Create a PULL socket that receives messages from upstream.
    
    PULL sockets collect messages from PUSH sockets in round-robin fashion.
    
    Args:
        context: ZeroMQ context
        address: Address to connect (e.g., "tcp://localhost:5555")
        
    Returns:
        zmq.Socket: Configured PULL socket
    """
    # TODO: Create a PULL socket using context.socket(zmq.PULL)
    # Connect to the specified address
    socket = None
    return socket


def send_message(push_socket, message):
    """
    Send a message through a PUSH socket.
    
    ZeroMQ handles reconnection, buffering, and delivery automatically.
    
    Args:
        push_socket: The PUSH socket
        message: String or bytes to send
    """
    # TODO: Send the message using socket.send()
    # Remember to encode strings to bytes: message.encode('utf-8')
    pass


def receive_message(pull_socket, timeout_ms=1000):
    """
    Receive a message from a PULL socket with timeout.
    
    Args:
        pull_socket: The PULL socket
        timeout_ms: Timeout in milliseconds
        
    Returns:
        bytes or None: Received message, or None if timeout
    """
    # TODO: Use socket.recv() with zmq.NOBLOCK or set RCVTIMEO
    # Handle zmq.Again exception which indicates timeout
    try:
        message = None
        return message
    except zmq.Again:
        return None


# =============================================================================
# Part 2: REQ/REP Pattern (Request-Reply)
# =============================================================================

def create_rep_socket(context, address):
    """
    Create a REP (reply) socket for server-side request handling.
    
    REP sockets must alternate between recv() and send().
    This enforces a strict request-reply pattern.
    
    Args:
        context: ZeroMQ context
        address: Address to bind
        
    Returns:
        zmq.Socket: Configured REP socket
    """
    # TODO: Create REP socket and bind
    socket = None
    return socket


def create_req_socket(context, address):
    """
    Create a REQ (request) socket for client-side requests.
    
    REQ sockets must alternate between send() and recv().
    Each send() blocks until the reply is received.
    
    Args:
        context: ZeroMQ context
        address: Address to connect
        
    Returns:
        zmq.Socket: Configured REQ socket
    """
    # TODO: Create REQ socket and connect
    socket = None
    return socket


def handle_request(rep_socket, handler_fn):
    """
    Handle a single request using a provided handler function.
    
    Args:
        rep_socket: The REP socket
        handler_fn: Function that takes request bytes and returns response bytes
    """
    # TODO: Receive request, call handler_fn, send response
    # REQ/REP pattern requires: recv() then send()
    request = None
    response = handler_fn(request)
    # TODO: Send response
    pass


# =============================================================================
# Part 3: Multipart Messages
# =============================================================================

def send_multipart_envelope(socket, envelope, data):
    """
    Send a multipart message with envelope pattern.
    
    Multipart messages are atomic — either all parts arrive or none.
    The envelope can contain routing info, message type, etc.
    
    Args:
        socket: ZeroMQ socket
        envelope: First part (e.g., message type, routing key)
        data: Second part (actual payload)
    """
    # TODO: Use socket.send_multipart([envelope, data])
    # Remember to encode strings to bytes
    pass


def receive_multipart_envelope(socket):
    """
    Receive a multipart message.
    
    Args:
        socket: ZeroMQ socket
        
    Returns:
        tuple: (envelope, data) as bytes, or (None, None) on timeout
    """
    # TODO: Use socket.recv_multipart() to receive all parts
    # Handle timeout appropriately
    try:
        parts = None
        if len(parts) >= 2:
            return parts[0], parts[1]
        return None, None
    except zmq.Again:
        return None, None


# =============================================================================
# Part 4: Polling (I/O Multiplexing)
# =============================================================================

def poll_multiple_sockets(sockets, timeout_ms=1000):
    """
    Wait for messages on multiple sockets simultaneously.
    
    Use case: A worker that receives tasks from multiple sources.
    
    Args:
        sockets: List of ZeroMQ sockets
        timeout_ms: Timeout in milliseconds
        
    Returns:
        list: Sockets that have messages ready
    """
    # TODO: Create zmq.Poller, register all sockets
    # Use poller.poll(timeout_ms) to wait
    # Return list of sockets with events
    poller = None  # Create Poller
    
    # TODO: Register each socket for POLLIN event
    
    # TODO: Poll and collect sockets with events
    events = None
    ready_sockets = []
    
    return ready_sockets


# =============================================================================
# Part 5: Practical Example - Task Queue
# =============================================================================

def run_task_queue_demo():
    """
    Demonstrate a simple task queue with PUSH/PULL.
    
    This runs a ventilator (push), workers (pull), and sink (collect results)
    in threads within the same process for demonstration.
    
    Returns:
        list: Results collected by the sink
    """
    context = zmq.Context()
    results = []
    
    # Create sockets
    ventilator = context.socket(zmq.PUSH)
    ventilator.bind("tcp://*:5557")
    
    worker_pull = context.socket(zmq.PULL)
    worker_pull.connect("tcp://localhost:5557")
    
    worker_push = context.socket(zmq.PUSH)
    worker_push.bind("tcp://*:5558")
    
    sink = context.socket(zmq.PULL)
    sink.bind("tcp://*:5558")
    
    # TODO: Send 5 tasks from ventilator
    # Tasks are strings like "task_1", "task_2", etc.
    for i in range(5):
        pass  # Send tasks
    
    # Give worker time to process
    time.sleep(0.1)
    
    # TODO: Collect results at sink
    # Use non-blocking receive with timeout
    for _ in range(5):
        result = receive_message(sink, timeout_ms=100)
        if result:
            results.append(result.decode('utf-8'))
    
    # Cleanup
    ventilator.close()
    worker_pull.close()
    worker_push.close()
    sink.close()
    context.term()
    
    return results


# =============================================================================
# Self-Check Functions
# =============================================================================

def check():
    """Run all checks to verify your implementations."""
    print("=" * 60)
    print("Module 05 - Exercise 01: Self-Check")
    print("=" * 60)
    
    # Check 1: Basic socket creation
    context = zmq.Context()
    
    push_sock = create_push_socket(context, "tcp://*:5560")
    assert push_sock.type == zmq.PUSH, f"Expected PUSH socket, got type {push_sock.type}"
    print("[PASS] create_push_socket")
    
    pull_sock = create_pull_socket(context, "tcp://localhost:5561")
    assert pull_sock.type == zmq.PULL, f"Expected PULL socket, got type {pull_sock.type}"
    print("[PASS] create_pull_socket")
    
    rep_sock = create_rep_socket(context, "tcp://*:5562")
    assert rep_sock.type == zmq.REP, f"Expected REP socket, got type {rep_sock.type}"
    print("[PASS] create_rep_socket")
    
    req_sock = create_req_socket(context, "tcp://localhost:5562")
    assert req_sock.type == zmq.REQ, f"Expected REQ socket, got type {req_sock.type}"
    print("[PASS] create_req_socket")
    
    # Check 2: Task queue demo
    results = run_task_queue_demo()
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    print("[PASS] run_task_queue_demo")
    
    # Cleanup
    push_sock.close()
    pull_sock.close()
    rep_sock.close()
    req_sock.close()
    context.term()
    
    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    check()
