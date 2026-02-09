# Distributed Concurrency - Hands-on Example (Python Version)
# This example simulates distributed systems concepts using multiple processes and threads

import threading
import multiprocessing
import queue
import time
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
import socket
import selectors
import pickle

# Simulated network message
@dataclass
class Message:
    from_node: int
    to_node: int
    msg_type: str  # "request", "response", "heartbeat", etc.
    data: str
    term: int = 0
    timestamp: int = 0

# Simple simulated network layer using queues
class NetworkLayer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.node_queues = [queue.Queue() for _ in range(num_nodes)]
        self.node_handlers = [None] * num_nodes
        self.running = True
        
    def register_node_handler(self, node_id, handler):
        self.node_handlers[node_id] = handler
        
    def send_message(self, msg):
        # Simulate network delay
        time.sleep(random.uniform(0.01, 0.05))  # 10-50ms delay
        
        if 0 <= msg.to_node < self.num_nodes:
            self.node_queues[msg.to_node].put(msg)
    
    def start_processing(self):
        def process_messages(node_id):
            while self.running:
                try:
                    msg = self.node_queues[node_id].get(timeout=0.1)
                    if self.node_handlers[node_id]:
                        self.node_handlers[node_id](msg)
                except queue.Empty:
                    continue
        
        for i in range(self.num_nodes):
            threading.Thread(target=process_messages, args=(i,), daemon=True).start()
    
    def stop(self):
        self.running = False

# Simple Raft consensus algorithm simulation
class RaftNode:
    def __init__(self, node_id, num_nodes, network):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.network = network
        
        # Node state
        self.role = "follower"  # "follower", "candidate", "leader"
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Timers
        self.heartbeat_timeout = 150  # milliseconds
        self.election_timeout = 300   # milliseconds
        self.last_heartbeat = time.time()
        
        # Register message handler
        self.network.register_node_handler(node_id, self.handle_message)
        
        self.running = True
        self.state_lock = threading.Lock()
    
    def start(self):
        threading.Thread(target=self.election_timer, daemon=True).start()
        threading.Thread(target=self.heartbeat_timer, daemon=True).start()
    
    def election_timer(self):
        while self.running:
            time.sleep(0.01)  # Check every 10ms
            
            with self.state_lock:
                if (self.role != "leader" and 
                    (time.time() - self.last_heartbeat) > self.election_timeout / 1000):
                    self.become_candidate()
                    self.start_election()
                    self.last_heartbeat = time.time()
    
    def heartbeat_timer(self):
        while self.running:
            if self.role == "leader":
                self.send_heartbeats()
            time.sleep(self.heartbeat_timeout / 2000)  # Half the heartbeat timeout
    
    def become_follower(self, term=None):
        with self.state_lock:
            self.role = "follower"
            if term is not None:
                self.current_term = term
            self.voted_for = None
            print(f"Node {self.node_id} became follower, term {self.current_term}")
    
    def become_candidate(self):
        with self.state_lock:
            self.role = "candidate"
            self.current_term += 1
            self.voted_for = self.node_id
            print(f"Node {self.node_id} became candidate, term {self.current_term}")
    
    def become_leader(self):
        with self.state_lock:
            self.role = "leader"
            print(f"Node {self.node_id} became leader, term {self.current_term}")
    
    def start_election(self):
        with self.state_lock:
            votes_received = 1  # Vote for self
            current_term_copy = self.current_term
            
            # Request votes from other nodes
            for i in range(self.num_nodes):
                if i != self.node_id:
                    vote_request = Message(
                        from_node=self.node_id,
                        to_node=i,
                        msg_type="vote_request",
                        data=f"{current_term_copy},{len(self.log)}",
                        term=current_term_copy
                    )
                    self.network.send_message(vote_request)
    
    def send_heartbeats(self):
        with self.state_lock:
            if self.role != "leader":
                return
            
            for i in range(self.num_nodes):
                if i != self.node_id:
                    heartbeat = Message(
                        from_node=self.node_id,
                        to_node=i,
                        msg_type="heartbeat",
                        data=f"{self.current_term},{self.commit_index}",
                        term=self.current_term
                    )
                    self.network.send_message(heartbeat)
    
    def handle_message(self, msg):
        with self.state_lock:
            # Update term if received message has higher term
            if msg.term > self.current_term:
                self.current_term = msg.term
                if self.role != "follower":
                    self.become_follower(self.current_term)
            
            if msg.msg_type == "vote_request":
                self.handle_vote_request(msg)
            elif msg.msg_type == "vote_response":
                self.handle_vote_response(msg)
            elif msg.msg_type == "heartbeat":
                self.handle_heartbeat(msg)
    
    def handle_vote_request(self, msg):
        # Parse message data: term,log_length
        parts = msg.data.split(',')
        request_term = int(parts[0])
        request_log_len = int(parts[1])
        
        grant_vote = False
        
        if (request_term >= self.current_term and 
            (self.voted_for is None or self.voted_for == msg.from_node) and
            request_log_len >= len(self.log)):
            grant_vote = True
            self.current_term = request_term
            self.voted_for = msg.from_node
        
        vote_response = Message(
            from_node=self.node_id,
            to_node=msg.from_node,
            msg_type="vote_response",
            data=f"{'yes' if grant_vote else 'no'},{self.current_term}",
            term=self.current_term
        )
        self.network.send_message(vote_response)
        
        if grant_vote:
            print(f"Node {self.node_id} granted vote to node {msg.from_node}")
    
    def handle_vote_response(self, msg):
        if self.role == "candidate" and msg.term == self.current_term:
            parts = msg.data.split(',')
            vote_granted = parts[0] == "yes"
            print(f"Node {self.node_id} received vote response from {msg.from_node}, granted: {vote_granted}")
    
    def handle_heartbeat(self, msg):
        if self.role in ["candidate", "follower"]:
            # Reset election timer
            if msg.term >= self.current_term:
                self.current_term = msg.term
                self.become_follower(msg.term)
            self.last_heartbeat = time.time()

# Distributed lock simulation
class DistributedLock:
    def __init__(self, node_id, network):
        self.node_id = node_id
        self.network = network
        self.locked = False
        self.lock_holder = None
        self.waiting_queue = []  # Queue of nodes waiting for the lock
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        # Register message handler
        self.network.register_node_handler(node_id, self.handle_lock_message)
    
    def acquire(self):
        # Request the lock from the coordinator (node 0)
        lock_request = Message(
            from_node=self.node_id,
            to_node=0,  # Coordinator is node 0
            msg_type="lock_request",
            data="acquire"
        )
        self.network.send_message(lock_request)
        
        # Wait for lock grant
        with self.condition:
            self.condition.wait_for(lambda: self.lock_holder == self.node_id)
        return True
    
    def release(self):
        with self.lock:
            if self.lock_holder == self.node_id:
                self.locked = False
                self.lock_holder = None
                
                # Grant lock to next waiting node
                if self.waiting_queue:
                    next_node = self.waiting_queue.pop(0)
                    
                    self.lock_holder = next_node
                    self.locked = True
                    
                    # Send lock grant message
                    grant_msg = Message(
                        from_node=0,  # Coordinator
                        to_node=next_node,
                        msg_type="lock_grant",
                        data="acquired"
                    )
                    self.network.send_message(grant_msg)
    
    def handle_lock_message(self, msg):
        if msg.msg_type == "lock_request":
            with self.lock:
                if not self.locked:
                    # Grant the lock immediately
                    self.locked = True
                    self.lock_holder = msg.from_node
                    
                    grant_msg = Message(
                        from_node=0,  # Coordinator
                        to_node=msg.from_node,
                        msg_type="lock_grant",
                        data="acquired"
                    )
                    self.network.send_message(grant_msg)
                else:
                    # Add to waiting queue
                    self.waiting_queue.append(msg.from_node)
        elif msg.msg_type == "lock_release":
            with self.lock:
                if self.lock_holder == msg.from_node:
                    self.locked = False
                    self.lock_holder = None
                    
                    # Grant to next in queue
                    if self.waiting_queue:
                        next_node = self.waiting_queue.pop(0)
                        
                        self.lock_holder = next_node
                        self.locked = True
                        
                        grant_msg = Message(
                            from_node=0,  # Coordinator
                            to_node=next_node,
                            msg_type="lock_grant",
                            data="acquired"
                        )
                        self.network.send_message(grant_msg)
        elif msg.msg_type == "lock_grant":
            if msg.from_node == 0 and msg.to_node == self.node_id:
                with self.lock:
                    self.lock_holder = self.node_id
                    self.locked = True
                    self.condition.notify_all()

# Simulate a distributed system with multiple nodes
def raft_consensus_simulation():
    print("\n=== Raft Consensus Simulation ===")
    
    num_nodes = 3
    network = NetworkLayer(num_nodes)
    network.start_processing()
    
    nodes = []
    
    # Create nodes
    for i in range(num_nodes):
        node = RaftNode(i, num_nodes, network)
        nodes.append(node)
    
    # Start nodes
    for node in nodes:
        node.start()
    
    # Let the simulation run for a while
    time.sleep(3)
    
    # Check roles after election
    print("\nFinal node states:")
    for i, node in enumerate(nodes):
        with node.state_lock:
            print(f"Node {i}: {node.role.upper()}, Term: {node.current_term}")
    
    # Clean up
    for node in nodes:
        node.running = False

def distributed_lock_simulation():
    print("\n=== Distributed Lock Simulation ===")
    
    num_nodes = 3
    network = NetworkLayer(num_nodes)
    network.start_processing()
    
    locks = []
    for i in range(num_nodes):
        locks.append(DistributedLock(i, network))
    
    # Simulate nodes competing for the lock
    def node_worker(node_id):
        print(f"Node {node_id} attempting to acquire lock...")
        
        start_time = time.time()
        locks[node_id].acquire()
        end_time = time.time()
        
        print(f"Node {node_id} acquired lock after {(end_time - start_time)*1000:.2f} ms")
        
        # Hold the lock for a bit
        time.sleep(0.5)
        
        print(f"Node {node_id} releasing lock")
        locks[node_id].release()
    
    threads = []
    for i in range(num_nodes):
        t = threading.Thread(target=node_worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    network.stop()

def leader_election_simulation():
    print("\n=== Leader Election Simulation ===")
    
    num_nodes = 5
    network = NetworkLayer(num_nodes)
    network.start_processing()
    
    nodes = []
    
    # Create nodes
    for i in range(num_nodes):
        node = RaftNode(i, num_nodes, network)
        nodes.append(node)
    
    # Start nodes
    for node in nodes:
        node.start()
    
    # Simulate a scenario
    time.sleep(2)
    
    # Check state after some time
    leader_count = 0
    for i, node in enumerate(nodes):
        with node.state_lock:
            if node.role == "leader":
                leader_count += 1
                print(f"Node {i} is the leader")
    
    print(f"Total leaders elected: {leader_count}")
    
    # Clean up
    for node in nodes:
        node.running = False
    
    network.stop()

# Simulate a simple distributed key-value store
class DistributedKVStore:
    def __init__(self, node_id, num_nodes, network):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.network = network
        self.data = {}
        self.leader_id = None
        self.is_leader = False
        
        # Register message handler
        self.network.register_node_handler(node_id, self.handle_kv_message)
    
    def handle_kv_message(self, msg):
        if msg.msg_type == "kv_get":
            key = msg.data
            value = self.data.get(key, "KEY_NOT_FOUND")
            response = Message(
                from_node=self.node_id,
                to_node=msg.from_node,
                msg_type="kv_get_response",
                data=f"{key}:{value}"
            )
            self.network.send_message(response)
        elif msg.msg_type == "kv_set":
            key, value = msg.data.split(':', 1)
            self.data[key] = value
            response = Message(
                from_node=self.node_id,
                to_node=msg.from_node,
                msg_type="kv_set_response",
                data="OK"
            )
            self.network.send_message(response)
    
    def get(self, key):
        req = Message(
            from_node=self.node_id,
            to_node=0,  # Ask node 0 (potential leader)
            msg_type="kv_get",
            data=key
        )
        self.network.send_message(req)
        # In a real implementation, we'd wait for the response
    
    def set(self, key, value):
        req = Message(
            from_node=self.node_id,
            to_node=0,  # Send to leader
            msg_type="kv_set",
            data=f"{key}:{value}"
        )
        self.network.send_message(req)

def distributed_kv_store_simulation():
    print("\n=== Distributed Key-Value Store Simulation ===")
    
    num_nodes = 3
    network = NetworkLayer(num_nodes)
    network.start_processing()
    
    stores = []
    for i in range(num_nodes):
        stores.append(DistributedKVStore(i, num_nodes, network))
    
    # Simulate setting and getting values
    stores[0].set("key1", "value1")
    stores[1].set("key2", "value2")
    
    # Wait a bit for propagation
    time.sleep(0.5)
    
    print("Set operations completed")
    
    # Clean up
    network.stop()

if __name__ == "__main__":
    print("Distributed Concurrency - Hands-on Example (Python)")
    
    raft_consensus_simulation()
    distributed_lock_simulation()
    leader_election_simulation()
    distributed_kv_store_simulation()
    
    print("\nAll distributed concurrency examples completed!")