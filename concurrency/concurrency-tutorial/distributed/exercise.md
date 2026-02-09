# Exercise: Distributed Concurrency

## Objective
Practice implementing and using distributed concurrency patterns to coordinate computation across multiple nodes.

## Tasks

1. **Run the examples**:
   - Compile and run the C++ example: `g++ -std=c++11 -pthread example.cpp -o example && ./example`
   - Run the Python example: `python3 example.py`

2. **Distributed Counter**:
   - Implement a distributed counter that maintains consistency across nodes
   - Use consensus algorithm to agree on counter value
   - Handle node failures and recoveries

3. **Gossip Protocol**:
   - Implement a gossip protocol for disseminating information across nodes
   - Nodes periodically exchange state with randomly selected peers
   - Simulate network partitions and healing

4. **Distributed Lock Manager**:
   - Implement a centralized or decentralized lock manager
   - Handle lock acquisition, release, and timeouts
   - Implement deadlock detection/prevention

## Advanced Challenge

Build a distributed hash table (DHT):
- Partition keys across nodes using consistent hashing
- Implement node join and leave operations
- Handle replication for fault tolerance
- Implement routing for key lookup

## Questions to Think About

1. How does the CAP theorem affect design decisions in distributed systems?
2. What are the challenges of implementing consensus in asynchronous networks?
3. How do you handle partial failures in distributed systems?
4. What are the trade-offs between strong and eventual consistency?
5. How do you test distributed systems for correctness?

## Solution Notes

This exercise demonstrates the complexity of distributed systems. Key takeaways include:
- Understanding the fundamental challenges of distributed coordination
- Learning to handle failures and network partitions
- Appreciating the trade-offs between consistency, availability, and partition tolerance
- Recognizing the importance of fault tolerance in distributed design