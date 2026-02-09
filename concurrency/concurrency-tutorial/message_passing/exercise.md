# Exercise: Message Passing

## Objective
Practice implementing and using message passing patterns to coordinate between concurrent entities.

## Tasks

1. **Run the examples**:
   - Compile and run the C++ example: `g++ -std=c++11 -pthread example.cpp -o example && ./example`
   - Run the Python example: `python3 example.py`

2. **Chat Application**:
   - Implement a simple chat application using message passing
   - Create a central message broker that routes messages between clients
   - Use channels/queues to handle incoming and outgoing messages

3. **Map-Reduce Simulation**:
   - Implement a simplified Map-Reduce framework using message passing
   - Create mapper actors/processes that transform data
   - Create reducer actors/processes that aggregate results
   - Use channels to pass intermediate data

4. **Pub-Sub System**:
   - Implement a publish-subscribe messaging system
   - Allow publishers to send messages to topics
   - Allow subscribers to register interest in topics
   - Route messages from publishers to interested subscribers

## Advanced Challenge

Build a fault-tolerant message passing system:
- Implement message acknowledgments
- Add retry mechanisms for failed deliveries
- Include timeouts for message processing
- Design a supervisor pattern for handling actor failures

## Questions to Think About

1. How does message passing differ from shared memory concurrency?
2. What are the trade-offs between synchronous and asynchronous channels?
3. When is the actor model preferable to traditional threading?
4. How do you handle backpressure in message passing systems?
5. What are the challenges of implementing message passing across networks?

## Solution Notes

This exercise demonstrates the power of message passing for building robust concurrent systems. Key takeaways include:
- Understanding the safety benefits of message passing
- Learning to structure applications around message flows
- Recognizing the performance trade-offs of message passing
- Appreciating the modularity benefits of the actor model