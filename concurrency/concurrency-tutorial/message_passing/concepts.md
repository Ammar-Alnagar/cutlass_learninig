# Message Passing

## Overview

Message passing is a concurrency paradigm where threads or processes communicate by sending messages to each other rather than sharing memory. This approach avoids many of the complexities of shared-memory concurrency like race conditions and deadlocks.

## Key Concepts

### Channels
Channels are the fundamental communication primitive in message-passing systems. They provide a way to send and receive messages between concurrent entities.

Types of channels:
- **Synchronous**: Sender blocks until receiver receives (rendezvous)
- **Asynchronous**: Sender doesn't block, messages stored in buffer
- **Bounded**: Limited capacity, sender blocks when full
- **Unbounded**: Unlimited capacity (potentially dangerous)

### Actor Model
A mathematical model of concurrent computation that treats actors as the universal primitives of concurrent digital computation. Each actor:
- Receives messages from other actors
- Processes messages in sequence
- Decides how to respond to the next message
- May create new actors
- May send messages to other actors

### Communicating Sequential Processes (CSP)
A formal language for describing patterns of interaction in concurrent systems. CSP emphasizes the importance of the communications, events and processes in a system.

### Mailboxes
Each concurrent entity has a mailbox where incoming messages are queued. The entity processes messages one at a time from its mailbox.

## Advantages of Message Passing

1. **Safety**: Eliminates shared mutable state, preventing race conditions
2. **Modularity**: Clear separation of concerns between concurrent entities
3. **Scalability**: Naturally suited for distributed systems
4. **Fault Isolation**: Errors in one actor don't directly affect others
5. **Simplicity**: Easier to reason about than shared-memory systems

## Disadvantages of Message Passing

1. **Performance**: Message passing has overhead compared to shared memory
2. **Memory Usage**: Copying messages can be expensive
3. **Complexity**: Can lead to complex protocols for coordination
4. **Latency**: Communication has inherent delays

## Languages and Frameworks Supporting Message Passing

- **Go**: Goroutines and channels
- **Erlang/Elixir**: Actor model
- **Akka**: Actor model for JVM
- **Rust**: Channels in standard library
- **Clojure**: Agents and core.async
- **JavaScript**: Web Workers with postMessage

## Design Patterns

### Producer-Consumer
One or more producers send messages to a channel, and one or more consumers receive them.

### Worker Pool
A dispatcher sends tasks to a pool of worker actors/channels.

### Pipeline
Messages flow through a series of processing stages.

### Pub/Sub
Publishers send messages to topics, subscribers receive messages from topics they're interested in.

## Performance Considerations

### Message Size
Larger messages have higher copying overhead but lower per-message overhead.

### Channel Capacity
Bounded channels provide backpressure but can cause blocking; unbounded channels don't block but can consume unlimited memory.

### Serialization
Messages often need to be serialized/deserialized, adding overhead.

## Error Handling

Message-passing systems need strategies for:
- Handling malformed messages
- Dealing with crashed receivers
- Managing timeouts
- Coordinating recovery