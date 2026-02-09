# Thread Basics

## What is a Thread?

A thread is the smallest unit of execution within a process. A process can contain multiple threads that share the same memory space, file handles, and other resources. Each thread has its own:
- Program counter
- Stack
- Registers
- State

## Thread Lifecycle

Threads go through several states during their lifetime:
1. **New**: Thread object created but not yet started
2. **Runnable**: Thread is eligible to run but waiting for CPU time
3. **Running**: Thread is currently executing
4. **Blocked/Waiting**: Thread is temporarily inactive
5. **Terminated**: Thread has finished execution

## Creating Threads

Different programming languages provide various ways to create threads:
- Using thread classes/functions
- Using thread pools
- Using higher-level abstractions (async/await)

## Thread Safety

Thread safety refers to the property of code that ensures correct behavior when accessed by multiple threads concurrently. This involves:
- Proper synchronization
- Avoiding race conditions
- Managing shared resources appropriately

## Common Thread Operations

- Starting a thread
- Waiting for thread completion (joining)
- Thread interruption/cancellation
- Setting thread priority
- Daemon vs user threads

## Thread Communication

Threads often need to communicate with each other:
- Shared memory
- Message passing
- Signaling mechanisms
- Thread-local storage

## Performance Considerations

- Thread creation overhead
- Context switching costs
- Scalability limits
- Resource contention