// Message Passing - Hands-on Example
// This example demonstrates message passing using channels and the actor model

#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <atomic>
#include <memory>
#include <chrono>

// Simple channel implementation for message passing
template<typename T>
class Channel {
private:
    std::queue<T> buffer;
    std::mutex mtx;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    size_t capacity;
    bool closed;

public:
    explicit Channel(size_t cap = 10) : capacity(cap), closed(false) {}

    // Send a message to the channel (blocks if full)
    bool send(const T& msg) {
        std::unique_lock<std::mutex> lock(mtx);
        
        // Wait until there's space or channel is closed
        not_full.wait(lock, [this] { return buffer.size() < capacity || closed; });
        
        if (closed) {
            return false; // Cannot send to closed channel
        }
        
        buffer.push(msg);
        not_empty.notify_one(); // Notify receivers
        return true;
    }

    // Receive a message from the channel (blocks if empty)
    bool receive(T& msg) {
        std::unique_lock<std::mutex> lock(mtx);
        
        // Wait until there's a message or channel is closed and empty
        not_empty.wait(lock, [this] { return !buffer.empty() || closed; });
        
        if (!buffer.empty()) {
            msg = buffer.front();
            buffer.pop();
            not_full.notify_one(); // Notify senders
            return true;
        }
        
        return false; // Channel closed and empty
    }

    // Close the channel (no more sends allowed)
    void close() {
        std::lock_guard<std::mutex> lock(mtx);
        closed = true;
        not_empty.notify_all();
        not_full.notify_all();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mtx);
        return closed;
    }
};

// Actor class that uses channels for message passing
class Actor {
protected:
    std::string name;
    Channel<std::function<void()>> mailbox;
    std::thread worker_thread;
    std::atomic<bool> running;

public:
    Actor(const std::string& actor_name, size_t mailbox_capacity = 10) 
        : name(actor_name), mailbox(mailbox_capacity), running(true) {
        worker_thread = std::thread(&Actor::process_messages, this);
    }

    virtual ~Actor() {
        stop();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    // Send a message (function) to the actor
    bool send_message(std::function<void()> msg) {
        return mailbox.send(msg);
    }

    // Stop the actor
    void stop() {
        running = false;
        mailbox.close();
    }

protected:
    // Process messages from the mailbox
    virtual void process_messages() {
        std::function<void()> msg;
        while (running) {
            if (mailbox.receive(msg)) {
                msg(); // Execute the message
            } else {
                break; // Channel closed
            }
        }
    }
};

// Specialized actor for mathematical operations
class MathActor : public Actor {
private:
    Channel<int> result_channel;

public:
    MathActor(const std::string& actor_name, Channel<int>& result_ch) 
        : Actor(actor_name), result_channel(result_ch) {}

    void perform_calculation(int a, int b, char op) {
        send_message([this, a, b, op]() {
            int result = 0;
            switch (op) {
                case '+': result = a + b; break;
                case '-': result = a - b; break;
                case '*': result = a * b; break;
                case '/': result = (b != 0) ? a / b : 0; break;
                default: result = 0; break;
            }
            
            std::cout << name << ": " << a << " " << op << " " << b << " = " << result << std::endl;
            result_channel.send(result);
        });
    }
};

// Producer-Consumer pattern using channels
void producer_consumer_example() {
    std::cout << "\n=== Producer-Consumer with Channels ===" << std::endl;
    
    Channel<int> ch(5); // Bounded channel with capacity 5
    
    // Producer thread
    std::thread producer([&ch]() {
        for (int i = 1; i <= 10; ++i) {
            std::cout << "Producer: sending " << i << std::endl;
            ch.send(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
        }
        ch.close(); // Signal end of production
        std::cout << "Producer: finished" << std::endl;
    });
    
    // Consumer thread
    std::thread consumer([&ch]() {
        int value;
        while (ch.receive(value)) {
            std::cout << "Consumer: received " << value << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(150)); // Simulate processing
        }
        std::cout << "Consumer: finished" << std::endl;
    });
    
    producer.join();
    consumer.join();
}

// Worker pool pattern using message passing
void worker_pool_example() {
    std::cout << "\n=== Worker Pool with Message Passing ===" << std::endl;
    
    Channel<std::pair<int, int>> task_channel(10);  // Tasks: (worker_id, task_id)
    Channel<std::pair<int, int>> result_channel(10); // Results: (worker_id, result)
    
    const int num_workers = 3;
    std::vector<std::thread> workers;
    
    // Create worker threads
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([&task_channel, &result_channel, i]() {
            std::pair<int, int> task;
            while (task_channel.receive(task)) {
                int worker_id = task.first;
                int task_id = task.second;
                
                std::cout << "Worker " << worker_id << " processing task " << task_id << std::endl;
                
                // Simulate work
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                
                // Send result
                int result = task_id * task_id; // Simple computation
                result_channel.send({worker_id, result});
                std::cout << "Worker " << worker_id << " completed task " << task_id 
                          << ", result: " << result << std::endl;
            }
            std::cout << "Worker " << worker_id << " shutting down" << std::endl;
        });
    }
    
    // Dispatcher thread
    std::thread dispatcher([&task_channel]() {
        // Send tasks to workers
        for (int i = 1; i <= 12; ++i) {
            task_channel.send({i % 3, i}); // Assign to one of 3 workers
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        task_channel.close(); // Signal end of tasks
        std::cout << "Dispatcher: all tasks sent" << std::endl;
    });
    
    // Collector thread
    std::thread collector([&result_channel, num_workers]() {
        std::pair<int, int> result;
        int results_received = 0;
        int expected_results = 12;
        
        while (results_received < expected_results && result_channel.receive(result)) {
            std::cout << "Collector: received result from worker " << result.first 
                      << ", value: " << result.second << std::endl;
            results_received++;
        }
        std::cout << "Collector: received all " << results_received << " results" << std::endl;
    });
    
    dispatcher.join();
    collector.join();
    
    // Wait for all workers to finish
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
}

// Actor model example
void actor_model_example() {
    std::cout << "\n=== Actor Model Example ===" << std::endl;
    
    Channel<int> result_channel(10);
    
    // Create math actors
    MathActor calc1("Calculator-1", result_channel);
    MathActor calc2("Calculator-2", result_channel);
    
    // Send calculation requests to actors
    calc1.perform_calculation(10, 5, '+');
    calc1.perform_calculation(20, 4, '*');
    calc2.perform_calculation(100, 25, '/');
    calc2.perform_calculation(50, 30, '-');
    
    // Collect results
    int result;
    for (int i = 0; i < 4; ++i) {
        if (result_channel.receive(result)) {
            std::cout << "Main: received calculation result: " << result << std::endl;
        }
    }
    
    // Stop actors
    calc1.stop();
    calc2.stop();
}

// Pipeline pattern using channels
void pipeline_example() {
    std::cout << "\n=== Pipeline Pattern ===" << std::endl;
    
    Channel<int> stage1_to_stage2(5);
    Channel<int> stage2_to_stage3(5);
    
    // Stage 1: Generate data
    std::thread generator([&stage1_to_stage2]() {
        for (int i = 1; i <= 5; ++i) {
            std::cout << "Generator: producing " << i << std::endl;
            stage1_to_stage2.send(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        stage1_to_stage2.close();
        std::cout << "Generator: finished" << std::endl;
    });
    
    // Stage 2: Process data
    std::thread processor([&stage1_to_stage2, &stage2_to_stage3]() {
        int value;
        while (stage1_to_stage2.receive(value)) {
            std::cout << "Processor: received " << value << ", processing..." << std::endl;
            // Simulate processing
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            int processed = value * 2;
            std::cout << "Processor: sending " << processed << std::endl;
            stage2_to_stage3.send(processed);
        }
        stage2_to_stage3.close();
        std::cout << "Processor: finished" << std::endl;
    });
    
    // Stage 3: Finalize data
    std::thread finalizer([&stage2_to_stage3]() {
        int value;
        while (stage2_to_stage3.receive(value)) {
            std::cout << "Finalizer: received " << value << ", finalizing..." << std::endl;
            // Simulate finalization
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "Finalizer: final result is " << (value + 10) << std::endl;
        }
        std::cout << "Finalizer: finished" << std::endl;
    });
    
    generator.join();
    processor.join();
    finalizer.join();
}

int main() {
    std::cout << "Message Passing - Hands-on Example" << std::endl;
    
    producer_consumer_example();
    worker_pool_example();
    actor_model_example();
    pipeline_example();
    
    std::cout << "\nAll message passing examples completed!" << std::endl;
    
    return 0;
}