// Real-World Case Studies - Hands-on Example
// This example implements simplified versions of real-world concurrent systems

#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <random>
#include <memory>
#include <atomic>
#include <map>
#include <algorithm>

// Case Study 1: Simplified Web Server with Thread Pool
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};

public:
    ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        
                        if (stop && tasks.empty()) return;
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    ~ThreadPool() {
        stop = true;
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }
};

// Simplified HTTP request handler
class HttpRequestHandler {
public:
    static void handle_request(int request_id) {
        std::cout << "Processing request " << request_id << " on thread " 
                  << std::this_thread::get_id() << std::endl;
        
        // Simulate processing time
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(100, 500);
        std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));
        
        std::cout << "Completed request " << request_id << std::endl;
    }
};

// Case Study 2: Database Connection Pool
class DatabaseConnection {
private:
    int connection_id;
    bool in_use;
    std::chrono::steady_clock::time_point last_used;
    
public:
    DatabaseConnection(int id) : connection_id(id), in_use(false) {
        last_used = std::chrono::steady_clock::now();
    }
    
    int get_id() const { return connection_id; }
    bool is_in_use() const { return in_use; }
    void set_in_use(bool use) { in_use = use; }
    std::chrono::steady_clock::time_point get_last_used() const { return last_used; }
    void update_last_used() { last_used = std::chrono::steady_clock::now(); }
    
    // Simulate database work
    void execute_query(const std::string& query) {
        std::cout << "Connection " << connection_id << " executing: " << query << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "Connection " << connection_id << " completed query" << std::endl;
    }
};

class ConnectionPool {
private:
    std::vector<std::unique_ptr<DatabaseConnection>> connections;
    std::queue<DatabaseConnection*> available_connections;
    std::mutex pool_mutex;
    std::condition_variable condition;
    int max_connections;
    std::atomic<int> active_connections{0};

public:
    ConnectionPool(int max_conns) : max_connections(max_conns) {
        for (int i = 0; i < max_conns; ++i) {
            connections.push_back(std::make_unique<DatabaseConnection>(i));
            available_connections.push(connections[i].get());
        }
    }
    
    DatabaseConnection* get_connection() {
        std::unique_lock<std::mutex> lock(pool_mutex);
        
        // Wait for available connection
        condition.wait(lock, [this] { 
            return !available_connections.empty() || active_connections < max_connections; 
        });
        
        if (!available_connections.empty()) {
            DatabaseConnection* conn = available_connections.front();
            available_connections.pop();
            conn->set_in_use(true);
            active_connections++;
            return conn;
        }
        
        return nullptr; // Should not happen in this simplified version
    }
    
    void return_connection(DatabaseConnection* conn) {
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            conn->set_in_use(false);
            conn->update_last_used();
            available_connections.push(conn);
            active_connections--;
        }
        condition.notify_one();
    }
    
    void print_stats() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        std::cout << "Pool stats: Available: " << available_connections.size() 
                  << ", Active: " << active_connections.load() 
                  << ", Max: " << max_connections << std::endl;
    }
};

// Case Study 3: Concurrent Cache with Lock Striping
template<typename K, typename V>
class ConcurrentCache {
private:
    struct Bucket {
        std::map<K, V> data;
        mutable std::shared_mutex mutex;
    };
    
    std::vector<Bucket> buckets;
    std::hash<K> hasher;
    
    size_t get_bucket_index(const K& key) const {
        return hasher(key) % buckets.size();
    }

public:
    ConcurrentCache(size_t num_buckets = 16) : buckets(num_buckets) {}
    
    void put(const K& key, const V& value) {
        size_t index = get_bucket_index(key);
        std::unique_lock<std::shared_mutex> lock(buckets[index].mutex);
        buckets[index].data[key] = value;
    }
    
    bool get(const K& key, V& value) const {
        size_t index = get_bucket_index(key);
        std::shared_lock<std::shared_mutex> lock(buckets[index].mutex);
        auto it = buckets[index].data.find(key);
        if (it != buckets[index].data.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
    
    bool remove(const K& key) {
        size_t index = get_bucket_index(key);
        std::unique_lock<std::shared_mutex> lock(buckets[index].mutex);
        return buckets[index].data.erase(key) > 0;
    }
    
    size_t size() const {
        size_t total_size = 0;
        for (const auto& bucket : buckets) {
            std::shared_lock<std::shared_mutex> lock(bucket.mutex);
            total_size += bucket.data.size();
        }
        return total_size;
    }
};

// Case Study 4: Parallel Data Processor
class ParallelDataProcessor {
private:
    ThreadPool& thread_pool;
    
public:
    ParallelDataProcessor(ThreadPool& pool) : thread_pool(pool) {}
    
    template<typename T, typename Func>
    void process_in_parallel(std::vector<T>& data, Func func) {
        std::vector<std::future<void>> futures;
        
        for (auto& item : data) {
            auto future = std::async(std::launch::async, [func, &item]() {
                func(item);
            });
            futures.push_back(std::move(future));
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
    
    // Alternative implementation using thread pool
    template<typename T, typename Func>
    void process_in_parallel_with_pool(std::vector<T>& data, Func func) {
        std::vector<std::promise<void>> promises(data.size());
        std::vector<std::future<void>> futures;
        
        for (size_t i = 0; i < data.size(); ++i) {
            futures.push_back(promises[i].get_future());
            
            thread_pool.enqueue([func, &data, i, promise = std::move(promises[i])]() mutable {
                func(data[i]);
                promise.set_value();
            });
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
};

// Case Study 5: Producer-Consumer with Bounded Buffer (common in many systems)
template<typename T>
class BoundedBuffer {
private:
    std::vector<T> buffer;
    size_t head, tail, count, capacity;
    mutable std::mutex mtx;
    std::condition_variable not_full, not_empty;

public:
    explicit BoundedBuffer(size_t cap) : buffer(cap), head(0), tail(0), count(0), capacity(cap) {}
    
    void put(const T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        not_full.wait(lock, [this] { return count < capacity; });
        
        buffer[tail] = item;
        tail = (tail + 1) % capacity;
        ++count;
        
        not_empty.notify_one();
    }
    
    T get() {
        std::unique_lock<std::mutex> lock(mtx);
        not_empty.wait(lock, [this] { return count > 0; });
        
        T item = buffer[head];
        head = (head + 1) % capacity;
        --count;
        
        not_full.notify_one();
        return item;
    }
    
    bool try_put(const T& item, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mtx);
        if (!not_full.wait_for(lock, timeout, [this] { return count < capacity; })) {
            return false; // Timeout
        }
        
        buffer[tail] = item;
        tail = (tail + 1) % capacity;
        ++count;
        
        not_empty.notify_one();
        return true;
    }
    
    bool try_get(T& item, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mtx);
        if (!not_empty.wait_for(lock, timeout, [this] { return count > 0; })) {
            return false; // Timeout
        }
        
        item = buffer[head];
        head = (head + 1) % capacity;
        --count;
        
        not_full.notify_one();
        return true;
    }
};

// Demonstration of Case Study 1: Web Server
void web_server_demo() {
    std::cout << "\n=== Case Study 1: Web Server with Thread Pool ===" << std::endl;
    
    ThreadPool pool(4); // 4 worker threads
    
    // Simulate incoming requests
    for (int i = 1; i <= 10; ++i) {
        pool.enqueue([i]() {
            HttpRequestHandler::handle_request(i);
        });
    }
    
    // Give time for all requests to process
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Web server demo completed" << std::endl;
}

// Demonstration of Case Study 2: Connection Pool
void connection_pool_demo() {
    std::cout << "\n=== Case Study 2: Database Connection Pool ===" << std::endl;
    
    ConnectionPool pool(3); // Pool with 3 connections
    
    // Simulate multiple clients requesting connections
    std::vector<std::thread> clients;
    
    for (int client_id = 1; client_id <= 5; ++client_id) {
        clients.emplace_back([&pool, client_id]() {
            DatabaseConnection* conn = pool.get_connection();
            if (conn) {
                std::cout << "Client " << client_id << " got connection " << conn->get_id() << std::endl;
                
                // Use the connection
                conn->execute_query("SELECT * FROM users WHERE id = " + std::to_string(client_id));
                
                // Return the connection to the pool
                std::cout << "Client " << client_id << " returning connection" << std::endl;
                pool.return_connection(conn);
            } else {
                std::cout << "Client " << client_id << " could not get connection" << std::endl;
            }
        });
    }
    
    // Print stats periodically
    for (int i = 0; i < 3; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        pool.print_stats();
    }
    
    // Wait for all clients to finish
    for (auto& client : clients) {
        client.join();
    }
    
    pool.print_stats();
    std::cout << "Connection pool demo completed" << std::endl;
}

// Demonstration of Case Study 3: Concurrent Cache
void concurrent_cache_demo() {
    std::cout << "\n=== Case Study 3: Concurrent Cache ===" << std::endl;
    
    ConcurrentCache<int, std::string> cache(8); // 8 buckets
    
    // Populate cache from multiple threads
    std::vector<std::thread> populators;
    for (int i = 0; i < 4; ++i) {
        populators.emplace_back([&cache, i]() {
            for (int j = 0; j < 10; ++j) {
                int key = i * 10 + j;
                cache.put(key, "Value_" + std::to_string(key));
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    
    // Query cache from multiple threads
    std::vector<std::thread> queriers;
    for (int i = 0; i < 2; ++i) {
        queriers.emplace_back([&cache, i]() {
            for (int j = 0; j < 20; ++j) {
                std::string value;
                if (cache.get(j, value)) {
                    std::cout << "Found: " << j << " -> " << value << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : populators) t.join();
    for (auto& t : queriers) t.join();
    
    std::cout << "Cache size: " << cache.size() << std::endl;
    std::cout << "Concurrent cache demo completed" << std::endl;
}

// Demonstration of Case Study 4: Parallel Data Processor
void parallel_processor_demo() {
    std::cout << "\n=== Case Study 4: Parallel Data Processor ===" << std::endl;
    
    ThreadPool pool(4);
    ParallelDataProcessor processor(pool);
    
    // Create a vector of numbers to process
    std::vector<int> data(20);
    std::iota(data.begin(), data.end(), 1); // Fill with 1, 2, 3, ..., 20
    
    std::cout << "Original data: ";
    for (int n : data) std::cout << n << " ";
    std::cout << std::endl;
    
    // Process data in parallel (square each number)
    processor.process_in_parallel_with_pool(data, [](int& n) {
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        n = n * n;
        std::cout << "Processed: " << n << " on thread " << std::this_thread::get_id() << std::endl;
    });
    
    std::cout << "Processed data: ";
    for (int n : data) std::cout << n << " ";
    std::cout << std::endl;
    std::cout << "Parallel processor demo completed" << std::endl;
}

// Demonstration of Case Study 5: Producer-Consumer
void producer_consumer_demo() {
    std::cout << "\n=== Case Study 5: Producer-Consumer with Bounded Buffer ===" << std::endl;
    
    BoundedBuffer<int> buffer(5); // Buffer with capacity 5
    
    // Producer thread
    std::thread producer([&buffer]() {
        for (int i = 1; i <= 10; ++i) {
            std::cout << "Producer: putting " << i << std::endl;
            buffer.put(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::cout << "Producer: finished" << std::endl;
    });
    
    // Consumer thread
    std::thread consumer([&buffer]() {
        for (int i = 0; i < 10; ++i) {
            int value = buffer.get();
            std::cout << "Consumer: got " << value << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        std::cout << "Consumer: finished" << std::endl;
    });
    
    producer.join();
    consumer.join();
    
    std::cout << "Producer-consumer demo completed" << std::endl;
}

int main() {
    std::cout << "Real-World Case Studies - Hands-on Example" << std::endl;
    
    web_server_demo();
    connection_pool_demo();
    concurrent_cache_demo();
    parallel_processor_demo();
    producer_consumer_demo();
    
    std::cout << "\nAll real-world case study examples completed!" << std::endl;
    
    return 0;
}