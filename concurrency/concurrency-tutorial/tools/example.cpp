// Tools and Profiling - Hands-on Example
// This example demonstrates how to profile and analyze concurrent code

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <random>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>

// Helper class for performance measurement
class Profiler {
private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::map<std::string, std::vector<double>> measurements;
    
public:
    void start_timer(const std::string& name) {
        start_times[name] = std::chrono::high_resolution_clock::now();
    }
    
    void stop_timer(const std::string& name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_times[name]).count();
        
        measurements[name].push_back(duration);
    }
    
    void print_summary() {
        std::cout << "\n=== Performance Summary ===" << std::endl;
        for (const auto& pair : measurements) {
            double total = 0;
            for (double val : pair.second) {
                total += val;
            }
            double avg = total / pair.second.size();
            std::cout << pair.first << ": " << pair.second.size() 
                      << " runs, avg: " << avg << " μs" << std::endl;
        }
    }
    
    void print_detailed_stats() {
        std::cout << "\n=== Detailed Statistics ===" << std::endl;
        for (const auto& pair : measurements) {
            if (pair.second.empty()) continue;
            
            double min_val = pair.second[0], max_val = pair.second[0], sum = 0;
            for (double val : pair.second) {
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
                sum += val;
            }
            double avg = sum / pair.second.size();
            
            std::cout << pair.first << ":" << std::endl;
            std::cout << "  Runs: " << pair.second.size() << std::endl;
            std::cout << "  Min: " << min_val << " μs" << std::endl;
            std::cout << "  Max: " << max_val << " μs" << std::endl;
            std::cout << "  Avg: " << avg << " μs" << std::endl;
        }
    }
};

// Example 1: Measuring lock contention
class ContentionBenchmark {
private:
    std::mutex mtx;
    std::atomic<int> atomic_counter{0};
    int regular_counter{0};
    Profiler& profiler;

public:
    ContentionBenchmark(Profiler& p) : profiler(p) {}
    
    void benchmark_mutex() {
        profiler.start_timer("mutex_benchmark");
        
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([this]() {
                for (int j = 0; j < 10000; ++j) {
                    std::lock_guard<std::mutex> lock(mtx);
                    regular_counter++;
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        profiler.stop_timer("mutex_benchmark");
    }
    
    void benchmark_atomic() {
        profiler.start_timer("atomic_benchmark");
        
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([this]() {
                for (int j = 0; j < 10000; ++j) {
                    atomic_counter.fetch_add(1);
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        profiler.stop_timer("atomic_benchmark");
    }
};

// Example 2: False sharing demonstration
struct BadlyAlignedData {
    int data1;  // This will likely share cache line with data2
    char padding[60];  // Padding to prevent false sharing
    int data2;
};

struct WellAlignedData {
    int data1;
    char padding1[60];  // Padding to align to cache line boundary
    int data2;
    char padding2[60];  // Padding to align to cache line boundary
};

void false_sharing_demo() {
    std::cout << "\n=== False Sharing Demo ===" << std::endl;
    
    // Without padding - potential false sharing
    std::vector<BadlyAlignedData> bad_data(2);
    std::vector<std::thread> bad_threads;
    
    auto start_bad = std::chrono::high_resolution_clock::now();
    
    bad_threads.emplace_back([&bad_data]() {
        for (int i = 0; i < 1000000; ++i) {
            bad_data[0].data1++;
        }
    });
    
    bad_threads.emplace_back([&bad_data]() {
        for (int i = 0; i < 1000000; ++i) {
            bad_data[1].data2++;
        }
    });
    
    for (auto& t : bad_threads) {
        t.join();
    }
    
    auto end_bad = std::chrono::high_resolution_clock::now();
    auto bad_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_bad - start_bad);
    
    // With padding - no false sharing
    std::vector<WellAlignedData> well_data(2);
    std::vector<std::thread> well_threads;
    
    auto start_well = std::chrono::high_resolution_clock::now();
    
    well_threads.emplace_back([&well_data]() {
        for (int i = 0; i < 1000000; ++i) {
            well_data[0].data1++;
        }
    });
    
    well_threads.emplace_back([&well_data]() {
        for (int i = 0; i < 1000000; ++i) {
            well_data[1].data2++;
        }
    });
    
    for (auto& t : well_threads) {
        t.join();
    }
    
    auto end_well = std::chrono::high_resolution_clock::now();
    auto well_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_well - start_well);
    
    std::cout << "Without padding (potential false sharing): " << bad_duration.count() << " μs" << std::endl;
    std::cout << "With padding (no false sharing): " << well_duration.count() << " μs" << std::endl;
    std::cout << "Speedup with padding: " << (double)bad_duration.count() / well_duration.count() << "x" << std::endl;
}

// Example 3: Thread pool performance analysis
class ThreadPoolProfiler {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    Profiler& profiler;
    std::atomic<int> tasks_executed{0};

public:
    ThreadPoolProfiler(size_t num_threads, Profiler& p) : profiler(p) {
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
                    tasks_executed++;
                }
            });
        }
    }
    
    void enqueue(std::function<void()> f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::move(f));
        }
        condition.notify_one();
    }
    
    void benchmark_throughput(int num_tasks) {
        profiler.start_timer("thread_pool_throughput");
        
        for (int i = 0; i < num_tasks; ++i) {
            enqueue([i]() {
                // Simulate work
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            });
        }
        
        // Wait for all tasks to complete
        while (tasks_executed.load() < num_tasks) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        profiler.stop_timer("thread_pool_throughput");
    }
    
    ~ThreadPoolProfiler() {
        stop = true;
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }
};

// Example 4: Lock contention analysis
class LockContentionAnalyzer {
private:
    std::mutex shared_resource;
    int resource_value = 0;
    Profiler& profiler;

public:
    LockContentionAnalyzer(Profiler& p) : profiler(p) {}
    
    void analyze_contention(int num_threads, int iterations) {
        std::string label = "contention_" + std::to_string(num_threads) + "_threads";
        profiler.start_timer(label);
        
        std::vector<std::thread> threads;
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([this, iterations]() {
                for (int j = 0; j < iterations; ++j) {
                    std::lock_guard<std::mutex> lock(shared_resource);
                    resource_value++;
                    // Simulate some work holding the lock
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        profiler.stop_timer(label);
    }
};

// Example 5: Memory access pattern analysis
void memory_access_analysis() {
    std::cout << "\n=== Memory Access Pattern Analysis ===" << std::endl;
    
    const size_t SIZE = 1000000;
    std::vector<int> data(SIZE);
    
    // Initialize data
    for (size_t i = 0; i < SIZE; ++i) {
        data[i] = i;
    }
    
    // Sequential access (cache-friendly)
    auto start_seq = std::chrono::high_resolution_clock::now();
    long long seq_sum = 0;
    for (size_t i = 0; i < SIZE; ++i) {
        seq_sum += data[i];
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    auto seq_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_seq - start_seq);
    
    // Random access (cache-unfriendly)
    std::vector<size_t> indices(SIZE);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    auto start_rand = std::chrono::high_resolution_clock::now();
    long long rand_sum = 0;
    for (size_t i = 0; i < SIZE; ++i) {
        rand_sum += data[indices[i]];
    }
    auto end_rand = std::chrono::high_resolution_clock::now();
    auto rand_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_rand - start_rand);
    
    std::cout << "Sequential access: " << seq_duration.count() << " μs" << std::endl;
    std::cout << "Random access: " << rand_duration.count() << " μs" << std::endl;
    std::cout << "Random access overhead: " << (double)rand_duration.count() / seq_duration.count() << "x" << std::endl;
}

int main() {
    std::cout << "Tools and Profiling - Hands-on Example" << std::endl;
    
    Profiler profiler;
    
    // Example 1: Compare mutex vs atomic performance
    ContentionBenchmark bench(profiler);
    bench.benchmark_mutex();
    bench.benchmark_atomic();
    
    // Example 2: False sharing demonstration
    false_sharing_demo();
    
    // Example 3: Thread pool throughput analysis
    {
        ThreadPoolProfiler pool(4, profiler);
        pool.benchmark_throughput(100);
    }
    
    // Example 4: Lock contention analysis with different thread counts
    LockContentionAnalyzer analyzer(profiler);
    analyzer.analyze_contention(2, 10000);
    analyzer.analyze_contention(4, 10000);
    analyzer.analyze_contention(8, 10000);
    
    // Example 5: Memory access pattern analysis
    memory_access_analysis();
    
    // Print results
    profiler.print_summary();
    profiler.print_detailed_stats();
    
    std::cout << "\nAll profiling examples completed!" << std::endl;
    
    return 0;
}