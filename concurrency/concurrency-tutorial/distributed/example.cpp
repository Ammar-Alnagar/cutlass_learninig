// Distributed Concurrency - Hands-on Example
// This example simulates distributed systems concepts using multiple threads/processes

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <map>
#include <queue>
#include <chrono>
#include <random>
#include <functional>
#include <memory>
#include <algorithm>
#include <atomic>

// Simulated network message
struct Message {
    int from_node;
    int to_node;
    std::string type;  // "request", "response", "heartbeat", etc.
    std::string data;
    int term;          // For consensus algorithms
    int timestamp;
    
    Message(int from, int to, const std::string& msg_type, const std::string& msg_data, int t = 0, int ts = 0)
        : from_node(from), to_node(to), type(msg_type), data(msg_data), term(t), timestamp(ts) {}
};

// Simulated network layer
class NetworkLayer {
private:
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<Message> message_queue;
    std::atomic<bool> running{true};
    std::vector<std::function<void(const Message&)>> node_handlers;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> delay_dist;

public:
    NetworkLayer() : gen(rd()), delay_dist(0.0, 0.1) {} // 0-100ms simulated network delay

    void register_node_handler(int node_id, std::function<void(const Message&)> handler) {
        if (node_id >= node_handlers.size()) {
            node_handlers.resize(node_id + 1);
        }
        node_handlers[node_id] = handler;
    }

    void send_message(const Message& msg) {
        // Simulate network delay
        double delay = delay_dist(gen);
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(delay * 1000)));
        
        std::lock_guard<std::mutex> lock(mtx);
        message_queue.push(msg);
        cv.notify_one();
    }

    void start_processing() {
        std::thread([this]() {
            while (running) {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return !message_queue.empty() || !running; });
                
                if (!running && message_queue.empty()) break;
                
                if (!message_queue.empty()) {
                    Message msg = message_queue.front();
                    message_queue.pop();
                    lock.unlock();
                    
                    // Deliver message to the appropriate node
                    if (msg.to_node >= 0 && msg.to_node < node_handlers.size() && node_handlers[msg.to_node]) {
                        node_handlers[msg.to_node](msg);
                    }
                }
            }
        }).detach();
    }

    void stop() {
        running = false;
        cv.notify_all();
    }
};

// Simple Raft consensus algorithm simulation
class RaftNode {
private:
    int node_id;
    int num_nodes;
    NetworkLayer& network;
    
    enum Role { FOLLOWER, CANDIDATE, LEADER };
    Role role;
    int current_term;
    int voted_for;
    std::vector<std::string> log;
    int commit_index;
    int last_applied;
    
    std::mutex state_mtx;
    std::condition_variable state_cv;
    std::atomic<int> heartbeat_timeout{150}; // milliseconds
    std::atomic<int> election_timeout{300};  // milliseconds
    std::atomic<bool> running{true};
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> timeout_dist;

public:
    RaftNode(int id, int total_nodes, NetworkLayer& net) 
        : node_id(id), num_nodes(total_nodes), network(net), gen(rd()), 
          timeout_dist(election_timeout.load()/2, election_timeout.load()) {
        role = FOLLOWER;
        current_term = 0;
        voted_for = -1;
        commit_index = 0;
        last_applied = 0;
        
        // Register message handler
        network.register_node_handler(node_id, [this](const Message& msg) { handle_message(msg); });
    }

    void start() {
        std::thread([this]() { run_election_timer(); }).detach();
        std::thread([this]() { run_heartbeat_timer(); }).detach();
    }

    void run_election_timer() {
        auto last_contact = std::chrono::steady_clock::now();
        
        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                if (role != LEADER && 
                    std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::steady_clock::now() - last_contact).count() > election_timeout) {
                    
                    become_candidate();
                    start_election();
                    last_contact = std::chrono::steady_clock::now();
                }
            }
        }
    }

    void run_heartbeat_timer() {
        while (running) {
            if (role == LEADER) {
                send_heartbeats();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(heartbeat_timeout/2));
        }
    }

    void become_follower(int term) {
        std::lock_guard<std::mutex> lock(state_mtx);
        role = FOLLOWER;
        current_term = term;
        voted_for = -1;
        std::cout << "Node " << node_id << " became follower, term " << term << std::endl;
    }

    void become_candidate() {
        std::lock_guard<std::mutex> lock(state_mtx);
        role = CANDIDATE;
        current_term++;
        voted_for = node_id;
        std::cout << "Node " << node_id << " became candidate, term " << current_term << std::endl;
    }

    void become_leader() {
        std::lock_guard<std::mutex> lock(state_mtx);
        role = LEADER;
        std::cout << "Node " << node_id << " became leader, term " << current_term << std::endl;
    }

    void start_election() {
        std::lock_guard<std::mutex> lock(state_mtx);
        
        int votes_received = 1; // Vote for self
        int current_term_copy = current_term;
        
        // Request votes from other nodes
        for (int i = 0; i < num_nodes; ++i) {
            if (i != node_id) {
                Message vote_request(node_id, i, "vote_request", 
                                   std::to_string(current_term) + "," + std::to_string(log.size()), 
                                   current_term);
                network.send_message(vote_request);
            }
        }
        
        // In a real implementation, we'd wait for responses and count votes
        // For simulation, we'll just check periodically
    }

    void send_heartbeats() {
        std::lock_guard<std::mutex> lock(state_mtx);
        if (role != LEADER) return;
        
        for (int i = 0; i < num_nodes; ++i) {
            if (i != node_id) {
                Message heartbeat(node_id, i, "heartbeat", 
                               std::to_string(current_term) + ",0", // term, commit_index
                               current_term);
                network.send_message(heartbeat);
            }
        }
    }

    void handle_message(const Message& msg) {
        std::lock_guard<std::mutex> lock(state_mtx);
        
        // Update term if received message has higher term
        if (msg.term > current_term) {
            current_term = msg.term;
            if (role != FOLLOWER) {
                role = FOLLOWER;
                voted_for = -1;
            }
        }
        
        if (msg.type == "vote_request") {
            handle_vote_request(msg);
        } else if (msg.type == "vote_response") {
            handle_vote_response(msg);
        } else if (msg.type == "heartbeat") {
            handle_heartbeat(msg);
        }
    }

    void handle_vote_request(const Message& msg) {
        // Parse message data: term,log_length
        size_t pos = msg.data.find(',');
        int request_term = std::stoi(msg.data.substr(0, pos));
        int request_log_len = std::stoi(msg.data.substr(pos + 1));
        
        bool grant_vote = false;
        
        if (request_term >= current_term && 
            (voted_for == -1 || voted_for == msg.from_node) &&
            request_log_len >= static_cast<int>(log.size())) {
            grant_vote = true;
            current_term = request_term;
            voted_for = msg.from_node;
        }
        
        Message vote_response(node_id, msg.from_node, "vote_response", 
                            grant_vote ? "yes," + std::to_string(current_term) : "no," + std::to_string(current_term),
                            current_term);
        network.send_message(vote_response);
        
        if (grant_vote) {
            std::cout << "Node " << node_id << " granted vote to node " << msg.from_node << std::endl;
        }
    }

    void handle_vote_response(const Message& msg) {
        if (role == CANDIDATE && msg.term == current_term) {
            // In a real implementation, we'd count votes
            // For simulation, just acknowledge
            std::cout << "Node " << node_id << " received vote response from " << msg.from_node 
                      << ", data: " << msg.data << std::endl;
        }
    }

    void handle_heartbeat(const Message& msg) {
        if (role == CANDIDATE || role == FOLLOWER) {
            // Reset election timer
            if (msg.term >= current_term) {
                current_term = msg.term;
                become_follower(msg.term);
            }
        }
    }

    void stop() {
        running = false;
    }
    
    Role get_role() const {
        std::lock_guard<std::mutex> lock(state_mtx);
        return role;
    }
    
    int get_term() const {
        std::lock_guard<std::mutex> lock(state_mtx);
        return current_term;
    }
};

// Distributed lock simulation
class DistributedLock {
private:
    int node_id;
    NetworkLayer& network;
    std::mutex lock_mtx;
    std::condition_variable lock_cv;
    bool locked;
    int lock_holder;
    std::queue<int> waiting_queue; // Queue of nodes waiting for the lock

public:
    DistributedLock(int id, NetworkLayer& net) : node_id(id), network(net), locked(false), lock_holder(-1) {
        network.register_node_handler(node_id, [this](const Message& msg) { handle_lock_message(msg); });
    }

    bool acquire() {
        // Request the lock from the coordinator (node 0)
        Message lock_request(node_id, 0, "lock_request", "acquire", 0);
        network.send_message(lock_request);
        
        // Wait for lock grant
        std::unique_lock<std::mutex> lock(lock_mtx);
        lock_cv.wait(lock, [this] { return lock_holder == node_id; });
        return true;
    }

    void release() {
        if (lock_holder == node_id) {
            std::lock_guard<std::mutex> lock(lock_mtx);
            locked = false;
            lock_holder = -1;
            
            // Grant lock to next waiting node
            if (!waiting_queue.empty()) {
                int next_node = waiting_queue.front();
                waiting_queue.pop();
                
                lock_holder = next_node;
                locked = true;
                
                // Send lock grant message
                Message grant_msg(0, next_node, "lock_grant", "acquired", 0);
                network.send_message(grant_msg);
            }
        }
    }

    void handle_lock_message(const Message& msg) {
        if (msg.type == "lock_request") {
            std::lock_guard<std::mutex> lock(lock_mtx);
            if (!locked) {
                // Grant the lock immediately
                locked = true;
                lock_holder = msg.from_node;
                
                Message grant_msg(0, msg.from_node, "lock_grant", "acquired", 0);
                network.send_message(grant_msg);
            } else {
                // Add to waiting queue
                waiting_queue.push(msg.from_node);
            }
        } else if (msg.type == "lock_release") {
            std::lock_guard<std::mutex> lock(lock_mtx);
            if (lock_holder == msg.from_node) {
                locked = false;
                lock_holder = -1;
                
                // Grant to next in queue
                if (!waiting_queue.empty()) {
                    int next_node = waiting_queue.front();
                    waiting_queue.pop();
                    
                    lock_holder = next_node;
                    locked = true;
                    
                    Message grant_msg(0, next_node, "lock_grant", "acquired", 0);
                    network.send_message(grant_msg);
                }
            }
        } else if (msg.type == "lock_grant") {
            if (msg.from_node == 0 && msg.to_node == node_id) {
                std::lock_guard<std::mutex> lock(lock_mtx);
                lock_holder = node_id;
                locked = true;
                lock_cv.notify_one();
            }
        }
    }
};

// Simulate a distributed system with multiple nodes
void raft_consensus_simulation() {
    std::cout << "\n=== Raft Consensus Simulation ===" << std::endl;
    
    const int num_nodes = 3;
    NetworkLayer network;
    network.start_processing();
    
    std::vector<std::unique_ptr<RaftNode>> nodes;
    
    // Create nodes
    for (int i = 0; i < num_nodes; ++i) {
        nodes.push_back(std::make_unique<RaftNode>(i, num_nodes, network));
    }
    
    // Start nodes
    for (auto& node : nodes) {
        node->start();
    }
    
    // Let the simulation run for a while
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Check roles after election
    std::cout << "\nFinal node states:" << std::endl;
    for (int i = 0; i < num_nodes; ++i) {
        RaftNode::Role role = nodes[i]->get_role();
        std::string role_str = (role == RaftNode::LEADER) ? "LEADER" : 
                              (role == RaftNode::CANDIDATE) ? "CANDIDATE" : "FOLLOWER";
        std::cout << "Node " << i << ": " << role_str << ", Term: " << nodes[i]->get_term() << std::endl;
    }
    
    // Clean up
    for (auto& node : nodes) {
        node->stop();
    }
    network.stop();
}

// Simulate distributed lock usage
void distributed_lock_simulation() {
    std::cout << "\n=== Distributed Lock Simulation ===" << std::endl;
    
    const int num_nodes = 3;
    NetworkLayer network;
    network.start_processing();
    
    std::vector<DistributedLock> locks;
    for (int i = 0; i < num_nodes; ++i) {
        locks.emplace_back(i, network);
    }
    
    // Simulate nodes competing for the lock
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_nodes; ++i) {
        threads.emplace_back([i, &locks]() {
            std::cout << "Node " << i << " attempting to acquire lock..." << std::endl;
            
            auto start = std::chrono::high_resolution_clock::now();
            locks[i].acquire();
            auto end = std::chrono::high_resolution_clock::now();
            
            std::cout << "Node " << i << " acquired lock after " 
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                      << " ms" << std::endl;
            
            // Hold the lock for a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            std::cout << "Node " << i << " releasing lock" << std::endl;
            locks[i].release();
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    network.stop();
}

// Simulate leader election
void leader_election_simulation() {
    std::cout << "\n=== Leader Election Simulation ===" << std::endl;
    
    const int num_nodes = 5;
    NetworkLayer network;
    network.start_processing();
    
    std::vector<std::unique_ptr<RaftNode>> nodes;
    
    // Create nodes
    for (int i = 0; i < num_nodes; ++i) {
        nodes.push_back(std::make_unique<RaftNode>(i, num_nodes, network));
    }
    
    // Start nodes
    for (auto& node : nodes) {
        node->start();
    }
    
    // Simulate a scenario where the leader fails
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Stop the current leader (in practice, we'd need to identify who the leader is)
    // For simulation, we'll just let the algorithm work naturally
    
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Check final state
    int leader_count = 0;
    for (int i = 0; i < num_nodes; ++i) {
        if (nodes[i]->get_role() == RaftNode::LEADER) {
            leader_count++;
            std::cout << "Node " << i << " is the leader" << std::endl;
        }
    }
    
    std::cout << "Total leaders elected: " << leader_count << std::endl;
    
    // Clean up
    for (auto& node : nodes) {
        node->stop();
    }
    network.stop();
}

int main() {
    std::cout << "Distributed Concurrency - Hands-on Example" << std::endl;
    
    raft_consensus_simulation();
    distributed_lock_simulation();
    leader_election_simulation();
    
    std::cout << "\nAll distributed concurrency examples completed!" << std::endl;
    
    return 0;
}