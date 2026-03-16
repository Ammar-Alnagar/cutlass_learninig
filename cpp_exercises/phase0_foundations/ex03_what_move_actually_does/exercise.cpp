// CONCEPT: What move actually does — strip the name, prove it moves
// FORMAT: IMPLEMENT
// TIME_TARGET: 15 min
// WHY_THIS_MATTERS: Move is a cast, not an operation. It changes type, not behavior.
// CUDA_CONNECTION: std::move in host code before kernel launch argument setup.

#include <iostream>
#include <utility>
#include <cstring>

// TODO: Implement a move-only Buffer class with these requirements:
// 1. Constructor allocates with new char[size]
// 2. Destructor prints "Destroying buffer at {address}" and deletes[]
// 3. Move constructor prints "Moving buffer from {other.address} to {this}"
//    then steals the pointer and sets other.data = nullptr
// 4. Move assignment prints "Move-assigning {other.address} to {this}"
//    then frees current data, steals other's data, sets other.data = nullptr
// 5. Copy constructor and copy assignment are DELETED (move-only type)
// 6. Add a print_state() method that shows: address, size, data[0..20]

class Buffer {
public:
    size_t size;
    char* data;
    
    // Implement constructor
    Buffer(size_t s) : size(s), data(new char[s]) {
        std::memset(data, 0, size);
    }
    
    // Implement destructor
    ~Buffer() {
        std::cout << "Destroying buffer at " << this << "\n";
        delete[] data;
    }
    
    // TODO: Implement move constructor
    // Print: "Moving buffer from {&other} to {this}"
    // Steal: this->data = other.data, this->size = other.size
    // Nullify: other.data = nullptr, other.size = 0
    
    // TODO: Implement move assignment
    // Check self-assignment: if (this == &other) return *this;
    // Free current: delete[] data;
    // Print: "Move-assigning {&other} to {this}"
    // Steal and nullify
    
    // TODO: Delete copy operations (move-only type)
    // Buffer(const Buffer&) = delete;
    // Buffer& operator=(const Buffer&) = delete;
    
    void print_state() const {
        std::cout << "Buffer@" << this << ": size=" << size 
                  << ", data=" << (data ? data : "nullptr") << "\n";
    }
};

// TODO: Implement this function to demonstrate move semantics
// It should:
// 1. Print "=== Inside create_and_move() ==="
// 2. Create a Buffer(32) called 'local'
// 3. Write "Moved from local" into local.data
// 4. Print local's state
// 5. Return std::move(local) — this invokes the move constructor
Buffer create_and_move() {
    std::cout << "=== Inside create_and_move() ===\n";
    
    // Implement here
    Buffer local(32);
    std::strcpy(local.data, "Moved from local");
    local.print_state();
    
    return std::move(local);
}

int main() {
    std::cout << "=== Step 1: Create buffer ===\n";
    Buffer buf1(64);
    std::strcpy(buf1.data, "Original data in buf1");
    buf1.print_state();
    
    std::cout << "\n=== Step 2: Move buf1 to buf2 ===\n";
    Buffer buf2(std::move(buf1));
    std::cout << "buf1 after move: ";
    buf1.print_state();
    std::cout << "buf2 after move: ";
    buf2.print_state();
    
    std::cout << "\n=== Step 3: Move-assign buf2 to buf3 ===\n";
    Buffer buf3(16);  // Will be destroyed (data freed)
    std::strcpy(buf3.data, "Old buf3 data");
    buf3.print_state();
    
    buf3 = std::move(buf2);
    std::cout << "buf2 after move-assign: ";
    buf2.print_state();
    std::cout << "buf3 after move-assign: ";
    buf3.print_state();
    
    std::cout << "\n=== Step 4: Return value from function ===\n";
    Buffer buf4 = create_and_move();
    std::cout << "buf4 in main: ";
    buf4.print_state();
    
    std::cout << "\n=== Step 5: Exiting main ===\n";
    return 0;
}

// VERIFY: Expected output pattern:
// === Step 1: Create buffer ===
// Buffer@0x...: size=64, data=Original data in buf1
//
// === Step 2: Move buf1 to buf2 ===
// Moving buffer from 0x... to 0x...
// buf1 after move: Buffer@0x...: size=0, data=nullptr
// buf2 after move: Buffer@0x...: size=64, data=Original data in buf1
//
// === Step 3: Move-assign buf2 to buf3 ===
// Destroying buffer at 0x...  (old buf3 data freed)
// Move-assigning 0x... to 0x...
// buf2 after move-assign: Buffer@0x...: size=0, data=nullptr
// buf3 after move-assign: Buffer@0x...: size=64, data=Original data in buf1
//
// === Step 4: Return value from function ===
// === Inside create_and_move() ===
// Moving buffer from 0x... to 0x...
// buf4 in main: Buffer@0x...: size=32, data=Moved from local
//
// === Step 5: Exiting main ===
// Destroying buffer at 0x... (buf4)
// Destroying buffer at 0x... (buf3)
// Destroying buffer at 0x... (buf2 - nullptr, safe)
// Destroying buffer at 0x... (buf1 - nullptr, safe)
