#include <iostream>
#include <vector>
#include <type_traits>
#include <memory>
#include <string>
#include <utility>
#include <tuple>
#include <algorithm>
#include <functional>
#include <cstring>

// Module 2: Template Fundamentals Practice
// Hands-on tutorial for template fundamentals needed for advanced metaprogramming
// TODO: Implement each exercise to learn template fundamentals

/*
 * EXERCISE 1: CLASS TEMPLATES
 * Creating generic classes that work with multiple types
 * 
 * LEARNING OBJECTIVES:
 * - Understand how to create generic classes using template<typename T>
 * - Learn how template parameters work in class definitions
 * - See how to use template types in member functions and variables
 * 
 * INSTRUCTIONS:
 * - Fill in the TODOs to implement a generic Stack class
 * - Each TODO represents a part of the implementation you need to complete
 * - Compare your implementation with the commented reference code
 */

// TODO: Implement this stack class template
template<typename T>
class Stack {
private:
    // TODO: What data structure should we use to store elements?
    std::vector<T> elements;  

public:
    // TODO: Implement the push function to add an element to the stack
    void push(const T& element) {
        // TODO: Add element to the data structure
        elements.push_back(element);
        // TODO: Print what was pushed (for debugging)
        std::cout << "Pushed: " << element << std::endl;
    }

    // TODO: Implement the pop function to remove the top element
    void pop() {
        // TODO: Check if stack is not empty
        if (!elements.empty()) {
            // TODO: Show what we're popping (for debugging)
            std::cout << "Popped: " << elements.back() << std::endl;
            // TODO: Remove the top element
            elements.pop_back();
        } else {
            // TODO: Handle empty stack case
            std::cout << "Stack is empty, cannot pop!" << std::endl;
        }
    }

    // TODO: Implement the top function to peek at the top element
    const T& top() const {
        // TODO: Check if stack is not empty
        if (!elements.empty()) {
            // TODO: Return the top element
            return elements.back();
        }
        // TODO: Throw exception for empty stack
        throw std::out_of_range("Stack is empty");
    }

    // TODO: Implement the empty function to check if stack has elements
    bool empty() const {
        // TODO: Return true if no elements, false otherwise
        return elements.empty();
    }

    // TODO: Implement the size function to get number of elements
    size_t size() const {
        // TODO: Return the number of elements
        return elements.size();
    }

    // TODO: Implement the print function to display all elements
    void print() const {
        // TODO: Print all elements in the stack
        std::cout << "Stack contents: ";
        for(const auto& elem : elements) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
};

// Original implementations kept as reference for learning
// Uncomment the following code to see a working implementation
/*
template<typename T>
class Stack {
private:
    std::vector<T> elements;

public:
    void push(const T& element) {
        elements.push_back(element);
        std::cout << "Pushed: " << element << std::endl;
    }

    void pop() {
        if (!elements.empty()) {
            std::cout << "Popped: " << elements.back() << std::endl;
            elements.pop_back();
        } else {
            std::cout << "Stack is empty, cannot pop!" << std::endl;
        }
    }

    const T& top() const {
        if (!elements.empty()) {
            return elements.back();
        }
        throw std::out_of_range("Stack is empty");
    }

    bool empty() const {
        return elements.empty();
    }

    size_t size() const {
        return elements.size();
    }

    void print() const {
        std::cout << "Stack contents: ";
        for(const auto& elem : elements) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
};
*/

// TODO: Implement this hash map class template with multiple parameters
// Template class with multiple parameters
template<typename Key, typename Value, typename Hash = std::hash<Key>>
class SimpleHashMap {
private:
    std::vector<std::pair<Key, Value>> data;  // TODO: What data structure should we use?

public:
    // TODO: Implement insert function
    void insert(const Key& k, const Value& v) {
        // Simple implementation without collision handling
        data.emplace_back(k, v);
    }

    // TODO: Implement find function
    bool find(const Key& k, Value& v) const {
        for(const auto& pair : data) {
            if(pair.first == k) {
                v = pair.second;
                return true;
            }
        }
        return false;
    }

    // TODO: Implement size function
    size_t size() const { return data.size(); }
};

// TODO: Implement this fixed array class template with non-type parameter
// Template class with non-type parameter
template<typename T, size_t N>
class FixedArray {
private:
    T data[N];      // TODO: What type should data be? What does N represent?
    size_t count = 0;  // TODO: What does count track?

public:
    // TODO: Implement push function
    void push(const T& element) {
        if(count < N) {
            data[count++] = element;
        } else {
            std::cout << "FixedArray is full!" << std::endl;
        }
    }

    // TODO: Implement operator[] for non-const objects
    T& operator[](size_t index) {
        if(index >= count) throw std::out_of_range("Index out of bounds");
        return data[index];
    }

    // TODO: Implement operator[] for const objects
    const T& operator[](size_t index) const {
        if(index >= count) throw std::out_of_range("Index out of bounds");
        return data[index];
    }

    // TODO: Implement size function
    size_t size() const { return count; }
    
    // TODO: Implement capacity function
    size_t capacity() const { return N; }

    // TODO: Implement print function
    void print() const {
        std::cout << "FixedArray contents: ";
        for(size_t i = 0; i < count; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
};

void exercise_class_templates() {
    std::cout << "\n=== Exercise 1: Class Templates ===" << std::endl;

    // TODO: Create a Stack of integers and test it
    Stack<int> int_stack;
    int_stack.push(42);
    int_stack.push(100);
    int_stack.push(-5);
    int_stack.print();

    std::cout << "Top element: " << int_stack.top() << std::endl;
    int_stack.pop();
    std::cout << "Size after pop: " << int_stack.size() << std::endl;

    // Stack with strings
    Stack<std::string> string_stack;
    string_stack.push("Hello");
    string_stack.push("World");
    string_stack.print();

    // FixedArray example
    FixedArray<int, 5> fixed_arr;
    fixed_arr.push(10);
    fixed_arr.push(20);
    fixed_arr.push(30);
    fixed_arr.print();
    std::cout << "Capacity: " << fixed_arr.capacity() << ", Size: " << fixed_arr.size() << std::endl;

    // Simple hash map
    SimpleHashMap<std::string, int> map;
    map.insert("one", 1);
    map.insert("two", 2);
    map.insert("three", 3);

    int value;
    if(map.find("two", value)) {
        std::cout << "Found 'two': " << value << std::endl;
    }
}

/*
 * EXERCISE 2: FUNCTION TEMPLATES
 * Creating generic functions that work with multiple types
 */
template<typename T>
T add(T a, T b) {
    return a + b;
}

template<typename T, typename U>
auto multiply(T t, U u) -> decltype(t * u) {
    return t * u;
}

// C++14 and later: auto return type deduction
template<typename T, typename U>
auto multiply_v2(T t, U u) {
    return t * u;
}

template<typename T>
void print_type_info(const T& value) {
    std::cout << "Value: " << value
              << ", Type info: " << typeid(T).name() << std::endl;
}

void exercise_function_templates() {
    std::cout << "\n=== Exercise 2: Function Templates ===" << std::endl;

    // Basic function template
    int sum_int = add(10, 20);
    double sum_double = add(3.5, 2.7);
    std::cout << "add(10, 20) = " << sum_int << std::endl;
    std::cout << "add(3.5, 2.7) = " << sum_double << std::endl;

    // Function template with multiple type parameters
    auto result1 = multiply(5, 2.5);      // int and double
    auto result2 = multiply_v2(3, 4.0);   // int and double
    std::cout << "multiply(5, 2.5) = " << result1 << std::endl;
    std::cout << "multiply_v2(3, 4.0) = " << result2 << std::endl;

    // Template argument deduction
    print_type_info(42);        // T is deduced as int
    print_type_info(3.14);      // T is deduced as double
    print_type_info("hello");   // T is deduced as const char (&)[6]

    // Explicit template argument specification
    print_type_info<int>(42.5); // T is explicitly int, value is converted
}

/*
 * EXERCISE 3: TEMPLATE PARAMETERS
 * Different types of template parameters
 */
template<int N>
class Buffer {
    char data[N];
public:
    size_t capacity() const { return N; }
    void fill_with_char(char c) {
        for(int i = 0; i < N; ++i) {
            data[i] = c;
        }
    }
    void print_sample() const {
        std::cout << "Buffer sample: ";
        for(int i = 0; i < std::min(N, 10); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << (N > 10 ? "... " : "") << "(capacity: " << N << ")" << std::endl;
    }
};

template<typename T, int SIZE>
class StaticVector {
    T data[SIZE];
    int count = 0;
public:
    void push(const T& item) {
        if (count < SIZE) {
            data[count++] = item;
            std::cout << "Added item, new size: " << count << std::endl;
        } else {
            std::cout << "StaticVector is full!" << std::endl;
        }
    }

    T& operator[](int index) { return data[index]; }
    const T& operator[](int index) const { return data[index]; }
    int size() const { return count; }
    int capacity() const { return SIZE; }
};

// Template template parameters
template<template<typename> class Container, typename T>
class GenericProcessor {
    Container<T> container;
public:
    void add(const T& item) {
        container.push_back(item);
        std::cout << "Added item to container, new size: " << container.size() << std::endl;
    }

    void print_size() const {
        std::cout << "Container size: " << container.size() << std::endl;
    }
};

void exercise_template_parameters() {
    std::cout << "\n=== Exercise 3: Template Parameters ===" << std::endl;

    // Non-type parameter example
    Buffer<20> buf;
    buf.fill_with_char('X');
    buf.print_sample();

    // Template with type and non-type parameters
    StaticVector<int, 5> sv;
    sv.push(10);
    sv.push(20);
    sv.push(30);
    std::cout << "StaticVector size: " << sv.size() << "/" << sv.capacity() << std::endl;

    // Template template parameter example
    GenericProcessor<std::vector, std::string> processor;
    processor.add("First");
    processor.add("Second");
    processor.print_size();
}

/*
 * EXERCISE 1: CLASS TEMPLATES
 * Creating generic classes that work with multiple types
 * 
 * LEARNING OBJECTIVES:
 * - Understand how to create generic classes using template<typename T>
 * - Learn how template parameters work in class definitions
 * - See how to use template types in member functions and variables
 * - Understand template classes with multiple parameters
 * - Learn about non-type template parameters (like size_t N)
 */
// TODO: Implement this stack class template
template<typename T>
class Stack {
private:
    std::vector<T> elements;  // TODO: What type should elements be?

public:
    // TODO: Implement push function
    void push(const T& element) {
        elements.push_back(element);
        std::cout << "Pushed: " << element << std::endl;
    }

    // TODO: Implement pop function
    void pop() {
        if (!elements.empty()) {
            std::cout << "Popped: " << elements.back() << std::endl;
            elements.pop_back();
        } else {
            std::cout << "Stack is empty, cannot pop!" << std::endl;
        }
    }

    // TODO: Implement top function
    const T& top() const {
        if (!elements.empty()) {
            return elements.back();
        }
        throw std::out_of_range("Stack is empty");
    }

    // TODO: Implement empty function
    bool empty() const {
        return elements.empty();
    }

    // TODO: Implement size function
    size_t size() const {
        return elements.size();
    }

    // TODO: Implement print function
    void print() const {
        std::cout << "Stack contents: ";
        for(const auto& elem : elements) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
};

// TODO: Implement this hash map class template with multiple parameters
// Template class with multiple parameters
template<typename Key, typename Value, typename Hash = std::hash<Key>>
class SimpleHashMap {
private:
    std::vector<std::pair<Key, Value>> data;  // TODO: What data structure should we use?

public:
    // TODO: Implement insert function
    void insert(const Key& k, const Value& v) {
        // Simple implementation without collision handling
        data.emplace_back(k, v);
    }

    // TODO: Implement find function
    bool find(const Key& k, Value& v) const {
        for(const auto& pair : data) {
            if(pair.first == k) {
                v = pair.second;
                return true;
            }
        }
        return false;
    }

    // TODO: Implement size function
    size_t size() const { return data.size(); }
};

// TODO: Implement this fixed array class template with non-type parameter
// Template class with non-type parameter
template<typename T, size_t N>
class FixedArray {
private:
    T data[N];      // TODO: What type should data be? What does N represent?
    size_t count = 0;  // TODO: What does count track?

public:
    // TODO: Implement push function
    void push(const T& element) {
        if(count < N) {
            data[count++] = element;
        } else {
            std::cout << "FixedArray is full!" << std::endl;
        }
    }

    // TODO: Implement operator[] for non-const objects
    T& operator[](size_t index) {
        if(index >= count) throw std::out_of_range("Index out of bounds");
        return data[index];
    }

    // TODO: Implement operator[] for const objects
    const T& operator[](size_t index) const {
        if(index >= count) throw std::out_of_range("Index out of bounds");
        return data[index];
    }

    // TODO: Implement size function
    size_t size() const { return count; }
    
    // TODO: Implement capacity function
    size_t capacity() const { return N; }

    // TODO: Implement print function
    void print() const {
        std::cout << "FixedArray contents: ";
        for(size_t i = 0; i < count; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
};

// Full specialization for char*
template<>
class Serializer<char*> {
public:
    static void serialize(const char* str, std::ostream& out) {
        if (str) {
            out << "Specialized serialization: String: " << str << std::endl;
            std::cout << "Specialized serialization: String: " << str << std::endl;
        } else {
            out << "Specialized serialization: Null string" << std::endl;
            std::cout << "Specialized serialization: Null string" << std::endl;
        }
    }
};

// Full specialization for int
template<>
class Serializer<int> {
public:
    static void serialize(const int& value, std::ostream& out) {
        out << "Specialized serialization: Integer: " << value << std::endl;
        std::cout << "Specialized serialization: Integer: " << value << std::endl;
    }
};

// Example of partial specialization for pointer types
template<typename T>
class Container {
    T value;
public:
    void print() {
        std::cout << "General Container: " << value << std::endl;
    }
};

template<typename T>
class Container<T*> {  // Partial specialization for pointer types
    T* ptr;
public:
    Container(T* p) : ptr(p) {}
    void print() {
        if (ptr) {
            std::cout << "Pointer Container: " << *ptr << std::endl;
        } else {
            std::cout << "Pointer Container: Null pointer" << std::endl;
        }
    }
};

void exercise_template_specialization() {
    std::cout << "\n=== Exercise 4: Template Specialization ===" << std::endl;

    // Test generic serializer
    Serializer<double> double_serializer;
    double_serializer.serialize(3.14159, std::cout);

    // Test specialized serializers
    Serializer<int> int_serializer;
    int_serializer.serialize(42, std::cout);

    char* str = const_cast<char*>("Hello, specialization!");
    Serializer<char*> char_ptr_serializer;
    char_ptr_serializer.serialize(str, std::cout);

    // Test partial specialization
    Container<int> general_container;
    general_container.print();

    int val = 100;
    Container<int*> ptr_container(&val);
    ptr_container.print();
}

/*
 * EXERCISE 5: VARIADIC TEMPLATES
 * Templates that accept a variable number of arguments
 */
// Base case (terminator)
template<typename T>
void print_variadic(T&& t) {
    std::cout << t << std::endl;
}

// Recursive case
template<typename T, typename... Args>
void print_variadic(T&& t, Args&&... args) {
    std::cout << t << " ";
    print_variadic(args...);  // Recursive call with remaining arguments
}

// Variadic class template example - Simple tuple
template<typename... Types>
class Tuple {};

// Specialization for single type
template<typename T>
class Tuple<T> {
    T value;
public:
    Tuple(const T& v) : value(v) {}
    T get() const { return value; }
    void print() const { std::cout << "Single tuple: " << value << std::endl; }
};

// Recursive tuple
template<typename Head, typename... Tail>
class Tuple<Head, Tail...> {
    Head head;
    Tuple<Tail...> tail;
public:
    Tuple(Head h, Tail... t) : head(h), tail(t...) {}

    Head get_head() const { return head; }
    auto get_tail() const -> decltype(tail) { return tail; }

    void print() const {
        std::cout << "Tuple: " << head << ", ";
        tail.print();
    }
};

// Specialization for empty tuple
template<>
class Tuple<> {
public:
    void print() const { std::cout << "Empty tuple" << std::endl; }
};

// Fold expressions (C++17) - if compiler supports it
template<typename... Args>
auto sum_fold(Args... args) {
    return (args + ... + 0);  // Unary right fold
}

template<typename... Args>
bool all_true_fold(Args... args) {
    return (args && ...);  // Unary right fold for logical AND
}

template<typename... Args>
void print_fold(Args... args) {
    ((std::cout << args << " "), ...);  // Pack expansion with fold
    std::cout << std::endl;
}

void exercise_variadic_templates() {
    std::cout << "\n=== Exercise 5: Variadic Templates ===" << std::endl;

    // Basic variadic function
    std::cout << "Variadic print: ";
    print_variadic(1, 2, 3, "hello", 4.5);

    // Tuple examples
    Tuple<int> single_tuple(42);
    single_tuple.print();

    Tuple<int, double, std::string> multi_tuple(10, 3.14, "world");
    std::cout << "Multi tuple head: " << multi_tuple.get_head() << std::endl;

    // Fold expressions
    auto sum_result = sum_fold(1, 2, 3, 4, 5);
    std::cout << "Sum using fold: " << sum_result << std::endl;

    bool all_true_result = all_true_fold(true, true, false, true);
    std::cout << "All true using fold: " << all_true_result << std::endl;

    std::cout << "Print using fold: ";
    print_fold("Fold", "expression", "example", 123);
}

/*
 * EXERCISE 6: PERFECT FORWARDING
 * Preserving the value category (lvalue/rvalue) of forwarded arguments
 */
class ExampleClass {
    int value;
    std::string text;
public:
    ExampleClass(int v, const std::string& t) : value(v), text(t) {
        std::cout << "ExampleClass constructed with value=" << value << ", text=\"" << text << "\"" << std::endl;
    }

    ExampleClass(const ExampleClass& other) : value(other.value), text(other.text) {
        std::cout << "ExampleClass copy constructed" << std::endl;
    }

    ExampleClass(ExampleClass&& other) noexcept : value(other.value), text(std::move(other.text)) {
        std::cout << "ExampleClass move constructed" << std::endl;
    }

    int get_value() const { return value; }
    const std::string& get_text() const { return text; }
};

template<typename T>
void wrapper_forward(T&& arg) {
    // Forward arg preserving its value category
    std::cout << "Wrapper called with: " << arg << std::endl;
    // In a real scenario, we would forward to another function
    // some_function(std::forward<T>(arg));
}

// More complex example with multiple arguments
template<typename Func, typename... Args>
auto call_with_forwarding(Func&& func, Args&&... args)
    -> decltype(func(std::forward<Args>(args)...)) {
    std::cout << "Calling function with " << sizeof...(Args) << " arguments" << std::endl;
    return func(std::forward<Args>(args)...);
}

// Example function to demonstrate forwarding
template<typename... Args>
std::unique_ptr<ExampleClass> make_example_class(Args&&... args) {
    std::cout << "Making ExampleClass with " << sizeof...(Args) << " arguments" << std::endl;
    return std::make_unique<ExampleClass>(std::forward<Args>(args)...);
}

void some_function(const std::string& s) {
    std::cout << "Called with lvalue string: " << s << std::endl;
}

void some_function(std::string&& s) {
    std::cout << "Called with rvalue string: " << s << std::endl;
}

void exercise_perfect_forwarding() {
    std::cout << "\n=== Exercise 6: Perfect Forwarding ===" << std::endl;

    // Demonstrate forwarding of lvalue
    std::string lval = "lvalue string";
    wrapper_forward(lval);

    // Demonstrate forwarding of rvalue
    wrapper_forward(std::string("rvalue string"));

    // Show the difference in function calls
    some_function(lval);  // Calls lvalue version
    some_function(std::string("temporary"));  // Calls rvalue version

    // Forwarding with constructor
    auto obj1 = make_example_class(42, std::string("Hello"));
    auto obj2 = make_example_class(100, "World");  // String literal will be converted

    std::cout << "Created objects with values: " << obj1->get_value()
              << " and " << obj2->get_value() << std::endl;

    // Example of forwarding in a generic context
    auto lambda = [](auto&&... args) {
        std::cout << "Lambda received " << sizeof...(args) << " arguments: ";
        print_variadic(std::forward<decltype(args)>(args)...);
    };

    call_with_forwarding(lambda, 1, 2.5, std::string("forwarded"));
}

/*
 * HANDS-ON CHALLENGES
 * Try implementing these exercises yourself
 */

// Challenge 1: Generic Pair Class
template<typename T, typename U>
class Pair {
public:
    T first;
    U second;

    // Default constructor
    Pair() : first{}, second{} {
        std::cout << "Default Pair constructed" << std::endl;
    }

    // Constructor with values
    Pair(const T& f, const U& s) : first(f), second(s) {
        std::cout << "Pair constructed with values: " << f << ", " << s << std::endl;
    }

    // Move constructor
    Pair(T&& f, U&& s) : first(std::forward<T>(f)), second(std::forward<U>(s)) {
        std::cout << "Pair move constructed" << std::endl;
    }

    // Copy constructor
    Pair(const Pair& other) : first(other.first), second(other.second) {
        std::cout << "Pair copy constructed" << std::endl;
    }

    // Move constructor
    Pair(Pair&& other) noexcept : first(std::move(other.first)), second(std::move(other.second)) {
        std::cout << "Pair move constructed from another Pair" << std::endl;
    }

    // Copy assignment
    Pair& operator=(const Pair& other) {
        if (this != &other) {
            first = other.first;
            second = other.second;
        }
        std::cout << "Pair copy assigned" << std::endl;
        return *this;
    }

    // Move assignment
    Pair& operator=(Pair&& other) noexcept {
        if (this != &other) {
            first = std::move(other.first);
            second = std::move(other.second);
        }
        std::cout << "Pair move assigned" << std::endl;
        return *this;
    }

    void print() const {
        std::cout << "Pair(" << first << ", " << second << ")" << std::endl;
    }
};

// Challenge 2: Specialized Container
#include <vector>
#include <string>

// Primary template
template<typename T>
class OptimizedContainer {
    std::vector<T> data;
public:
    void add(const T& item) {
        data.push_back(item);
        std::cout << "Added item to generic container" << std::endl;
    }
    size_t size() const { return data.size(); }
    const T& operator[](size_t idx) const { return data[idx]; }

    void print() const {
        std::cout << "Generic container contents: ";
        for(const auto& item : data) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
};

// Specialization for bool (bit packing)
template<>
class OptimizedContainer<bool> {
    std::vector<uint8_t> data;
    size_t count;

public:
    OptimizedContainer() : count(0) {}

    void add(bool value) {
        size_t byte_idx = count / 8;
        size_t bit_idx = count % 8;

        if (byte_idx >= data.size()) {
            data.push_back(0);
        }

        if (value) {
            data[byte_idx] |= (1 << bit_idx);
        } else {
            data[byte_idx] &= ~(1 << bit_idx);
        }
        ++count;
        std::cout << "Added boolean value to optimized container" << std::endl;
    }

    bool operator[](size_t idx) const {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        return (data[byte_idx] >> bit_idx) & 1;
    }

    size_t size() const { return count; }

    void print() const {
        std::cout << "Boolean container bits: ";
        for(size_t i = 0; i < count; ++i) {
            std::cout << (*this)[i] << " ";
        }
        std::cout << std::endl;
    }
};

// Challenge 3: Variadic Function Calculator
template<typename... Args>
auto sum_calc(Args... args) {
    return (args + ...);
}

template<typename... Args>
auto product_calc(Args... args) {
    return (args * ...);
}

template<typename... Args>
auto max_calc(Args... args) {
    return std::max({args...});
}

template<typename... Args>
auto calc(const std::string& op, Args... args) {
    if (op == "sum") {
        return sum_calc(args...);
    } else if (op == "product") {
        return product_calc(args...);
    } else if (op == "max") {
        return max_calc(args...);
    } else if (op == "min") {
        return std::min({args...});
    } else {
        throw std::invalid_argument("Unknown operation: " + op);
    }
}

int main() {
    std::cout << "Module 2: Template Fundamentals Practice - Hands-on Tutorial" << std::endl;

    // Run all exercises
    exercise_class_templates();
    exercise_function_templates();
    exercise_template_parameters();
    exercise_template_specialization();
    exercise_variadic_templates();
    exercise_perfect_forwarding();

    // Try the challenges
    std::cout << "\n=== Challenge Solutions ===" << std::endl;

    // Challenge 1: Pair class
    std::cout << "\nChallenge 1 - Pair class:" << std::endl;
    Pair<int, std::string> p1(42, "Hello");
    p1.print();

    Pair<double, int> p2(3.14, 100);
    p2.print();

    // Challenge 2: Specialized containers
    std::cout << "\nChallenge 2 - Specialized containers:" << std::endl;
    OptimizedContainer<int> int_container;
    int_container.add(10);
    int_container.add(20);
    int_container.add(30);
    int_container.print();

    OptimizedContainer<bool> bool_container;
    bool_container.add(true);
    bool_container.add(false);
    bool_container.add(true);
    bool_container.add(true);
    bool_container.print();

    // Challenge 3: Calculator
    std::cout << "\nChallenge 3 - Calculator:" << std::endl;
    auto sum_result = calc("sum", 1, 2, 3, 4, 5);
    auto product_result = calc("product", 2, 3, 4);
    auto max_result = calc("max", 5, 2, 8, 1);
    auto min_result = calc("min", 5, 2, 8, 1);

    std::cout << "Sum of 1,2,3,4,5: " << sum_result << std::endl;
    std::cout << "Product of 2,3,4: " << product_result << std::endl;
    std::cout << "Max of 5,2,8,1: " << max_result << std::endl;
    std::cout << "Min of 5,2,8,1: " << min_result << std::endl;

    return 0;
}