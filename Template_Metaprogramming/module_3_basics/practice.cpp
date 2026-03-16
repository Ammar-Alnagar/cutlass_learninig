#include <iostream>
#include <vector>
#include <type_traits>
#include <chrono>
#include <string>
#include <deque>
#include <iterator>

// Module 3: Template Metaprogramming Basics Practice
// Hands-on tutorial for compile-time computation and type manipulation
// TODO: Implement each exercise to learn template metaprogramming concepts

/*
 * EXERCISE 1: COMPILE-TIME VS RUNTIME COMPUTATION
 * Understanding the differences and benefits of compile-time computation
 * 
 * LEARNING OBJECTIVES:
 * - Understand how template metaprogramming enables compile-time computation
 * - Learn the difference between runtime and compile-time calculations
 * - See how template specialization works for base cases
 */
// TODO: Implement compile-time factorial calculation using template metaprogramming
// Hint: This uses template recursion where each instantiation depends on the previous one
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;  // What happens when N=0?
};

// TODO: Add base case specialization to stop the recursion
// Hint: What should the factorial of 0 be?
template<>
struct Factorial<0> {
    static constexpr int value = 1;  // Why is this the base case?
};

// Original implementations kept as reference for learning
// Uncomment the following code to see a working implementation
/*
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};
*/

// TODO: Implement runtime factorial calculation for comparison
// Hint: This is a regular recursive function that runs at runtime
int runtime_factorial(int n) {
    if (n <= 1) return 1;
    return n * runtime_factorial(n - 1);  // TODO: Trace through this recursion for n=3
}

void exercise_compile_time_vs_runtime() {
    std::cout << "\n=== Exercise 1: Compile-Time vs Runtime ===" << std::endl;

    // Runtime calculation
    auto start = std::chrono::high_resolution_clock::now();
    int rt_result = runtime_factorial(10);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // TODO: Calculate factorial at compile time
// Hint: This should be instantaneous at runtime since it's computed during compilation
    constexpr int ct_result = Factorial<10>::value;  // What value should this be?

    std::cout << "Runtime result: " << rt_result << std::endl;
    std::cout << "Compile-time result: " << ct_result << std::endl;
    std::cout << "Runtime calculation took: " << duration.count() << " nanoseconds" << std::endl;
    std::cout << "Compile-time calculation: instantaneous at runtime" << std::endl;

    // Compile-time assertion
    static_assert(Factorial<5>::value == 120, "Factorial of 5 should be 120");
    std::cout << "Compile-time assertion passed: 5! = " << Factorial<5>::value << std::endl;
}

/*
 * EXERCISE 2: TEMPLATE RECURSION
 * Implementing compile-time algorithms using template recursion
 */
// TODO: Implement compile-time Fibonacci sequence using template metaprogramming
// Hint: Similar to factorial but depends on two previous values (N-1 and N-2)
// What happens when we try to instantiate Fibonacci<0> or Fibonacci<1>? They would try to access negative indices which don't exist!
template<int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N-1>::value + Fibonacci<N-2>::value;  // What happens for N=0 or N=1?
};

// TODO: Add base case specializations to stop the recursion
// Hint: What should Fibonacci<0> and Fibonacci<1> be? (0 and 1 by definition)
template<>
struct Fibonacci<0> {
    static constexpr int value = 0;  // Why is this the base case? (By definition, F(0) = 0)
};

template<>
struct Fibonacci<1> {
    static constexpr int value = 1;  // Why is this the base case? (By definition, F(1) = 1)
};

// Sum of numbers from 1 to N at compile time
template<int N>
struct SumToN {
    static constexpr int value = N + SumToN<N-1>::value;
};

template<>
struct SumToN<1> {
    static constexpr int value = 1;
};

// Alternative with function templates (C++17)
template<int N>
constexpr int sum_to_n() {
    if constexpr (N <= 1) {
        return N;
    } else {
        return N + sum_to_n<N-1>();
    }
}

void exercise_template_recursion() {
    std::cout << "\n=== Exercise 2: Template Recursion ===" << std::endl;

    // Fibonacci examples
    constexpr int fib10 = Fibonacci<10>::value;  // 55
    constexpr int fib5 = Fibonacci<5>::value;    // 5
    std::cout << "Fibonacci<10>::value = " << fib10 << std::endl;
    std::cout << "Fibonacci<5>::value = " << fib5 << std::endl;

    // Sum examples
    constexpr int sum10 = SumToN<10>::value;  // 1+2+3+...+10 = 55
    std::cout << "SumToN<10>::value = " << sum10 << std::endl;

    // Function template example
    constexpr int func_sum5 = sum_to_n<5>();  // 1+2+3+4+5 = 15
    std::cout << "sum_to_n<5>() = " << func_sum5 << std::endl;
}

/*
 * EXERCISE 3: TYPE TRAITS AND STD::ENABLE_IF
 * Using type traits for conditional compilation
 */
// Check if type is integral
template<typename T>
void process_integral(T value) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    std::cout << "Processing integral: " << value << std::endl;
}

// Check if type is floating point
template<typename T>
void process_floating(T value) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    std::cout << "Processing float: " << value << std::endl;
}

// Check if types are the same
template<typename T, typename U>
void compare_types() {
    if constexpr (std::is_same_v<T, U>) {
        std::cout << "Types are the same" << std::endl;
    } else {
        std::cout << "Types are different" << std::endl;
    }
}

// Check if type has begin() and end() methods
template<typename T, typename = void>
struct is_iterable : std::false_type {};

template<typename T>
struct is_iterable<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

// Check if type is callable
template<typename T, typename = void>
struct is_callable : std::false_type {};

template<typename T>
struct is_callable<T, std::void_t<
    decltype(std::declval<T>()(std::declval<int>()))
>> : std::true_type {};

// Using std::enable_if - Method 1: Return type
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
abs_value(T val) {
    return val < 0 ? -val : val;
}

// Using std::enable_if - Method 2: Template parameter
template<typename T,
         typename = std::enable_if_t<std::is_arithmetic_v<T>>>
T abs_value_v2(T val) {
    return val < 0 ? -val : val;
}

// Using SFINAE with function parameters
template<typename T>
void process_number(T val, std::enable_if_t<std::is_integral_v<T>, int> = 0) {
    std::cout << "Processing integer: " << val << std::endl;
}

template<typename T>
void process_number(T val, std::enable_if_t<std::is_floating_point_v<T>, int> = 0) {
    std::cout << "Processing float: " << val << std::endl;
}

void exercise_type_traits() {
    std::cout << "\n=== Exercise 3: Type Traits and std::enable_if ===" << std::endl;

    // Testing type checking
    process_integral(42);
    process_floating(3.14f);

    compare_types<int, int>();
    compare_types<int, double>();

    // Testing custom type traits
    std::cout << "std::vector<int> is iterable: " << is_iterable<std::vector<int>>::value << std::endl;
    std::cout << "int is iterable: " << is_iterable<int>::value << std::endl;

    // Testing std::enable_if
    std::cout << "abs_value(-5): " << abs_value(-5) << std::endl;
    std::cout << "abs_value(-3.14): " << abs_value(-3.14) << std::endl;

    process_number(42);
    process_number(3.14);
}

/*
 * EXERCISE 4: CONDITIONAL COMPILATION WITH TEMPLATES
 * Using templates for compile-time branching
 */
template<typename T>
void conditional_process(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Processing integer: " << value << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Processing float: " << value << std::endl;
    } else {
        std::cout << "Processing other type" << std::endl;
    }
}

// Complex conditional processing
template<typename Container>
auto get_size(Container& c) {
    if constexpr (requires { c.size(); }) {
        std::cout << "Using size() method" << std::endl;
        return c.size();
    } else if constexpr (requires { c.length(); }) {
        std::cout << "Using length() method" << std::endl;
        return c.length();
    } else {
        static_assert(false, "Container must have size() or length() method");
    }
}

void exercise_conditional_compilation() {
    std::cout << "\n=== Exercise 4: Conditional Compilation ===" << std::endl;

    conditional_process(42);
    conditional_process(3.14);
    conditional_process(std::string("Hello"));

    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::string str = "Hello, World!";

    auto vec_size = get_size(vec);
    auto str_size = get_size(str);

    std::cout << "Vector size: " << vec_size << std::endl;
    std::cout << "String size: " << str_size << std::endl;
}

/*
 * EXERCISE 5: DEPENDENT NAMES AND TEMPLATE TEMPLATE PARAMETERS
 * Understanding dependent names and template template parameters
 */
template<typename T>
struct Base {
    static constexpr int value = 42;
    using type = double;
};

template<typename T>
struct Derived : Base<T> {
    // Without 'typename', the compiler doesn't know 'type' is a type
    typename Base<T>::type member_var;

    void access_base_value() {
        // 'this->' needed to access dependent base members
        int val = this->value;  // Same as Base<T>::value
        std::cout << "Accessed base value: " << val << std::endl;
    }
};

// Template template parameter example
template<template<typename> typename Container, typename T>
class GenericWrapper {
    Container<T> storage;
public:
    void add(const T& item) {
        storage.push_back(item);
        std::cout << "Added item to container, new size: " << storage.size() << std::endl;
    }

    size_t size() const {
        return storage.size();
    }

    void print_info() const {
        std::cout << "Container type: " << typeid(storage).name() << std::endl;
        std::cout << "Size: " << size() << std::endl;
    }
};

void exercise_dependent_names() {
    std::cout << "\n=== Exercise 5: Dependent Names and Template Template Parameters ===" << std::endl;

    Derived<int> derived;
    derived.access_base_value();

    // Template template parameter examples
    GenericWrapper<std::vector, int> vector_wrapper;
    vector_wrapper.add(10);
    vector_wrapper.add(20);
    vector_wrapper.print_info();

    GenericWrapper<std::deque, std::string> deque_wrapper;
    deque_wrapper.add("Hello");
    deque_wrapper.add("World");
    deque_wrapper.print_info();
}

/*
 * EXERCISE 6: EXPRESSION SFINAE
 * Checking if expressions are valid at compile time
 */
// Check if a type has a serialize method
template<typename T>
class HasSerialize {
    template<typename U>
    static auto test(int) -> decltype(std::declval<U>().serialize(), std::true_type{});

    template<typename>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

// Modern C++17 approach using std::void_t
template<typename T, typename = void>
struct has_serialize_method : std::false_type {};

template<typename T>
struct has_serialize_method<T, std::void_t<
    decltype(std::declval<T>().serialize())
>> : std::true_type {};

// A class with serialize method
class SerializableClass {
public:
    void serialize() const {
        std::cout << "SerializableClass serialized" << std::endl;
    }
};

// A class without serialize method
class NonSerializableClass {
public:
    void print() const {
        std::cout << "NonSerializableClass printed" << std::endl;
    }
};

void exercise_expression_sfinae() {
    std::cout << "\n=== Exercise 6: Expression SFINAE ===" << std::endl;

    std::cout << "SerializableClass has serialize: " << HasSerialize<SerializableClass>::value << std::endl;
    std::cout << "NonSerializableClass has serialize: " << HasSerialize<NonSerializableClass>::value << std::endl;

    std::cout << "Using std::void_t approach:" << std::endl;
    std::cout << "SerializableClass has serialize: " << has_serialize_method<SerializableClass>::value << std::endl;
    std::cout << "NonSerializableClass has serialize: " << has_serialize_method<NonSerializableClass>::value << std::endl;

    // Example of using the trait
    if constexpr (has_serialize_method<SerializableClass>::value) {
        SerializableClass obj;
        obj.serialize();
    }
}

/*
 * HANDS-ON CHALLENGES
 * Try implementing these exercises yourself
 */

// Challenge 1: Compile-Time Prime Checker
template<int N, int Divisor = N-1>
struct IsPrime {
    static constexpr bool value =
        (N % Divisor != 0) && IsPrime<N, Divisor-1>::value;
};

// Base cases
template<int N>
struct IsPrime<N, 1> {
    static constexpr bool value = true;
};

template<int N>
struct IsPrime<N, 0> {
    static constexpr bool value = false;
};

// Special cases
template<>
struct IsPrime<0> {
    static constexpr bool value = false;
};

template<>
struct IsPrime<1> {
    static constexpr bool value = false;
};

template<>
struct IsPrime<2> {
    static constexpr bool value = true;
};

// Helper variable template
template<int N>
inline constexpr bool is_prime_v = IsPrime<N>::value;

// Challenge 2: Type Trait for Container Detection
#include <type_traits>

// Check if type has begin() and end() methods
template<typename T, typename = void>
struct has_begin_end : std::false_type {};

template<typename T>
struct has_begin_end<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

// Check if type has size() method
template<typename T, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<
    decltype(std::declval<T>().size())
>> : std::true_type {};

// Check if type is container-like (has begin/end and size)
template<typename T>
struct is_container_like {
    static constexpr bool value =
        has_begin_end<T>::value && has_size<T>::value;
};

// Convenience variable templates
template<typename T>
inline constexpr bool has_begin_end_v = has_begin_end<T>::value;

template<typename T>
inline constexpr bool has_size_v = has_size<T>::value;

template<typename T>
inline constexpr bool is_container_like_v = is_container_like<T>::value;

// Challenge 3: Conditional Function Dispatcher
// Handler functions
template<typename T>
void arithmetic_handler(T value) {
    std::cout << "Handling arithmetic: " << value << std::endl;
}

template<typename T>
void pointer_handler(T ptr) {
    if (ptr) {
        std::cout << "Handling pointer: " << *ptr << std::endl;
    } else {
        std::cout << "Handling null pointer" << std::endl;
    }
}

template<typename T>
void serializable_handler(T& obj) {
    std::cout << "Handling serializable object" << std::endl;
    obj.serialize();  // This should compile if T has serialize method
}

// Main dispatcher
template<typename T>
void dispatch_handler(T value) {
    if constexpr (std::is_arithmetic_v<T>) {
        arithmetic_handler(value);
    } else if constexpr (std::is_pointer_v<T>) {
        pointer_handler(value);
    } else if constexpr (has_serialize_method<T>::value) {
        serializable_handler(value);
    } else {
        std::cout << "Unsupported type for dispatch" << std::endl;
    }
}

int main() {
    std::cout << "Module 3: Template Metaprogramming Basics Practice - Hands-on Tutorial" << std::endl;

    // Run all exercises
    exercise_compile_time_vs_runtime();
    exercise_template_recursion();
    exercise_type_traits();
    exercise_conditional_compilation();
    exercise_dependent_names();
    exercise_expression_sfinae();

    // Try the challenges
    std::cout << "\n=== Challenge Solutions ===" << std::endl;

    // Challenge 1: Prime checker
    std::cout << "\nChallenge 1 - Prime checker:" << std::endl;
    std::cout << "Is 2 prime? " << is_prime_v<2> << std::endl;
    std::cout << "Is 17 prime? " << is_prime_v<17> << std::endl;
    std::cout << "Is 4 prime? " << is_prime_v<4> << std::endl;
    std::cout << "Is 11 prime? " << is_prime_v<11> << std::endl;

    // Challenge 2: Container detection
    std::cout << "\nChallenge 2 - Container detection:" << std::endl;
    std::cout << "std::vector<int> has begin/end: " << has_begin_end_v<std::vector<int>> << std::endl;
    std::cout << "std::vector<int> has size: " << has_size_v<std::vector<int>> << std::endl;
    std::cout << "std::vector<int> is container-like: " << is_container_like_v<std::vector<int>> << std::endl;
    std::cout << "int is container-like: " << is_container_like_v<int> << std::endl;

    // Challenge 3: Dispatcher
    std::cout << "\nChallenge 3 - Dispatcher:" << std::endl;
    dispatch_handler(42);  // arithmetic
    dispatch_handler(3.14);  // arithmetic
    int x = 100;
    dispatch_handler(&x);  // pointer
    SerializableClass serial_obj;
    dispatch_handler(serial_obj);  // serializable
    dispatch_handler(std::string("test"));  // unsupported

    return 0;
}