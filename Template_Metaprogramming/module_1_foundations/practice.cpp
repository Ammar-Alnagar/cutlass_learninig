#include <iostream>
#include <vector>
#include <type_traits>
#include <memory>
#include <utility>
#include <string>
#include <fstream>

// Module 1: Foundations Practice
// Hands-on tutorial for modern C++ features needed for template metaprogramming
// TODO: Implement each exercise to learn modern C++ features

/*
 * EXERCISE 1: AUTO KEYWORD PRACTICE
 * Learn how the auto keyword performs type deduction
 * 
 * LEARNING OBJECTIVES:
 * - Understand how auto deduces types from initializers
 * - See how auto works with complex types like iterators
 * - Learn about auto with references and const qualifiers
 */
void exercise_auto_keyword() {
    std::cout << "\n=== Exercise 1: Auto Keyword ===" << std::endl;

    // Basic type deduction - TODO: Determine what types these variables will have
    auto x = 42;           // TODO: What type is x? (Hint: look at the initializer)
    auto y = 42.5;         // TODO: What type is y?
    auto z = "hello";      // TODO: What type is z? (Hint: string literals have a specific type)

    std::cout << "x = " << x << " (type deduced as int)" << std::endl;
    std::cout << "y = " << y << " (type deduced as double)" << std::endl;
    std::cout << "z = " << z << " (type deduced as const char*)" << std::endl;

    // With complex types - TODO: Understand how auto works with STL containers
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin(); // TODO: What type is it? (Hint: it's an iterator type)
    std::cout << "First element via iterator: " << *it << std::endl;

    // Using auto with references - TODO: See how references work with auto
    auto& ref_x = x;       // TODO: What type is ref_x? (Hint: the & makes it a reference)
    ref_x = 100;
    std::cout << "After modifying ref_x, x = " << x << std::endl;

    // Using auto with const - TODO: Understand how const works with auto
    const auto cx = x;     // TODO: What type is cx? (Hint: const makes it read-only)
    std::cout << "cx = " << cx << " (const int)" << std::endl;
}

/*
 * EXERCISE 2: CONSTEXPR FUNCTIONS
 * Functions that can be evaluated at compile time
 */
constexpr int square(int x) {
    return x * x;
}

void exercise_constexpr_functions() {
    std::cout << "\n=== Exercise 2: Constexpr Functions ===" << std::endl;

    constexpr int result = square(5);  // Computed at compile time
    std::cout << "square(5) = " << result << " (computed at compile time)" << std::endl;

    int arr[result];                   // Valid array size in C++
    std::cout << "Created array of size " << result << " using constexpr" << std::endl;

    // More complex constexpr function
    constexpr auto power = [](int base, int exp) constexpr {
        int result = 1;
        for(int i = 0; i < exp; ++i) {
            result *= base;
        }
        return result;
    };

    constexpr int pow_result = power(2, 4);
    std::cout << "2^4 = " << pow_result << " (computed at compile time)" << std::endl;
}

/*
 * EXERCISE 3: LAMBDA FUNCTIONS
 * Anonymous functions with capture capabilities
 */
void exercise_lambda_functions() {
    std::cout << "\n=== Exercise 3: Lambda Functions ===" << std::endl;

    std::vector<int> vec = {5, 2, 8, 1, 9};

    // Sort in descending order using lambda
    std::sort(vec.begin(), vec.end(), [](int a, int b) {
        return a > b;
    });

    std::cout << "Sorted in descending order: ";
    for(const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    // Capture variables by value
    int multiplier = 10;
    auto multiply_by_ten = [multiplier](int x) {
        return x * multiplier;
    };

    std::cout << "multiply_by_ten(7) = " << multiply_by_ten(7) << std::endl;

    // Capture by reference
    auto increment_and_multiply = [&multiplier](int x) {
        multiplier++;
        return x * multiplier;
    };

    int result = increment_and_multiply(5);
    std::cout << "increment_and_multiply(5) = " << result
              << ", multiplier is now " << multiplier << std::endl;

    // Generic lambda (C++14 and later)
    auto generic_lambda = [](auto value) {
        std::cout << "Generic lambda called with: " << value << std::endl;
    };

    generic_lambda(42);
    generic_lambda(3.14);
    generic_lambda("Hello");
}

/*
 * EXERCISE 4: RAII AND SMART POINTERS
 * Resource Acquisition Is Initialization principle
 */
class FileManager {
private:
    std::unique_ptr<std::ifstream> file;

public:
    FileManager(const std::string& filename) {
        file = std::make_unique<std::ifstream>(filename);
        std::cout << "Opened file: " << filename << std::endl;
    }

    ~FileManager() {
        std::cout << "File automatically closed when object goes out of scope" << std::endl;
    }

    bool is_open() const {
        return file && file->is_open();
    }
};

void exercise_raii_smart_pointers() {
    std::cout << "\n=== Exercise 4: RAII and Smart Pointers ===" << std::endl;

    // Smart pointers examples
    // unique_ptr - exclusive ownership
    auto ptr1 = std::make_unique<int>(42);
    std::cout << "unique_ptr value: " << *ptr1 << std::endl;

    // shared_ptr - shared ownership
    auto ptr2 = std::make_shared<int>(42);
    {
        auto ptr3 = ptr2;  // Both ptr2 and ptr3 share ownership
        std::cout << "shared_ptr value: " << *ptr2 << std::endl;
        std::cout << "Reference count: " << ptr2.use_count() << std::endl;
    } // ptr3 goes out of scope here
    std::cout << "After ptr3 goes out of scope, reference count: " << ptr2.use_count() << std::endl;

    // weak_ptr - non-owning reference to shared_ptr
    std::weak_ptr<int> weak_ref = ptr2;
    if(auto locked_ptr = weak_ref.lock()) {
        std::cout << "weak_ptr locked, value: " << *locked_ptr << std::endl;
    }

    // RAII example with FileManager
    {
        FileManager fm("nonexistent.txt");  // Will try to open, but fail silently in this example
        std::cout << "FileManager created inside scope" << std::endl;
    } // FileManager destructor called here, demonstrating RAII
}

/*
 * EXERCISE 5: MOVE SEMANTICS
 * Efficient transfer of resources
 */
class MyVector {
private:
    std::vector<int> data;

public:
    // Constructor
    MyVector(std::initializer_list<int> init) : data(init) {
        std::cout << "MyVector constructed with " << data.size() << " elements" << std::endl;
    }

    // Copy constructor
    MyVector(const MyVector& other) : data(other.data) {
        std::cout << "MyVector copied" << std::endl;
    }

    // Move constructor
    MyVector(MyVector&& other) noexcept : data(std::move(other.data)) {
        std::cout << "MyVector moved" << std::endl;
        // other.data is now empty
    }

    // Move assignment operator
    MyVector& operator=(MyVector&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            std::cout << "MyVector move assigned" << std::endl;
        }
        return *this;
    }

    void print() const {
        std::cout << "MyVector contents: ";
        for(const auto& val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
};

void exercise_move_semantics() {
    std::cout << "\n=== Exercise 5: Move Semantics ===" << std::endl;

    // Create a vector
    MyVector v1{1, 2, 3, 4, 5};
    v1.print();

    // Move constructor
    MyVector v2 = std::move(v1);  // v1's resources are moved to v2
    std::cout << "After moving v1 to v2:" << std::endl;
    v2.print();
    // Note: v1 is now in a valid but unspecified state

    // Create another vector
    MyVector v3{10, 20, 30};
    v3.print();

    // Move assignment
    v3 = std::move(v2);
    std::cout << "After move assigning v2 to v3:" << std::endl;
    v3.print();
}

/*
 * EXERCISE 6: FUNCTION TEMPLATES
 * Generic functions that work with multiple types
 */
template<typename T>
T max_func(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T, typename U>
auto multiply(T t, U u) -> decltype(t * u) {
    return t * u;
}

// C++14 and later can use auto return type deduction
template<typename T, typename U>
auto multiply_v2(T t, U u) {
    return t * u;
}

void exercise_function_templates() {
    std::cout << "\n=== Exercise 6: Function Templates ===" << std::endl;

    int i = max_func(5, 10);        // Instantiated as max_func<int>
    double d = max_func(3.14, 2.71); // Instantiated as max_func<double>

    std::cout << "max_func(5, 10) = " << i << std::endl;
    std::cout << "max_func(3.14, 2.71) = " << d << std::endl;

    // Multiple template parameters
    auto result1 = multiply(5, 2.5);      // int and double
    auto result2 = multiply_v2(3, 4.0);   // int and double

    std::cout << "multiply(5, 2.5) = " << result1 << std::endl;
    std::cout << "multiply_v2(3, 4.0) = " << result2 << std::endl;
}

/*
 * EXERCISE 7: TYPE DEDUCTION
 * Understanding how the compiler deduces types
 */
void exercise_type_deduction() {
    std::cout << "\n=== Exercise 7: Type Deduction ===" << std::endl;

    int x = 42;
    const int cx = x;
    const int& crx = x;

    auto a1 = x;        // a1 is int
    auto a2 = cx;       // a2 is int (top-level const is ignored)
    auto a3 = crx;      // a3 is int (reference is ignored, top-level const is ignored)
    auto& a4 = x;       // a4 is int&
    auto& a5 = cx;      // a5 is const int&
    auto& a6 = crx;     // a6 is const int&

    std::cout << "a1 (auto x = 42) is type: int, value: " << a1 << std::endl;
    std::cout << "a2 (auto cx = const int) is type: int, value: " << a2 << std::endl;
    std::cout << "a4 (auto& x = 42) is type: int&, value: " << a4 << std::endl;
    std::cout << "a5 (auto& cx = const int) is type: const int&, value: " << a5 << std::endl;

    decltype(x) b1 = 42;     // b1 is int
    //decltype((x)) b2 = x;    // b2 would be int& (parentheses matter!)

    std::cout << "b1 (decltype(x)) is type: int, value: " << b1 << std::endl;
}

/*
 * EXERCISE 8: SFINAE BASICS
 * Substitution Failure Is Not An Error
 */
#include <type_traits>

// Enable function only for integral types
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
process(T value) {
    std::cout << "Processing integral: " << static_cast<long long>(value) << std::endl;
}

// Enable function only for floating-point types
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
process(T value) {
    std::cout << "Processing float: " << static_cast<double>(value) << std::endl;
}

// Alternative C++17 approach using if constexpr
template<typename T>
void process_v2(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Processing integral (v2): " << static_cast<long long>(value) << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Processing float (v2): " << static_cast<double>(value) << std::endl;
    } else {
        std::cout << "Processing other type (v2): " << typeid(T).name() << std::endl;
    }
}

void exercise_sfinae() {
    std::cout << "\n=== Exercise 8: SFINAE Basics ===" << std::endl;

    process(42);      // Calls integral version
    process(3.14f);   // Calls floating-point version

    process_v2(42);      // Calls integral branch
    process_v2(3.14);    // Calls floating-point branch
    process_v2("text");  // Calls else branch
}

/*
 * HANDS-ON CHALLENGES
 * Try implementing these exercises yourself
 */

// Challenge 1: Implement generic min/max functions
template<typename T>
const T& min_func(const T& a, const T& b) {
    return (a < b) ? a : b;
}

template<typename T>
const T& max_func_challenge(const T& a, const T& b) {
    return (a < b) ? b : a;
}

// Challenge 2: Simple smart pointer implementation
template<typename T>
class SimplePtr {
private:
    T* ptr;

public:
    explicit SimplePtr(T* p = nullptr) : ptr(p) {}

    ~SimplePtr() {
        delete ptr;
    }

    // Move constructor
    SimplePtr(SimplePtr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    // Move assignment
    SimplePtr& operator=(SimplePtr&& other) noexcept {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    // Delete copy operations to prevent double deletion
    SimplePtr(const SimplePtr&) = delete;
    SimplePtr& operator=(const SimplePtr&) = delete;

    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
    T* get() const { return ptr; }
};

// Challenge 3: Type traits detector
template<typename T>
void process_value(T value) {
    if constexpr (std::is_arithmetic_v<T>) {
        std::cout << "Arithmetic: " << value << std::endl;
    } else {
        std::cout << "Non-arithmetic: " << typeid(T).name() << std::endl;
    }
}

int main() {
    std::cout << "Module 1: Foundations Practice - Hands-on Tutorial" << std::endl;

    // Run all exercises
    exercise_auto_keyword();
    exercise_constexpr_functions();
    exercise_lambda_functions();
    exercise_raii_smart_pointers();
    exercise_move_semantics();
    exercise_function_templates();
    exercise_type_deduction();
    exercise_sfinae();

    // Try the challenges
    std::cout << "\n=== Challenge Solutions ===" << std::endl;

    // Challenge 1
    std::cout << "Challenge 1 - min/max: min(15, 10) = " << min_func(15, 10) << std::endl;

    // Challenge 2
    {
        SimplePtr<int> sp(std::make_unique<int>(100).release()); // Manual memory management for demo
        if(sp.get()) {
            std::cout << "Challenge 2 - SimplePtr: " << *sp << std::endl;
        }
    } // sp destructor called here, memory freed

    // Challenge 3
    std::cout << "Challenge 3 - Type detection: ";
    process_value(42);
    process_value(3.14);
    process_value(std::string("Hello"));

    return 0;
}