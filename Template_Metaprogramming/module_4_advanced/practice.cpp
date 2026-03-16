#include <iostream>
#include <vector>
#include <type_traits>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <iterator>
#include <memory>
#include <algorithm>

// Module 4: Advanced Template Metaprogramming Techniques Practice
// Hands-on tutorial for sophisticated TMP patterns and techniques
// TODO: Implement each exercise to learn advanced template metaprogramming

/*
 * EXERCISE 1: TEMPLATE METAPROGRAMMING PATTERNS
 * Implementing common TMP patterns like Enable-if, Tag dispatching, and Policy-based design
 * 
 * LEARNING OBJECTIVES:
 * - Understand how to use SFINAE (Substitution Failure Is Not An Error) with std::enable_if
 * - Learn tag dispatching for selecting different implementations based on types
 * - See how policy-based design enables customizable behavior through template parameters
 */
// TODO: Implement safe multiplication that handles different types differently
// Hint: Use std::enable_if to create type-specific overloads
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T>
multiply_safe(T a, T b) {
    // Safe multiplication for integers - check for overflow
    // TODO: What condition checks for multiplication overflow?
    if (a != 0 && b > std::numeric_limits<T>::max() / a) {
        throw std::overflow_error("Multiplication overflow");
    }
    return a * b;
}

template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, T>
multiply_safe(T a, T b) {
    // Different logic for floating-point types
    // TODO: Why don't we need overflow checking for floating-point?
    return a * b;  // Floating-point handles overflow differently (to infinity)
}

// Multiple conditions
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T> && sizeof(T) >= 4, T>
process_large_numbers(T value) {
    // Process only large arithmetic types
    return value * 2;
}

// Tag Dispatching Pattern
struct random_access_tag {};
struct bidirectional_tag {};
struct forward_tag {};

// Implementation for random access iterators
template<typename Iterator>
void advance_impl(Iterator& it, int n, random_access_tag) {
    it += n;  // O(1) operation
}

// Implementation for bidirectional iterators
template<typename Iterator>
void advance_impl(Iterator& it, int n, bidirectional_tag) {
    if (n >= 0) {
        while (n--) ++it;
    } else {
        while (n++) --it;
    }
}

// Implementation for forward iterators
template<typename Iterator>
void advance_impl(Iterator& it, int n, forward_tag) {
    // Only forward movement allowed
    while (n-- > 0) ++it;
}

// Public interface that dispatches based on iterator category
template<typename Iterator>
void advance(Iterator& it, int n) {
    using category = typename std::iterator_traits<Iterator>::iterator_category;
    advance_impl(it, n, category{});
}

// Policy-Based Design Pattern
// Memory management policies
struct heap_policy {
    template<typename T>
    static T* allocate(size_t n) {
        return new T[n];
    }

    template<typename T>
    static void deallocate(T* ptr) {
        delete[] ptr;
    }
};

struct stack_policy {
    template<typename T>
    static T* allocate(size_t n) {
        // TODO: Replace static_assert with runtime check since n is a runtime parameter
        // Hint: What should we do if the requested size is too large?
        if (n > 1024) {
            throw std::length_error("Stack allocation too large");
        }
        thread_local static T buffer[1024];
        return buffer;
    }

    template<typename T>
    static void deallocate(T* ptr) {
        // Nothing to do for stack allocation
    }
};

// Threading policies
struct single_threaded_policy {
    template<typename Func>
    static void parallel_for(size_t start, size_t end, Func&& f) {
        for (size_t i = start; i < end; ++i) {
            f(i);
        }
    }
};

// Container that uses policies
template<typename T,
         typename MemoryPolicy = heap_policy,
         typename ThreadingPolicy = single_threaded_policy>
class PolicyBasedContainer {
private:
    T* data;
    size_t capacity;
    size_t size_;

public:
    PolicyBasedContainer(size_t cap) : capacity(cap), size_(0) {
        data = MemoryPolicy::allocate<T>(capacity);
    }

    ~PolicyBasedContainer() {
        MemoryPolicy::deallocate(data);
    }

    void fill(const T& value) {
        ThreadingPolicy::parallel_for(0, capacity, [this, &value](size_t i) {
            data[i] = value;
        });
        size_ = capacity;
    }

    T& operator[](size_t idx) { return data[idx]; }
    const T& operator[](size_t idx) const { return data[idx]; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity; }
};

void exercise_tmp_patterns() {
    std::cout << "\n=== Exercise 1: Template Metaprogramming Patterns ===" << std::endl;

    // Enable-If pattern
    int safe_mult = multiply_safe(10, 20);
    double float_mult = multiply_safe(3.14, 2.0);
    std::cout << "Safe multiplication (int): " << safe_mult << std::endl;
    std::cout << "Safe multiplication (float): " << float_mult << std::endl;

    // Large number processing
    long large_num = process_large_numbers(42L);
    std::cout << "Processed large number: " << large_num << std::endl;

    // Tag dispatching
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin();
    advance(it, 2);  // Advance by 2 positions
    std::cout << "After advancing iterator by 2: " << *it << std::endl;

    // Policy-based design
    PolicyBasedContainer<int, heap_policy, single_threaded_policy> heap_container(5);
    heap_container.fill(42);
    std::cout << "Heap container [0]: " << heap_container[0] << ", size: " << heap_container.size() << std::endl;
}

/*
 * EXERCISE 2: EXPRESSION TEMPLATES
 * Implementing lazy evaluation for mathematical expressions
 */
// Base expression template
template<typename Derived>
struct Expression {
    const Derived& impl() const { return static_cast<const Derived&>(*this); }

    template<typename T>
    auto operator+(const T& other) const {
        return AddExpr<Derived, T>(impl(), other);
    }

    template<typename T>
    auto operator*(const T& other) const {
        return MultExpr<Derived, T>(impl(), other);
    }
};

// Scalar value expression
template<typename T>
struct ScalarExpr : Expression<ScalarExpr<T>> {
    T value;

    ScalarExpr(T v) : value(v) {}

    auto operator[](size_t) const { return value; }
    size_t size() const { return 1; }
};

// Vector expression
template<typename T>
struct VectorExpr : Expression<VectorExpr<T>> {
    std::vector<T> data;

    VectorExpr(std::initializer_list<T> init) : data(init) {}
    VectorExpr(size_t size, T init_val = T{}) : data(size, init_val) {}

    auto operator[](size_t i) const { return data[i]; }
    auto operator[](size_t i) { return data[i]; }
    size_t size() const { return data.size(); }
};

// Addition expression
template<typename Left, typename Right>
struct AddExpr : Expression<AddExpr<Left, Right>> {
    const Left& left;
    const Right& right;

    AddExpr(const Left& l, const Right& r) : left(l), right(r) {}

    auto operator[](size_t i) const {
        return left[i % left.size()] + right[i % right.size()];
    }

    size_t size() const {
        return std::max(left.size(), right.size());
    }
};

// Multiplication expression
template<typename Left, typename Right>
struct MultExpr : Expression<MultExpr<Left, Right>> {
    const Left& left;
    const Right& right;

    MultExpr(const Left& l, const Right& r) : left(l), right(r) {}

    auto operator[](size_t i) const {
        return left[i % left.size()] * right[i % right.size()];
    }

    size_t size() const {
        return std::max(left.size(), right.size());
    }
};

// Evaluation function
template<typename Expr>
auto evaluate(const Expression<Expr>& expr) {
    const auto& e = expr.impl();
    VectorExpr<typename std::decay_t<decltype(e[0])>> result(e.size());
    for (size_t i = 0; i < e.size(); ++i) {
        result[i] = e[i];
    }
    return result;
}

void exercise_expression_templates() {
    std::cout << "\n=== Exercise 2: Expression Templates ===" << std::endl;

    VectorExpr<double> v1{1.0, 2.0, 3.0};
    VectorExpr<double> v2{4.0, 5.0, 6.0};
    ScalarExpr<double> s(2.0);

    // This creates an expression tree, no computation yet
    auto expr = v1 + v2 * s;

    // Now the computation happens
    auto result = evaluate(expr);

    std::cout << "Result of v1 + v2 * s: ";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";  // Should print: 9 12 15
    }
    std::cout << std::endl;
}

/*
 * EXERCISE 1: TEMPLATE METAPROGRAMMING PATTERNS
 * Implementing common TMP patterns like Enable-if, Tag dispatching, and Policy-based design
 * 
 * LEARNING OBJECTIVES:
 * - Understand how to use SFINAE (Substitution Failure Is Not An Error) with std::enable_if
 * - Learn tag dispatching for selecting different implementations based on types
 * - See how policy-based design enables customizable behavior through template parameters
 */
// TODO: Implement safe multiplication that handles different types differently
// Hint: Use std::enable_if to create type-specific overloads
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T>
multiply_safe(T a, T b) {
    // Safe multiplication for integers - check for overflow
    // TODO: What condition checks for multiplication overflow?
    if (a != 0 && b > std::numeric_limits<T>::max() / a) {
        throw std::overflow_error("Multiplication overflow");
    }
    return a * b;
}

template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, T>
multiply_safe(T a, T b) {
    // Different logic for floating-point types
    // TODO: Why don't we need overflow checking for floating-point?
    return a * b;  // Floating-point handles overflow differently (to infinity)
}

// Multiple conditions
// TODO: Implement a function that only works for large arithmetic types (>= 4 bytes)
// Hint: Combine multiple type traits with && in std::enable_if
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T> && sizeof(T) >= 4, T>
process_large_numbers(T value) {
    // Process only large arithmetic types
    // TODO: Why would we want to restrict this to only large types?
    return value * 2;
}

// Tag Dispatching Pattern
// TODO: Define tags for different iterator categories
// Hint: These are empty structs used purely for type dispatching
struct random_access_tag {};
struct bidirectional_tag {};
struct forward_tag {};

// Implementation for random access iterators
// TODO: Complete this implementation for random access iterators (should be O(1))
// Hint: Random access iterators can jump directly to any position
template<typename Iterator>
void advance_impl(Iterator& it, int n, random_access_tag) {
    // TODO: Implement random access iterator advancement (should be O(1))
    // For random access iterators, we can just add n to the iterator
    it += n;  // O(1) operation
}

// Implementation for bidirectional iterators
// TODO: Complete this implementation for bidirectional iterators
// Hint: Bidirectional iterators can move both forward and backward, but only one step at a time
template<typename Iterator>
void advance_impl(Iterator& it, int n, bidirectional_tag) {
    if (n >= 0) {
        // TODO: Implement forward traversal (move forward n steps)
        while (n--) ++it;
    } else {
        // TODO: Implement backward traversal (move backward |n| steps)
        while (n++) --it;
    }
}

// Implementation for forward iterators
// TODO: Complete this implementation for forward iterators (can only move forward)
// Hint: Forward iterators can only move forward, one step at a time
template<typename Iterator>
void advance_impl(Iterator& it, int n, forward_tag) {
    // TODO: Implement forward-only movement
    // Only forward movement allowed, and we need to check n > 0
    while (n-- > 0) ++it;
}

// Public interface that dispatches based on iterator category
// TODO: Understand how this selects the right implementation based on iterator type
// Hint: std::iterator_traits::iterator_category gives us the category of an iterator
template<typename Iterator>
void advance(Iterator& it, int n) {
    // TODO: Get the iterator category tag for dispatch
    // This extracts the category tag from the iterator type
    using category = typename std::iterator_traits<Iterator>::iterator_category;
    // TODO: Dispatch to the appropriate implementation based on category
    // The category{} creates a temporary object of the iterator's category type
    advance_impl(it, n, category{});
}

// Policy-Based Design Pattern
// Memory management policies
// TODO: Understand how policies encapsulate different behaviors that can be swapped
struct heap_policy {
    template<typename T>
    static T* allocate(size_t n) {
        return new T[n];
    }

    template<typename T>
    static void deallocate(T* ptr) {
        delete[] ptr;
    }
};

struct stack_policy {
    template<typename T>
    static T* allocate(size_t n) {
        // TODO: Replace static_assert with runtime check since n is a runtime parameter
        // Hint: What should we do if the requested size is too large?
        if (n > 1024) {
            throw std::length_error("Stack allocation too large");
        }
        thread_local static T buffer[1024];
        return buffer;
    }

    template<typename T>
    static void deallocate(T* ptr) {
        // Nothing to do for stack allocation
    }
};

// Threading policies
// TODO: Understand how threading policies can enable different execution strategies
struct single_threaded_policy {
    template<typename Func>
    static void parallel_for(size_t start, size_t end, Func&& f) {
        for (size_t i = start; i < end; ++i) {
            f(i);
        }
    }
};

// Container that uses policies
// TODO: Understand how policies customize behavior through template parameters
// This is the core of policy-based design: behavior is customized by template parameters
template<typename T,
         typename MemoryPolicy = heap_policy,
         typename ThreadingPolicy = single_threaded_policy>
class PolicyBasedContainer {
private:
    T* data;
    size_t capacity;
    size_t size_;

public:
    PolicyBasedContainer(size_t cap) : capacity(cap), size_(0) {
        // TODO: Use the memory policy to allocate storage
        // This is where the policy customizes behavior - different policies allocate differently
        data = MemoryPolicy::allocate<T>(capacity);
    }

    ~PolicyBasedContainer() {
        // TODO: Use the memory policy to deallocate storage
        // Again, the policy customizes how deallocation happens
        MemoryPolicy::deallocate(data);
    }

    void fill(const T& value) {
        // TODO: Use the threading policy to parallelize the fill operation
        // The threading policy determines how (or if) the loop is parallelized
        ThreadingPolicy::parallel_for(0, capacity, [this, &value](size_t i) {
            data[i] = value;
        });
        size_ = capacity;
    }

    T& operator[](size_t idx) { return data[idx]; }
    const T& operator[](size_t idx) const { return data[idx]; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity; }
};

template<int A>
struct GCD<A, 0> {
    static constexpr int value = A;
};

// Compile-time least common multiple
template<int A, int B>
struct LCM {
    static constexpr int value = (A * B) / GCD<A, B>::value;
};

// Compile-time string hashing
template<size_t N>
struct StrHash {
    static constexpr uint64_t hash(const char (&str)[N], uint64_t seed = 0xcbf29ce484222325ULL) {
        uint64_t hash = seed;
        for (size_t i = 0; i < N - 1; ++i) {
            hash ^= str[i];
            hash *= 0x100000001b3ULL;
        }
        return hash;
    }
};

// Compile-time type list
template<typename... Ts>
struct TypeList {
    static constexpr size_t size = sizeof...(Ts);
};

// Type list operations
template<typename List, typename T>
struct Append;

template<typename... Ts, typename T>
struct Append<TypeList<Ts...>, T> {
    using type = TypeList<Ts..., T>;
};

template<typename List, typename T>
using Append_t = typename Append<List, T>::type;

// Find type in type list
template<typename List, typename T>
struct Contains;

template<typename T>
struct Contains<TypeList<>, T> : std::false_type {};

template<typename U, typename... Ts, typename T>
struct Contains<TypeList<U, Ts...>, T> : Contains<TypeList<Ts...>, T> {};

template<typename T, typename... Ts>
struct Contains<TypeList<T, Ts...>, T> : std::true_type {};

void exercise_compile_time_computations() {
    std::cout << "\n=== Exercise 3: Compile-Time Computations ===" << std::endl;

    // GCD and LCM
    constexpr int gcd_result = GCD<48, 18>::value;
    constexpr int lcm_result = LCM<12, 18>::value;
    std::cout << "GCD(48, 18): " << gcd_result << std::endl;
    std::cout << "LCM(12, 18): " << lcm_result << std::endl;

    // String hashing
    constexpr uint64_t hash_result = StrHash<6>::hash("hello");
    std::cout << "Hash of 'hello': " << hash_result << std::endl;

    // Type list operations
    using MyTypes = TypeList<int, float, double, char, bool>;
    std::cout << "TypeList size: " << MyTypes::size << std::endl;

    using ExtendedTypes = Append_t<MyTypes, long>;
    std::cout << "Extended TypeList size: " << ExtendedTypes::size << std::endl;

    std::cout << "Contains<int> in MyTypes: " << Contains<MyTypes, int>::value << std::endl;
    std::cout << "Contains<std::string> in MyTypes: " << Contains<MyTypes, std::string>::value << std::endl;
}

/*
 * EXERCISE 4: TYPE LISTS AND OPERATIONS
 * Working with type lists and performing operations on them
 */
// Basic type list operations
template<typename List>
struct Length;

template<typename... Ts>
struct Length<TypeList<Ts...>> {
    static constexpr size_t value = sizeof...(Ts);
};

template<typename List>
struct Front;

template<typename T, typename... Ts>
struct Front<TypeList<T, Ts...>> {
    using type = T;
};

template<typename List>
struct PopFront;

template<typename T, typename... Ts>
struct PopFront<TypeList<T, Ts...>> {
    using type = TypeList<Ts...>;
};

template<typename List, typename T>
struct PushBack;

template<typename... Ts, typename T>
struct PushBack<TypeList<Ts...>, T> {
    using type = TypeList<Ts..., T>;
};

// Transform operation on type list
template<typename List, template<typename> typename F>
struct Transform;

template<typename... Ts, template<typename> typename F>
struct Transform<TypeList<Ts...>, F> {
    using type = TypeList<F<Ts>...>;
};

// Filter operation on type list
template<typename List, template<typename> typename Predicate>
struct Filter;

template<template<typename> typename Predicate>
struct Filter<TypeList<>, Predicate> {
    using type = TypeList<>;
};

template<typename T, typename... Ts, template<typename> typename Predicate>
struct Filter<TypeList<T, Ts...>, Predicate> {
    using rest = typename Filter<TypeList<Ts...>, Predicate>::type;
    using type = std::conditional_t<
        Predicate<T>::value,
        typename PushBack<rest, T>::type,
        rest
    >;
};

// Example predicates
template<typename T>
struct IsIntegral {
    static constexpr bool value = std::is_integral_v<T>;
};

template<typename T>
struct IsFloatingPoint {
    static constexpr bool value = std::is_floating_point_v<T>;
};

void exercise_type_lists() {
    std::cout << "\n=== Exercise 4: Type Lists and Operations ===" << std::endl;

    using MyTypes = TypeList<int, float, double, char, bool>;
    std::cout << "Original TypeList length: " << Length<MyTypes>::value << std::endl;

    using FrontType = typename Front<MyTypes>::type;
    std::cout << "Front type is integral: " << IsIntegral<FrontType>::value << std::endl;

    using RestTypes = typename PopFront<MyTypes>::type;
    std::cout << "Rest TypeList length: " << Length<RestTypes>::value << std::endl;

    using ExtendedTypes = typename PushBack<RestTypes, long>::type;
    std::cout << "Extended TypeList length: " << Length<ExtendedTypes>::value << std::endl;

    using IntegralTypes = typename Filter<MyTypes, IsIntegral>::type;
    std::cout << "Integral types count: " << Length<IntegralTypes>::value << std::endl;

    using TransformedTypes = typename Transform<IntegralTypes, std::add_const>::type;
    std::cout << "Transformed types count: " << Length<TransformedTypes>::value << std::endl;
}

/*
 * EXERCISE 5: HIGHER-ORDER TEMPLATE FUNCTIONS
 * Templates that operate on other templates
 */
// Higher-order function: applies a template to each type in a type list
template<template<typename> typename F, typename... Ts>
constexpr auto apply_to_types(TypeList<Ts...>) {
    return TypeList<F<Ts>...>{};
}

// Higher-order function: combines two types
template<template<typename, typename> typename F, typename... Ts, typename... Us>
constexpr auto zip_with(TypeList<Ts...>, TypeList<Us...>) {
    static_assert(sizeof...(Ts) == sizeof...(Us), "Type lists must have same size");
    return TypeList<F<Ts, Us>...>{};
}

// Example: create pointer types for each type in list
template<typename T>
struct MakePointer {
    using type = T*;
};

// Example: pair two types together
template<typename T, typename U>
struct TypePair {
    using first = T;
    using second = U;
};

void exercise_higher_order() {
    std::cout << "\n=== Exercise 5: Higher-Order Template Functions ===" << std::endl;

    using OriginalTypes = TypeList<int, float, double>;
    using PointerTypes = decltype(apply_to_types<MakePointer>(OriginalTypes{}));
    std::cout << "Original types count: " << Length<OriginalTypes>::value << std::endl;
    std::cout << "Pointer types count: " << Length<PointerTypes>::value << std::endl;

    using SecondTypes = TypeList<char, bool, long>;
    using PairedTypes = decltype(zip_with<TypePair>(OriginalTypes{}, SecondTypes{}));
    std::cout << "Paired types count: " << Length<PairedTypes>::value << std::endl;
}

/*
 * EXERCISE 6: TEMPLATE ALIASING AND TYPE MANIPULATION
 * Using template aliases to simplify complex type manipulations
 */
// Template aliases for complex types
template<typename T>
using Vec = std::vector<T>;

template<typename T>
using VecVec = std::vector<std::vector<T>>;

// Conditional type aliasing
template<bool Condition, typename TrueType, typename FalseType>
using conditional_t = std::conditional_t<Condition, TrueType, FalseType>;

// Remove reference and const
template<typename T>
using remove_cref = std::remove_const_t<std::remove_reference_t<T>>;

// Complex type manipulation
template<typename T>
struct SmartTypeSelector {
    using type = conditional_t<
        std::is_integral_v<T>,
        int,
        conditional_t<
            std::is_floating_point_v<T>,
            double,
            std::string
        >
    >;
};

template<typename T>
using SmartType = typename SmartTypeSelector<T>::type;

void exercise_template_aliasing() {
    std::cout << "\n=== Exercise 6: Template Aliasing and Type Manipulation ===" << std::endl;

    Vec<int> int_vec = {1, 2, 3};
    std::cout << "Vector size: " << int_vec.size() << std::endl;

    using IntSmartType = SmartType<int>;
    using DoubleSmartType = SmartType<double>;
    using StringSmartType = SmartType<std::string>;

    std::cout << "SmartType<int> is int: " << std::is_same_v<IntSmartType, int> << std::endl;
    std::cout << "SmartType<double> is double: " << std::is_same_v<DoubleSmartType, double> << std::endl;
    std::cout << "SmartType<string> is string: " << std::is_same_v<StringSmartType, std::string> << std::endl;
}

/*
 * HANDS-ON CHALLENGES
 * Try implementing these exercises yourself
 */

// Challenge 1: Matrix Expression Templates
#include <vector>

// Base matrix expression
template<typename Derived>
struct MatrixExpression {
    const Derived& impl() const { return static_cast<const Derived&>(*this); }

    size_t rows() const { return impl().rows(); }
    size_t cols() const { return impl().cols(); }

    template<typename Other>
    auto operator+(const MatrixExpression<Other>& other) const {
        return MatrixAddExpr<Derived, Other>(impl(), other.impl());
    }

    template<typename Scalar>
    auto operator*(const Scalar& scalar) const {
        return MatrixScalarMultExpr<Derived, Scalar>(impl(), scalar);
    }
};

// Dense matrix implementation
template<typename T>
struct DenseMatrix : MatrixExpression<DenseMatrix<T>> {
    std::vector<std::vector<T>> data;
    size_t rows_, cols_;

    DenseMatrix(size_t r, size_t c, T init_val = T{})
        : rows_(r), cols_(c), data(r, std::vector<T>(c, init_val)) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T& operator()(size_t i, size_t j) { return data[i][j]; }
    const T& operator()(size_t i, size_t j) const { return data[i][j]; }
};

// Matrix addition expression
template<typename Left, typename Right>
struct MatrixAddExpr : MatrixExpression<MatrixAddExpr<Left, Right>> {
    const Left& left;
    const Right& right;

    MatrixAddExpr(const Left& l, const Right& r) : left(l), right(r) {
        if (l.rows() != r.rows() || l.cols() != r.cols()) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
    }

    size_t rows() const { return left.rows(); }
    size_t cols() const { return left.cols(); }

    auto operator()(size_t i, size_t j) const {
        return left(i, j) + right(i, j);
    }
};

// Matrix-scalar multiplication expression
template<typename Matrix, typename Scalar>
struct MatrixScalarMultExpr : MatrixExpression<MatrixScalarMultExpr<Matrix, Scalar>> {
    const Matrix& matrix;
    const Scalar& scalar;

    MatrixScalarMultExpr(const Matrix& m, const Scalar& s) : matrix(m), scalar(s) {}

    size_t rows() const { return matrix.rows(); }
    size_t cols() const { return matrix.cols(); }

    auto operator()(size_t i, size_t j) const {
        return matrix(i, j) * scalar;
    }
};

// Evaluation function
template<typename T, typename Expr>
void evaluate_to(DenseMatrix<T>& result, const MatrixExpression<Expr>& expr) {
    const auto& e = expr.impl();
    if (result.rows() != e.rows() || result.cols() != e.cols()) {
        result = DenseMatrix<T>(e.rows(), e.cols());
    }

    for (size_t i = 0; i < e.rows(); ++i) {
        for (size_t j = 0; j < e.cols(); ++j) {
            result(i, j) = e(i, j);
        }
    }
}

// Challenge 2: Type List Algorithms
// Concatenate two type lists
template<typename List1, typename List2>
struct Concat;

template<typename... Ts, typename... Us>
struct Concat<TypeList<Ts...>, TypeList<Us...>> {
    using type = TypeList<Ts..., Us...>;
};

template<typename List1, typename List2>
using Concat_t = typename Concat<List1, List2>::type;

// Find index of type in type list
template<typename List, typename T>
struct IndexOf;

template<typename T, typename... Ts>
struct IndexOf<TypeList<T, Ts...>, T> {
    static constexpr size_t value = 0;
};

template<typename U, typename T, typename... Ts>
struct IndexOf<TypeList<U, Ts...>, T> {
    static constexpr size_t value = 1 + IndexOf<TypeList<Ts...>, T>::value;
};

template<typename T>
struct IndexOf<TypeList<>, T> {
    static constexpr size_t value = static_cast<size_t>(-1);  // Not found
};

// Challenge 3: Policy-Based Matrix Class
// Storage policies
struct HeapStoragePolicy {
    template<typename T>
    static std::vector<T> create_storage(size_t size) {
        return std::vector<T>(size);
    }
};

// Layout policies
struct RowMajorLayout {
    size_t operator()(size_t rows, size_t cols, size_t i, size_t j) const {
        return i * cols + j;
    }
};

// Policy-based matrix
template<
    typename T,
    typename StoragePolicy = HeapStoragePolicy,
    typename LayoutPolicy = RowMajorLayout
>
class PolicyMatrix {
private:
    std::vector<T> data;
    size_t rows_;
    size_t cols_;
    LayoutPolicy layout_;

public:
    PolicyMatrix(size_t rows, size_t cols)
        : data(StoragePolicy::template create_storage<T>(rows * cols)),
          rows_(rows), cols_(cols) {}

    T& operator()(size_t i, size_t j) {
        return data[layout_(rows_, cols_, i, j)];
    }

    const T& operator()(size_t i, size_t j) const {
        return data[layout_(rows_, cols_, i, j)];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
};

int main() {
    std::cout << "Module 4: Advanced Template Metaprogramming Techniques Practice - Hands-on Tutorial" << std::endl;

    // Run all exercises
    exercise_tmp_patterns();
    exercise_expression_templates();
    exercise_compile_time_computations();
    exercise_type_lists();
    exercise_higher_order();
    exercise_template_aliasing();

    // Try the challenges
    std::cout << "\n=== Challenge Solutions ===" << std::endl;

    // Challenge 1: Matrix expression templates
    std::cout << "\nChallenge 1 - Matrix Expression Templates:" << std::endl;
    DenseMatrix<double> mat1(2, 2, 1.0);  // Initialize 2x2 matrix with 1.0
    DenseMatrix<double> mat2(2, 2, 2.0);  // Initialize 2x2 matrix with 2.0

    // Fill matrices with specific values
    mat1(0, 0) = 1.0; mat1(0, 1) = 2.0;
    mat1(1, 0) = 3.0; mat1(1, 1) = 4.0;

    mat2(0, 0) = 5.0; mat2(0, 1) = 6.0;
    mat2(1, 0) = 7.0; mat2(1, 1) = 8.0;

    // Create expression: mat1 + mat2 * 2.0
    auto expr = mat1 + mat2 * 2.0;

    DenseMatrix<double> result(2, 2);
    evaluate_to(result, expr);

    std::cout << "Result of mat1 + mat2 * 2:" << std::endl;
    for (size_t i = 0; i < result.rows(); ++i) {
        for (size_t j = 0; j < result.cols(); ++j) {
            std::cout << result(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // Challenge 2: Type list algorithms
    std::cout << "\nChallenge 2 - Type List Algorithms:" << std::endl;
    using List1 = TypeList<int, float>;
    using List2 = TypeList<double, char>;
    using Concatenated = Concat_t<List1, List2>;
    std::cout << "Concatenated list size: " << Length<Concatenated>::value << std::endl;

    std::cout << "Index of float in List1: " << IndexOf<List1, float>::value << std::endl;
    std::cout << "Index of double in List1: " << IndexOf<List1, double>::value << std::endl;

    // Challenge 3: Policy-based matrix
    std::cout << "\nChallenge 3 - Policy-Based Matrix:" << std::endl;
    PolicyMatrix<int, HeapStoragePolicy, RowMajorLayout> policy_mat(2, 3);
    policy_mat(0, 0) = 10;
    policy_mat(0, 1) = 20;
    policy_mat(1, 2) = 30;

    std::cout << "Policy matrix (2x3) values:" << std::endl;
    for (size_t i = 0; i < policy_mat.rows(); ++i) {
        for (size_t j = 0; j < policy_mat.cols(); ++j) {
            std::cout << policy_mat(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}