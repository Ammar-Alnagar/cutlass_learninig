// CONCEPT: if constexpr for compile-time dispatch
// FORMAT: IMPLEMENT
// TIME_TARGET: 20 min
// WHY_THIS_MATTERS: if constexpr enables clean template specialization without SFINAE.
// CUDA_CONNECTION: Type-specific kernel dispatch without template explosion.

#include <iostream>
#include <type_traits>
#include <string>

// TODO: Implement a generic process function using if constexpr
// Requirements:
// 1. For integral types: print "Integral: {value}"
// 2. For floating-point types: print "Float: {value}" with 2 decimal places
// 3. For string types: print "String: {value}"
// 4. For all other types: print "Unknown type"

template<typename T>
void process_value(T value) {
    // TODO: Use if constexpr to dispatch based on type traits
    // Pattern:
    //   if constexpr (std::is_integral_v<T>) {
    //       // Integral handling
    //   } else if constexpr (std::is_floating_point_v<T>) {
    //       // Float handling
    //   } else if constexpr (std::is_same_v<T, std::string>) {
    //       // String handling
    //   } else {
    //       // Fallback
    //   }
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integral: " << value << "\n";
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Float: " << std::fixed << std::setprecision(2) << value << "\n";
    } else if constexpr (std::is_same_v<T, std::string>) {
        std::cout << "String: " << value << "\n";
    } else {
        std::cout << "Unknown type\n";
    }
}

// TODO: Implement a generic serialize function
// For arithmetic types: return std::to_string(value)
// For string: return the string as-is
// For other types: return "[unserializable]"
template<typename T>
std::string serialize(T value) {
    if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return value;
    } else {
        return "[unserializable]";
    }
}

// TODO: Implement a type-specific kernel launcher
// This simulates CUDA kernel dispatch based on element type
template<typename T>
void launch_kernel(const char* kernel_name) {
    std::cout << "Launching " << kernel_name << " for ";
    
    if constexpr (std::is_same_v<T, float>) {
        std::cout << "float (FP32 CUDA cores)\n";
    } else if constexpr (std::is_same_v<T, double>) {
        std::cout << "double (FP64 CUDA cores)\n";
    } else if constexpr (std::is_same_v<T, __half>) {
        std::cout << "__half (FP16 Tensor Cores)\n";
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        std::cout << "__nv_bfloat16 (BF16 Tensor Cores)\n";
    } else {
        std::cout << "unknown type (fallback kernel)\n";
    }
}

// TODO: Implement a compile-time type printer
// Use if constexpr to print type name at compile time
template<typename T>
constexpr const char* type_name() {
    if constexpr (std::is_same_v<T, int>) {
        return "int";
    } else if constexpr (std::is_same_v<T, float>) {
        return "float";
    } else if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "std::string";
    } else {
        return "unknown";
    }
}

int main() {
    std::cout << "=== Process Value (if constexpr dispatch) ===\n";
    process_value(42);           // Integral
    process_value(3.14159);      // Float
    process_value(std::string("hello"));  // String
    process_value(42LL);         // Integral (long long)
    
    std::cout << "\n=== Serialize ===\n";
    std::cout << "serialize(42) = " << serialize(42) << "\n";
    std::cout << "serialize(3.14) = " << serialize(3.14) << "\n";
    std::cout << "serialize(hello) = " << serialize(std::string("hello")) << "\n";
    
    std::cout << "\n=== Type Names ===\n";
    std::cout << "type_name<int>() = " << type_name<int>() << "\n";
    std::cout << "type_name<float>() = " << type_name<float>() << "\n";
    std::cout << "type_name<std::string>() = " << type_name<std::string>() << "\n";
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "if constexpr vs regular if:\n";
    std::cout << "  - Regular if: both branches must compile\n";
    std::cout << "  - if constexpr: false branch is DISCARDED (not compiled)\n";
    std::cout << "\nThis enables type-specific code in templates without SFINAE!\n";
    
    return 0;
}

// VERIFY: Expected output:
// === Process Value (if constexpr dispatch) ===
// Integral: 42
// Float: 3.14
// String: hello
// Integral: 42
//
// === Serialize ===
// serialize(42) = 42
// serialize(3.14) = 3.140000
// serialize(hello) = hello
//
// === Type Names ===
// type_name<int>() = int
// type_name<float>() = float
// type_name<std::string>() = std::string
//
// === KEY LEARNING ===
// ...
