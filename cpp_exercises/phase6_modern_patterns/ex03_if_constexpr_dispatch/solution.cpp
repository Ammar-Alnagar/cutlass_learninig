// SOLUTION: ex03_if_constexpr_dispatch
// Complete implementation with if constexpr

#include <iostream>
#include <type_traits>
#include <string>
#include <iomanip>

// Generic process function using if constexpr dispatch
template<typename T>
void process_value(T value) {
    // Key insight: if constexpr discards the false branch at compile time
    // This means code that wouldn't compile for type T is never compiled
    if constexpr (std::is_integral_v<T>) {
        // Only compiled when T is integral (int, long, etc.)
        std::cout << "Integral: " << value << "\n";
    } else if constexpr (std::is_floating_point_v<T>) {
        // Only compiled when T is floating-point (float, double)
        std::cout << "Float: " << std::fixed << std::setprecision(2) << value << "\n";
    } else if constexpr (std::is_same_v<T, std::string>) {
        // Only compiled when T is exactly std::string
        std::cout << "String: " << value << "\n";
    } else {
        // Fallback for all other types
        std::cout << "Unknown type\n";
    }
}

// Generic serialize function
template<typename T>
std::string serialize(T value) {
    if constexpr (std::is_arithmetic_v<T>) {
        // Arithmetic = integral OR floating-point
        return std::to_string(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return value;
    } else {
        return "[unserializable]";
    }
}

// Type-specific kernel launcher simulation
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

// Compile-time type name printer
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
    
    std::cout << "\n=== Kernel Dispatch ===\n";
    launch_kernel<float>("gemm");
    launch_kernel<__half>("gemm");
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "if constexpr vs regular if:\n";
    std::cout << "  - Regular if: both branches must compile\n";
    std::cout << "  - if constexpr: false branch is DISCARDED (not compiled)\n";
    std::cout << "\nExample:\n";
    std::cout << "  template<typename T>\n";
    std::cout << "  void foo(T x) {\n";
    std::cout << "    if constexpr (is_float<T>) x.sin();  // Only compiles for float\n";
    std::cout << "    if (is_float<T>) x.sin();            // ERROR: int has no sin()\n";
    std::cout << "  }\n";
    
    return 0;
}

// KEY_INSIGHT:
// if constexpr (C++17) enables compile-time branching in templates.
//
// Key difference from regular if:
// - Regular if: BOTH branches must be valid code
// - if constexpr: FALSE BRANCH IS DISCARDED (never compiled)
//
// This enables type-specific code without SFINAE or overloading:
//   template<typename T>
//   void process(T x) {
//       if constexpr (std::is_integral_v<T>) {
//           // Integer-specific code
//       } else if constexpr (std::is_floating_point_v<T>) {
//           // Float-specific code
//       }
//   }
//
// CUDA mapping: Type dispatch for kernel launchers:
//   template<typename T>
//   void launchGemm() {
//       if constexpr (std::is_same_v<T, float>) {
//           launchFp32Kernel();
//       } else if constexpr (std::is_same_v<T, __half>) {
//           launchFp16Kernel();  // Uses Tensor Cores
//       }
//   }
//
// CUTLASS 3.x uses if constexpr extensively for cleaner code
// compared to CUTLASS 2.x SFINAE-based dispatch.
