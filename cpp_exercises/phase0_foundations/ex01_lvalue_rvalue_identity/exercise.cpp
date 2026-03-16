// CONCEPT: lvalue vs rvalue identity — addressability test
// FORMAT: SCAFFOLD
// TIME_TARGET: 10 min
// WHY_THIS_MATTERS: You cannot understand move semantics without knowing what can be moved.
// CUDA_CONNECTION: Kernel launch arguments — temporaries vs named variables in device code.

#include <iostream>
#include <utility>

// Rule: If you can take the address of it with &, it is an lvalue.
// If you cannot take the address, it is an rvalue.

void print_category(const char* label, int& ref) {
    std::cout << label << " is an lvalue, address = " << &ref << "\n";
}

void print_category(const char* label, int&& ref) {
    std::cout << label << " is an rvalue reference (bound to rvalue), address = " << &ref << "\n";
}

int get_value() {
    return 42;
}

int main() {
    int x = 10;
    
    // TODO 1: Print address of x (named variable)
    // Use: std::cout << "x: " << &x << "\n";
    std::cout << "x: " << &x << "\n";
    
    // TODO 2: Print address of literal 42
    // This will NOT compile — uncomment and observe the error:
    // std::cout << "42: " << &42 << "\n";
    
    // TODO 3: Print address of function return value
    // This will NOT compile — uncomment and observe the error:
    // std::cout << "get_value(): " << &get_value() << "\n";
    
    // TODO 4: Call print_category with x (lvalue)
    // The lvalue overload should be selected
    print_category("x", x);
    
    // TODO 5: Call print_category with std::move(x)
    // std::move casts x to an rvalue reference type
    // Which overload is selected now?
    print_category("std::move(x)", std::move(x));
    
    // TODO 6: Call print_category with literal 99
    // Literals are rvalues — which overload fires?
    print_category("99", 99);
    
    // TODO 7: Call print_category with get_value()
    // Function return is a temporary (rvalue)
    print_category("get_value()", get_value());
    
    std::cout << "\n=== KEY OBSERVATION ===\n";
    std::cout << "Named variables (x) are lvalues — you can take their address.\n";
    std::cout << "Temporaries (42, get_value()) are rvalues — no address.\n";
    std::cout << "std::move(x) casts a named variable to rvalue reference type.\n";
    
    return 0;
}

// VERIFY: Expected output pattern:
// x: 0x... (some address)
// x is an lvalue, address = 0x...
// std::move(x) is an rvalue reference (bound to rvalue), address = 0x... (same as x)
// 99 is an rvalue reference (bound to rvalue), address = 0x...
// get_value() is an rvalue reference (bound to rvalue), address = 0x...
// 
// Lines with &42 and &get_value() must NOT compile (commented out).
