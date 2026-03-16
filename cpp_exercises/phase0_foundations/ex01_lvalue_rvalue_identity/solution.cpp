// SOLUTION: ex01_lvalue_rvalue_identity
// Complete implementation with explanations

#include <iostream>
#include <utility>

// Overload resolution is based on the VALUE CATEGORY of the argument,
// not the type. int& binds to lvalues, int&& binds to rvalues.

void print_category(const char* label, int& ref) {
    // This overload is selected when argument is an lvalue
    std::cout << label << " is an lvalue, address = " << &ref << "\n";
}

void print_category(const char* label, int&& ref) {
    // This overload is selected when argument is an rvalue
    // Note: ref itself is a named variable inside this function,
    // so inside the function body, ref is an lvalue!
    std::cout << label << " is an rvalue reference (bound to rvalue), address = " << &ref << "\n";
}

int get_value() {
    return 42;
}

int main() {
    int x = 10;
    
    // TODO 1: Print address of x (named variable)
    // Named variables always have an address — they are lvalues
    std::cout << "x: " << &x << "\n";
    
    // TODO 2: Print address of literal 42
    // CANNOT DO THIS: literals are rvalues, they have no storage location
    // std::cout << "42: " << &42 << "\n";  // ERROR: lvalue required as unary '&' operand
    
    // TODO 3: Print address of function return value
    // CANNOT DO THIS: return value is a temporary (rvalue)
    // std::cout << "get_value(): " << &get_value() << "\n";  // ERROR
    
    // TODO 4: Call print_category with x (lvalue)
    // x is a named variable → lvalue → int& overload selected
    print_category("x", x);
    
    // TODO 5: Call print_category with std::move(x)
    // std::move(x) returns int&& — casts x to rvalue reference type
    // int&& overload is selected, even though x is still the same object
    print_category("std::move(x)", std::move(x));
    
    // TODO 6: Call print_category with literal 99
    // 99 is a prvalue (pure rvalue) → int&& overload selected
    print_category("99", 99);
    
    // TODO 7: Call print_category with get_value()
    // Function returning by value produces a temporary → rvalue
    print_category("get_value()", get_value());
    
    std::cout << "\n=== KEY OBSERVATION ===\n";
    std::cout << "Named variables (x) are lvalues — you can take their address.\n";
    std::cout << "Temporaries (42, get_value()) are rvalues — no address.\n";
    std::cout << "std::move(x) casts a named variable to rvalue reference type.\n";
    
    return 0;
}

// KEY_INSIGHT:
// The single rule: "If it has a name, it is an lvalue."
// 
// This explains everything:
// - x is an lvalue (it has a name)
// - 42 is an rvalue (no name, it's a literal)
// - get_value() is an rvalue (the return value is unnamed/anonymous)
// - std::move(x) produces an rvalue REFERENCE TYPE, but x itself is still named
//
// CUDA mapping: When you pass a kernel argument, the same rules apply.
// A named device pointer is an lvalue. A temporary from a function call
// is an rvalue. Understanding this matters when you write generic CUDA
// wrappers that forward arguments — you need to preserve value category.
