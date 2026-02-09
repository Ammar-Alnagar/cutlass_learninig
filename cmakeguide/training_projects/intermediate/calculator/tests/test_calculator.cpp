#include <iostream>
#include <cassert>
#include "calculator.h"

bool test_add() {
    calc::Calculator calc;
    double result = calc.add(2.0, 3.0);
    bool success = (result == 5.0);
    std::cout << "add(2, 3) = " << result << " (expected: 5) " 
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool test_multiply() {
    calc::Calculator calc;
    double result = calc.multiply(4.0, 5.0);
    bool success = (result == 20.0);
    std::cout << "multiply(4, 5) = " << result << " (expected: 20) " 
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool test_divide() {
    calc::Calculator calc;
    double result = calc.divide(10.0, 2.0);
    bool success = (result == 5.0);
    std::cout << "divide(10, 2) = " << result << " (expected: 5) " 
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool test_divide_by_zero() {
    calc::Calculator calc;
    bool exception_thrown = false;
    try {
        calc.divide(5.0, 0.0);
    } catch (const std::exception&) {
        exception_thrown = true;
    }
    std::cout << "divide by zero throws exception: " 
              << (exception_thrown ? "PASS" : "FAIL") << std::endl;
    return exception_thrown;
}

bool test_power() {
    calc::Calculator calc;
    double result = calc.power(2.0, 3.0);
    bool success = (result == 8.0);
    std::cout << "power(2, 3) = " << result << " (expected: 8) " 
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

int main() {
    int failed = 0;
    
    if (!test_add()) failed++;
    if (!test_multiply()) failed++;
    if (!test_divide()) failed++;
    if (!test_divide_by_zero()) failed++;
    if (!test_power()) failed++;
    
    std::cout << "\nTests run: 5, Failed: " << failed << std::endl;
    
    if (failed == 0) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}