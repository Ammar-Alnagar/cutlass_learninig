#include <iostream>
#include "mathlib.h"

int main() {
    double a = 5.0, b = 3.0;
    std::cout << a << " + " << b << " = " << mathlib::add(a, b) << std::endl;
    std::cout << a << " * " << b << " = " << mathlib::multiply(a, b) << std::endl;
    return 0;
}