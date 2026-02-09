#include <iostream>
#include <string>
#include "calculator.h"

int main() {
    calc::Calculator calc;
    
    std::cout << "Advanced Calculator Demo" << std::endl;
    
    try {
        double result1 = calc.add(10.5, 5.3);
        std::cout << "10.5 + 5.3 = " << result1 << std::endl;
        
        double result2 = calc.multiply(result1, 2.0);
        std::cout << result1 << " * 2.0 = " << result2 << std::endl;
        
        double result3 = calc.divide(result2, 3.0);
        std::cout << result2 << " / 3.0 = " << result3 << std::endl;
        
        // Show history
        std::cout << "\nCalculation History:" << std::endl;
        for (const auto& entry : calc.get_history()) {
            std::cout << "  " << entry << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}